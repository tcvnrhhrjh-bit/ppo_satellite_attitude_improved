import argparse
import csv
import os

import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from baseline_pd import attitude_error_vector, pd_action, update_integral_error
from main import make_demo_start_state, parse_angle_list, set_global_seed
from telemetry_loader import load_telemetry_states, write_telemetry_state_summary
from torque_dynamics import TorqueDynamics
from utils import plot_episode_diagnostics, save_episode_csv


DEFAULT_BENCHMARK_ENV = {
    "add_9": 0.1,
    "integrator": "rk4",
    "actuator_deadzone": 0.012,
    "coulomb_friction": 0.006,
    "viscous_friction": 0.04,
    "actuator_efficiency": 0.82,
    "disturbance_level": 0.004,
        "randomize_disturbance": True,
        "wheel_model": "none",
        "progress_weight": 8.0,
    "success_bonus": 100.0,
    "action_weight": 0.001,
    "omega_weight": 0.05,
    "near_target_weight": 0.15,
    "fine_pointing_weight": 0.40,
    "regression_weight": 3.0,
    "stagnation_weight": 0.03,
}


SCENARIO_PRESETS = {
    "nominal_nonlinear": {},
    "residual_friendly": {
        "actuator_deadzone": 0.018,
        "coulomb_friction": 0.008,
        "viscous_friction": 0.05,
        "actuator_efficiency": 0.76,
        "disturbance_level": 0.008,
        "randomize_disturbance": False,
    },
    "random_disturbance": {
        "actuator_deadzone": 0.014,
        "coulomb_friction": 0.006,
        "viscous_friction": 0.04,
        "actuator_efficiency": 0.82,
        "disturbance_level": 0.006,
        "randomize_disturbance": True,
    },
}


def scenario_env_defaults(name):
    if name not in SCENARIO_PRESETS:
        raise ValueError(f"Unknown scenario: {name}")
    env = dict(DEFAULT_BENCHMARK_ENV)
    env.update(SCENARIO_PRESETS[name])
    return env


def make_env(max_steps, seed, env_kwargs):
    kwargs = dict(env_kwargs)
    kwargs["max_steps"] = max_steps
    kwargs["seed"] = seed
    return TorqueDynamics(0.1, np.array([1.0, 0.0, 0.0, 0.0]), **kwargs)


def nearest_action(action_space, torque):
    distances = np.linalg.norm(action_space - np.asarray(torque).reshape(1, 3), axis=1)
    return action_space[int(np.argmin(distances))].copy()


def ppo_action(observation):
    from ppo_satellite_orientation import actor

    logits = actor(observation.reshape(1, -1))
    action_index = int(np.argmax(logits.numpy()[0]))
    env = TorqueDynamics(0.1, np.array([1.0, 0.0, 0.0, 0.0]))
    return env.action_space[action_index].copy()


def attitude_error_deg_from_observation(observation):
    q = np.asarray(observation[:4], dtype=float)
    norm_q = np.linalg.norm(q)
    if norm_q <= 0:
        return 180.0
    q = q / norm_q
    return float(2.0 * np.rad2deg(np.arccos(np.clip(abs(q[0]), -1.0, 1.0))))


def gated_residual_torque(
    observation,
    u_pd,
    u_residual,
    residual_scale,
    gate_mode="off",
    min_error_deg=0.0,
    alignment_threshold=0.0,
):
    if gate_mode == "off":
        return residual_scale * u_residual

    error_deg = attitude_error_deg_from_observation(observation)
    if error_deg < min_error_deg:
        return np.zeros(3)

    pd_norm = np.linalg.norm(u_pd)
    residual_norm = np.linalg.norm(u_residual)
    if pd_norm <= 1e-12 or residual_norm <= 1e-12:
        return np.zeros(3)

    pd_unit = u_pd / pd_norm
    alignment = float(np.dot(u_residual, pd_unit) / residual_norm)
    if gate_mode == "suppress":
        if alignment < alignment_threshold:
            return np.zeros(3)
        return residual_scale * u_residual
    if gate_mode == "project":
        aligned_magnitude = max(0.0, float(np.dot(u_residual, pd_unit)))
        return residual_scale * aligned_magnitude * pd_unit

    raise ValueError(f"Unknown residual gate mode: {gate_mode}")


def controller_action(
    name,
    env,
    observation,
    residual_scale,
    pd_kp,
    pd_kd,
    pd_ki,
    integral_error,
    torque_limit,
    combined_torque_limit=None,
    residual_gate_mode="off",
    residual_gate_min_error_deg=0.0,
    residual_gate_alignment=0.0,
):
    u_pd = pd_action(
        env,
        observation,
        kp=pd_kp,
        kd=pd_kd,
        ki=pd_ki,
        integral_error=integral_error,
        torque_limit=torque_limit,
    )
    if name == "pd":
        return u_pd

    u_ppo = ppo_action(observation)
    if name == "ppo":
        return u_ppo

    if name == "pd_ppo_residual":
        output_limit = combined_torque_limit if combined_torque_limit is not None else torque_limit
        gated_residual = gated_residual_torque(
            observation,
            u_pd,
            u_ppo,
            residual_scale,
            gate_mode=residual_gate_mode,
            min_error_deg=residual_gate_min_error_deg,
            alignment_threshold=residual_gate_alignment,
        )
        u_total = u_pd + gated_residual
        return np.clip(u_total, -output_limit, output_limit)

    raise ValueError(f"Unknown controller: {name}")


def run_trial(
    controller,
    start_state,
    max_steps,
    seed,
    env_kwargs,
    residual_scale,
    pd_kp,
    pd_kd,
    pd_ki,
    integral_limit,
    torque_limit,
    telemetry_disturbance=None,
    combined_torque_limit=None,
    residual_gate_mode="off",
    residual_gate_min_error_deg=0.0,
    residual_gate_alignment=0.0,
):
    env = make_env(max_steps=max_steps, seed=seed, env_kwargs=env_kwargs)
    if telemetry_disturbance is not None:
        env.telemetry_disturbance = np.asarray(telemetry_disturbance, dtype=float).reshape(3,)
    observation = env.reset(start_state)
    integral_error = np.zeros(3)
    total_return = 0.0
    info = {
        "x": env.history,
        "t": env.t,
        "phi": env.phi_history,
        "reward": env.reward_history,
        "action": env.action_history,
        "effective_action": env.effective_action_history,
        "disturbance": env.disturbance_history,
        "omega_norm": env.omega_norm_history,
        "wheel_saturation": env.wheel_saturation_history,
        "wheel_power": env.wheel_power_history,
    }

    for _ in range(max_steps):
        if controller in {"pd", "pd_ppo_residual"}:
            error_vector = attitude_error_vector(env, observation)
            integral_error = update_integral_error(integral_error, error_vector, env.dt, integral_limit)
        action = controller_action(
            controller,
            env,
            observation,
            residual_scale,
            pd_kp,
            pd_kd,
            pd_ki,
            integral_error,
            torque_limit,
            combined_torque_limit=combined_torque_limit,
            residual_gate_mode=residual_gate_mode,
            residual_gate_min_error_deg=residual_gate_min_error_deg,
            residual_gate_alignment=residual_gate_alignment,
        )
        observation, reward, done, info = env.step(action)
        total_return += reward
        if done:
            break

    actions = np.asarray(info["action"]) if len(info["action"]) else np.zeros((0, 3))
    effective_actions = np.asarray(info.get("effective_action", [])) if len(info.get("effective_action", [])) else np.zeros((0, 3))
    disturbances = np.asarray(info.get("disturbance", [])) if len(info.get("disturbance", [])) else np.zeros((0, 3))
    command_norms = np.linalg.norm(actions, axis=1) if len(actions) else np.zeros(0)
    applied_norms = np.linalg.norm(effective_actions, axis=1) if len(effective_actions) else np.zeros(0)
    disturbance_norms = np.linalg.norm(disturbances, axis=1) if len(disturbances) else np.zeros(0)
    mean_command_norm = float(np.mean(command_norms)) if len(command_norms) else 0.0
    mean_applied_norm = float(np.mean(applied_norms)) if len(applied_norms) else 0.0
    command_to_applied_ratio = mean_applied_norm / mean_command_norm if mean_command_norm > 0 else 0.0
    omega_norm = np.asarray(info["omega_norm"])
    wheel_saturation = np.asarray(info.get("wheel_saturation", []), dtype=bool)
    wheel_power = np.asarray(info.get("wheel_power", []), dtype=float)
    metrics = {
        "controller": controller,
        "return": total_return,
        "final_attitude_error_deg": info["attitude_error_deg"],
        "final_omega_norm": info["omega_norm_current"],
        "success": bool(info["success"]),
        "settling_time_s": info["t"][-1] if info["success"] else np.nan,
        "episode_steps": len(info["reward"]),
        "control_energy": float(np.sum(command_norms ** 2) * env.dt),
        "applied_control_energy": float(np.sum(applied_norms ** 2) * env.dt),
        "mean_command_torque_norm": mean_command_norm,
        "mean_applied_torque_norm": mean_applied_norm,
        "command_to_applied_ratio": float(command_to_applied_ratio),
        "mean_disturbance_torque_norm": float(np.mean(disturbance_norms)) if len(disturbance_norms) else 0.0,
        "peak_omega_norm": float(np.max(omega_norm)) if len(omega_norm) else 0.0,
        "wheel_saturation_fraction": float(np.mean(np.any(wheel_saturation, axis=1))) if wheel_saturation.size else 0.0,
        "mean_wheel_power_w": float(np.mean(wheel_power)) if wheel_power.size else 0.0,
    }
    return metrics, info


def summarize(rows):
    controllers = sorted(set(row["controller"] for row in rows))
    summary = []
    for controller in controllers:
        group = [row for row in rows if row["controller"] == controller]
        success_rate = np.mean([row["success"] for row in group])
        finite_settling = [row["settling_time_s"] for row in group if np.isfinite(row["settling_time_s"])]
        summary.append({
            "controller": controller,
            "trials": len(group),
            "success_rate": success_rate,
            "mean_final_attitude_error_deg": np.mean([row["final_attitude_error_deg"] for row in group]),
            "median_final_attitude_error_deg": np.median([row["final_attitude_error_deg"] for row in group]),
            "mean_final_omega_norm": np.mean([row["final_omega_norm"] for row in group]),
            "mean_control_energy": np.mean([row["control_energy"] for row in group]),
            "mean_applied_control_energy": np.mean([row["applied_control_energy"] for row in group]),
            "mean_command_torque_norm": np.mean([row["mean_command_torque_norm"] for row in group]),
            "mean_applied_torque_norm": np.mean([row["mean_applied_torque_norm"] for row in group]),
            "mean_command_to_applied_ratio": np.mean([row["command_to_applied_ratio"] for row in group]),
            "mean_disturbance_torque_norm": np.mean([row["mean_disturbance_torque_norm"] for row in group]),
            "mean_peak_omega_norm": np.mean([row["peak_omega_norm"] for row in group]),
            "mean_wheel_saturation_fraction": np.mean([row["wheel_saturation_fraction"] for row in group]),
            "mean_wheel_power_w": np.mean([row["mean_wheel_power_w"] for row in group]),
            "mean_settling_time_s": np.mean(finite_settling) if finite_settling else np.nan,
        })
    return summary


def initial_error_deg_from_state(state):
    q = np.asarray(state[:4], dtype=float)
    q = q / np.linalg.norm(q)
    return float(2.0 * np.rad2deg(np.arccos(np.clip(abs(q[0]), -1.0, 1.0))))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path, summary_rows):
    if plt is None:
        print("matplotlib is not installed; skipping benchmark summary plot.")
        return
    controllers = [row["controller"] for row in summary_rows]
    x = np.arange(len(controllers))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)
    panels = [
        ("success_rate", "Success rate"),
        ("mean_final_attitude_error_deg", "Mean final attitude error [deg]"),
        ("mean_control_energy", "Mean control energy"),
        ("mean_peak_omega_norm", "Mean peak angular-rate norm"),
    ]
    for ax, (field, title) in zip(axes, panels):
        values = [row[field] for row in summary_rows]
        ax.bar(x, values, color=["tab:blue", "tab:orange", "tab:green"])
        ax.set_xticks(x)
        ax.set_xticklabels(controllers, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("PD vs PPO vs PD+PPO Residual Benchmark")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PD, PPO, and PD+PPO residual control under nonlinear disturbances.")
    parser.add_argument("--output-dir", default="residual_benchmark_outputs")
    parser.add_argument("--model", default="", help="Optional PPO/residual model weights pickle to load before benchmarking.")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--angles", default="20,30,40,50,60,70")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--residual-gate-mode", choices=["off", "suppress", "project"], default="off")
    parser.add_argument("--residual-gate-min-error-deg", type=float, default=0.0)
    parser.add_argument("--residual-gate-alignment", type=float, default=0.0)
    parser.add_argument("--scenario", choices=sorted(SCENARIO_PRESETS), default="nominal_nonlinear")
    parser.add_argument("--add-9", type=float, default=0.1)
    parser.add_argument("--pd-kp", type=float, default=0.08)
    parser.add_argument("--pd-kd", type=float, default=0.80)
    parser.add_argument("--pd-ki", type=float, default=0.0)
    parser.add_argument("--integral-limit", type=float, default=0.25)
    parser.add_argument("--pd-torque-limit", type=float, default=0.1)
    parser.add_argument("--combined-torque-limit", type=float, default=None)
    parser.add_argument("--disturbance-level", type=float, default=None)
    parser.add_argument("--actuator-deadzone", type=float, default=None)
    parser.add_argument("--coulomb-friction", type=float, default=None)
    parser.add_argument("--viscous-friction", type=float, default=None)
    parser.add_argument("--actuator-efficiency", type=float, default=None)
    parser.add_argument("--wheel-model", choices=["none", "rw-0.01", "rw-0.03"], default="none")
    parser.add_argument("--wheel-torque-limit", type=float, default=None)
    parser.add_argument("--wheel-momentum-limit", type=float, default=None)
    parser.add_argument("--wheel-power-limit", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=None)
    parser.add_argument("--success-bonus", type=float, default=None)
    parser.add_argument("--action-weight", type=float, default=None)
    parser.add_argument("--omega-weight", type=float, default=None)
    parser.add_argument("--near-target-weight", type=float, default=None)
    parser.add_argument("--fine-pointing-weight", type=float, default=None)
    parser.add_argument("--regression-weight", type=float, default=None)
    parser.add_argument("--stagnation-weight", type=float, default=None)
    parser.add_argument("--use-telemetry-reset", action="store_true")
    parser.add_argument("--telemetry-dir", default="")
    parser.add_argument("--telemetry-gyro-unit", choices=["deg/s", "rad/s"], default="deg/s")
    parser.add_argument("--telemetry-scalar-component", choices=["auto", "q0", "q1", "q2", "q3"], default="auto")
    parser.add_argument("--telemetry-min-initial-error", type=float, default=None)
    parser.add_argument("--telemetry-max-initial-error", type=float, default=None)
    parser.add_argument("--use-housekeeping", action="store_true")
    parser.add_argument("--housekeeping-dir", default="")
    parser.add_argument("--housekeeping-wheel-momentum-limit", type=float, default=0.04)
    parser.add_argument("--housekeeping-wheel-speed-reference", type=float, default=6500.0)
    parser.add_argument("--housekeeping-disturbance-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.use_telemetry_reset and not args.telemetry_dir:
        raise ValueError("--telemetry-dir is required when --use-telemetry-reset is enabled.")
    if args.use_housekeeping and not args.use_telemetry_reset:
        raise ValueError("--use-housekeeping requires --use-telemetry-reset so HK rows can be synchronized by clock.")

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model:
        from ppo_satellite_orientation import load_model

        load_model(args.model)
        print(f"Loaded model weights: {args.model}")
    angles = parse_angle_list(args.angles)
    telemetry_states = None
    telemetry_records = None
    telemetry_metadata = None
    telemetry_disturbances = None
    if args.use_telemetry_reset:
        telemetry_states, telemetry_records, telemetry_metadata, telemetry_disturbances = load_telemetry_states(
            args.telemetry_dir,
            gyro_unit=args.telemetry_gyro_unit,
            scalar_component=args.telemetry_scalar_component,
            use_housekeeping=args.use_housekeeping,
            housekeeping_dir=args.housekeeping_dir or args.telemetry_dir,
            housekeeping_wheel_momentum_limit=args.housekeeping_wheel_momentum_limit,
            housekeeping_wheel_speed_reference=args.housekeeping_wheel_speed_reference,
            housekeeping_disturbance_scale=args.housekeeping_disturbance_scale,
            telemetry_min_initial_error=args.telemetry_min_initial_error,
            telemetry_max_initial_error=args.telemetry_max_initial_error,
        )
        write_telemetry_state_summary(
            os.path.join(args.output_dir, "telemetry_benchmark_states.csv"),
            telemetry_records,
            telemetry_metadata,
        )
        print(
            "Loaded telemetry benchmark states: "
            f"{telemetry_metadata['rows']} rows, "
            f"clock {telemetry_metadata['first_clock']} to {telemetry_metadata['last_clock']}, "
            f"scalar={telemetry_metadata['scalar_component']}, "
            f"gyro_unit={telemetry_metadata['gyro_unit']}, "
            f"error_range={telemetry_metadata['min_attitude_error_deg']:.2f}-{telemetry_metadata['max_attitude_error_deg']:.2f} deg, "
            f"housekeeping={telemetry_metadata['use_housekeeping']}"
        )
    env_kwargs = scenario_env_defaults(args.scenario)
    env_kwargs["add_9"] = args.add_9
    optional_overrides = {
        "disturbance_level": args.disturbance_level,
        "actuator_deadzone": args.actuator_deadzone,
        "coulomb_friction": args.coulomb_friction,
        "viscous_friction": args.viscous_friction,
        "actuator_efficiency": args.actuator_efficiency,
        "wheel_model": args.wheel_model,
        "wheel_torque_limit": args.wheel_torque_limit,
        "wheel_momentum_limit": args.wheel_momentum_limit,
        "wheel_power_limit": args.wheel_power_limit,
        "progress_weight": args.progress_weight,
        "success_bonus": args.success_bonus,
        "action_weight": args.action_weight,
        "omega_weight": args.omega_weight,
        "near_target_weight": args.near_target_weight,
        "fine_pointing_weight": args.fine_pointing_weight,
        "regression_weight": args.regression_weight,
        "stagnation_weight": args.stagnation_weight,
    }
    env_kwargs.update({key: value for key, value in optional_overrides.items() if value is not None})

    rows = []
    controllers = ["pd", "ppo", "pd_ppo_residual"]
    for trial in range(args.trials):
        if args.use_telemetry_reset:
            telemetry_index = trial % len(telemetry_states)
            start_state = telemetry_states[telemetry_index].copy()
            telemetry_disturbance = telemetry_disturbances[telemetry_index].copy()
            angle = initial_error_deg_from_state(start_state)
        else:
            angle = angles[trial % len(angles)]
            start_state = make_demo_start_state(angle_deg=angle)
            telemetry_disturbance = None
        for controller in controllers:
            metrics, info = run_trial(
                controller,
                start_state,
                max_steps=args.steps,
                seed=args.seed + trial,
                env_kwargs=env_kwargs,
                residual_scale=args.residual_scale,
                pd_kp=args.pd_kp,
                pd_kd=args.pd_kd,
                pd_ki=args.pd_ki,
                integral_limit=args.integral_limit,
                torque_limit=args.pd_torque_limit,
                telemetry_disturbance=telemetry_disturbance,
                combined_torque_limit=args.combined_torque_limit,
                residual_gate_mode=args.residual_gate_mode,
                residual_gate_min_error_deg=args.residual_gate_min_error_deg,
                residual_gate_alignment=args.residual_gate_alignment,
            )
            metrics["trial"] = trial
            metrics["initial_angle_deg"] = angle
            rows.append(metrics)
            if trial == 0:
                prefix = os.path.join(args.output_dir, controller)
                plot_episode_diagnostics(info, filename=f"{prefix}_diagnostics.pdf")
                save_episode_csv(info, filename=f"{prefix}_history.csv")

    metric_fields = [
        "trial", "controller", "initial_angle_deg", "return", "final_attitude_error_deg",
        "final_omega_norm", "success", "settling_time_s", "episode_steps",
        "control_energy", "applied_control_energy", "mean_command_torque_norm",
        "mean_applied_torque_norm", "command_to_applied_ratio", "mean_disturbance_torque_norm",
        "peak_omega_norm",
        "wheel_saturation_fraction", "mean_wheel_power_w",
    ]
    summary_rows = summarize(rows)
    summary_fields = list(summary_rows[0].keys())

    write_csv(os.path.join(args.output_dir, "benchmark_trials.csv"), rows, metric_fields)
    write_csv(os.path.join(args.output_dir, "benchmark_summary.csv"), summary_rows, summary_fields)
    plot_summary(os.path.join(args.output_dir, "benchmark_summary.pdf"), summary_rows)

    print("Residual-control benchmark summary")
    for row in summary_rows:
        print(
            f"{row['controller']}: success_rate={row['success_rate']:.2f}, "
            f"mean_final_error={row['mean_final_attitude_error_deg']:.3f} deg, "
            f"mean_energy={row['mean_control_energy']:.6f}, "
            f"applied_energy={row['mean_applied_control_energy']:.6f}, "
            f"cmd_tau={row['mean_command_torque_norm']:.6f}, "
            f"applied_tau={row['mean_applied_torque_norm']:.6f}, "
            f"applied/cmd={row['mean_command_to_applied_ratio']:.3f}, "
            f"mean_peak_omega={row['mean_peak_omega_norm']:.6f}, "
            f"wheel_sat={row['mean_wheel_saturation_fraction']:.3f}"
        )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()



