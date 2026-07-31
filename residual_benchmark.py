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

from baseline_pd import pd_action
from main import make_demo_start_state, parse_angle_list, set_global_seed
from torque_dynamics import TorqueDynamics
from utils import plot_episode_diagnostics, save_episode_csv


DEFAULT_BENCHMARK_ENV = {
    "integrator": "rk4",
    "actuator_deadzone": 0.012,
    "coulomb_friction": 0.006,
    "viscous_friction": 0.04,
    "actuator_efficiency": 0.82,
    "disturbance_level": 0.004,
    "randomize_disturbance": True,
    "progress_weight": 8.0,
    "success_bonus": 100.0,
    "action_weight": 0.001,
    "omega_weight": 0.05,
}


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


def controller_action(name, env, observation, residual_scale, pd_kp, pd_kd, torque_limit):
    u_pd = pd_action(env, observation, kp=pd_kp, kd=pd_kd, torque_limit=torque_limit)
    if name == "pd":
        return u_pd

    u_ppo = ppo_action(observation)
    if name == "ppo":
        return u_ppo

    if name == "pd_ppo_residual":
        u_total = u_pd + residual_scale * u_ppo
        return nearest_action(env.action_space, np.clip(u_total, -torque_limit, torque_limit))

    raise ValueError(f"Unknown controller: {name}")


def run_trial(controller, start_state, max_steps, seed, env_kwargs, residual_scale, pd_kp, pd_kd, torque_limit):
    env = make_env(max_steps=max_steps, seed=seed, env_kwargs=env_kwargs)
    observation = env.reset(start_state)
    total_return = 0.0
    info = {
        "x": env.history,
        "t": env.t,
        "phi": env.phi_history,
        "reward": env.reward_history,
        "action": env.action_history,
        "omega_norm": env.omega_norm_history,
    }

    for _ in range(max_steps):
        action = controller_action(controller, env, observation, residual_scale, pd_kp, pd_kd, torque_limit)
        observation, reward, done, info = env.step(action)
        total_return += reward
        if done:
            break

    actions = np.asarray(info["action"]) if len(info["action"]) else np.zeros((0, 3))
    omega_norm = np.asarray(info["omega_norm"])
    metrics = {
        "controller": controller,
        "return": total_return,
        "final_attitude_error_deg": info["attitude_error_deg"],
        "final_omega_norm": info["omega_norm_current"],
        "success": bool(info["success"]),
        "settling_time_s": info["t"][-1] if info["success"] else np.nan,
        "episode_steps": len(info["reward"]),
        "control_energy": float(np.sum(np.linalg.norm(actions, axis=1) ** 2) * env.dt),
        "peak_omega_norm": float(np.max(omega_norm)) if len(omega_norm) else 0.0,
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
            "mean_peak_omega_norm": np.mean([row["peak_omega_norm"] for row in group]),
            "mean_settling_time_s": np.mean(finite_settling) if finite_settling else np.nan,
        })
    return summary


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
    parser.add_argument("--pd-kp", type=float, default=0.08)
    parser.add_argument("--pd-kd", type=float, default=0.80)
    parser.add_argument("--pd-torque-limit", type=float, default=0.1)
    parser.add_argument("--disturbance-level", type=float, default=0.004)
    parser.add_argument("--actuator-deadzone", type=float, default=0.012)
    parser.add_argument("--coulomb-friction", type=float, default=0.006)
    parser.add_argument("--viscous-friction", type=float, default=0.04)
    parser.add_argument("--actuator-efficiency", type=float, default=0.82)
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.model:
        from ppo_satellite_orientation import load_model

        load_model(args.model)
        print(f"Loaded model weights: {args.model}")
    angles = parse_angle_list(args.angles)
    env_kwargs = dict(DEFAULT_BENCHMARK_ENV)
    env_kwargs.update({
        "disturbance_level": args.disturbance_level,
        "actuator_deadzone": args.actuator_deadzone,
        "coulomb_friction": args.coulomb_friction,
        "viscous_friction": args.viscous_friction,
        "actuator_efficiency": args.actuator_efficiency,
    })

    rows = []
    controllers = ["pd", "ppo", "pd_ppo_residual"]
    for trial in range(args.trials):
        angle = angles[trial % len(angles)]
        start_state = make_demo_start_state(angle_deg=angle)
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
                torque_limit=args.pd_torque_limit,
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
        "control_energy", "peak_omega_norm",
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
            f"mean_peak_omega={row['mean_peak_omega_norm']:.6f}"
        )
    print(f"Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
