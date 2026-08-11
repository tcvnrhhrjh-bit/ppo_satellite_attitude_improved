import argparse
import os

import numpy as np

from baseline_pd import attitude_error_vector, pd_action, run_pd_baseline, update_integral_error
from main import make_demo_start_state, parse_angle_list, set_global_seed
from residual_benchmark import run_trial, scenario_env_defaults
from telemetry_loader import load_telemetry_states, write_telemetry_state_summary
from torque_dynamics import TorqueDynamics
from utils import (
    Buffer,
    curriculum_start_state,
    logprobabilities,
    plot_episode_diagnostics,
    plot_rewards_curve,
    save_episode_csv,
    train_policy,
    train_value_function,
)


def residual_env_kwargs(args, max_steps):
    kwargs = scenario_env_defaults(args.scenario)
    kwargs.update({
        "max_steps": max_steps,
        "integrator": args.integrator,
        "add_9": args.add_9,
        "progress_weight": args.progress_weight,
        "success_bonus": args.success_bonus,
        "action_weight": args.action_weight,
        "omega_weight": args.omega_weight,
        "near_target_weight": args.near_target_weight,
        "fine_pointing_weight": args.fine_pointing_weight,
        "regression_weight": args.regression_weight,
        "stagnation_weight": args.stagnation_weight,
        "wheel_model": args.wheel_model,
    })
    optional_overrides = {
        "disturbance_level": args.disturbance_level,
        "actuator_deadzone": args.actuator_deadzone,
        "coulomb_friction": args.coulomb_friction,
        "viscous_friction": args.viscous_friction,
        "actuator_efficiency": args.actuator_efficiency,
        "wheel_torque_limit": args.wheel_torque_limit,
        "wheel_momentum_limit": args.wheel_momentum_limit,
        "wheel_power_limit": args.wheel_power_limit,
    }
    kwargs.update({key: value for key, value in optional_overrides.items() if value is not None})
    return kwargs


def sample_residual_action(observation):
    import tensorflow as tf
    from ppo_satellite_orientation import actor

    logits = actor(observation.reshape(1, -1))
    action_index = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action_index


def deterministic_residual_action(observation):
    import tensorflow as tf
    from ppo_satellite_orientation import actor

    logits = actor(observation.reshape(1, -1))
    return int(tf.argmax(logits, axis=1).numpy()[0])


def combined_residual_torque(
    env,
    observation,
    action_index,
    residual_scale,
    pd_kp,
    pd_kd,
    pd_ki,
    integral_error,
    pd_torque_limit,
    combined_torque_limit=None,
):
    u_pd = pd_action(
        env,
        observation,
        kp=pd_kp,
        kd=pd_kd,
        ki=pd_ki,
        integral_error=integral_error,
        torque_limit=pd_torque_limit,
    )
    u_residual = env.action_space[int(action_index)]
    output_limit = combined_torque_limit if combined_torque_limit is not None else pd_torque_limit
    u_total = np.clip(u_pd + residual_scale * u_residual, -output_limit, output_limit)
    return u_total


def telemetry_initial_error_deg(state):
    q = np.asarray(state[:4], dtype=float)
    q = q / np.linalg.norm(q)
    return float(2.0 * np.rad2deg(np.arccos(np.clip(abs(q[0]), -1.0, 1.0))))


def train_residual(args):
    import ppo_satellite_orientation as ppo_config
    from ppo_satellite_orientation import actor, critic, observation_dimensions, save_model

    ppo_config.steps_per_epoch = args.steps_per_epoch
    env_kwargs = residual_env_kwargs(args, max_steps=args.train_max_steps)
    env = TorqueDynamics(0.1, np.array([1.0, 0.0, 0.0, 0.0]), **env_kwargs)
    buffer = Buffer(observation_dimensions, args.steps_per_epoch)
    curriculum_angles = parse_angle_list(args.curriculum_angles)
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
            os.path.join(args.output_dir, "telemetry_reset_states.csv"),
            telemetry_records,
            telemetry_metadata,
        )
        print(
            "Loaded telemetry reset states: "
            f"{telemetry_metadata['rows']} rows, "
            f"clock {telemetry_metadata['first_clock']} to {telemetry_metadata['last_clock']}, "
            f"scalar={telemetry_metadata['scalar_component']}, "
            f"gyro_unit={telemetry_metadata['gyro_unit']}, "
            f"error_range={telemetry_metadata['min_attitude_error_deg']:.2f}-{telemetry_metadata['max_attitude_error_deg']:.2f} deg, "
            f"housekeeping={telemetry_metadata['use_housekeeping']}"
        )
    returns = []
    stage_index = 0
    stage_epoch_count = 0
    stage_returns = []

    def start_state_for_stage(epoch):
        if args.use_telemetry_reset:
            return telemetry_states[epoch % len(telemetry_states)].copy()
        if args.adaptive_curriculum:
            return make_demo_start_state(
                angle_deg=curriculum_angles[stage_index],
                axis=(1.0, 1.0, 0.5),
                omega=(0.04, -0.03, 0.02),
            )
        return curriculum_start_state(
            epoch, args.epochs, curriculum_angles, (1.0, 1.0, 0.5), (0.04, -0.03, 0.02)
        )

    def apply_telemetry_disturbance(index):
        if args.use_telemetry_reset and telemetry_disturbances is not None:
            env.telemetry_disturbance = telemetry_disturbances[index % len(telemetry_disturbances)].copy()
        else:
            env.telemetry_disturbance = np.zeros(3)

    current_start = start_state_for_stage(0)
    apply_telemetry_disturbance(0)
    observation = env.reset(current_start)
    episode_return = 0.0
    episode_length = 0

    for epoch in range(args.epochs):
        current_start = start_state_for_stage(epoch)
        apply_telemetry_disturbance(epoch)
        observation = env.reset(current_start)
        integral_error = np.zeros(3)
        episode_return = 0.0
        episode_length = 0
        sum_return = 0.0
        sum_length = 0
        num_episodes = 0
        telemetry_initial_errors = []
        if args.use_telemetry_reset:
            telemetry_initial_errors.append(telemetry_initial_error_deg(current_start))

        for t in range(args.steps_per_epoch):
            logits, action_index = sample_residual_action(observation)
            action_scalar = int(action_index.numpy()[0])
            error_vector = attitude_error_vector(env, observation)
            integral_error = update_integral_error(
                integral_error,
                error_vector,
                env.dt,
                args.integral_limit,
            )
            total_torque = combined_residual_torque(
                env,
                observation,
                action_scalar,
                residual_scale=args.residual_scale,
                pd_kp=args.pd_kp,
                pd_kd=args.pd_kd,
                pd_ki=args.pd_ki,
                integral_error=integral_error,
                pd_torque_limit=args.pd_torque_limit,
                combined_torque_limit=args.combined_torque_limit,
            )
            observation_new, reward, done, _ = env.step(total_torque)

            episode_return += reward
            episode_length += 1
            value_t = critic(observation.reshape(1, -1))
            logprobability_t = logprobabilities(logits, action_index)
            buffer.store(observation, action_index, reward, value_t, logprobability_t)
            observation = observation_new

            if done or t == args.steps_per_epoch - 1:
                last_val = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_val)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                if args.use_telemetry_reset:
                    telemetry_index = (epoch + num_episodes) % len(telemetry_states)
                    current_start = telemetry_states[telemetry_index].copy()
                    apply_telemetry_disturbance(telemetry_index)
                    telemetry_initial_errors.append(telemetry_initial_error_deg(current_start))
                observation = env.reset(current_start)
                integral_error = np.zeros(3)
                episode_return = 0.0
                episode_length = 0

        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()
        for _ in range(ppo_config.train_policy_iterations):
            kl = train_policy(obs_buf, act_buf, logp_buf, adv_buf)
            if kl > 1.5 * ppo_config.target_kl:
                break
        for _ in range(ppo_config.train_value_iterations):
            train_value_function(obs_buf, ret_buf)

        if args.adaptive_curriculum:
            display_stage_index = stage_index
        else:
            display_stage_index = min(len(curriculum_angles) - 1, int(epoch * len(curriculum_angles) / max(1, args.epochs)))
        mean_return = sum_return / max(1, num_episodes)
        mean_length = sum_length / max(1, num_episodes)
        returns.append(mean_return)
        if args.use_telemetry_reset:
            mean_initial_error = float(np.mean(telemetry_initial_errors)) if telemetry_initial_errors else float("nan")
            min_initial_error = float(np.min(telemetry_initial_errors)) if telemetry_initial_errors else float("nan")
            max_initial_error = float(np.max(telemetry_initial_errors)) if telemetry_initial_errors else float("nan")
            print(
                f" Epoch: {epoch + 1}. Mean Return: {mean_return}. "
                f"Mean Length: {mean_length}. Telemetry initial error: "
                f"mean={mean_initial_error:.2f} deg, range={min_initial_error:.2f}-{max_initial_error:.2f} deg"
            )
        else:
            print(
                f" Epoch: {epoch + 1}. Mean Return: {mean_return}. "
                f"Mean Length: {mean_length}. Residual curriculum angle: {curriculum_angles[display_stage_index]} deg"
            )
        if args.adaptive_curriculum and stage_index < len(curriculum_angles) - 1:
            stage_returns.append(mean_return)
            stage_epoch_count += 1
            window = stage_returns[-args.advance_window:]
            rolling_mean = float(np.mean(window))
            enough_epochs = stage_epoch_count >= args.min_epochs_per_angle
            reached_score = len(window) >= args.advance_window and rolling_mean >= args.advance_return_threshold
            waited_too_long = stage_epoch_count >= args.max_epochs_per_angle
            if enough_epochs and (reached_score or waited_too_long):
                old_angle = curriculum_angles[stage_index]
                stage_index += 1
                new_angle = curriculum_angles[stage_index]
                reason = "return threshold reached" if reached_score else "max epochs reached"
                print(
                    f"  Adaptive curriculum advanced: {old_angle} deg -> {new_angle} deg "
                    f"({reason}; rolling mean return = {rolling_mean:.3f})"
                )
                stage_epoch_count = 0
                stage_returns = []

    plot_rewards_curve(returns)
    save_model(args.model_out)
    np.savetxt("residual_training_returns.csv", np.asarray(returns), delimiter=",", header="mean_return", comments="")
    return returns


def evaluate_residual(args):
    eval_kwargs = residual_env_kwargs(args, max_steps=args.steps)
    telemetry_disturbance = None
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
        eval_index = args.telemetry_eval_index % len(telemetry_states)
        start_state = telemetry_states[eval_index].copy()
        telemetry_disturbance = telemetry_disturbances[eval_index].copy()
        print(
            "Using telemetry evaluation state: "
            f"index={eval_index}, clock={telemetry_records[eval_index]['clock']}, "
            f"initial_error={telemetry_records[eval_index]['attitude_error_deg']:.3f} deg"
        )
    else:
        start_state = make_demo_start_state(angle_deg=args.test_angle_deg)
    metrics, info = run_trial(
        "pd_ppo_residual",
        start_state,
        max_steps=args.steps,
        seed=args.seed + 999,
        env_kwargs=eval_kwargs,
        residual_scale=args.residual_scale,
        pd_kp=args.pd_kp,
        pd_kd=args.pd_kd,
        pd_ki=args.pd_ki,
        integral_limit=args.integral_limit,
        torque_limit=args.pd_torque_limit,
        telemetry_disturbance=telemetry_disturbance,
        combined_torque_limit=args.combined_torque_limit,
    )
    plot_episode_diagnostics(info, filename="residual_policy_diagnostics.pdf")
    save_episode_csv(info, filename="residual_policy_history.csv")
    run_pd_baseline(
        start_state=start_state,
        max_steps=args.steps,
        kp=args.pd_kp,
        kd=args.pd_kd,
        ki=args.pd_ki,
        integral_limit=args.integral_limit,
        torque_limit=args.pd_torque_limit,
        integrator=args.integrator,
        output_prefix="pd_baseline_eval",
        env_kwargs=eval_kwargs,
        telemetry_disturbance=telemetry_disturbance,
    )
    with open("residual_training_summary.txt", "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write(f"model_out: {args.model_out}\n")
    print("Residual PPO evaluation")
    print(f"final attitude error: {metrics['final_attitude_error_deg']:.3f} deg")
    print(f"final angular-rate norm: {metrics['final_omega_norm']:.6f}")
    print(f"success: {metrics['success']}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO as a residual correction on top of a PD attitude controller.")
    parser.add_argument("--output-dir", default="residual_training_outputs")
    parser.add_argument("--model-out", default="residual_model_weights.pickle")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--train-max-steps", type=int, default=120)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--add-9", type=float, default=0.1)
    parser.add_argument("--curriculum-angles", default="10,15,20,25,30,40,50,60,70")
    parser.add_argument("--test-angle-deg", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--residual-scale", type=float, default=0.35)
    parser.add_argument("--scenario", choices=["nominal_nonlinear", "random_disturbance", "residual_friendly"], default="residual_friendly")
    parser.add_argument("--pd-kp", type=float, default=0.08)
    parser.add_argument("--pd-kd", type=float, default=0.80)
    parser.add_argument("--pd-ki", type=float, default=0.0)
    parser.add_argument("--integral-limit", type=float, default=0.25)
    parser.add_argument("--pd-torque-limit", type=float, default=0.1)
    parser.add_argument("--combined-torque-limit", type=float, default=None)
    parser.add_argument("--integrator", choices=["rk4", "solve_ivp"], default="rk4")
    parser.add_argument("--disturbance-level", type=float, default=None)
    parser.add_argument("--actuator-deadzone", type=float, default=None)
    parser.add_argument("--coulomb-friction", type=float, default=None)
    parser.add_argument("--viscous-friction", type=float, default=None)
    parser.add_argument("--actuator-efficiency", type=float, default=None)
    parser.add_argument("--wheel-model", choices=["none", "rw-0.01", "rw-0.03"], default="none")
    parser.add_argument("--wheel-torque-limit", type=float, default=None)
    parser.add_argument("--wheel-momentum-limit", type=float, default=None)
    parser.add_argument("--wheel-power-limit", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=8.0)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--action-weight", type=float, default=0.001)
    parser.add_argument("--omega-weight", type=float, default=0.05)
    parser.add_argument("--near-target-weight", type=float, default=0.15)
    parser.add_argument("--fine-pointing-weight", type=float, default=0.40)
    parser.add_argument("--regression-weight", type=float, default=3.0)
    parser.add_argument("--stagnation-weight", type=float, default=0.03)
    parser.add_argument("--adaptive-curriculum", action="store_true")
    parser.add_argument("--advance-return-threshold", type=float, default=-150.0)
    parser.add_argument("--advance-window", type=int, default=5)
    parser.add_argument("--min-epochs-per-angle", type=int, default=8)
    parser.add_argument("--max-epochs-per-angle", type=int, default=50)
    parser.add_argument("--use-telemetry-reset", action="store_true")
    parser.add_argument("--telemetry-dir", default="")
    parser.add_argument("--telemetry-gyro-unit", choices=["deg/s", "rad/s"], default="deg/s")
    parser.add_argument("--telemetry-scalar-component", choices=["auto", "q0", "q1", "q2", "q3"], default="auto")
    parser.add_argument("--telemetry-eval-index", type=int, default=-1)
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
    if args.telemetry_dir:
        args.telemetry_dir = os.path.abspath(args.telemetry_dir)

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    train_residual(args)
    evaluate_residual(args)


if __name__ == "__main__":
    main()


