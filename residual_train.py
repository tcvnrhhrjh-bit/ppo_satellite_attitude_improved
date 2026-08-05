import argparse
import os

import numpy as np

from baseline_pd import attitude_error_vector, pd_action, run_pd_baseline, update_integral_error
from main import make_demo_start_state, parse_angle_list, set_global_seed
from residual_benchmark import run_trial, scenario_env_defaults
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
    })
    optional_overrides = {
        "disturbance_level": args.disturbance_level,
        "actuator_deadzone": args.actuator_deadzone,
        "coulomb_friction": args.coulomb_friction,
        "viscous_friction": args.viscous_friction,
        "actuator_efficiency": args.actuator_efficiency,
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
    torque_limit,
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
    u_residual = env.action_space[int(action_index)]
    u_total = np.clip(u_pd + residual_scale * u_residual, -torque_limit, torque_limit)
    return u_total


def train_residual(args):
    import ppo_satellite_orientation as ppo_config
    from ppo_satellite_orientation import actor, critic, observation_dimensions, save_model

    ppo_config.steps_per_epoch = args.steps_per_epoch
    env_kwargs = residual_env_kwargs(args, max_steps=args.train_max_steps)
    env = TorqueDynamics(0.1, np.array([1.0, 0.0, 0.0, 0.0]), **env_kwargs)
    buffer = Buffer(observation_dimensions, args.steps_per_epoch)
    curriculum_angles = parse_angle_list(args.curriculum_angles)
    returns = []

    current_start = curriculum_start_state(
        0, args.epochs, curriculum_angles, (1.0, 1.0, 0.5), (0.04, -0.03, 0.02)
    )
    observation = env.reset(current_start)
    episode_return = 0.0
    episode_length = 0

    for epoch in range(args.epochs):
        current_start = curriculum_start_state(
            epoch, args.epochs, curriculum_angles, (1.0, 1.0, 0.5), (0.04, -0.03, 0.02)
        )
        observation = env.reset(current_start)
        integral_error = np.zeros(3)
        episode_return = 0.0
        episode_length = 0
        sum_return = 0.0
        sum_length = 0
        num_episodes = 0

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
                torque_limit=args.pd_torque_limit,
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

        stage_index = min(len(curriculum_angles) - 1, int(epoch * len(curriculum_angles) / max(1, args.epochs)))
        mean_return = sum_return / max(1, num_episodes)
        mean_length = sum_length / max(1, num_episodes)
        returns.append(mean_return)
        print(
            f" Epoch: {epoch + 1}. Mean Return: {mean_return}. "
            f"Mean Length: {mean_length}. Residual curriculum angle: {curriculum_angles[stage_index]} deg"
        )

    plot_rewards_curve(returns)
    save_model(args.model_out)
    np.savetxt("residual_training_returns.csv", np.asarray(returns), delimiter=",", header="mean_return", comments="")
    return returns


def evaluate_residual(args):
    eval_kwargs = residual_env_kwargs(args, max_steps=args.steps)
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
    parser.add_argument("--integrator", choices=["rk4", "solve_ivp"], default="rk4")
    parser.add_argument("--disturbance-level", type=float, default=None)
    parser.add_argument("--actuator-deadzone", type=float, default=None)
    parser.add_argument("--coulomb-friction", type=float, default=None)
    parser.add_argument("--viscous-friction", type=float, default=None)
    parser.add_argument("--actuator-efficiency", type=float, default=None)
    parser.add_argument("--progress-weight", type=float, default=8.0)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--action-weight", type=float, default=0.001)
    parser.add_argument("--omega-weight", type=float, default=0.05)
    parser.add_argument("--near-target-weight", type=float, default=0.15)
    parser.add_argument("--fine-pointing-weight", type=float, default=0.40)
    parser.add_argument("--regression-weight", type=float, default=3.0)
    parser.add_argument("--stagnation-weight", type=float, default=0.03)
    args = parser.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    train_residual(args)
    evaluate_residual(args)


if __name__ == "__main__":
    main()
