import argparse
import os
import random

import numpy as np

from torque_dynamics import TorqueDynamics
from utils import (
    train,
    plot_euler_angles,
    plot_episode_diagnostics,
    save_episode_csv,
    quat2rpy_deg,
    warm_start_policy,
)
from baseline_pd import run_pd_baseline


def make_demo_start_state(angle_deg=70.0, axis=(1.0, 1.0, 0.5), omega=(0.04, -0.03, 0.02)):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half_angle = np.deg2rad(angle_deg) / 2
    q = np.concatenate([[np.cos(half_angle)], axis * np.sin(half_angle)])
    return np.concatenate([q, np.asarray(omega, dtype=float), np.zeros(3)])


def parse_angle_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def run_ppo_test(start_state, add_9, max_steps, output_prefix="ppo", env_kwargs=None):
    from ppo_satellite_orientation import deterministic_action

    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    env_kwargs.pop("max_steps", None)
    env = TorqueDynamics(
        dt=0.1,
        q_req=np.array([1.0, 0.0, 0.0, 0.0]),
        add_9=add_9,
        max_steps=max_steps,
        **env_kwargs,
    )
    observation = env.reset(start_state)
    episode_return = 0.0
    info = {
        "x": env.history,
        "t": env.t,
        "phi": env.phi_history,
        "reward": env.reward_history,
        "action": env.action_history,
        "omega_norm": env.omega_norm_history,
    }

    for _ in range(max_steps):
        action = deterministic_action(observation.reshape(1, -1))
        observation, reward, done, info = env.step(action[0].numpy())
        episode_return += reward
        if done:
            break

    x = np.asarray(info["x"])
    t = np.asarray(info["t"])
    roll, pitch, yaw = quat2rpy_deg(x[:, 0], x[:, 1], x[:, 2], x[:, 3])
    plot_euler_angles(t, roll, pitch, yaw, filename=f"{output_prefix}_angles.pdf", inset=False)
    plot_episode_diagnostics(info, filename=f"{output_prefix}_diagnostics.pdf")
    save_episode_csv(info, filename=f"{output_prefix}_history.csv")

    print(f"PPO total return: {episode_return:.3f}")
    print(f"PPO final attitude error: {info['attitude_error_deg']:.3f} deg")
    print(f"PPO final angular-rate norm: {info['omega_norm_current']:.6f}")
    print(f"PPO success: {info['success']}")
    return episode_return, info


def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ModuleNotFoundError:
        pass


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Improved PPO satellite attitude-control demo.")
    parser.add_argument("--epochs", type=int, default=100, help="PPO training epochs.")
    parser.add_argument("--steps", type=int, default=500, help="Max test steps.")
    parser.add_argument("--add-9", type=float, default=0.1, help="Reward bonus attitude-error threshold in radians.")
    parser.add_argument("--output-dir", default="outputs_improved", help="Folder for generated plots and CSV files.")
    parser.add_argument("--skip-training", action="store_true", help="Only run PPO test and PD baseline.")
    parser.add_argument("--pd-only", action="store_true", help="Run only the PD baseline. Useful when TensorFlow is not installed.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible PPO training/evaluation.")
    parser.add_argument("--pd-kp", type=float, default=0.08, help="PD proportional gain for the baseline controller.")
    parser.add_argument("--pd-kd", type=float, default=0.80, help="PD derivative gain for the baseline controller.")
    parser.add_argument("--pd-torque-limit", type=float, default=0.1, help="PD torque limit. Default matches PPO action range.")
    parser.add_argument("--test-angle-deg", type=float, default=70.0, help="Initial attitude error used for final PPO/PD evaluation.")
    parser.add_argument("--curriculum-angles", default="10,15,20,25,30,40,50,60,70", help="Comma-separated initial attitude angles for PPO curriculum training.")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning and train only from --test-angle-deg.")
    parser.add_argument("--train-max-steps", type=int, default=120, help="Training episode step limit. Evaluation still uses --steps.")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="PPO environment interaction steps collected per epoch.")
    parser.add_argument("--warm-start-epochs", type=int, default=20, help="Supervised PD teacher warm-start epochs before PPO training. Use 0 to disable.")
    parser.add_argument("--warm-start-samples-per-angle", type=int, default=3, help="PD teacher trajectories collected per curriculum angle.")
    parser.add_argument("--warm-start-lr", type=float, default=1e-3, help="Learning rate for supervised PD teacher warm-start.")
    parser.add_argument("--progress-weight", type=float, default=8.0, help="Reward weight for reducing attitude error from one step to the next.")
    parser.add_argument("--success-bonus", type=float, default=100.0, help="Reward bonus when attitude and angular-rate tolerances are met.")
    parser.add_argument("--action-weight", type=float, default=0.001, help="Penalty on commanded torque magnitude.")
    parser.add_argument("--omega-weight", type=float, default=0.05, help="Penalty on angular-rate norm.")
    parser.add_argument("--integrator", choices=["rk4", "solve_ivp"], default="rk4", help="Dynamics integrator. rk4 is faster for PPO rollouts; solve_ivp is slower but adaptive.")
    args = parser.parse_args()

    set_global_seed(args.seed)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    train_env_kwargs = {
        "max_steps": args.train_max_steps,
        "progress_weight": args.progress_weight,
        "success_bonus": args.success_bonus,
        "action_weight": args.action_weight,
        "omega_weight": args.omega_weight,
        "integrator": args.integrator,
    }
    eval_env_kwargs = {
        "progress_weight": args.progress_weight,
        "success_bonus": args.success_bonus,
        "action_weight": args.action_weight,
        "omega_weight": args.omega_weight,
        "integrator": args.integrator,
    }
    curriculum_angles = [] if args.no_curriculum else parse_angle_list(args.curriculum_angles)
    start_state = make_demo_start_state(angle_deg=args.test_angle_deg)

    if args.pd_only:
        run_pd_baseline(
            start_state=start_state,
            max_steps=args.steps,
            kp=args.pd_kp,
            kd=args.pd_kd,
            torque_limit=args.pd_torque_limit,
            integrator=args.integrator,
            output_prefix="pd_baseline",
        )
        return

    try:
        import tensorflow  # noqa: F401
    except ModuleNotFoundError:
        print("TensorFlow is not installed, so PPO cannot run in this environment.")
        print("Running the PD baseline instead. Install requirements.txt to train PPO.")
        run_pd_baseline(
            start_state=start_state,
            max_steps=args.steps,
            kp=args.pd_kp,
            kd=args.pd_kd,
            torque_limit=args.pd_torque_limit,
            integrator=args.integrator,
            output_prefix="pd_baseline",
        )
        return

    if not args.skip_training:
        import ppo_satellite_orientation as ppo_config
        ppo_config.steps_per_epoch = args.steps_per_epoch
        if args.warm_start_epochs > 0:
            teacher_angles = curriculum_angles if curriculum_angles else [args.test_angle_deg]
            warm_start_policy(
                teacher_angles,
                env_kwargs=train_env_kwargs,
                samples_per_angle=args.warm_start_samples_per_angle,
                epochs=args.warm_start_epochs,
                learning_rate=args.warm_start_lr,
                kp=args.pd_kp,
                kd=args.pd_kd,
                torque_limit=args.pd_torque_limit,
            )
        returns = train(
            args.epochs,
            args.add_9,
            start=start_state,
            env_kwargs=train_env_kwargs,
            curriculum_angles=curriculum_angles,
        )
        np.savetxt("ppo_training_returns.csv", np.asarray(returns), delimiter=",", header="mean_return", comments="")

    ppo_return, ppo_info = run_ppo_test(start_state, args.add_9, args.steps, output_prefix="ppo", env_kwargs=eval_env_kwargs)
    pd_return, pd_info = run_pd_baseline(
        start_state=start_state,
        max_steps=args.steps,
        kp=args.pd_kp,
        kd=args.pd_kd,
        torque_limit=args.pd_torque_limit,
        integrator=args.integrator,
        output_prefix="pd_baseline",
    )

    with open("comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("Improved satellite attitude-control comparison\n")
        f.write(f"PPO return: {ppo_return:.6f}\n")
        f.write(f"PPO success: {ppo_info['success']}\n")
        f.write(f"PPO final attitude error deg: {ppo_info['attitude_error_deg']:.6f}\n")
        f.write(f"PPO final angular-rate norm: {ppo_info['omega_norm_current']:.6f}\n")
        f.write(f"PD return: {pd_return:.6f}\n")
        f.write(f"PD success: {pd_info['success']}\n")
        f.write(f"PD final attitude error deg: {pd_info['attitude_error_deg']:.6f}\n")
        f.write(f"PD final angular-rate norm: {pd_info['omega_norm_current']:.6f}\n")
        if curriculum_angles:
            f.write(f"PPO curriculum angles deg: {curriculum_angles}\n")
        f.write(f"Final evaluation initial angle deg: {args.test_angle_deg:.6f}\n")
        f.write(f"Training max steps: {args.train_max_steps}\n")
        f.write(f"Steps per epoch: {args.steps_per_epoch}\n")
        f.write(f"PD teacher warm-start epochs: {args.warm_start_epochs}\n")
        f.write(f"PD teacher warm-start learning rate: {args.warm_start_lr:.6g}\n")
        f.write(f"Dynamics integrator: {args.integrator}\n")
        f.write("Use ppo_diagnostics.pdf and pd_baseline_diagnostics.pdf to compare convergence.\n")


if __name__ == "__main__":
    main()
