import numpy as np

from torque_dynamics import TorqueDynamics, quat_product
from utils import quat_conjugate, plot_episode_diagnostics, save_episode_csv


def pd_action(env, observation, kp=0.08, kd=0.25, torque_limit=0.5):
    """Simple quaternion PD controller for comparison with PPO."""
    q = observation[:4]
    omega = observation[4:7]
    q_error = quat_product(env.q_req_conj, q)

    # Keep the shortest rotation direction.
    if q_error[0] < 0:
        q_error *= -1

    torque = -kp * q_error[1:4] - kd * omega
    return np.clip(torque, -torque_limit, torque_limit)


def run_pd_baseline(
    start_state=None,
    dt=0.1,
    max_steps=500,
    kp=0.08,
    kd=0.80,
    torque_limit=0.1,
    output_prefix="pd_baseline",
):
    env = TorqueDynamics(
        dt=dt,
        q_req=np.array([1.0, 0.0, 0.0, 0.0]),
        max_steps=max_steps,
    )
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
        action = pd_action(env, observation, kp=kp, kd=kd, torque_limit=torque_limit)
        observation, reward, done, info = env.step(action)
        total_return += reward
        if done:
            break

    plot_episode_diagnostics(info, filename=f"{output_prefix}_diagnostics.pdf")
    save_episode_csv(info, filename=f"{output_prefix}_history.csv")
    print(f"PD total return: {total_return:.3f}")
    print(f"PD final attitude error: {info['attitude_error_deg']:.3f} deg")
    print(f"PD final angular-rate norm: {info['omega_norm_current']:.6f}")
    print(f"PD success: {info['success']}")
    return total_return, info


if __name__ == "__main__":
    run_pd_baseline()
