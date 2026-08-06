import numpy as np

from torque_dynamics import TorqueDynamics, quat_product
from utils import quat_conjugate, plot_episode_diagnostics, save_episode_csv


def attitude_error_vector(env, observation):
    q = observation[:4]
    q_error = quat_product(env.q_req_conj, q)

    # Keep the shortest rotation direction.
    if q_error[0] < 0:
        q_error *= -1

    return q_error[1:4]


def update_integral_error(integral_error, error_vector, dt, integral_limit=0.25):
    """Accumulate attitude error with a simple norm clamp for anti-windup."""
    integral_error = np.asarray(integral_error, dtype=float) + error_vector * dt
    norm = np.linalg.norm(integral_error)
    if norm > integral_limit:
        integral_error = integral_error * (integral_limit / norm)
    return integral_error


def pd_action(env, observation, kp=0.08, kd=0.25, ki=0.0, integral_error=None, torque_limit=0.5):
    """Quaternion PD/PID-like controller for comparison with PPO."""
    omega = observation[4:7]
    error_vector = attitude_error_vector(env, observation)
    if integral_error is None:
        integral_error = np.zeros(3)

    torque = -kp * error_vector - kd * omega - ki * integral_error
    return np.clip(torque, -torque_limit, torque_limit)


def run_pd_baseline(
    start_state=None,
    dt=0.1,
    max_steps=500,
    kp=0.08,
    kd=0.80,
    ki=0.0,
    integral_limit=0.25,
    torque_limit=0.1,
    integrator="rk4",
    output_prefix="pd_baseline",
    env_kwargs=None,
    telemetry_disturbance=None,
):
    env_options = dict(env_kwargs or {})
    env_options.setdefault("max_steps", max_steps)
    env_options.setdefault("integrator", integrator)
    env = TorqueDynamics(
        dt=dt,
        q_req=np.array([1.0, 0.0, 0.0, 0.0]),
        **env_options,
    )
    if telemetry_disturbance is not None:
        env.telemetry_disturbance = np.asarray(telemetry_disturbance, dtype=float).reshape(3,)
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

    integral_error = np.zeros(3)
    for _ in range(max_steps):
        error_vector = attitude_error_vector(env, observation)
        integral_error = update_integral_error(integral_error, error_vector, env.dt, integral_limit)
        action = pd_action(
            env,
            observation,
            kp=kp,
            kd=kd,
            ki=ki,
            integral_error=integral_error,
            torque_limit=torque_limit,
        )
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
