import numpy as np
try:
    import gym
except ModuleNotFoundError:
    class _Box:
        def __init__(self, low, high, shape):
            self.low = low
            self.high = high
            self.shape = shape

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape)

    class gym:
        class Env:
            pass

        class spaces:
            Box = _Box
from types import SimpleNamespace
from utils import normalize, quat_conjugate, rotate_vec_with_quat, cross_product
try:
    from scipy.integrate import solve_ivp
except ModuleNotFoundError:
    solve_ivp = None

# Physical constants and a small CubeSat-like rigid body model.
const = SimpleNamespace()
const.mu = 3.986e14
const.Re = 6371.0e3

sat = SimpleNamespace()
sat.J = np.diag(np.array([4.0, 5.0, 3.0]))
sat.J_inv = np.linalg.inv(sat.J)
sat.altitude = 600e3
sat.mean_motion = np.sqrt(const.mu / (const.Re + sat.altitude) ** 3)


class TorqueDynamics(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(10,))

    def __init__(
        self,
        dt,
        q_req,
        add_9=0.25,
        attitude_tolerance_deg=5.0,
        rate_tolerance=0.01,
        max_steps=500,
        omega_weight=0.05,
        action_weight=0.001,
        progress_weight=8.0,
        success_bonus=100.0,
        fuel_weight=0.0,
        time_weight=0.0,
        disturbance_level=0.0,
        randomize_disturbance=False,
        actuator_efficiency=1.0,
        actuator_deadzone=0.0,
        coulomb_friction=0.0,
        viscous_friction=0.0,
        friction_smoothing=1e-3,
        seed=None,
    ):
        self.state = None
        self.dt = dt
        self.q_req = np.asarray(q_req, dtype=float)
        self.q_req = normalize(self.q_req)
        self.q_req_conj = quat_conjugate(self.q_req)
        self.w_req = np.zeros(3)
        self.history = []
        self.t = []
        self.action_space = self.init_actions()
        self.q_prev = None
        self.add_9 = add_9
        self.attitude_tolerance = np.deg2rad(attitude_tolerance_deg)
        self.rate_tolerance = rate_tolerance
        self.max_steps = max_steps
        self.omega_weight = omega_weight
        self.action_weight = action_weight
        self.progress_weight = progress_weight
        self.success_bonus = success_bonus
        self.fuel_weight = fuel_weight
        self.time_weight = time_weight
        self.disturbance_level = disturbance_level
        self.randomize_disturbance = randomize_disturbance
        self.actuator_efficiency = actuator_efficiency
        self.actuator_deadzone = actuator_deadzone
        self.coulomb_friction = coulomb_friction
        self.viscous_friction = viscous_friction
        self.friction_smoothing = friction_smoothing
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.phi_history = []
        self.reward_history = []
        self.action_history = []
        self.effective_action_history = []
        self.disturbance_history = []
        self.omega_norm_history = []
        self.disturbance_bias = np.zeros(3)
        self.disturbance_phase = np.zeros(3)

    @staticmethod
    def init_actions():
        a = np.linspace(-1, 1, 21)
        a1 = a / 10
        a2 = a / 100
        a = np.concatenate((a, a1, a2))
        a = np.unique(a.round(10))
        a = a[a >= -0.1]
        a = a[a <= 0.1]
        s = a.shape
        a = np.vstack((a, np.zeros(s)))
        a = np.vstack((a, np.zeros(s)))
        a = a.T
        aroll1 = np.roll(a, 1)
        aroll2 = np.roll(a, 2)
        a = np.concatenate((a, aroll1))
        a = np.concatenate((a, aroll2))
        return np.unique(a, axis=0)

    def reset(self, state=None):
        if state is not None:
            self.state = np.array(state, dtype=float).copy()
            self.state[:4] = normalize(self.state[:4])
        else:
            self.state = self.observation_space.sample()
            phi = self.state[0] * np.pi
            self.state[0] = np.cos(phi / 2)
            self.state[1:4] = normalize(self.state[1:4]) * np.sin(phi / 2)
            self.state[4:] = 0

        self.history = [self.state.copy()]
        self.t = [0.0]
        self.q_prev = self.state[:4].copy()
        self.multiplier = 1.0
        self.step_count = 0
        self.phi_history = [self.attitude_error(self.state[:4])]
        self.reward_history = []
        self.action_history = []
        self.effective_action_history = []
        self.disturbance_history = []
        self.omega_norm_history = [np.linalg.norm(self.state[4:7])]
        if self.randomize_disturbance and self.disturbance_level > 0:
            self.disturbance_bias = self.rng.uniform(
                -self.disturbance_level, self.disturbance_level, size=3
            )
            self.disturbance_phase = self.rng.uniform(0.0, 2.0 * np.pi, size=3)
        else:
            self.disturbance_bias = np.array(
                [0.6, -0.35, 0.25], dtype=float
            ) * self.disturbance_level
            self.disturbance_phase = np.array([0.0, 1.7, 3.1], dtype=float)
        return self.state

    def attitude_error(self, q_current):
        q_error = quat_product(self.q_req_conj, q_current)
        q_error = np.clip(q_error, -1, 1)
        return 2 * np.arccos(np.abs(q_error[0]))

    def terminal_bonus(self, reward, phi):
        return reward + 9 if phi <= self.add_9 else reward

    def shaped_reward(self, phi, prev_phi, omega, action, success):
        progress = prev_phi - phi
        fuel_use = np.linalg.norm(action, ord=1) * self.dt
        reward = (
            -0.35 * phi
            - 0.05 * phi**2
            - self.omega_weight * np.linalg.norm(omega)
            - self.action_weight * np.linalg.norm(action)
            - self.fuel_weight * fuel_use
            - self.time_weight
            + self.progress_weight * progress
        )
        if phi < np.deg2rad(20.0):
            reward += 0.05
        if phi < np.deg2rad(10.0):
            reward += 0.15
        reward = self.terminal_bonus(reward, phi)
        if success:
            reward += self.success_bonus
        return reward

    def step(self, action):
        if isinstance(action, (int, np.integer)):
            action = self.action_space[action].copy()

        action = np.asarray(action, dtype=float).reshape(3,)
        action *= self.multiplier
        effective_action = self.apply_actuator_model(action, self.state[4:7])
        disturbance = self.external_disturbance(self.t[-1])

        if solve_ivp is not None:
            sol = solve_ivp(
                lambda t, x: rhs(t, x, sat, effective_action, disturbance),
                (0, self.dt),
                self.state,
            )
            observation = sol.y.T[-1]
        else:
            observation = rk4_step(self.state, self.dt, effective_action, disturbance)
        observation[:4] = normalize(observation[:4])

        self.state = observation
        self.history.append(observation.copy())
        self.t.append(self.t[-1] + self.dt)

        q_current = observation[:4]
        w_current = observation[4:7]
        phi = self.attitude_error(q_current)
        prev_phi = self.attitude_error(self.q_prev)
        omega_norm = np.linalg.norm(w_current)
        action_norm = np.linalg.norm(action)

        success = phi < self.attitude_tolerance and omega_norm < self.rate_tolerance
        self.step_count += 1
        timeout = self.step_count >= self.max_steps
        done = success or timeout
        reward = self.shaped_reward(phi, prev_phi, w_current, action, success)

        self.multiplier = 1 if phi > np.pi / 8 else np.sin(4 * phi)
        self.q_prev = q_current.copy()

        self.phi_history.append(phi)
        self.reward_history.append(reward)
        self.action_history.append(action.copy())
        self.effective_action_history.append(effective_action.copy())
        self.disturbance_history.append(disturbance.copy())
        self.omega_norm_history.append(omega_norm)

        info = {
            "x": self.history,
            "t": self.t,
            "phi": self.phi_history,
            "reward": self.reward_history,
            "action": self.action_history,
            "effective_action": self.effective_action_history,
            "disturbance": self.disturbance_history,
            "omega_norm": self.omega_norm_history,
            "success": success,
            "timeout": timeout,
            "attitude_error_deg": np.rad2deg(phi),
            "omega_norm_current": omega_norm,
            "action_norm_current": action_norm,
        }
        return observation, reward, done, info

    def render(self):
        pass

    def apply_actuator_model(self, command, omega):
        command = np.asarray(command, dtype=float).reshape(3,)
        command = np.where(np.abs(command) < self.actuator_deadzone, 0.0, command)
        wheel_torque = self.actuator_efficiency * command
        friction = (
            self.coulomb_friction * np.tanh(omega / self.friction_smoothing)
            + self.viscous_friction * omega
        )
        return wheel_torque - friction

    def external_disturbance(self, t):
        if self.disturbance_level <= 0:
            return np.zeros(3)
        periodic = 0.35 * self.disturbance_level * np.sin(
            0.07 * t + self.disturbance_phase
        )
        return self.disturbance_bias + periodic


def quat_product(q1, q2):
    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + cross_product(q1[1:], q2[1:])
    return q


def rhs(t, x, sat_model, action, disturbance=None):
    quat = x[:4] / np.linalg.norm(x[:4])
    omega = x[4:7]
    if disturbance is None:
        disturbance = np.zeros(3)

    omega_rel = omega - rotate_vec_with_quat(quat, np.array([0.0, sat_model.mean_motion, 0.0]))
    Ae3 = rotate_vec_with_quat(quat, np.array([0.0, 0.0, 1.0]))
    trq_gg = 3.0 * sat_model.mean_motion**2 * cross_product(Ae3, sat_model.J @ Ae3)

    x_dot = np.zeros(10)
    x_dot[0] = -0.5 * np.dot(quat[1:], omega_rel)
    x_dot[1:4] = 0.5 * (quat[0] * omega_rel + cross_product(quat[1:], omega_rel))
    x_dot[4:7] = sat_model.J_inv @ (
        trq_gg + action + disturbance - cross_product(omega, sat_model.J @ omega)
    )
    x_dot[7:10] = -action
    return x_dot


def rk4_step(state, dt, action, disturbance=None):
    k1 = rhs(0.0, state, sat, action, disturbance)
    k2 = rhs(dt / 2, state + dt * k1 / 2, sat, action, disturbance)
    k3 = rhs(dt / 2, state + dt * k2 / 2, sat, action, disturbance)
    k4 = rhs(dt, state + dt * k3, sat, action, disturbance)
    return state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
