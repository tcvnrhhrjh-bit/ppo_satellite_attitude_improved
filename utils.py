# utils.py
# Spyder-friendly PPO utils module

import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ModuleNotFoundError:
    class _TensorFlowStub:
        @staticmethod
        def function(fn):
            return fn

    tf = _TensorFlowStub()
    layers = None
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True
try:
    import scipy.signal
except ModuleNotFoundError:
    scipy = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
import csv

# === Support Functions ===
def rotate_vec_with_quat(q, vec):
    q_conj = quat_conjugate(q)
    qxvec = cross_product(q_conj[1:], vec)
    return (q_conj[1:].dot(vec)) * q_conj[1:] + q_conj[0]**2 * vec + 2. * q_conj[0] * qxvec + cross_product(q_conj[1:], qxvec)

def cross_product(a, b):
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def plot_euler_angles(t, roll, pitch, yaw, filename=None, bbox=(60,100,-1,1), inset=True):
    if plt is None:
        print("matplotlib is not installed; skipping Euler-angle plot.")
        return
    fig, ax = plt.subplots(figsize=(16,8))
    ax.set_title("Euler Angles")
    ax.plot(t, roll, label='roll', color='red')
    ax.plot(t, pitch, label='pitch', color='green')
    ax.plot(t, yaw, label='yaw', color='blue')

    if inset:
        axins = ax.inset_axes([0.3, 0.55, 0.5, 0.25])
        axins.set_xlim(bbox[0], bbox[1])
        axins.set_ylim(bbox[2], bbox[3])
        axins.plot(t, roll, color='red')
        axins.plot(t, pitch, color='green')
        axins.plot(t, yaw, color='blue')
        ax.indicate_inset_zoom(axins)

    ax.set_ylabel('Angles [deg]')
    ax.set_xlabel('Time [s]')
    ax.grid(True)
    ax.legend()
    if filename:
        fig.savefig(filename)
    plt.close(fig)

# === New: Plot reward curve ===
def plot_rewards_curve(reward_list):
    if plt is None:
        print("matplotlib is not installed; skipping reward plot.")
        return
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(reward_list, marker='o')
    ax.set_title("Reward per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Reward")
    ax.grid(True)
    fig.savefig("reward_curve.pdf")
    plt.close(fig)

def plot_episode_diagnostics(info, filename="episode_diagnostics.pdf"):
    if plt is None:
        print("matplotlib is not installed; skipping episode diagnostics plot.")
        return
    t = np.asarray(info["t"])
    x = np.asarray(info["x"])
    phi_deg = np.rad2deg(np.asarray(info["phi"]))
    omega_norm = np.asarray(info["omega_norm"])
    actions = np.asarray(info["action"]) if len(info["action"]) else np.zeros((0, 3))
    rewards = np.asarray(info["reward"])
    roll, pitch, yaw = quat2rpy_deg(x[:, 0], x[:, 1], x[:, 2], x[:, 3])

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    axes[0].plot(t, roll, label="roll")
    axes[0].plot(t, pitch, label="pitch")
    axes[0].plot(t, yaw, label="yaw")
    axes[0].set_ylabel("Euler angle [deg]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t, phi_deg, color="tab:purple")
    axes[1].set_ylabel("Attitude error [deg]")
    axes[1].grid(True)

    axes[2].plot(t, omega_norm, color="tab:orange")
    axes[2].set_ylabel("Angular rate norm")
    axes[2].grid(True)

    if len(actions):
        action_t = t[1:1 + len(actions)]
        axes[3].plot(action_t, actions[:, 0], label="Tx")
        axes[3].plot(action_t, actions[:, 1], label="Ty")
        axes[3].plot(action_t, actions[:, 2], label="Tz")
        axes[3].legend()
    axes[3].set_ylabel("Control torque")
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(True)

    fig.suptitle(f"Episode diagnostics | return={rewards.sum():.3f}")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def save_episode_csv(info, filename="episode_history.csv"):
    t = np.asarray(info["t"])
    x = np.asarray(info["x"])
    phi_deg = np.rad2deg(np.asarray(info["phi"]))
    omega_norm = np.asarray(info["omega_norm"])
    actions = np.asarray(info["action"]) if len(info["action"]) else np.zeros((0, 3))
    rewards = np.asarray(info["reward"])

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "q0", "q1", "q2", "q3", "wx", "wy", "wz", "hx", "hy", "hz",
            "attitude_error_deg", "omega_norm", "reward", "tx", "ty", "tz",
        ])
        for i in range(len(t)):
            action = actions[i - 1] if 0 < i <= len(actions) else [0.0, 0.0, 0.0]
            reward = rewards[i - 1] if 0 < i <= len(rewards) else 0.0
            writer.writerow([
                t[i], *x[i], phi_deg[i], omega_norm[i], reward, *action,
            ])

def quat2rpy(q0, q1, q2, q3):
    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1**2 + q2**2))
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2**2 + q3**2))
    return [roll, pitch, yaw]

def quat2rpy_deg(q0, q1, q2, q3):
    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1**2 + q2**2)) * 180 / np.pi
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3)) * 180 / np.pi
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2**2 + q3**2)) * 180 / np.pi
    return [roll, pitch, yaw]

def normalize(obj):
    return obj / np.linalg.norm(obj)

def quat_conjugate(q):
    q_new = np.copy(q)
    q_new[1:] *= -1
    return q_new

# === Log-probability ===
def logprobabilities(logits, a):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required for PPO log probabilities.")
    from ppo_satellite_orientation import num_actions
    logprob_all = tf.nn.log_softmax(logits)
    return tf.reduce_sum(tf.one_hot(a, num_actions) * logprob_all, axis=1)

# === MLP ===
def mlp(x, sizes, activation=None, output_activation=None):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required to build PPO neural networks.")
    if activation is None:
        activation = tf.nn.relu
    for size in sizes[:-1]:
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)

# === Discounted cumulative sums ===
def discounted_cumulative_sums(x, discount):
    if scipy is not None:
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    out = np.zeros_like(x, dtype=float)
    running = 0.0
    for i in range(len(x) - 1, -1, -1):
        running = x[i] + discount * running
        out[i] = running
    return out

# === Train ===
@tf.function
def sample_action(observation):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required for PPO sampling. Install tensorflow to train or test PPO.")
    from ppo_satellite_orientation import actor
    logits = actor(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

@tf.function
def train_policy(obs_buf, act_buf, logp_buf, adv_buf):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required for PPO training. Install tensorflow to train PPO.")
    from ppo_satellite_orientation import actor, clip_ratio, policy_optimizer
    with tf.GradientTape() as tape:
        logits = actor(obs_buf)
        logp = logprobabilities(logits, act_buf)
        ratio = tf.exp(logp - logp_buf)
        min_adv = tf.where(adv_buf > 0, (1 + clip_ratio) * adv_buf, (1 - clip_ratio) * adv_buf)
        loss_pi = -tf.reduce_mean(tf.minimum(ratio * adv_buf, min_adv))
    grads = tape.gradient(loss_pi, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(grads, actor.trainable_variables))
    kl = tf.reduce_mean(logp_buf - logp)
    return tf.reduce_sum(kl)

@tf.function
def train_value_function(obs_buf, ret_buf):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required for PPO training. Install tensorflow to train PPO.")
    from ppo_satellite_orientation import critic, value_optimizer
    with tf.GradientTape() as tape:
        loss_v = tf.reduce_mean((ret_buf - critic(obs_buf)) ** 2)
    grads = tape.gradient(loss_v, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

def make_axis_angle_state(angle_deg, axis=(1.0, 1.0, 0.5), omega=(0.04, -0.03, 0.02)):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half_angle = np.deg2rad(angle_deg) / 2.0
    q = np.concatenate([[np.cos(half_angle)], axis * np.sin(half_angle)])
    return np.concatenate([q, np.asarray(omega, dtype=float), np.zeros(3)])

def curriculum_start_state(epoch, epochs, curriculum_angles, axis, omega):
    if not curriculum_angles:
        return None
    stage_count = len(curriculum_angles)
    stage_index = min(stage_count - 1, int(epoch * stage_count / max(1, epochs)))
    return make_axis_angle_state(curriculum_angles[stage_index], axis=axis, omega=omega)

def quat_product_local(q1, q2):
    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + cross_product(q1[1:], q2[1:])
    return q

def pd_teacher_torque(env, observation, kp=0.08, kd=0.80, torque_limit=0.1):
    q = observation[:4]
    omega = observation[4:7]
    q_error = quat_product_local(env.q_req_conj, q)
    if q_error[0] < 0:
        q_error *= -1
    torque = -kp * q_error[1:4] - kd * omega
    return np.clip(torque, -torque_limit, torque_limit)

def nearest_action_index(action_space, torque):
    distances = np.linalg.norm(action_space - np.asarray(torque).reshape(1, 3), axis=1)
    return int(np.argmin(distances))

def warm_start_policy(
    curriculum_angles,
    env_kwargs=None,
    samples_per_angle=4,
    epochs=20,
    batch_size=64,
    learning_rate=1e-3,
    kp=0.08,
    kd=0.80,
    torque_limit=0.1,
):
    if not TENSORFLOW_AVAILABLE:
        raise ModuleNotFoundError("TensorFlow is required for PPO warm-start.")
    from ppo_satellite_orientation import actor
    from torque_dynamics import TorqueDynamics

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    env = TorqueDynamics(0.1, np.array([1, 0, 0, 0]), **env_kwargs)
    observations = []
    labels = []

    for angle in curriculum_angles:
        for _ in range(samples_per_angle):
            observation = env.reset(make_axis_angle_state(angle))
            for _ in range(env.max_steps):
                torque = pd_teacher_torque(env, observation, kp=kp, kd=kd, torque_limit=torque_limit)
                action_index = nearest_action_index(env.action_space, torque)
                observations.append(observation.copy())
                labels.append(action_index)
                observation, _, done, _ = env.step(torque)
                if done:
                    break

    if not observations:
        return 0.0

    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(labels, dtype=np.int32)
    dataset_size = len(obs)
    last_loss = 0.0

    for _ in range(epochs):
        order = np.random.permutation(dataset_size)
        for start_idx in range(0, dataset_size, batch_size):
            idx = order[start_idx:start_idx + batch_size]
            with tf.GradientTape() as tape:
                logits = actor(obs[idx])
                loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(
                        act[idx], logits, from_logits=True
                    )
                )
            grads = tape.gradient(loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor.trainable_variables))
            last_loss = float(loss.numpy())

    print(f"PD teacher warm-start samples: {dataset_size}. Final imitation loss: {last_loss:.4f}")
    return last_loss

def train(
    epochs,
    add_9,
    start=None,
    env_kwargs=None,
    curriculum_angles=None,
    curriculum_axis=(1.0, 1.0, 0.5),
    curriculum_omega=(0.04, -0.03, 0.02),
):
    from ppo_satellite_orientation import (
        observation_dimensions, steps_per_epoch, actor, critic,
        train_policy_iterations, train_value_iterations, target_kl
    )
    from torque_dynamics import TorqueDynamics
    buffer = Buffer(observation_dimensions, steps_per_epoch)
    env_kwargs = {} if env_kwargs is None else dict(env_kwargs)
    env = TorqueDynamics(0.1, np.array([1, 0, 0, 0]), add_9, **env_kwargs)
    current_start = start
    if curriculum_angles:
        current_start = curriculum_start_state(0, epochs, curriculum_angles, curriculum_axis, curriculum_omega)
    observation, episode_return, episode_length = env.reset(current_start), 0, 0
    returns = []

    for epoch in range(epochs):
        if curriculum_angles:
            current_start = curriculum_start_state(epoch, epochs, curriculum_angles, curriculum_axis, curriculum_omega)
            observation, episode_return, episode_length = env.reset(current_start), 0, 0

        sum_return = 0
        sum_length = 0
        num_episodes = 0

        for t in range(steps_per_epoch):
            logits, action = sample_action(observation.reshape(1, -1))
            observation_new, reward, done, info = env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1
            value_t = critic(observation.reshape(1, -1))
            logprobability_t = logprobabilities(logits, action)
            buffer.store(observation, action, reward, value_t, logprobability_t)
            observation = observation_new

            if done or t == steps_per_epoch - 1:
                last_val = 0 if done else critic(observation.reshape(1, -1))
                buffer.finish_trajectory(last_val)
                sum_return += episode_return
                sum_length += episode_length
                num_episodes += 1
                observation, episode_return, episode_length = env.reset(current_start), 0, 0

        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buffer.get()

        for _ in range(train_policy_iterations):
            kl = train_policy(obs_buf, act_buf, logp_buf, adv_buf)
            if kl > 1.5 * target_kl:
                break

        for _ in range(train_value_iterations):
            train_value_function(obs_buf, ret_buf)

        if curriculum_angles:
            angle_msg = f". Curriculum angle: {curriculum_angles[min(len(curriculum_angles) - 1, int(epoch * len(curriculum_angles) / max(1, epochs)))]} deg"
        else:
            angle_msg = ""
        print(f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}{angle_msg}")
        returns.append(sum_return / num_episodes)

    plot_rewards_curve(returns)
    return returns

# === Experience buffer ===

class Buffer:
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        self.observation_buffer = np.zeros((size, observation_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    @staticmethod
    def _scalar(x, dtype=float):
        if hasattr(x, "numpy"):
            x = x.numpy()
        x = np.asarray(x).reshape(-1)[0]
        return dtype(x)

    def store(self, observation, action, reward, value, logprobability):
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = self._scalar(action, int)
        self.reward_buffer[self.pointer] = self._scalar(reward, float)
        self.value_buffer[self.pointer] = self._scalar(value, float)
        self.logprobability_buffer[self.pointer] = self._scalar(logprobability, float)
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        last_value = self._scalar(last_value, float)
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(deltas, self.gamma * self.lam)
        self.return_buffer[path_slice] = discounted_cumulative_sums(rewards, self.gamma)[:-1]
        self.trajectory_start_index = self.pointer

    def get(self):
        self.pointer, self.trajectory_start_index = 0, 0
        adv_mean, adv_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)
        if adv_std < 1e-8:
            self.advantage_buffer = self.advantage_buffer - adv_mean
        else:
            self.advantage_buffer = (self.advantage_buffer - adv_mean) / adv_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
