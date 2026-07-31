#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils import (
    Buffer, discounted_cumulative_sums, mlp, logprobabilities,
    sample_action, train_policy, train_value_function,
    plot_euler_angles, quat2rpy_deg, rotate_vec_with_quat,
    cross_product, quat2rpy, quat_conjugate
)
from torque_dynamics import TorqueDynamics

# === Hyperparameters ===
global observation_dimensions, num_actions
steps_per_epoch = 500
epochs = 1000
gamma = 0.99
clip_ratio = 0.2
policy_learning_rate = 3e-4
value_function_learning_rate = 1e-3
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.95
target_kl = 0.01
hidden_sizes = (400, 300)
render = False

# === Physics: Dynamics Equations ===
def gravity_gradient_torque(quat, inertia_tensor, mean_motion):
    Ae3 = rotate_vec_with_quat(quat, np.array([0., 0., 1.]))
    return 3. * mean_motion ** 2. * cross_product(Ae3, inertia_tensor.dot(Ae3))

def rhs(t, x, sat, action):
    quat = x[:4] / np.linalg.norm(x[:4])
    omega = x[4:7]
    h = x[7:10]

    omega_rel = omega - rotate_vec_with_quat(
        quat, np.array([0., sat.mean_motion, 0.])
    )
    trq_gg = gravity_gradient_torque(quat, sat.J, sat.mean_motion)

    x_dot = np.zeros(10)
    action = action.reshape(3,)

    x_dot[0] = -0.5 * quat[1:].dot(omega_rel)
    x_dot[1:4] = 0.5 * (
        quat[0] * omega_rel + cross_product(quat[1:], omega_rel)
    )

    tmp3 = trq_gg + action - cross_product(omega, sat.J.dot(omega))
    x_dot[4:7] = sat.J_inv.dot(tmp3)
    x_dot[7:10] = -action

    return x_dot

# === Initialization ===
env = TorqueDynamics(0.1, np.array([1, 0, 0, 0]), 0.05)
observation_dimensions = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

# === Actor-Critic models ===
observation_input = keras.Input(
    shape=(observation_dimensions,),
    dtype=tf.float32
)

# Actor
logits = mlp(
    observation_input,
    list(hidden_sizes) + [num_actions],
    tf.nn.relu,
    None
)
actor = keras.Model(
    inputs=observation_input,
    outputs=logits
)

# Critic
net_out = mlp(
    observation_input,
    list(hidden_sizes) + [1],
    tf.nn.relu,
    None
)

try:
    value = tf.keras.ops.squeeze(net_out, axis=1)
except AttributeError:
    import keras as standalone_keras
    value = standalone_keras.ops.squeeze(net_out, axis=1)

critic = keras.Model(
    inputs=observation_input,
    outputs=value
)

# === Optimizers ===
policy_optimizer = keras.optimizers.Adam(
    learning_rate=policy_learning_rate
)
value_optimizer = keras.optimizers.Adam(
    learning_rate=value_function_learning_rate
)

# === Save Weights Example ===
def save_model(name='model_weights.pickle'):
    actor_config = actor.get_config()
    actor_weights = actor.get_weights()
    critic_config = critic.get_config()
    critic_weights = critic.get_weights()

    import pickle
    with open(name, 'wb') as f:
        pickle.dump(
            (actor_config, actor_weights, critic_config, critic_weights),
            f
        )

# === Load Weights Example ===
def load_model(name='model_weights.pickle'):
    import pickle

    with open(name, 'rb') as f:
        ac, aw, cc, cw = pickle.load(f)

    global actor, critic

    actor = keras.Model.from_config(ac)
    actor.set_weights(aw)

    critic = keras.Model.from_config(cc)
    critic.set_weights(cw)


def deterministic_action(observation):
    """Return the highest-probability action index for evaluation.

    PPO training samples from the policy distribution for exploration, but
    evaluation should be deterministic; otherwise a trained policy can still
    look like a random controller during the final demo.
    """
    logits = actor(observation)
    return tf.argmax(logits, axis=1)
