import gymnasium as gym
import numpy as np
import toymeta

env = gym.make(
    "ContextualBandit", context_dim=2, arm_embeds=np.random.randn(2, 2), num_arms=2
)
print(env.reset())
print(env.step(1))

env = gym.make("HAD-Dark-Room", available_actions=np.array([1, 2, 3]), action_seq_len=3)
print(env.reset())
print(env.step(1))

env = gym.make("BernoulliBandit", arms_mean=np.array([0.5, 0.9]), num_arms=2)
print(env.reset())
print(env.step(1))
