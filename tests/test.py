import unittest

import gymnasium as gym
import numpy as np
import toymeta


class TestEnvs(unittest.TestCase):
    def test_contextual(self):
        try:
            env = gym.make(
                "ContextualBandit",
                context_dim=2,
                arm_embeds=np.random.randn(2, 2),
                num_arms=2,
            )
            env.reset()
            env.step(1)
        except Exception as e:
            self.fail(f"Some error occured: {e}")

    def test_bernoulli(self):
        try:
            env = gym.make(
                "BernoulliBandit", arms_mean=np.array([0.5, 0.9]), num_arms=2
            )
            env.reset()
            env.step(1)
        except Exception as e:
            self.fail(f"Some error occured: {e}")

    def test_had_dark_room(self):
        try:
            env = gym.make(
                "HAD-Dark-Room", available_actions=np.array([1, 2, 3]), action_seq_len=3
            )
            env.reset()
            env.step(1)
        except Exception as e:
            self.fail(f"Some error occured: {e}")


if __name__ == "__main__":
    unittest.main()
