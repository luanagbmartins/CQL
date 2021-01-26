import gym
import numpy as np


class ACPulse(gym.Env):
    def __init__(self, observation_shape, observation_dtype):
        self.observation_shape = observation_shape
        self.observation_dtype = observation_dtype
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.observation_shape,
            dtype=self.observation_dtype,
        )
