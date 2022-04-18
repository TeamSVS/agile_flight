import gym
import torch as th
import torch.nn as nn
import random
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper

class CompassFE(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, env: wrapper, features_dim: int = 5):
        super(CompassFE, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.env = env

        self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # x = self.env.getImage()
        self.env.render(0)
        # print(x)
        # print(self.env.getImage().shape[0])
        return observations
