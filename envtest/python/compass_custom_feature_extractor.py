import gym
import torch as th
import torch.nn as nn
import random
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper


class SimpleCNNFE(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 4):
        super(SimpleCNNFE, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.linear = nn.Sequential(nn.Linear(n_input_channels, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        self.linear.forward()
        return observations
