#!/usr/bin/env python3
import argparse
import math
#
import os
import time

import cv2

import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

from flightmare.flightpy.flightrl.rpg_baselines.torch.common.ppo import PPO
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper1 as wrapper
from flightmare.flightpy.flightrl.rpg_baselines.torch.common.util import test_policy
from compass_custom_feature_extractor import SimpleCNNFE
from threading import Thread
from dronenavigation.models.compass.compass_model import CompassModel
from customCallback import CustomCallback

cfg = YAML().load(
        open(
                os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
)


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    return parser


def main():
    args = parser().parse_args()
    from flightmare.flightpy.flightrl.rpg_baselines.torch.common.ppo import PPO

    cfg["unity"]["render"] = "yes"
    cfg["rgb_camera"]["on"] = "yes"
    train_env = wrapper.FlightEnvVec(
            VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )
    # set random seed
    # MI mancava questo riga ecco perche non parteva
    configure_random_seed(args.seed, env=train_env)

    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    train_env.connectUnity()
    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    cfg["unity"]["render"] = "no"
    cfg["rgb_camera"]["on"] = "yes"
    cfg["simulation"]["num_envs"] = 1

    # create evaluation environment
    # old_num_envs = cfg["simulation"]["num_envs"]
    # cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
            VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    )

    configure_random_seed(args.seed, env=eval_env)
    # cfg["simulation"]["num_envs"] = old_num_envs

    custom_callback = CustomCallback(train_env)
    """
    for i in range(1):
        obs_dim = train_env.obs_dim
    act_dim = train_env.act_dim
    num_env = train_env.num_envs

    # generate dummy action [-1, 1]
    train_env.reset()
    dummy_actions = np.random.rand(num_env, act_dim) * 2 - np.ones(shape=(num_env, act_dim))

    z = train_env.getObs()
    x = train_env.step(dummy_actions)
    y = train_env.getObs()
    h = train_env.getImage()
    print(h)
    train_env.render(i)
    if i == 10:
        train_env.reset(True)
    time.sleep(0.2)
    """

    model = PPO(
            tensorboard_log=log_dir,
            policy="MlpPolicy",
            policy_kwargs=dict(
                    features_extractor_class=CompassModel,
                    features_extractor_kwargs=dict(linear_prob=True,
                                                   pretrained_encoder_path=os.environ["COMPASS_CKPT"],
                                                   feature_size=256),
                    # features_extractor_class=SimpleCNNFE,
                    # features_extractor_kwargs=dict(
                    #        features_dim=256),
                    activation_fn=torch.nn.ReLU,
                    net_arch=[256, dict(pi=[128, 128], vf=[256, 256])],
                    log_std_init=-0.5,
            ),
            env=train_env,
            eval_env=eval_env,
            use_tanh_act=True,
            gae_lambda=0.95,
            gamma=0.99,
            n_steps=100,
            ent_coef=0.002,
            vf_coef=0.5,
            max_grad_norm=0.5,
            batch_size=100,  # num batch != num env!! to use train env, as eval env need to use 1 num env!
            clip_range=0.2,
            use_sde=False,
            env_cfg=cfg,
            verbose=1,
    )
    model.learn(total_timesteps=int(5 * 1e7), log_interval=(1, 5), callback=custom_callback)

    print("ENDED!!!")


if __name__ == "__main__":
    main()
