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
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from flightmare.flightpy.flightrl.rpg_baselines.torch.common.util import test_policy
from compass_custom_feature_extractor import CompassFE
from threading import Thread
from dronenavigation.models.compass.compass_model import CompassModel

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


def renderBig(env):
    time.sleep(5)
    while True:
        env.render(0)
        time.sleep(0.01)


def main():
    args = parser().parse_args()

    cfg["unity"]["render"] = "yes"
    cfg["rgb_camera"]["on"] = "yes"
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

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
    # cfg["simulation"]["num_envs"] = old_num_envs

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
            features_extractor_kwargs=dict(linear_prob=True, pretrained_encoder_path="/home/cam/Desktop/compass"
                                                                                     "/Modello/new-checkpoint.pth"
                                                                                     ".tar", feature_size=4),
            #                activation_fn=torch.nn.ReLU,
            net_arch=[4,dict(pi=[256, 256], vf=[512, 512])],
            log_std_init=-0.5,
        ),
        env=train_env,
        eval_env=eval_env,
        use_tanh_act=True,
        gae_lambda=0.95,
        gamma=0.99,
        n_steps=250,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=train_env.num_envs * 250,
        clip_range=0.2,
        use_sde=False,  # don't use (gSDE), doesn't work
        env_cfg=cfg,
        verbose=1,
    )
    # Thread(target=renderBig, args=[train_env]).start()
    model.learn(total_timesteps=int(5 * 1e7), log_interval=(10, 50))
    print("ENDED!!!")


if __name__ == "__main__":
    main()
