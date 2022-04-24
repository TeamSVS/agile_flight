#!/usr/bin/env python3
import argparse
import math
#
import os
import random
import time

import cv2

import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

# from flightmare.flightpy.flightrl.rpg_baselines.torch.common.ppo import PPO
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from flightmare.flightpy.flightrl.rpg_baselines.torch.common.util import test_policy
from compass_custom_feature_extractor import SimpleCNNFE
from threading import Thread
from stable_baselines3 import PPO
from dronenavigation.models.compass.compass_model import CompassModel
from customCallback import CustomCallback
from threading import Thread
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)

cfg2 = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train", type=int, default=1, help="Train the policy or evaluate the policy")
    parser.add_argument("--render", type=int, default=0, help="Render with Unity")
    parser.add_argument("--trial", type=int, default=1, help="PPO trial number")
    parser.add_argument("--iter", type=int, default=100, help="PPO iter number")
    return parser


def main():
    args = parser().parse_args()

    ###############--LOAD CFG ENV 1--###############

    cfg["simulation"]["num_envs"] = 2
    cfg["unity"]["render"] = "yes"
    cfg["rgb_camera"]["on"] = "yes"
    cfg["unity"]["input_port"] = 10253
    cfg["unity"]["output_port"] = 10254

    train_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False), "train")

    configure_random_seed(args.seed, env=train_env)
    os.system(
        os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 -input-port 10253 -output-port 10254 &")
    train_env.connectUnity()

    ###############--LOAD CFG ENV 2--###############
    """
    cfg2["unity"]["render"] = "no"
    cfg2["rgb_camera"]["on"] = "yes"
    cfg2["simulation"]["num_envs"] = 1
    cfg2["unity"]["input_port"] = 10255
    cfg2["unity"]["output_port"] = 10256

    # create evaluation environment
    # old_num_envs = cfg["simulation"]["num_envs"]
    # cfg["simulation"]["num_envs"] = 1
    eval_env = wrapper.FlightEnvVec(
        VisionEnv_v1(dump(cfg2, Dumper=RoundTripDumper), False), "eval")

    configure_random_seed(args.seed, env=eval_env)

    # os.system( os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 -input-port 10255 -output-port
    # 10256 &")

    """
    ###############--SETUP FOLDERS--###############
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"

    os.makedirs(log_dir, exist_ok=True)

    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = log_dir + "/tensorboard/"
    os.makedirs(tensorboard_dir, exist_ok=True)
    best_dir = log_dir + "/best_model/"
    os.makedirs(best_dir, exist_ok=True)
    model_dir = log_dir + "/model/"
    os.makedirs(model_dir, exist_ok=True)

    custom_callback = CustomCallback(train_env)

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
        # eval_env=train_env, OLD PPO
        # eval_env=eval_env, OLD PPO
        # use_tanh_act=True, OLD PPO
        # gae_lambda=0.95,
        gamma=0.99,
        seed=args.seed,
        n_steps=10,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.5,
        batch_size=10,  # num batch != num env!! to use train env, as eval env need to use 1 num env!
        clip_range=0.2,
        use_sde=False,
        # env_cfg=cfg, OLD PPO
        verbose=1,
    )

    eval_callback = EvalCallback(train_env, best_model_save_path=best_dir,
                                 log_path=tensorboard_dir, eval_freq=6000,
                                 n_eval_episodes=10, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=model_dir,
                                             name_prefix='ppo_model')

    model.learn(total_timesteps=int(5 * 1e7), log_interval=5,
                callback=[eval_callback, checkpoint_callback, custom_callback])

    print("Train ended!!!")


if __name__ == "__main__":
    main()
