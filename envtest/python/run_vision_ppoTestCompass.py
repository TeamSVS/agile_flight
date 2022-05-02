#!/usr/bin/env python3
import argparse
import glob
import logging
import os

import random
from typing import Callable

import numpy as np
import torch
from ruamel.yaml import YAML
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import sys

sys.path.insert(0, '/home/students/COMPASS-RL/icra22_competition_ws/src/agile_flight')
from customCallback import CustomCallback
from dronenavigation.models.compass.compass_model import CompassModel
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper

logging.basicConfig(level=logging.WARNING)

######################################
##########--COSTANT VALUES--##########
######################################


ENVIRONMENT_CHANGE_THRESHOLD = 50000

MODE = "depth"  # depth,rgb,both

FRAME = 3  # clip-length
cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)




def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """

        return  0.9995 * initial_value

    return func


def train_loop(model, callback, log=50, easy=1, medium=2, total=10):
    diff = ["easy", "medium", "hard"]
    for unit in range(total):

        if unit < easy:
            new_diff = diff[0]

        elif easy <= unit <= (easy + medium):
            new_diff = diff[1]
        else:
            new_diff = diff[2]

        new_lvl = random.randint(0, 100)

        # new_lvl = random.randint(0, 100)
        # new_diff = diff[random.randint(0, 2)]
        model.get_env().change_obstacles(level=new_lvl, difficult=new_diff)  # )
        model.learn(total_timesteps=ENVIRONMENT_CHANGE_THRESHOLD, log_interval=log,
                    callback=callback)


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
    parser.add_argument("--load", type=str, default="", help="load and train an existing model.")
    return parser


def main():
    args = parser().parse_args()
    ################################################
    ###############--LOAD CFG ENV 1--###############
    ################################################
    train_env = wrapper.FlightEnvVec(cfg, name="train", mode=MODE, n_frames=FRAME)

    train_env.spawn_flightmare(10253, 10254)
    train_env.connectUnity()
    configure_random_seed(42, train_env)

    ###############################################
    ###############--SETUP FOLDERS--###############
    ###############################################
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    semipath = rsg_root + "/saved/"

    list_dir = glob.glob(semipath + "/PPO_*")
    if list_dir:
        num_dir = max(list_dir)
    else:
        num_dir = "0"
    num_dir = num_dir[-4:]
    log_dir = (semipath + "/PPO_{:04d}").format(int(num_dir) + 1)
    os.makedirs(log_dir)

    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = log_dir + "/tensorboard/"
    os.makedirs(tensorboard_dir, exist_ok=True)
    best_dir = log_dir + "/best_model/"
    os.makedirs(best_dir, exist_ok=True)
    model_dir = log_dir + "/model/"
    os.makedirs(model_dir, exist_ok=True)

    #################################################
    ###############--SETUP CALLBACKS--###############
    #################################################

    custom_callback = CustomCallback(trigg_freq=ENVIRONMENT_CHANGE_THRESHOLD)
    eval_callback = EvalCallback(train_env, best_model_save_path=best_dir,
                                 log_path=tensorboard_dir, eval_freq=6000,
                                 n_eval_episodes=10, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=model_dir,
                                             name_prefix='ppo_model')
    #################################################
    ###############--SETUP PPO-MODEL--###############
    #################################################

    number_feature = (256 + FRAME * 13)
    pi_arch = [number_feature, int(number_feature / 2), int(number_feature / 4)]
    vi_arch = [number_feature, int(number_feature / 2), int(number_feature / 4)]
    if args.load:
        load_path = semipath + "/PPO_" + args.load + "/best_model/best_model.zip"
        model = PPO.load(load_path, env=train_env, device='cuda:0', custom_objects=None, print_system_info=True,
                     force_reset=True)
    else:
        model = PPO(
            tensorboard_log=log_dir,
            policy="MultiInputPolicy",
            policy_kwargs=dict(
                features_extractor_class=CompassModel,
                features_extractor_kwargs=dict(mode=MODE,
                                               pretrained_encoder_path=os.environ["COMPASS_CKPT"],
                                               feature_size=number_feature),
                # features_extractor_class=SimpleCNNFE,
                # features_extractor_kwargs=dict(
                #        features_dim=256),
                activation_fn=torch.nn.ReLU,
                net_arch=[(256 + FRAME * 13), dict(pi=pi_arch, vf=vi_arch)],
                # Number hidden layer 1-3 TODO last layer?
                log_std_init=-0.5,
                normalize_images=False,
                optimizer_kwargs=dict(weight_decay=0, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, maximize=False),
                # Adam optimizer TODO
                optimizer_class=torch.optim.Adam
            ),

            env=train_env,
            gamma=0.99,  # Discout factor old 0.99 IMPORTANT 0.8,0.9997-0.99
            seed=args.seed,
            ent_coef=0.002,  # Range:  0 - 0.01
            vf_coef=0.75,  # OLD 0.5 Range 0.5-1
            max_grad_norm=0.5,
            clip_range=0.25,  # OLD 0.2
            learning_rate=linear_schedule(0.001),  # OLD 0.0003 Range: 1e-5 - 1e-3
            gae_lambda=0.9,  # OLD 95 Range 0.9-1
            use_sde=False,  # action noise exploration vs GsDSE(true)
            target_kl=None,  # Range: 0.003 - 0.03 IMPORTANT?? TODO
            verbose=1,
            n_epochs=10,  # Range: 3 - 30
            batch_size=600,  # num batch != num env!! to use train env, as eval env need to use 1 num env!
            n_steps=200,  # Ragne: 512-5000

            # env_cfg=cfg, OLD PPO
            # eval_env=train_env, OLD PPO
            # eval_env=eval_env, OLD PPO
            # use_tanh_act=True, OLD PPO
            # gae_lambda=0.95,

        )
    # model.learn(total_timesteps=int(5 * 1e7), log_interval=5,
    #             callback=[custom_callback, eval_callback, checkpoint_callback])
    train_loop(model, callback=[custom_callback, eval_callback, checkpoint_callback],
               log=5, easy=2, medium=20, total=50)

    logging.info("Train ended!!!")


if __name__ == "__main__":
    main()
