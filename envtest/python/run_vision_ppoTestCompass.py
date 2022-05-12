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

# sys.path.insert(0, '/home/students/COMPASS-RL/icra22_competition_ws/src/agile_flight')
from customCallback import CustomCallback
from dronenavigation.models.compass.compass_model import CompassModel
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper

logging.basicConfig(level=logging.WARNING)

######################################
##########--COSTANT VALUES--##########
######################################


ENVIRONMENT_CHANGE_THRESHOLD = 50000

STARTING_LR = 0.001  # clip-length
cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)

actual_lr = STARTING_LR


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
        global actual_lr
        actual_lr = actual_lr * 0.9995 + 0.000001

        return actual_lr

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
        model.learn(total_timesteps=ENVIRONMENT_CHANGE_THRESHOLD, reset_num_timesteps=False, log_interval=log,
                    callback=callback)

        obs = model.get_env().change_obstacles(level=new_lvl, difficult=new_diff)  # )
        model._last_obs = obs
        model._last_episode_starts = np.full([model.get_env().num_envs], False)


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--iport", type=int, default=10277, help="Input port for simulation")
    parser.add_argument("--oport", type=int, default=10278, help="Output port for simulation")
    parser.add_argument("--nframe", type=int, default=3, help="Number of frame")
    parser.add_argument("--load", type=str, default=None, help="load and train an existing model.")
    parser.add_argument("--mode", type=str, default="rgb", help="the compass net input")  # depth,rgb,both
    parser.add_argument("--gpu", type=int, default=None, help="the gpu used by torch")
    return parser


def main():
    args = parser().parse_args()
    ################################################
    ###############--LOAD CFG ENV 1--###############
    ################################################
    train_env = wrapper.FlightEnvVec(cfg, name="train", mode=args.mode, n_frames=args.nframe, in_port=args.iport,
                                     out_port=args.oport)

    train_env.connectUnity()
    configure_random_seed(args.seed, train_env)

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
                                 log_path=tensorboard_dir, eval_freq=int(ENVIRONMENT_CHANGE_THRESHOLD / 10),
                                 n_eval_episodes=10, deterministic=True)
    checkpoint_callback = CheckpointCallback(save_freq=3000, save_path=model_dir,
                                             name_prefix='ppo_model')
    #################################################
    ###############--SETUP PPO-MODEL--###############
    #################################################

    number_feature = (256 + args.nframe * 13)
    pi_arch = [number_feature, int(number_feature / 2), int(number_feature / 4)]
    vi_arch = [number_feature, int(number_feature / 2), int(number_feature / 4)]

    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu)

    if args.load is not None:
        load_path = semipath + "/PPO_" + args.load + "/best_model/best_model.zip"
        model = PPO.load(load_path, env=train_env, device=("cuda:{0}".format(args.gpu)), custom_objects=None,
                         print_system_info=True,
                         force_reset=True)
    else:
        if args.mode != "obs":
            kwargs = dict(
                features_extractor_class=CompassModel,
                features_extractor_kwargs=dict(mode=args.mode,
                                               pretrained_encoder_path=os.environ["COMPASS_CKPT"],
                                               feature_size=number_feature),
                # features_extractor_class=SimpleCNNFE,
                # features_extractor_kwargs=dict(
                #        features_dim=256),
                activation_fn=torch.nn.Tanh,
                net_arch=[ dict(pi=pi_arch, vf=vi_arch)],
                # Number hidden layer 1-3 TODO last layer?
                log_std_init=-0.5,
                normalize_images=False,
                optimizer_kwargs=dict(weight_decay=0, betas=(0.9, 0.999), eps=1e-08, amsgrad=False),  # , maximize=False
                # Adam optimizer TODO
                optimizer_class=torch.optim.Adam
            )
        else:
            kwargs = None

        model = PPO(
            tensorboard_log=log_dir,
            policy="MultiInputPolicy",
            policy_kwargs=kwargs,
            env=train_env,
            gamma=0.999,  # Discout factor old 0.99 IMPORTANT 0.8,0.9997-0.99
            seed=args.seed,
            ent_coef=0.002,  # Range:  0 - 0.01
            vf_coef=0.75,  # OLD 0.5 Range 0.5-1
            max_grad_norm=0.5,
            clip_range=0.35,  # OLD 0.2
            learning_rate=0.0003,  # OLD 0.0003 Range: 1e-5 - 1e-3
            gae_lambda=0.9,  # OLD 95 Range 0.9-1
            use_sde=False,  # action noise exploration vs GsDSE(true)
            target_kl=None,  # Range: 0.003 - 0.03 IMPORTANT?? TODO
            verbose=1,
            n_epochs=10,  # Range: 3 - 30
            batch_size=500,  # num batch != num env!! to use train env, as eval env need to use 1 num env!
            n_steps=500,  # Ragne: 512-5000

            # env_cfg=cfg, OLD PPO
            # eval_env=train_env, OLD PPO
            # eval_env=eval_env, OLD PPO
            # use_tanh_act=True, OLD PPO
            # gae_lambda=0.95,

        )
    # model.learn(total_timesteps=int(5 * 1e7), log_interval=5,
    #             callback=[custom_callback, eval_callback, checkpoint_callback])
    train_loop(model, callback=[custom_callback, eval_callback, checkpoint_callback],
               log=5, easy=0, medium=50, total=90)

    logging.info("Train ended!!!")


if __name__ == "__main__":
    main()
