#!/usr/bin/env python3
import argparse
import glob
import logging
import os
import sys
import random

sys.path.append('/home/zaks/icra22_competition_ws/src/agile_flight/')


import numpy as np
import torch
from ruamel.yaml import YAML
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from customCallback import CustomCallback
from dronenavigation.models.compass.compass_model import CompassModel
# from flightmare.flightpy.flightrl.rpg_baselines.torch.common.ppo import PPO
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper
logging.basicConfig(level=logging.WARNING)

######################################
##########--COSTANT VALUES--##########
######################################

ENVIRONMENT_CHANGE_THRESHOLD = 50000

cfg = YAML().load(
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
    ################################################
    ###############--LOAD CFG ENV 1--###############
    ################################################


    train_env = wrapper.FlightEnvVec(cfg, "train", "rgb")
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

    model = PPO(
        tensorboard_log=log_dir,
        policy="MultiInputPolicy",
        policy_kwargs=dict(
            features_extractor_class=CompassModel,
            features_extractor_kwargs=dict(linear_prob=True,
                                           pretrained_encoder_path=os.environ["COMPASS_CKPT"],
                                           feature_size=269),
            # features_extractor_class=SimpleCNNFE,
            # features_extractor_kwargs=dict(
            #        features_dim=256),
            activation_fn=torch.nn.ReLU,
            net_arch=[269, dict(pi=[269, 134, 67], vf=[269, 134, 67])],
            # Number hidden layer 1-3 TODO last layer?
            log_std_init=-0.5,
            normalize_images=False,
            # optimizer_kwargs=dict(weight_decay=0, betas=0.9),  # Adam optimizer TODO
        ),

        env=train_env,
        gamma=0.98,  # Discout factor old 0.99 IMPORTANT 0.8,0.9997-0.99
        seed=args.seed,
        ent_coef=0.002,  # Range:  0 - 0.01
        vf_coef=0.75,  # OLD 0.5 Range 0.5-1
        max_grad_norm=0.5,
        clip_range=0.25,  # OLD 0.2
        learning_rate=0.0003,  # OLD 0.0003 Range: 1e-5 - 1e-3
        gae_lambda=0.9,  # OLD 95 Range 0.9-1
        use_sde=False,  # action noise exploration vs GsDSE(true)
        target_kl=None,  # Range: 0.003 - 0.03 IMPORTANT?? TODO
        verbose=1,
        n_epochs=10,  # Range: 3 - 30
        batch_size=10,  # num batch != num env!! to use train env, as eval env need to use 1 num env!
        n_steps=10,  # Ragne: 512-5000

        # env_cfg=cfg, OLD PPO
        # eval_env=train_env, OLD PPO
        # eval_env=eval_env, OLD PPO
        # use_tanh_act=True, OLD PPO
        # gae_lambda=0.95,

    )

    model.learn(total_timesteps=int(5 * 1e7), log_interval=5,
                callback=[custom_callback, eval_callback, checkpoint_callback])

    logging.info("Train ended!!!")


if __name__ == "__main__":
    main()
