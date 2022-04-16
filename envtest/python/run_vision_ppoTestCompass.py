#!/usr/bin/env python3
import argparse
import math
#
import os
import cv2

import numpy as np
import torch
from flightgym import VisionEnv_v1
from ruamel.yaml import YAML, RoundTripDumper, dump
from stable_baselines3.common.utils import get_device
from stable_baselines3.ppo.policies import MlpPolicy

from rpg_baselines.torch.common.ppo import PPO
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from rpg_baselines.torch.common.util import test_policy
from compass_custom_feature_extractor import CompassFE
from threading import Thread

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

def printer1(env):
    import time
    env.connectUnity()

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    num_env = env.num_envs
    for frame_id in range(1,100000):
        print("Simulation step: {0}".format(frame_id))
        # generate dummy action [-1, 1]
        dummy_actions = np.random.rand(num_env, act_dim) * 2 - np.ones(shape=(num_env, act_dim))

        # A standard OpenAI gym style interface for reinforcement learning.
       # obs, rew, done, info = env.step(dummy_actions)

        #
        receive_frame_id = env.render(frame_id=frame_id)
        print("sending frame id: ", frame_id, "received frame id: ", receive_frame_id)

        # ====== Retrieve RGB Image From the simulator=========
        raw_rgb_img = env.getImage(rgb=True)
        print(raw_rgb_img)
        num_img = raw_rgb_img.shape[0]
        num_col = 1
        num_row = int(num_img / num_col)

        rgb_img_list = []
        for col in range(num_col):
            rgb_img_list.append([])
            for row in range(num_row):
                rgb_img = np.reshape(
                    raw_rgb_img[col * num_row + row], (env.img_height, env.img_width, 3))
                rgb_img_list[col] += [rgb_img]

        rgb_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in rgb_img_list])
        cv2.imshow("rgb_img", rgb_img_tile)
        # cv2.imwrite("./images/img_{0:05d}.png".format(frame_id), rgb_img_tile)
        # wait for the purpose of using open cv visualization
        cv2.waitKey(500)

        # ======Retrieve Depth Image=========
        raw_depth_images = env.getDepthImage()[0]
        depth_img_list = []
        for col in range(num_col):
            depth_img_list.append([])
            for row in range(num_row):
                depth_img = np.reshape(
                    raw_depth_images, (env.img_height, env.img_width))
                depth_img_list[col] += [depth_img]

        depth_img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in depth_img_list])
        cv2.imshow("depth_img", depth_img_tile)
        # wait for the purpose of using open cv visualization
        cv2.waitKey(500)

def printer2(env):
    import time
    while 100:
        time.sleep(1)
        print(env.getObs())
        print(env.getExtraInfo())


def main():
    args = parser().parse_args()



    # load configurations

    if not args.train:
        cfg["simulation"]["num_envs"] = 1

        # create training environment
    train_env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    train_env = wrapper.FlightEnvVec(train_env)

    # set random seed
    configure_random_seed(args.seed, env=train_env)

    if args.render:
        cfg["unity"]["render"] = "yes"
        cfg["rgb_camera"]["on"] = "yes"
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    # create evaluation environment
    cfg["simulation"]["num_envs"] = 1

    #old_num_envs = cfg["simulation"]["num_envs"]
    #eval_env = wrapper.FlightEnvVec(
     #   VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    #)
   # cfg["simulation"]["num_envs"] = old_num_envs

    #configure_random_seed(args.seed, env=eval_env)
    configure_random_seed(args.seed, env=train_env)


    # save the configuration and other files
    rsg_root = os.path.dirname(os.path.abspath(__file__))
    log_dir = rsg_root + "/saved"
    os.makedirs(log_dir, exist_ok=True)

    train_env.connectUnity()

    if args.train:

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

            model = PPO(

                tensorboard_log=log_dir,
                policy="MlpPolicy",
                policy_kwargs=dict(
                    features_extractor_class=CompassFE,
                    features_extractor_kwargs=dict(features_dim=5,env = train_env),
    #                activation_fn=torch.nn.ReLU,
                    net_arch=[256, dict(pi=[256, 256], vf=[512, 512])],
                    log_std_init=-0.5,
                ),
                env=train_env,
                 #eval_env=eval_env,
                use_tanh_act=True,
                gae_lambda=0.95,
                gamma=0.99,
                n_steps=250,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                batch_size=25000,
                clip_range=0.2,
                use_sde=False,  # don't use (gSDE), doesn't work
                env_cfg=cfg,
               verbose=1,
            )

            #Thread(target=printer1, args=[train_env]).start()
            model.learn(total_timesteps=int(5 * 1e7), log_interval=(10, 50))


            """
            obs_dim = eval_env.obs_dim
            act_dim = eval_env.act_dim
            num_env = eval_env.num_envs
    
    
            while 1000:
                dummy_actions = np.random.rand(num_env, act_dim) * 2 - np.ones(shape=(num_env, act_dim))
                eval_env.step(dummy_actions)
                print(train_env.getImage())
            
            """




    else:
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")
        #
        weight = rsg_root + "/saved/PPO_{0}/Policy/iter_{1:05d}.pth".format(args.trial, args.iter)
        env_rms = rsg_root + "/saved/PPO_{0}/RMS/iter_{1:05d}.npz".format(args.trial, args.iter)

        device = get_device("auto")
        saved_variables = torch.load(weight, map_location=device)
        # Create policy object
        policy = MlpPolicy(**saved_variables["data"])
        #
        policy.action_net = torch.nn.Sequential(policy.action_net, torch.nn.Tanh())
        # Load weights
        policy.load_state_dict(saved_variables["state_dict"], strict=False)
        policy.to(device)
        # 
        #eval_env.load_rms(env_rms)
        test_policy(eval_env, policy, render=args.render)


if __name__ == "__main__":
    main()
