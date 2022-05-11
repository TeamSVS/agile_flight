#!/usr/bin/env python3

import argparse
import os
import random
import logging
import cv2
import numpy as np
from flightgym import VisionEnv_v1
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper_old as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump
import time

def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--render", type=int, default=1, help="Render with Unity")
    return parser


def main():
    args = parser().parse_args()

    # load configurations
    cfg = YAML().load(
        open(
            os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
        )
    )
    cfg["unity"]["input_port"] = 10253
    cfg["unity"]["output_port"] = 10254

    if args.render:
        # to connect unity
        cfg["unity"]["render"] = "yes"
        # to simulate rgb camera
        cfg["rgb_camera"]["on"] = "yes"

    # load the Unity standardalone, make sure you have downloaded it.
    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 &")

    # define the number of environment for parallelization simulation
    cfg["simulation"]["num_envs"] = 1

    # create training environment
    env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    env = wrapper.FlightEnvVec(env)

    ep_length = 1000000

    obs_dim = env.obs_dim
    act_dim = env.act_dim
    num_env = env.num_envs

    env.reset(random=True)

    # connect unity
    if args.render:
        env.connectUnity()

    test = False
    for frame_id in range(ep_length):
        logging.info("Simuation step: {0}".format(frame_id))
        # generate dummmy action [-1, 1]
        dummy_actions = np.random.rand(num_env, act_dim) * 2 - np.ones(shape=(num_env, act_dim))

        # A standard OpenAI gym style interface for reinforcement learning.
        time.sleep(0.05)
        dummy_actions[0][0] = -0.697 #-0.697
        dummy_actions[0][1] = 0 # ruota a destra
        dummy_actions[0][2] = 0.2 # ruota avanti
        dummy_actions[0][3] = 0.0  # su se stesso

        obs, rew, done, info = env.step(dummy_actions)
        #print(env.getQuadState()[0][3:7])  # 0-1,5-6, 6-7
        #
        receive_frame_id = env.render(frame_id=frame_id)


if __name__ == "__main__":
    main()
