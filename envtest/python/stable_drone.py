#!/usr/bin/env python3

import argparse
import os
import random
import logging
import cv2
import numpy as np
from flightgym import VisionEnv_v1
from rpg_baselines.torch.envs import vec_env_wrapper as wrapper
from ruamel.yaml import YAML, RoundTripDumper, dump


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

    if args.render:
        # to connect unity
        cfg["unity"]["render"] = "yes"
        # to simulate rgb camera
        cfg["rgb_camera"]["on"] = "yes"

    # load the Unity standardalone, make sure you have downloaded it.
    os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 -batchmode &")

    # define the number of environment for parallelization simulation
    cfg["simulation"]["num_envs"] = 1

    # create training environment
    env = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
    env = wrapper.FlightEnvVec(env)

    ep_length = 100000

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

        dummy_actions[0][0] = -0.693
        dummy_actions[0][1] = 0.  # ruota a destra
        dummy_actions[0][2] = 0.  # ruota avanti
        dummy_actions[0][3] = 0.  # su se stesso

        obs, rew, done, info = env.step(dummy_actions)

        #
        receive_frame_id = env.render(frame_id=frame_id)


if __name__ == "__main__":
    main()
