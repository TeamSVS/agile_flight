#!/usr/bin/python3


import glob
import os
import sys

import numpy as np
from ruamel.yaml import YAML
from stable_baselines3 import PPO

from rl_example import rl_example
from utils import AgileCommand

sys.path.insert(0, '/home/students/COMPASS-RL/icra22_competition_ws/src/agile_flight')
from flightmare.flightpy.flightrl.rpg_baselines.torch.envs import vec_env_wrapper as wrapper

sys.path.insert(0, '/home/students/COMPASS-RL/icra22_competition_ws/src/agile_flight')

######################################
##########--COSTANT VALUES--##########
######################################


ENVIRONMENT_CHANGE_THRESHOLD = 300

MODE = "depth"  # depth,rgb,both
STARTING_LR = 0.001  # clip-length
FRAME = 3  # clip-length
cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)

actual_lr = STARTING_LR
stacked_drone_state = []
stacked_depth_imgs = []
stacked_rgb_imgs = []
DEPTH_CHANNELS = 1
RGB_CHANNELS = 3
mode = "depth"
num_envs = 1
img_height = cfg["rgb_camera"]["height"]

img_width = cfg["rgb_camera"]["width"]

n_frames = 3

def _normalize_img(obs: np.ndarray) -> np.ndarray:
    return obs / 255

def load_model(path=""):
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
    load_path = semipath + "/PPO_" + path + "/best_model/best_model.zip"
    model = PPO.load(load_path, device='cuda:0', custom_objects=None, print_system_info=True)

    return model


def _stack_frames(frame_list, new_frame):
    if len(frame_list) == 0:
        frame_list = [new_frame for _ in range(n_frames)]
    else:
        frame_list = frame_list[:n_frames - 1]
        frame_list.insert(0, new_frame)
    return frame_list


def get_obs(drone_state, img):
    new_obs = {}
    global stacked_drone_state

    stacked_drone_state = _stack_frames(stacked_drone_state, drone_state)
    new_obs['state'] = np.array(stacked_drone_state).swapaxes(0, 1).swapaxes(1, 2)
    if 'depth' == mode or 'both' == mode:
        global stacked_depth_imgs
        depth_imgs = img.reshape((num_envs, 1, img_height, img_width))
        stacked_depth_imgs = _stack_frames(stacked_depth_imgs, depth_imgs)
        new_obs['depth'] = np.array(stacked_depth_imgs).swapaxes(0, 1).swapaxes(1, 2)
    if 'rgb' == mode or 'both' == mode:
        global stacked_rgb_imgs
        rgb_imgs = _normalize_img(
            np.reshape(img, (num_envs, RGB_CHANNELS, img_width, img_height)))
        stacked_rgb_imgs = _stack_frames(stacked_rgb_imgs, rgb_imgs)
        new_obs['rgb'] = np.array(stacked_rgb_imgs).swapaxes(0, 1).swapaxes(1, 2)
    return new_obs


def compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################

    print("Computing command vision-based!")
    if 'model' not in vars():
        model = load_model("0166")

    obs = get_obs(state, img)
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = PPO.predict(obs)
    #

    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    # If you want to test your RL policy
    if rl_policy is not None:
        command = rl_example(state, obstacles, rl_policy)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command
