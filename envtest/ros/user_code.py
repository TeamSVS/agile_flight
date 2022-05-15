#!/usr/bin/python3
import os
from pickle import NONE
from ruamel.yaml import YAML
from stable_baselines3 import PPO
import numpy as np
from utils import AgileCommandMode, AgileCommand
from rl_example import rl_example

###################### LOAD_YAML ###################
cfg = YAML().load(
    open(
        os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
    )
)
###################### ACTION_NORMALIZATION_PARAMS ###################
quad_mass = cfg["quadrotor_dynamics"]["mass"]
omega_max = cfg["quadrotor_dynamics"]["omega_max"]
thrust_max = 4 * cfg["quadrotor_dynamics"]["thrust_map"][0] * \
             cfg["quadrotor_dynamics"]["motor_omega_max"] * \
             cfg["quadrotor_dynamics"]["motor_omega_max"]
act_mean = np.array([thrust_max / quad_mass / 2, 0.0, 0.0, 0.0])[np.newaxis, :]
act_std = np.array([thrust_max / quad_mass / 2, \
                    omega_max[0], omega_max[1], omega_max[2]])[np.newaxis, :]

###################### MODEL_VARIABLE ###################
dict_state_key = "state"
dict_image_depth_key = "depth"
dict_image_rgb_key = "rgb"
dict_obs_key = "obs"

is_defined = False
###################### STACKED_VARIABLE ###################
# stacked_drone_state = []
stacked_depth_imgs = []
stacked_rgb_imgs = []
n_frames = 3


def _normalize(value, min, max):
    return 2 * (value - min) / (max - min) - 1


def normalize_img_obs(drone_state):
    drone_state[:, 0] = _normalize(drone_state[:, 0], -10, 65)
    drone_state[:, 1] = _normalize(drone_state[:, 1], -10, 10)
    drone_state[:, 2] = _normalize(drone_state[:, 2], 0, 10)
    drone_state[:, 7] = _normalize(drone_state[:, 7], -3, 65)
    drone_state[:, 8] = _normalize(drone_state[:, 8], -25, 30)
    drone_state[:, 9] = _normalize(drone_state[:, 9], -20, 20)
    drone_state[:, 10] = _normalize(drone_state[:, 10], -9, 9)
    drone_state[:, 11] = _normalize(drone_state[:, 11], -9, 9)
    drone_state[:, 12] = _normalize(drone_state[:, 12], -9, 9)

    return drone_state.copy()


def normalize_state_obs(obstacles):
    obstacles[:, 0::4] = _normalize(obstacles[:, 0::4], -8, 1010)
    obstacles[:, 1::4] = _normalize(obstacles[:, 1::4], -20, 1008)
    obstacles[:, 2::4] = _normalize(obstacles[:, 2::4], -10, 1000)
    obstacles[:, 3::4] = _normalize(obstacles[:, 3::4], 0, 1.5)

    return obstacles.copy()


def _stack_frames(frame_list, new_frame):
    if len(frame_list) == 0:
        frame_list = [new_frame for _ in range(n_frames)]
    else:
        frame_list = frame_list[:n_frames - 1]
        frame_list.insert(0, new_frame)
    return frame_list


def load_state(state, buggy=False):
    new_state = []
    state_elem = [state.pos, state.att, state.vel, state.omega]
    # new_state.append(state.t)  # For old buggy model
    for el in state_elem:
        new_state.extend(el)
    return new_state


# load_path = "/home/cam/Desktop/sim-ros2/icra22_competition_ws/src/agile_flight/envtest/python/saved/PPO_0301/best_model/best_model.zip"  # depth 3 frame
load_path = "/home/cam/Desktop/sim-ros2/icra22_competition_ws/src/agile_flight/envtest/python/saved/PPO_0300/best_model/best_model.zip"  # OBS

model_ppo = PPO.load(load_path, device="cuda:0", custom_objects=None,
                     print_system_info=True,
                     force_reset=True)


def compute_command_vision_based(state, img):
    global model_ppo
    #    global stacked_drone_state
    global stacked_depth_imgs
    global stacked_rgb_imgs
    ############ INIT_MODEL #########
    print("Computing command VISION")

    ############ LOAD_STATE #########

    # stacked_drone_state = _stack_frames(stacked_drone_state, load_state(state))
    stacked_new_state = np.array(load_state(state))
    stacked_new_state = np.expand_dims(stacked_new_state, 0)
    stacked_new_state = normalize_img_obs(stacked_new_state)
    # stacked_new_state = stacked_new_state.swapaxes(0, 1)

    ############ LOAD_IMAGE #########

    stacked_depth_imgs = _stack_frames(stacked_depth_imgs, img)
    new_img = np.array(stacked_depth_imgs)
    new_img = np.expand_dims(new_img, 0)

    obs = {
        dict_state_key: stacked_new_state,
        dict_image_depth_key: new_img
    }

    action, _ = model_ppo.predict(obs, deterministic=True)
    ############ NORMALIZE_ACTION #########
    action = (action * act_std + act_mean)[0, :]

    # CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = action[0]
    command.bodyrates = [action[1], action[2], action[3]]
    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    global model_ppo
    # global stacked_drone_state
    global stacked_depth_imgs
    global stacked_rgb_imgs

    ############ INIT_MODEL #########
    print("Computing command STATE")

    # stacked_drone_state = _stack_frames(stacked_drone_state, load_state(state))
    stacked_new_state = np.array(load_state(state))
    stacked_new_state = np.expand_dims(stacked_new_state, 0)

    stacked_new_state = normalize_state_obs(stacked_new_state)
    # stacked_new_state = stacked_new_state.swapaxes(0, 1)
    new_obs = []
    for x in obstacles.obstacles:
        partial_vec1 = [x.position.x, x.position.y, x.position.z, x.scale]
        new_obs.append(partial_vec1)

    obs = {
        dict_state_key: stacked_new_state,
        dict_obs_key: np.array(new_obs)
    }

    action, _ = model_ppo.predict(obs, deterministic=True)
    ############ NORMALIZE_ACTION #########
    action = (action * act_std + act_mean)[0, :]

    # CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = action[0]
    command.bodyrates = [action[1], action[2], action[3]]
    return command
