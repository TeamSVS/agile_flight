#!/usr/bin/python3
import os
import time
from pickle import NONE
from threading import Thread, Event

from ruamel.yaml import YAML
from stable_baselines3 import PPO
import numpy as np

from mpc_files.MPC2 import actual_mpc
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
VISION_EVAL = 1

###################### MODEL_VARIABLE ###################
dict_state_key = "state"
dict_image_depth_key = "depth"
dict_image_rgb_key = "rgb"
dict_obs_key = "obs"

is_defined = False
###################### STACKED_VARIABLE ###################
stacked_drone_state = []
stacked_depth_imgs = []
stacked_rgb_imgs = []
n_frames = 3

stopFlag = Event()
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
last_state_vision = {
    dict_state_key: np.zeros((1, 13)),
    dict_image_depth_key: np.zeros((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH))
}

last_state_st_based = {
    dict_state_key: np.zeros((1, 13)),
    dict_obs_key: np.zeros((1, 40))
}

last_action_vision = np.array([[0.68, 0, 0, 0]])
last_action_st_based = np.array([[0, 0, 0, 0]])
last_seen_state = None


class PingThreadVision(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        global last_action_vision
        global last_seen_state
        while True:
            time.sleep(0.00005)
            while not self.stopped.wait(0.00001):
                ############ NORMALIZE_ACTION #########
                # action = (action * act_std + act_mean)[0, :]
                if last_seen_state is not None:
                    action, _ = model_ppo.predict(last_state_vision, deterministic=True)
                    width = (action[0][0] * 1) - int(cfg["rgb_camera"]["width"] / 2)  # width
                    height = (action[0][1] * 1) - int(cfg["rgb_camera"]["height"] / 2)  # height
                    depth = action[0][2]
                    pos = last_seen_state.pos
                    vel = last_seen_state.vel
                    att = last_seen_state.att
                    omega = last_seen_state.omega
                    last_action_vision = actual_mpc(float(depth), width, height, pos, vel, att, omega)


class PingThreadStateBased(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        global last_action_st_based
        while True:
            time.sleep(0.00005)
            while not self.stopped.wait(0.00001):
                last_action_st_based, _ = model_ppo.predict(last_state_st_based, deterministic=True)


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


# load_path = "/home/cam/Modelli-decenti/best_model/best_model.zip"  # OBS
load_path = "/home/cam/Modelli-decenti/PPO_0528_Xline_mpc_butCrashesDown/best_model/best_model.zip"  # OBS

model_ppo = PPO.load(load_path, device="cuda:0", custom_objects=None,
                     print_system_info=True,
                     kwargs=dict(policy_kwargs=dict(
                         features_extractor_kwargs=dict(
                             pretrained_encoder_path="PIPPO",
                         ),
                     )),
                     force_reset=True)

if VISION_EVAL != 0:
    thread = PingThreadVision(stopFlag)
    model_ppo.predict(last_state_vision, deterministic=True)
else:
    thread = PingThreadStateBased(stopFlag)
    model_ppo.predict(last_state_st_based, deterministic=True)

thread.daemon = True
thread.start()


def compute_command_vision_based(state, img):
    global model_ppo
    global last_seen_state
    global last_action_vision
    #    global stacked_drone_state
    global stacked_depth_imgs
    global stacked_rgb_imgs
    global last_state_vision
    ############ INIT_MODEL #########
    print("Computing command VISION")
    last_seen_state = state
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

    last_state_vision = {
        dict_state_key: stacked_new_state,
        dict_image_depth_key: new_img
    }

    real_action = last_action_vision
    # CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = real_action[0]
    command.bodyrates = [real_action[1], real_action[2], real_action[3]]
    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    global model_ppo
    global stacked_drone_state
    global stacked_depth_imgs
    global stacked_rgb_imgs

    ############ INIT_MODEL #########
    print("Computing command STATE")

    stacked_drone_state = _stack_frames(stacked_drone_state, load_state(state))
    stacked_new_state = np.array(stacked_drone_state)
    # stacked_new_state = np.expand_dims(stacked_new_state, 0)

    stacked_new_state = normalize_state_obs(stacked_new_state)
    stacked_new_state = stacked_new_state.swapaxes(0, 1)
    new_obs = []
    for x in obstacles.obstacles:
        partial_vec1 = [x.position.x, x.position.y, x.position.z, x.scale]
        new_obs.append(x.position.x)
        new_obs.append(x.position.y)
        new_obs.append(x.position.z)
        new_obs.append(x.scale)

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
    command.collective_thrust = 0.68
    command.bodyrates = [0, 0, 0]
    return command
