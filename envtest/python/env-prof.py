
from typing import Any, List, Type
import random
import os

import numpy as np
import gym
from gym import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices

from ruamel.yaml import YAML, RoundTripDumper, dump
from flightgym import VisionEnv_v1


IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
MAX_EPISODE_LENGTH = 50

X_PROGRESS_REWARD_WEIGHT = 1

class VisualFlightEnvVec(VecEnv):

    def __init__(self, num_envs, obs_type="combined", seed=None):
        self.num_envs = num_envs
        self.obs_type = obs_type
        self._seed = seed if seed is not None else random.randrange(2**31)
        ######################################################
        # creation of the original environment (wrapper)
        ######################################################
        cfg = YAML().load(
            open(
                os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
            ))
        cfg["unity"]["render"] = "yes"
        cfg["rgb_camera"]["on"] = "yes"
        cfg["rgb_camera"]["enable_depth"] = "yes"
        cfg["rgb_camera"]["width"] = IMAGE_WIDTH
        cfg["rgb_camera"]["height"] = IMAGE_HEIGHT
        cfg["simulation"]["max_t"] = MAX_EPISODE_LENGTH
        cfg["simulation"]["num_envs"] = num_envs
        cfg["simulation"]["seed"] = self._seed

        self.wrapper = VisionEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
        self.wrapper.setSeed(self._seed)
        os.system(os.environ["FLIGHTMARE_PATH"] + "/flightrender/RPG_Flightmare.x86_64 -input-port " + str(cfg["unity"]["input_port"]) + " -output-port " + str(cfg["unity"]["output_port"]) + " &")
        self.wrapper.connectUnity()

        ######################################################
        # setup data structures used as a bridge with the original environment
        #
        # note that the contents and the dimensions of obs, reward, done computed in the method "step"
        # of this class differs from the one returned by the original environment
        ######################################################
        self.act_dim = self.wrapper.getActDim()
        self.obs_dim = self.wrapper.getObsDim()
        self.rew_dim = self.wrapper.getRewDim()
        self.img_width = self.wrapper.getImgWidth()
        self.img_height = self.wrapper.getImgHeight()

        self._observation = np.zeros([self.num_envs, self.obs_dim], dtype=np.float64)
        self._rgb_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height * 3], dtype=np.uint8)
        self._gray_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height], dtype=np.uint8)
        self._depth_img_obs = np.zeros([self.num_envs, self.img_width * self.img_height], dtype=np.float32)
        self._reward_components = np.zeros([self.num_envs, self.rew_dim], dtype=np.float64)
        self._done = np.zeros((self.num_envs), dtype=np.bool)
        self._extraInfo = np.zeros([self.num_envs, len(self.wrapper.getExtraInfoNames())], dtype=np.float64)
        self._quadstate = np.zeros([self.num_envs, 25], dtype=np.float64)
        self._quadact = np.zeros([self.num_envs, 4], dtype=np.float64)
        self._flightmodes = np.zeros([self.num_envs, 1], dtype=np.float64)

        # list to track episode rewards to log the results
        self.rewards = [[] for _ in range(self.num_envs)]

        self.frame_id = 0
        self.max_x_reached = np.zeros((self.num_envs))

        ######################################################
        # Gym spaces of the environment (the actual dimensions for this environment's obs and action)
        ######################################################
        depth_space = spaces.Box(
            low=0, high=1,
            shape=(1, self.img_height, self.img_width), dtype=np.float64
        )
        drone_state_space = spaces.Box(
            low=-np.Inf, high=np.Inf,
            shape=(13,), dtype=np.float64
        )
        combined_space = spaces.Dict(
            spaces={
                "drone_state": drone_state_space,
                "depth": depth_space,
            }
        )

        if self.obs_type == "combined":
            self.observation_space = combined_space
        elif self.obs_type == "depth":
            self.observation_space = depth_space

        self.action_space = spaces.Box(
            low=np.ones(self.act_dim) * -1.0,
            high=np.ones(self.act_dim) * 1.0,
            dtype=np.float64,
        )


    def step(self, action):
        if action.ndim <= 1:
            action = action.reshape((-1, self.act_dim))
        self.wrapper.step(
            action,
            self._observation,
            self._reward_components,
            self._done,
            self._extraInfo,
        )
        self.frame_id = self.wrapper.updateUnity(self.frame_id)

        obs = self.get_obs()
        reward = self.get_rewards()
        done = self._done.copy()
        info = self.get_info(reward)
        return (obs, reward, done, info)


    def reset(self, random=True):
        self.wrapper.reset(self._observation, random)
        self.frame_id = 0
        self.max_x_reached = np.zeros((self.num_envs))
        return self.get_obs()




    def get_obs(self):
        depth_imgs = self.getDepthImage().reshape((self.num_envs, 1, self.img_height, self.img_width))
        # position (z, x, y) = [0:3], attitude=[3:7], linear_velocity=[7:10], angular_velocity=[10:13]
        drone_state = self.getQuadState()[:,:13]
        if self.obs_type == "combined":
            obs = {"drone_state": drone_state, "depth": depth_imgs}
        elif self.obs_type == "depth":
            obs = depth_imgs
        return obs

    def get_rewards(self):
        rewards = []
        for i in range(self.num_envs):
            if self._done[i]:
                #if done, last reward component is
                # -1 for collision or exiting bounding box
                # 0 if max time elapsed
                rewards.append(self._reward_components[i, -1])
                self.max_x_reached[i] = 0
            else:
                x_pos = self._quadstate[i, 1]
                x_progress = x_pos - self.max_x_reached[i]
                if x_pos > self.max_x_reached[i]:
                    self.max_x_reached[i] = x_pos
                x_progress_reward = x_progress if x_progress > 0 else 0
                rewards.append(x_progress_reward * X_PROGRESS_REWARD_WEIGHT)
        return rewards

    def get_info(self, reward):
        info = [{} for _ in range(self.num_envs)]
        # log episode total reward and lenght
        for i in range(self.num_envs):
            self.rewards[i].append(reward[i])
            if self._done[i]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                info[i]["episode"] = {"r": eprew, "l": eplen}
                self.rewards[i].clear()
        return info



    def getImage(self, rgb=False):
        if rgb:
            self.wrapper.getImage(self._rgb_img_obs, True)
            return self._rgb_img_obs.copy()
        else:
            self.wrapper.getImage(self._gray_img_obs, False)
            return self._gray_img_obs.copy()

    def getDepthImage(self):
        self.wrapper.getDepthImage(self._depth_img_obs)
        return self._depth_img_obs.copy()

    def getQuadState(self):
        self.wrapper.getQuadState(self._quadstate)
        return self._quadstate

    def getQuadAct(self):
        self.wrapper.getQuadAct(self._quadact)
        return self._quadact




    def close(self):
        self.wrapper.disconnectUnity()
        self.wrapper.close()

    def seed(self, seed=None):
        if seed is None:
            seed = random.randrange(2**31)
        self.wrapper.setSeed(seed)

    def sample_actions(self):
        actions = []
        for _ in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float64)





############################
#
# Required by abstract class VecEnv
#
############################

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        raise RuntimeError("This method is not implemented")

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        # the implementation in the original file gives runtime error
        # here I return true as I don't have access to the single env
        # but it should be considered when a callback using this method is used
        return [True]

    def step_async(self):
        raise RuntimeError("This method is not implemented")

    def step_wait(self):
        raise RuntimeError("This method is not implemented")

    def get_attr(self, attr_name, indices=None):
        raise RuntimeError("This method is not implemented")

    def set_attr(self, attr_name, value, indices=None):
        raise RuntimeError("This method is not implemented")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise RuntimeError("This method is not implemented")
