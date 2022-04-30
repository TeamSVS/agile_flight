from flightgym import VisionEnv_v1

from stable_baselines3.common.callbacks import BaseCallback
import random

import signal
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

QUADSTATE_NAMES = ["0_pos_z", "1_pos_x", "2_pos_y", "3_att_w", "4_att_z", "5_att_x", "6_att_y", "7_vel_z", "8_vel_x",
                   "9_vel_y", "10_ome_z", "11_ome_x", "12_ome_y"]
QUADSTATE_RANGE = len(QUADSTATE_NAMES)


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0, trigg_freq=0):
        super(CustomCallback, self).__init__(verbose)
        self.trigg_freq = trigg_freq
        self.start_counter = trigg_freq
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        #
        # if self.num_timesteps >= self.start_counter:
        #
        #     diff = ["easy", "medium", "hard"]
        #     if self.start_counter < 500000:
        #         new_diff = diff[0]
        #
        #     elif 500000 < self.start_counter < 1500000:
        #         new_diff = diff[1]
        #     else:
        #         new_diff = diff[2]
        #
        #     new_lvl = random.randint(0, 100)
        #
        #     # new_lvl = random.randint(0, 100)
        #     # new_diff = diff[random.randint(0, 2)]
        #     self.training_env.change_obstacles(level=new_lvl, difficult=new_diff)
        #     self.start_counter += self.trigg_freq
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

    def _init_callback(self) -> None:
        self.drone_states = [[[] for _ in range(QUADSTATE_RANGE)] for _ in range(self.training_env.num_envs)]
        self.total_speed = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        for drone_id in range(self.training_env.num_envs):
            if self.locals["dones"][drone_id] == False:
                for state_id in range(QUADSTATE_RANGE):
                    self.drone_states[drone_id][state_id].append(self.training_env._quadstate[drone_id, state_id])
                self.total_speed[drone_id].append(np.sum(np.abs(self.training_env._quadstate[drone_id, 7:10])))
            else:
                for state_id in range(QUADSTATE_RANGE):
                    if len(self.drone_states[drone_id][state_id]) > 0:
                        self.logger.record("state/" + QUADSTATE_NAMES[state_id] + "_max",
                                           np.max(self.drone_states[drone_id][state_id]), exclude="stdout")
                        self.logger.record("state/" + QUADSTATE_NAMES[state_id] + "_min",
                                           np.min(self.drone_states[drone_id][state_id]), exclude="stdout")

                if len(self.total_speed[drone_id]) > 0:
                    self.logger.record("speed/" + "speed_max", np.max(self.total_speed[drone_id]))
                    self.logger.record("speed/" + "speed_avg", np.mean(self.total_speed[drone_id]))
                    self.logger.record("speed/" + "z_speed_avg", np.mean(self.drone_states[drone_id][7]),
                                       exclude="stdout")
                    self.logger.record("speed/" + "x_speed_avg", np.mean(self.drone_states[drone_id][8]),
                                       exclude="stdout")
                    self.logger.record("speed/" + "y_speed_avg", np.mean(self.drone_states[drone_id][9]),
                                       exclude="stdout")

                self.drone_states[drone_id] = [[] for _ in range(QUADSTATE_RANGE)]
                self.total_speed[drone_id] = []

        return True


class HandleControlC(BaseCallback):
    def __init__(self, verbose=0):
        super(HandleControlC, self).__init__(verbose)
        self.continue_training = True
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print('Control-C detected.')
        print('Ending training and saving model... Please wait.')
        self.continue_training = False
        self.training_env.controlC_flag = True

    def _on_step(self) -> bool:
        return self.continue_training
