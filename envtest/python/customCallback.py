from flightgym import VisionEnv_v1

from stable_baselines3.common.callbacks import BaseCallback
import random


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

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        if self.num_timesteps >= self.start_counter:

            diff = ["easy", "medium", "hard"]
            if self.start_counter < 500000:
                new_diff = diff[0]

            elif 500000 < self.start_counter < 1500000:
                new_diff = diff[1]
            else:
                new_diff = diff[2]

            new_lvl = random.randint(0, 100)

            # new_lvl = random.randint(0, 100)
            # new_diff = diff[random.randint(0, 2)]
            self.training_env.change_obstacles(level=new_lvl, difficult=new_diff)
            self.start_counter += self.trigg_freq
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
