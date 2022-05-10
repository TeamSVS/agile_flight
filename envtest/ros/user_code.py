#!/usr/bin/python3


from pickle import NONE
from utils import AgileCommandMode, AgileCommand
from rl_example import rl_example


def compute_command_vision_based(state, img, rl_model):
    print("Computing command vision-based!")
    new_state = []
    state_elem = [state.pos, state.att, state.vel, state.omega]
    new_state.append(state.t)
    for el in state_elem:
        new_state.extend(el)

    # print(state)
    # print("Image shape: ", img.shape)
    # state.t,state.pos,state.att,state.vel,state.omega
    # rl_model.predict()
    # Example of CTBR command
    # print(state.shape)
    print(img.shape)
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

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
