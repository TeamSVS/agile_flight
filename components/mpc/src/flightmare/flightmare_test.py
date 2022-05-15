import time
import numpy as np 
import os
from scipy.spatial.transform.rotation import Rotation as R

from src.model.quadrotor_model_generation import generate
from src.model_test.quadrotor_model_test import plot_solution
from src.controllers.setpoint_controller import init_controller, control_motor, control_ctbr

import src.controllers.back_and_forth as back_and_forth

from src.params import flightmare
from .flightmare_common import obs_to_state, create_env


env = create_env()
# Uses FLU frame
# Thrust is acceleration

params = flightmare.params()
generate(params)
controller = init_controller()

env.connectUnity()
done = False
control_ctbr_not_motors_directly = True


state = np.array([
    0, 0, 10, # pos
    1, 0, 0, 0, # quat
    0, 0, 0, # vel
    0, 0, 0, # ang vel
])
state_full = np.array([
    *state,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
], dtype=np.float32)
env.wrapper.setResetState(state_full)
obs = env.reset()
steps = 0
start = time.time()
states = [obs_to_state(obs, params)]
actions = []
infos = []
ts = [0]
while True or not done:
    steps += 1
    x = obs_to_state(obs, params)


    if control_ctbr_not_motors_directly:
        thrust, w_source, info = control_ctbr(controller, x, back_and_forth.setpoint(x), params)
        infos.append(info)
        w_target = params["toTargetRotationMatrix"] @ w_source
        acceleration = thrust / params["mass"]

        # act = np.array([[9.81, 0, 0, 0]], dtype=np.float32)
        act = np.array([[acceleration, *w_target]], dtype=np.float32)
        # print(f"acceleration: {acceleration}")
        # print(f"velocity {np.round(v, 2)}")
    else:
        thrusts = control_motor(x, params)
        act = np.array([[*thrusts]], dtype=np.float32)
        # m = params["mass"]
        # act = np.array([[9.81/4*m, 9.81/4*m, 9.81/4*m, 9.81/4*m]], dtype=np.float32)
    actions.append(act[0])
    obs, rew, done, infos = env.step(act)
    states.append(obs_to_state(obs, params))
    ts.append(ts[-1] + params["sim_dt"])

    # Hack to make the execution about realtime
    now = time.time()
    avg_dt = (now - start)/steps
    if avg_dt < params["sim_dt"]:
        time.sleep(params["sim_dt"])

    if steps % 100 == 0:
        print(f"{steps} steps in {now - start} seconds ({steps / (now - start)} steps/s)")
    if steps > 100:
        break


env.disconnectUnity()

plot_solution(np.array(ts), np.array(states), np.array(actions), save_fig="tf1-N10-flip-recovery-opt-time-0.006-other.pdf")

# print(f"average optimization time: {np.mean(optimization_times)}")