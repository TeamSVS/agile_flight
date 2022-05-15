from matplotlib.pyplot import plot
import numpy as np 
from ruamel.yaml import YAML, dump, RoundTripDumper

from src.model.quadrotor_model_generation import generate
from src.controllers import setpoint_controller
from src.model_test.quadrotor_model_test import plot_solution

from src.params import flightmare_step_response
from src.flightmare.flightmare_common import obs_to_state, create_env

import time

env = create_env()
# Uses FLU frame
# Thrust is acceleration

params = flightmare_step_response.params()
generate(params)
controller = setpoint_controller.init_controller()

done = False
control_ctbr_not_motors_directly = True


#FLU
state = np.array([
    0, 0, 10, # pos
    0, 1, 0, 0, # quat
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

#FRD
env.wrapper.setResetState(state_full)
obs = env.reset()
x = obs_to_state(obs, params)
x_target = np.array([
    0, 0, -10, # pos
    1, 0, 0, 0, # quat
    0, 0, 0, # vel
    0, 0, 0, # ang vel
])

ts, simX, simU, info = setpoint_controller.control(controller, x, x_target, params)
realX = []
realX.append(obs_to_state(obs, params))
env.connectUnity()
for t, x, u in zip(ts, simX, simU):
    w_source = simX[1, (3+4+3):]
    w_target = params["toTargetRotationMatrix"] @ w_source
    orientationSetpointTarget = params["toTargetRotation"] * np.quaternion(*simX[1, 3:7]) * params["fromTargetRotation"]

    # create AttitudeTarget message containing the control input u (angular velocity) and setting the angular velocity flags
    # thrust_target = (toTargetRotation @ combined_thrust(propeller_thrust_model(u, params)))[2]
    thrust_target = u.sum()
    acceleration = thrust_target / params["mass"]
    act = np.array([[acceleration, *w_target]], dtype=np.float32)
    obs, rew, done, infos = env.step(act)
    realX.append(obs_to_state(obs, params))
    time.sleep(0.1)

env.disconnectUnity()

realX = np.array(realX)

plot_solution(ts, simX, simU, realX=realX, save_fig="flip-recovery-step-response-other.pdf")


print("end")