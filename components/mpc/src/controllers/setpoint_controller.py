from ..model.quadrotor_model_execution import create_capsule, control_trajectory
from ..model.quadrotor_model_generation import propeller_thrust_model, combined_thrust, generate
# from quadrotor_model_test import plot_solution
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R
import json
import h5py
import time

def init_controller():
    global capsule, pos_target_source_initial
    pos_target_source_initial = None
    capsule = create_capsule()
    return capsule

def control(capsule, x, x_target, params, u_target=None):
    if u_target is None:
        u_target = np.zeros(len(params["rotor_positions"]))

    sp = np.concatenate([x_target, u_target])
    traj = np.repeat(sp.reshape((1, -1)), params["N"]+1, axis=0)

    start = time.time()
    t, simX, simU = control_trajectory(capsule, x.astype(float), traj.astype(float), params["N"], params["Tf"])
    end = time.time()
    return t, simX, simU, {
        "optimization_time": end - start,
    }

def control_ctbr(capsule, x, x_target, params, u_target=None):
    t, simX, simU, info = control(capsule, x, x_target, params, u_target=u_target)
    # u = simU[3, :].mean(axis=0)
    u = simU[0, :]
    w_source = simX[1, (3+4+3):]
    thrust_target = u.sum()

    return thrust_target, w_source, {
        "optimization_time": info["optimization_time"],
        "t": t,
        "simX": simX,
        "simU": simU,
    }

def control_motor(capsule, x, x_target, params, u_target=None):
    t, simX, simU, info = control(capsule, x, x_target, params, u_target=u_target)
    u = simU[0, :]
    return u


    