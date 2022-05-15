import numpy as np
import quaternion

FRDinRFU = np.array([
    [0, 1,  0],
    [1, 0,  0],
    [0, 0, -1]
])
RFUinFRD = FRDinRFU.T

RFUinFLU = np.array([
    [ 0, 1, 0],
    [-1, 0, 0],
    [ 0, 0, 1]
])
FLUinRFU = RFUinFLU.T

FRDinFLU = RFUinFLU @ FRDinRFU
FLUinFRD = FRDinFLU.T


toTargetRotation = quaternion.from_rotation_matrix(FRDinFLU)


def params():
    # using FRD
    mass = 0.752

    # # Planning config
    # Tf = 1
    # N = 20
    # nlp_solver_max_iter = 100

    # Tracking config
    Tf = 1
    N = 10
    nlp_solver_max_iter = 20


    J = np.diag([0.0025, 0.0021, 0.0043])
    Q = np.array([
        10, 10, 10, # position,
        1, 1, 1, 1, # orientation
        0.001, 0.001, 0.001, # linear velocity
        0.01, 0.01, 0.01, # angular velocity
    ])
    R = np.array([
        1, 1, 1, 1, # propeller thrust
    ]) * 0.10 * 0.000001
    Qe = np.array([
        10, 10, 10, # position,
        10, 10, 10, 10, # orientation
        1, 1, 1, # linear velocity
        1, 1, 1, # angular velocity
    ])
    Qe = Q

    params = {
        "sim_dt": 0.02,
        "Tf": Tf,
        "N": N,
        "nlp_solver_max_iter": nlp_solver_max_iter,
        "g": 9.81,
        "mass": mass,
        "motor_thrust_constants": np.array([1.562522e-6, 0.0, 0.0]),
        "motor_torque_constant": 0.022,
        "control_level": "thrust",
        # rotor directions: positive for counter clockwise rotation (viewed from top / camera at e.g. 0, 0, -1 in the FRD body frame)
        "rotor_thrust_directions": np.array([
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1],
            [0, 0, -1]
        ]),
        "rotor_torque_directions": np.array([
            [0, 0, -1],
            [0, 0,  1],
            [0, 0, -1],
            [0, 0,  1]
        ]),
        "rotor_positions": [
            np.array([ 0.075, -0.1, 0]),
            np.array([ 0.075,  0.1, 0]),
            np.array([-0.075,  0.1, 0]),
            np.array([-0.075, -0.1, 0]),
        ],
        "J": J,
        # max output: thrusts directly snappy
        "max_output": 8.5, #62 <=> 2g
        "min_output": 0, #43 <=> 1g
        # # max output: thrusts directly
        # "max_output": 40/4, #62 <=> 2g
        # "min_output": 5/4, #43 <=> 1g
        # max output: rpm (gazebo)
        # "max_rpm": 62, #62 <=> 2g
        # "min_rpm": 20, #43 <=> 1g
        "Qe": Qe,
        "Q": Q/N,
        "R": R,
        "toTargetRotation": toTargetRotation,
        "fromTargetRotation": ~toTargetRotation,
        "toTargetRotationMatrix": quaternion.as_rotation_matrix(toTargetRotation),
        "fromTargetRotationMatrix": quaternion.as_rotation_matrix(~toTargetRotation),
    }
    return params