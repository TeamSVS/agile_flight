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
    mass = 2
    inertia_body_radius=0.20
    inertia_body_height=0.05
    arm_length = 0.23

    # # Planning config
    # Tf = 1
    # N = 20
    # nlp_solver_max_iter = 100

    # Tracking config
    Tf = 1
    N = 10
    nlp_solver_max_iter = 50

    J = np.diag([
        mass * (3 * inertia_body_radius * inertia_body_radius + inertia_body_height * inertia_body_height) / 12,
        mass * (3 * inertia_body_radius * inertia_body_radius + inertia_body_height * inertia_body_height) / 12,
        mass * inertia_body_radius * inertia_body_radius / 2
    ])
    Qe = np.array([
        10, 10, 10, # position,
        10, 10, 10, 10, # orientation
        1, 1, 1, # linear velocity
        1, 1, 1, # angular velocity
    ]) * 0.5
    Q = np.array([
        10, 10, 10, # position,
        10, 10, 10, 10, # orientation
        1, 1, 1, # linear velocity
        1, 1, 1, # angular velocity
    ])
    R = np.array([
        1, 1, 1, 1, # propeller thrust
    ]) * 0.10 * 0.001

    params = {
        "sim_dt": 0.02,
        "Tf": Tf,
        "N": N,
        "nlp_solver_max_iter": nlp_solver_max_iter,
        "g": 9.81,
        "mass": mass,
        "motor_thrust_constants": np.array([10.04544 * (0.0159236**2), 0, 0]),
        "motor_torque_constant": 0.016,
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
            [0, 0, -1],
            [0, 0,  1],
            [0, 0,  1]
        ]),
        "rotor_positions": [
            np.array([ 1,  1, 0])/np.sqrt(2)*arm_length,
            np.array([-1, -1, 0])/np.sqrt(2)*arm_length,
            np.array([ 1, -1, 0])/np.sqrt(2)*arm_length,
            np.array([-1,  1, 0])/np.sqrt(2)*arm_length,
        ],
        "J": J,
        # max output: thrusts directly snappy
        "max_output": 2*9.81*mass/4, #62 <=> 2g
        "min_output": 0.1*3*9.81*mass/4, #43 <=> 1g
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