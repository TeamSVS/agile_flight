from typing import Tuple

import numpy as np
import yaml
from components.mpc.src.model.quadrotor_model_generation import generate
from components.mpc.src.controllers.setpoint_controller import init_controller, control_ctbr
from components.mpc.src.params import flightmare


def get_target_coordinate(att, x_final, y_final, z_final, pos):
    """
    This is a getter method. This method is XXX . It is a global method.

    :param att: XXX
    :param x_final: XXX
    :param y_final: XXX
    :param z_final: XXX
    :param pos: XXX
    """

    att = att.flatten()
    R = quat_rot_mat(att[0], att[1], att[2], att[3]).squeeze()
    transposed_coords = np.array([[x_final, y_final, z_final]]).T
    return R @ transposed_coords + pos[:, np.newaxis]


class Quadrotor:
    """
    This class XXX .

    Methods:
    :method __init__: XXX
    :method get_vel_command: XXX
    :method RK45: XXX
    :method state_dot: XXX
    :method set_lambda_w: XXX
    :method state_dot_s: XXX
    :method from_yaml: XXX
    :method Heun_s: XXX


    :param pop_size: XXX
    :type pop_size: int
    """

    def __init__(self, params):
        """
        This overrides the built-in object Initializator. It is a class method of Quadrotor.

        :param pop_size: XXX
        :type pop_size: int
        """

        self.params = params
        self.controller = init_controller()

    def get_vel_command(self, state, target):
        """
        This is a getter method. This method is XXX . It is a class method of Quadrotor.

        :param state: XXX
        :param target: XXX
        """
        target_con = world2mpc(np.hstack((target,
                                          np.array([0.9659258262890683, 0, 0.25881904510252074, 0,
                                                    4, 0, 0,
                                                    0, 0, 0])
                                          ))
                               , self.params)
        x = world2mpc(state, self.params)
        thrust, w_source, _ = control_ctbr(self.controller, x, target_con, self.params)
        collective_thrust = thrust  # / self.params["mass"]
        bodyrates = self.params["toTargetRotationMatrix"] @ w_source
        return [collective_thrust, *bodyrates]


    def from_yaml(self, path_to_config: str):
        """
        This method is XXX . It is a class method of Quadrotor.

        :param path_to_config: XXX
        :type path_to_config: str
        :raises Exception: XXX
        :raises : XXX
        :raises Exception: XXX
        :raises Exception: XXX
        :raises : XXX
        :raises Exception: XXX
        """

        print(
            "#####################################\n"
            "###### load quadrotor dynamics ######\n"
            "#####################################"
        )
        with open(path_to_config, "r") as quad_file:
            quad_str = quad_file.read()
        quad_yaml = yaml.safe_load(quad_str)
        quad = Quadrotor(self.learner.Np)
        try:
            my_object = quad_yaml["quadrotor_dynamics"]
        except Exception as e:
            my_object = quad_yaml
        try:
            for key, value in my_object.items():
                try:
                    setattr(quad, key, value)
                    print(f"Set {key} to: {value}")
                except:
                    print(f"Couldn't set {key}, {value}")
        except Exception as e:
            print("Didn't load quadrotor_dynamics: \n", e.with_traceback())
        return quad


def quat_rot_mat(a, b, c, d):
    """
    This method is XXX . It is a global method.

    :param a: XXX
    :param b: XXX
    :param c: XXX
    :param d: XXX
    """

    rot = np.array(
        [
            [
                a ** 2 + b ** 2 - c ** 2 - d ** 2,
                2 * b * c - 2 * a * d,
                2 * b * d + 2 * a * c,
            ],
            [
                2 * b * c + 2 * a * d,
                a ** 2 - b ** 2 + c ** 2 - d ** 2,
                2 * c * d - 2 * a * b,
            ],
            [
                2 * b * d - 2 * a * c,
                2 * c * d + 2 * a * b,
                a ** 2 - b ** 2 - c ** 2 + d ** 2,
            ],
        ],
        dtype=np.float64,
    )
    return rot


def world2mpc(state, params):
    pt = state[:3]
    qt = state[3:3 + 4]
    vt = state[3 + 4:3 + 4 + 3]
    wt = state[3 + 4 + 3:]

    p = params["fromTargetRotationMatrix"] @ pt
    orientationTarget = np.quaternion(*qt)
    q = params["fromTargetRotation"] * orientationTarget * params["toTargetRotation"]
    v = params["fromTargetRotationMatrix"] @ vt
    w = params["fromTargetRotationMatrix"] @ wt
    return np.concatenate([p, np.array([q.w, q.x, q.y, q.z]), v, w])


params = flightmare.params()
generate(params)
mpc_model = Quadrotor(params)
scale = 0.5
start_rescaling = False

def actual_mpc(
        x_final, y_final, z_final, pos, vel, att, omega
) -> Tuple[float, float, float, float]:
    """
    This method is XXX . It is a global method.

    :param x_final: XXX
    :param y_final: XXX
    :param z_final: XXX
    :param pos: XXX
    :param vel: XXX
    :param att: XXX
    :param omega: XXX
    :returns: Tuple[float, float, float, float] - XXX
    """
    global scale
    global start_rescaling
    # transposed_coords = np.array([[]]).T
    transposed_coords = np.array([[x_final, y_final, z_final]]).T

    vel = list(vel)
    if vel[0] < 1:
        scale = 0.5

        if x_final > 10:
            start_rescaling = True

    if start_rescaling:
        scale += 0.1
        start_rescaling = True if scale < 1.5 else False

    # scale = 1.5
    vel = np.array(vel)
    att = att.flatten()
    R = quat_rot_mat(att[0], att[1], att[2], att[3]).squeeze()

    norm = np.linalg.norm(transposed_coords)
    transposed_coords = np.true_divide(transposed_coords, norm, out=transposed_coords, where=norm != 0)
    tar_wf = R @ transposed_coords

    # norm_rel = np.linalg.norm(tar_wf)
    # tar_dir = np.true_divide(tar_wf, norm_rel, out=tar_wf, where=norm_rel != 0)
    # norm_vel = np.linalg.norm(vel)
    # vel_dir = np.true_divide(vel, norm_vel, out=vel, where=norm_vel != 0)
    # scale = scale + np.inner(tar_dir.flatten(), vel_dir.flatten())

    target_wf = tar_wf*scale + pos[:, np.newaxis]

    state_vec = np.hstack((pos, att, vel, omega))

    cmd_ctbr = mpc_model.get_vel_command(state_vec, target_wf.flatten())

    return tuple(cmd_ctbr)
