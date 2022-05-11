#!/usr/bin/python3
import argparse
import logging
import os

import numpy as np
import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from cv_bridge import CvBridge
import cv_bridge
from geometry_msgs.msg import TwistStamped
from ruamel.yaml import YAML
from sensor_msgs.msg import Image
from stable_baselines3 import PPO
from std_msgs.msg import Empty

from envsim_msgs.msg import ObstacleArray
from rl_example import load_rl_policy
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState

print("\t\t" + rospy.__file__)
print("\t\t" + cv_bridge.__file__)


class CompassRl:

    def __init__(self, model_path, cuda_device=0):
        self.model = PPO.load(model_path, env=None, device=("cuda:{0}".format(cuda_device)), custom_objects=None,
                              print_system_info=True)

        self.cfg = YAML().load(
            open(
                os.environ["FLIGHTMARE_PATH"] + "/flightpy/configs/vision/config.yaml", "r"
            )
        )

        quad_mass = self.cfg["quadrotor_dynamics"]["mass"]
        omega_max = self.cfg["quadrotor_dynamics"]["omega_max"]
        thrust_max = 4 * self.cfg["quadrotor_dynamics"]["thrust_map"][0] * \
                     self.cfg["quadrotor_dynamics"]["motor_omega_max"] * \
                     self.cfg["quadrotor_dynamics"]["motor_omega_max"]
        self.act_mean = np.array([thrust_max / quad_mass / 2, 0.0, 0.0, 0.0])[np.newaxis, :]
        self.act_std = np.array([thrust_max / quad_mass / 2, \
                                 omega_max[0], omega_max[1], omega_max[2]])[np.newaxis, :]

    def predict(self, obs):
        action = self.model.predict(obs)
        action = (action * self.act_std + self.act_mean)[0, :]
        return action


class AgilePilotNode:
    def __init__(self, vision_based=False, m_path=None, gpu=0):
        print("Initializing agile_pilot_node...")
        rospy.init_node('agile_pilot_node', anonymous=False)

        self.vision_based = vision_based
        self.m_path = m_path
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.model = CompassRl(self.m_path, gpu)
        quad_name = 'kingfisher'

        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/state", QuadState, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        self.img_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/unity/depth", Image, self.img_callback,
                                        queue_size=1, tcp_nodelay=True)
        # self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/dodgeros_pilot/groundtruth/obstacles", ObstacleArray,
        #                                     self.obstacle_callback, queue_size=1, tcp_nodelay=True)

        # Command publishers
        self.cmd_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/feedthrough_command", Command, queue_size=1)
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/dodgeros_pilot/velocity_command", TwistStamped,
                                          queue_size=1)
        print("Initialization completed!")

    def img_callback(self, img_data):
        if not self.vision_based:
            return
        if self.state is None:
            return
        cv_image = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding='passthrough')
        command = compute_command_vision_based(self.state, cv_image, rl_model=self.model)

        self.publish_command(command)

    def state_callback(self, state_data):
        self.state = AgileQuadState(state_data)

    def obstacle_callback(self, obs_data):
        if self.vision_based:
            return
        if self.state is None:
            return
        rl_policy = None
        if self.m_path is not None:
            rl_policy = load_rl_policy(self.m_path)
        command = compute_command_state_based(state=self.state, obstacles=obs_data, rl_policy=rl_policy)
        self.publish_command(command)

    def publish_command(self, command):
        if command.mode == AgileCommandMode.SRT:
            assert len(command.rotor_thrusts) == 4
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = True
            cmd_msg.thrusts = command.rotor_thrusts
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.CTBR:
            assert len(command.bodyrates) == 3
            cmd_msg = Command()
            cmd_msg.t = command.t
            cmd_msg.header.stamp = rospy.Time(command.t)
            cmd_msg.is_single_rotor_thrust = False
            cmd_msg.collective_thrust = command.collective_thrust
            cmd_msg.bodyrates.x = command.bodyrates[0]
            cmd_msg.bodyrates.y = command.bodyrates[1]
            cmd_msg.bodyrates.z = command.bodyrates[2]
            if self.publish_commands:
                self.cmd_pub.publish(cmd_msg)
                return
        elif command.mode == AgileCommandMode.LINVEL:
            vel_msg = TwistStamped()
            vel_msg.header.stamp = rospy.Time(command.t)
            vel_msg.twist.linear.x = command.velocity[0]
            vel_msg.twist.linear.y = command.velocity[1]
            vel_msg.twist.linear.z = command.velocity[2]
            vel_msg.twist.angular.x = 0.0
            vel_msg.twist.angular.y = 0.0
            vel_msg.twist.angular.z = command.yawrate
            if self.publish_commands:
                self.linvel_pub.publish(vel_msg)
                return
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("Start publishing commands!")
        self.publish_commands = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agile Pilot.')
    parser.add_argument('--vision_based', help='Fly vision-based', required=False, dest='vision_based',
                        action='store_true')
    parser.add_argument('--m_path', help='Neural network model to load', type=str, required=True, default=None)
    parser.add_argument('--gpu', help='The gpu used for the rl model', type=int, required=False, default=0)
    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, m_path=args.m_path, gpu=args.gpu)
    rospy.spin()
