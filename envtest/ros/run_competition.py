#!/usr/bin/python3
import argparse

import rospy
from agiros_msgs.msg import Command
from agiros_msgs.msg import QuadState
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

from envsim_msgs.msg import ObstacleArray
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState


class AgilePilotNode:
    def __init__(self, vision_based=False):
        print("Initializing agile_pilot_node...")
        rospy.init_node('agile_pilot_node', anonymous=False)

        self.vision_based = vision_based
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None

        quad_name = 'kingfisher'

        # Logic subscribers
        self.start_sub = rospy.Subscriber("/" + quad_name + "/start_navigation", Empty, self.start_callback,
                                          queue_size=1, tcp_nodelay=True)

        # Observation subscribers
        self.odom_sub = rospy.Subscriber("/" + quad_name + "/agiros_pilot/state", QuadState, self.state_callback,
                                         queue_size=1, tcp_nodelay=True)

        self.img_sub = rospy.Subscriber("/" + quad_name + "/agiros_pilot/unity/depth", Image, self.img_callback,
                                        queue_size=1, tcp_nodelay=True)
        self.obstacle_sub = rospy.Subscriber("/" + quad_name + "/agiros_pilot/groundtruth/obstacles", ObstacleArray,
                                             self.obstacle_callback, queue_size=1, tcp_nodelay=True)

        # Command publishers
        self.cmd_pub = rospy.Publisher("/" + quad_name + "/agiros_pilot/feedthrough_command", Command, queue_size=1)
        self.linvel_pub = rospy.Publisher("/" + quad_name + "/agiros_pilot/velocity_command", TwistStamped,
                                          queue_size=1)

    def img_callback(self, data):
        if not self.vision_based:
            return
        if self.state is None:
            return
        cv_image = self.cv_bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        command = compute_command_vision_based(self.state, cv_image)
        self.publish_command(command)

    def state_callback(self, data):
        self.state = AgileQuadState(data)

    def obstacle_callback(self, data):
        if self.vision_based:
            return
        if self.state is None:
            return
        obstacles = None
        command = compute_command_state_based(state=self.state, obstacles=obstacles)
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

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based)
    rospy.spin()
