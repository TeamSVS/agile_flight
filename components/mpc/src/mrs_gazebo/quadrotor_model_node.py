import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from src.model.quadrotor_model_generation import generate
from src.model.quadrotor_model_execution import create_capsule, control_trajectory
from src.params import mrs_gazebo

# from quadrotor_model_test import plot_solution
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R
import json
import h5py
import time


params = mrs_gazebo.params()
generate(params)
capsule = create_capsule()


lastImuMsg = None
lastOdomMsg = None
pos_target_source_initial = None

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


# toTargetRotation = FRDinFLU
# fromTargetRotation = FLUinFRD

toTargetRotation = quaternion.from_rotation_matrix(FRDinFLU)
fromTargetRotation = ~toTargetRotation 


fromTargetRotationMatrix = quaternion.as_rotation_matrix(fromTargetRotation)
toTargetRotationMatrix = quaternion.as_rotation_matrix(toTargetRotation)

def imuCallback(msg):
    global lastImuMsg
    lastImuMsg = msg

def odomCallback(msg):
    global lastOdomMsg
    lastOdomMsg = msg


def timerCallback(control_pub):
    global pos_target_source_initial
    if lastImuMsg is not None and lastOdomMsg is not None:
        pos = lastOdomMsg.pose.pose.position
        pos_source = fromTargetRotationMatrix @ [pos.x, pos.y, pos.z]
        orientationTarget = lastOdomMsg.pose.pose.orientation
        orientationTarget = np.quaternion(orientationTarget.w, orientationTarget.x, orientationTarget.y, orientationTarget.z)
        orientationSource = fromTargetRotation * orientationTarget * toTargetRotation
        vel = lastOdomMsg.twist.twist.linear
        vel_source = fromTargetRotationMatrix @ [vel.x, vel.y, vel.z]
        angular_vel = lastOdomMsg.twist.twist.angular
        angular_vel_source = fromTargetRotationMatrix @ [angular_vel.x, angular_vel.y, angular_vel.z]
        x0 = np.array([
            *pos_source,
            orientationSource.w, orientationSource.x, orientationSource.y, orientationSource.z,
            *vel_source,
            *angular_vel_source,
        ]).astype(float)
        # print(f"current: {vel_source}")

        pos_target_source_initial = pos_source.copy() if pos_target_source_initial is None else pos_target_source_initial
        pos_target_source = pos_target_source_initial.copy()
        pos_target_source[2] = -5

        # if pos_source[2] < -3:
        #     pos_target_source[1] = 1 if np.floor(time.time() % 10) > 5 else -1

        pos_target_source_diff = pos_target_source - pos_source
        if np.linalg.norm(pos_target_source_diff) > 0.5:
            pos_target_source = pos_source + pos_target_source_diff/np.linalg.norm(pos_target_source_diff) * 0.5

        

        phi = 0/180*np.pi

        setpoint = np.array([
            *pos_target_source,
            # 1, 0, 0, 0,
            np.cos(phi/2), *(np.sin(phi/2)*np.array([0, 0, 1])),
            0, 0, 0,
            0, 0, 0,
            0, 0, 0, 0
        ]).astype(float)
        traj = np.repeat(setpoint.reshape((1, -1)), params["N"]+1, axis=0)

        # print(f"target: {setpoint}")

        t, simX, simU = control_trajectory(capsule, x0, traj, params["N"], params["Tf"])
        u = simU[0, :]
        w_source = simX[1, (3+4+3):]
        w_target = toTargetRotationMatrix @ w_source
        orientationSetpointTarget = toTargetRotation * np.quaternion(*simX[1, 3:7]) * fromTargetRotation
        # create AttitudeTarget message containing the control input u (angular velocity) and setting the angular velocity flags
        # thrust_target = (toTargetRotation @ combined_thrust(propeller_thrust_model(u, params)))[2]
        thrust_target = u.sum()
        thrust_target_scaled =  thrust_target / (9.81 * params["mass"]) / 2 #thrust_target / params["mass"] / (9.81 * 2)
        thrust_target_scaled = thrust_target_scaled if thrust_target_scaled < 1 else 1
        thrust_target_scaled = thrust_target_scaled if thrust_target_scaled > 0 else 0

        msg = AttitudeTarget()
        msg.header.stamp = rospy.Time.now()
        msg.type_mask = 128
        msg.orientation.w = orientationSetpointTarget.w
        msg.orientation.x = orientationSetpointTarget.x
        msg.orientation.y = orientationSetpointTarget.y
        msg.orientation.z = orientationSetpointTarget.z
        msg.body_rate.x = w_target[0]
        msg.body_rate.y = w_target[1]
        msg.body_rate.z = w_target[2]
        msg.thrust = thrust_target_scaled
        print(f"thrust: {msg.thrust} w: {np.round(w_source, 2)}")
        control_pub.publish(msg)

        # print(f"thrust {msg.thrust}")
        # h5f = h5py.File(f"solutions/solution_{time.time()}.h5", 'w')
        # h5f.create_dataset('simX', data=simX)
        # h5f.create_dataset('simU', data=simU)
        # h5f.create_dataset('t', data=t)
        # h5f.close()
        # exit()


    
def listener():

    rospy.init_node('payload_mpc', anonymous=True)

    rospy.Subscriber("imu", Imu , imuCallback)
    rospy.Subscriber("/uav1/mavros/odometry/in", Odometry, odomCallback)
    control_pub = rospy.Publisher('control_output', AttitudeTarget, queue_size=10)
    timer = rospy.Timer(rospy.Duration(0.01), lambda sth: timerCallback(control_pub))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()