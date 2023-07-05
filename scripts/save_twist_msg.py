#!/usr/bin/env python
import sys
sys.path.insert(0,"/home/zhen/imu_ws/devel/lib/python3/dist-packages")
import rospy
import time
from sensor_msgs.msg import Imu
from zr_msgs.msg import motor_info
from nav_msgs.msg import Odometry
import logging
import threading
import numpy as np

log = logging.Logger("filter")
log.setLevel(logging.DEBUG)
hander = logging.FileHandler(
    '/home/zhen/imu_ws/src/imu_odom/scripts/datasets.log', mode='w')
hander.setLevel(logging.DEBUG)
log.addHandler(hander)

imu_que = []
odo_que = []
r_que= []
# 回调函数，接收双轮速度消息
def wheel_speed_callback(msg):
    global imu_que, odo_que
    # 获取左右轮速度
    left_wheel_speed = msg.left_vel
    right_wheel_speed = msg.right_vel

    # 根据双轮速度计算车体速度和角度
    linear_velocity = (left_wheel_speed + right_wheel_speed) / 2.0 * 0.95
    angular_velocity = (right_wheel_speed - left_wheel_speed) / 0.5

    # 创建并发布车体速度和角度的消息
    odom_msg = Odometry()
    odom_msg.twist.twist.linear.x = linear_velocity
    odom_msg.twist.twist.angular.z = angular_velocity
    odom_msg.header.stamp = msg.header.stamp
    lock.acquire()
    odo_que.append(odom_msg)
    lock.release()

def imu_callback(msg):
    global imu_que, odo_que, r_que
    lock.acquire()
    imu_que.append(msg)
    lock.release()

def process():
    global imu_que, odo_que
    num = 0
    start = False
    # 运行ROS节点
    while not rospy.is_shutdown():
        time.sleep(0.02)
        if not (len(imu_que)>2 and len(odo_que)>2):
            continue
        imu = imu_que[0]
        odo = odo_que[0]
        delta_t = (imu.header.stamp - odo.header.stamp).to_sec()
        
        if delta_t > 0.1 and not start:
            lock.acquire()
            odo_que = odo_que[1:]   
            lock.release()
            num +=10
            continue

        elif delta_t < 0.0001 and not start:
            lock.acquire()
            imu_que = imu_que[1:]
            lock.release()
            num +=10
            continue

        else:
            start = True

            # if(delta_t) > 0.05:
            #     lock.acquire()
            #     odo_que = odo_que[1:]
            #     lock.release()
            #     odo = odo_que[0]
            #     delta_t = (imu.header.stamp - odo.header.stamp).to_sec()

            r = np.array([[ 9.94398102e-01,  8.76919099e-04, -1.05696004e-01],
                        [ 8.76919099e-04,  9.99862727e-01,  1.65456152e-02],
                        [ 1.05696004e-01, -1.65456152e-02,  9.94260830e-01]])
            
            acc = [imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z]

            gyro = [imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z]
            acc_ = np.dot(r,acc)
            gyro_ = np.dot(r,gyro)
    
            accel_x = acc_[0]
            accel_y = acc_[1]
            accel_z = acc_[2]
            gyro_x = gyro_[0]
            gyro_y = gyro_[1]
            gyro_z = gyro_[2]

            angular_z_i = odo.twist.twist.angular.z
            linear_x_i = odo.twist.twist.linear.x

            print(linear_x_i)

            angular_z_j = odo_que[1].twist.twist.angular.z
            linear_x_j = odo_que[1].twist.twist.linear.x
            
            if odo.twist.twist.linear.x == 0.0:
                r_que.append([accel_x,accel_y,accel_z])
            

            t_i_j = (odo_que[1].header.stamp - odo.header.stamp).to_sec()

            angular_z = angular_z_i + (angular_z_j - angular_z_i)*delta_t/t_i_j
            linear_x = linear_x_i + (linear_x_j - linear_x_i)*delta_t/t_i_j

            print(linear_x)

            log_content = str(accel_x)+" "+str(accel_y)+" "+str(accel_z)+" "+str(gyro_x)+" "+str(gyro_y)+" "+str(gyro_z)+" " +str(linear_x)+" "+str(angular_z)+" "+str(odo.header.stamp.to_sec())+" "+str(imu.header.stamp.to_sec())
            log.debug(log_content)

            lock.acquire()
            odo_que = odo_que[2:]
            imu_que = imu_que[1:]
            lock.release()

        num+=1
        print(num)

def rotation_matrix(v1, v2):
    v1 = v1 / np.linalg.norm(v1)  # 将向量v1归一化
    v2 = v2 / np.linalg.norm(v2)  # 将向量v2归一化

    # 计算旋转轴
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    # 如果旋转轴的长度为0，则两个向量共线，返回单位矩阵
    if axis_norm == 0:
        return np.eye(3)
    
    # 归一化旋转轴
    axis = axis / axis_norm
    
    # 计算旋转角度
    angle = np.arccos(np.dot(v1, v2))
    
    # 计算旋转矩阵
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_matrix = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + sin_theta * cross_matrix + (1 - cos_theta) * np.dot(cross_matrix, cross_matrix)
    
    return rotation_matrix

if __name__ == "__main__":
    # 初始化ROS节点
    rospy.init_node("odometry_publisher")
    # 设置双轮速度消息的订阅器
    rospy.Subscriber("/motor_info", motor_info, wheel_speed_callback)
    rospy.Subscriber("/imu_data", Imu, imu_callback)
    # 创建车体速度和角度的消息发布器
    lock  =  threading.Lock()

    odom_pub = rospy.Publisher("/wheel_odom", Odometry, queue_size=10)
    threading.Thread(target=process).start()
    rospy.spin()
    
    q = np.mean(np.array(r_que), axis=0)

    q = q / (np.linalg.norm(q)/9.81)
    p = np.array([0.0,0.0,9.81])

    rotation_matrix = rotation_matrix(q, p)

    print(q)
    print(rotation_matrix)
    print(np.dot(rotation_matrix, q))