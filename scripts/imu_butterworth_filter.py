#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import rospy
from sensor_msgs.msg import Imu
import logging
import time
from scipy.signal import butter, filtfilt
import sys
import datetime
# log = logging.Logger("filter")
# log.setLevel(logging.DEBUG)
# hander = logging.FileHandler(
#     '/home/zhen/imu_ws/src/imu_to_odom/scripts/var.log', mode='w')
# hander.setLevel(logging.DEBUG)
# log.addHandler(hander)

last_state = True
count = 0


def butterworth_filter(data, cutoff_freq, sample_freq, order=4):
    nyquist_freq = 0.5 * sample_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def is_static(accel_data, gyro_data, sigma_acc_noise, sigma_acc_drift, sigma_gyro_noise, sigma_gyro_drift):
    # 提取 X 轴加速度和 Z 轴角速度数据
    accel_x = accel_data[:, 0]
    gyro_z = gyro_data[:, 2]
    # 计算加速度和角速度的噪声方差
    accel_noise_var = np.var(accel_x)
    gyro_noise_var = np.var(gyro_z)

    rospy.loginfo("accel_noise_var:{}".format(accel_noise_var))
    # 计算加速度和角速度的游走方差
    accel_drift_var = np.var(np.cumsum(accel_x))

    rospy.loginfo("accel_drift_var:{}".format(accel_drift_var))
    gyro_drift_var = np.var(np.cumsum(gyro_z))

    # 判断是否为静止状态
    is_static_accel = accel_noise_var < 3 * \
        sigma_acc_noise or accel_drift_var < 3*sigma_acc_drift
    is_static_gyro = gyro_noise_var < 3 * \
        sigma_gyro_noise or gyro_drift_var < 3*sigma_gyro_drift

    return is_static_accel and is_static_gyro


def imu_callback(imu_data):
    global imu_data_array, imu_data_index, last_state, count

    # 提取需要的加速度和角速度数据
    accel_x = imu_data.linear_acceleration.x
    accel_y = imu_data.linear_acceleration.y
    accel_z = imu_data.linear_acceleration.z
    gyro_x = imu_data.angular_velocity.x
    gyro_y = imu_data.angular_velocity.y
    gyro_z = imu_data.angular_velocity.z
    t = time.time()
    # 将数据存储到数组中
    imu_data_array[imu_data_index, :] = [
        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, t]

    # 检查滑窗内的数据是否足够
    if imu_data_index >= window_size - 1:
        # 获取滑窗内的数据
        accel_data = imu_data_array[:, :3]  # 加速度数据
        gyro_data = imu_data_array[:, 3:-1]  # 角速度数据

        # 调用动静判断函数
        static = is_static(accel_data, gyro_data, sigma_acc_noise,
                           sigma_acc_drift, sigma_gyro_noise, sigma_gyro_drift)

        if static == last_state:
            count += 1
        else:
            count = 0

        last_state = static

        # 打印动静判断结果
        if count > 2:
            if static:
                rospy.loginfo("IMU is in static state")
                imu_data.linear_acceleration.z = 0.0

            else:
                rospy.loginfo("IMU is in dynamic state")
                # #均值滤波
                # imu_data_array[-1,:3] = np.mean(accel_data.transpose(), axis=1)
                # imu_data_array[-1,3:-1] = np.mean(gyro_data.transpose(), axis=1)

                # 巴特沃斯滤波
                for i in range(6):
                    # 加速度截止频率
                    cutoff_freq = 0.5
                    if i > 2:
                        # 角速度截止频率
                        cutoff_freq = 2
                    imu_data_array[:, i] = butterworth_filter(
                        imu_data_array[:, i], cutoff_freq=cutoff_freq, sample_freq=10, order=2)

                # b =  ' '.join([str(num) for num in imu_data_array[-1]])
                # log_content = str(accel_x)+" "+str(accel_y)+" "+str(accel_z)+" "+str(gyro_x)+" "+str(gyro_y)+" "+str(gyro_z)+" "+b
                # log.debug(log_content)

                imu_data.linear_acceleration.x = imu_data_array[-1, 0]
                imu_data.linear_acceleration.y = imu_data_array[-1, 1]
                imu_data.linear_acceleration.z = imu_data_array[-1, 2]
                # imu_data.angular_velocity.x = imu_data_array[-1, 3]
                # imu_data.angular_velocity.y = imu_data_array[-1, 4]
                # imu_data.angular_velocity.z = imu_data_array[-1, 5]

        # 发布topic
        imu_pub.publish(imu_data)
        # 移除队列中最早的数据，保持滑窗大小
        imu_data_array[:-1] = imu_data_array[1:]

    else:
        imu_data_index += 1


if __name__ == "__main__":
    rospy.init_node("imu_butterworth_filter")

    robot_id = rospy.get_param("/robot_id", "ZR1001")

    sigma_acc_noise = rospy.get_param(
        "~{}/imu0/sigma_acc_noise".format(robot_id), 0.01)
    sigma_acc_drift = rospy.get_param(
        "~{}/imu0/sigma_acc_bias".format(robot_id), 0.01)

    sigma_gyro_noise = rospy.get_param(
        "~{}/imu0/sigma_gyr_noise".format(robot_id), 0.01)
    sigma_gyro_drift = rospy.get_param(
        "~{}/imu0/sigma_gyr_bias".format(robot_id), 0.01)

    sub_topic = rospy.get_param("~sub_topic", "/imu_calibrated")
    pub_topic = rospy.get_param("~pub_topic", "/imu_data_filtered")

    # 设置滑窗大小和阈值 初始化滑窗队列
    window_size = 10
    imu_data_array = np.zeros((window_size, 7))
    imu_data_index = 0

    rospy.loginfo("robot_id: {}".format(robot_id))
    rospy.loginfo("sigma_acc_noise: {}".format(sigma_acc_noise))
    rospy.loginfo("sigma_acc_drift: {}".format(sigma_acc_drift))
    rospy.loginfo("sigma_gyro_noise: {}".format(sigma_gyro_noise))
    rospy.loginfo("sigma_gyro_drift: {}".format(sigma_gyro_drift))
    rospy.loginfo("sub_topic: {}".format(sub_topic))
    rospy.loginfo("pub_topic: {}".format(pub_topic))

    imu_sub = rospy.Subscriber(sub_topic, Imu, imu_callback)
    imu_pub = rospy.Publisher(pub_topic, Imu, queue_size=100)

    rospy.spin()
