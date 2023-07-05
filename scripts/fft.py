from scipy import fft
import rospy

import numpy as np

from scipy.signal import butter, filtfilt

from sensor_msgs.msg import Imu

import time

import matplotlib.pyplot as plt

import logging


log = logging.Logger("filter")
log.setLevel(logging.DEBUG)
hander=logging.FileHandler('/home/zhen/imu_ws/src/imu_to_odom/scripts/fft.log',mode='w')
hander.setLevel(logging.DEBUG)
log.addHandler(hander)

class IMUFFT:

    def __init__(self):

        self.imu_sub = rospy.Subscriber(
            '/imu_data', Imu, self.imu_callback)

        self.imu_pub = rospy.Publisher('/imu_filtered', Imu, queue_size=100)

        self.sample_count = 0

        self.buffer_size = 1000

        self.accel_buffer = np.zeros((self.buffer_size, 3))

        self.gyro_buffer = np.zeros((self.buffer_size, 3))

        self.data_buffer = [None for i in range(self.buffer_size)]

        self.t_buffer = np.zeros(self.buffer_size)

    def imu_callback(self, msg):
        # 将加速度和角速度数据转换为 numpy 数组

        accel = np.array([msg.linear_acceleration.x,

                          msg.linear_acceleration.y,

                          msg.linear_acceleration.z])

        gyro = np.array([msg.angular_velocity.x,

                         msg.angular_velocity.y,

                         msg.angular_velocity.z])

        self.t_buffer[self.sample_count] = time.time()

        self.accel_buffer[self.sample_count] = accel

        self.gyro_buffer[self.sample_count] = gyro

        self.data_buffer[self.sample_count] = msg

        self.sample_count += 1
        
        if self.sample_count >= 896:

            acc_before = np.transpose(self.accel_buffer)

            gyro_before = np.transpose(self.gyro_buffer)

            print("acc_x_cotoff_f:" + str(calculate_cutoff_frequency(acc_before[1], 10, 0.8)))
            print("gyr_z_cotoff_f:" + str(calculate_cutoff_frequency(gyro_before[2], 10, 0.8)))

            for i in range(3):

                plt.figure("acc "+ str(i))
                Y = fft.fft(acc_before[i])

                N = len(Y)  # 频谱的长度
                Fs = 100

                f = Fs * np.arange(0, N/2) / N  # 计算频率范围（仅取正频率部分）
                magnitude = 2 * np.abs(Y[:N//2]) / N  # 计算幅值谱
                xf = fft.fftfreq(896, 1 / 100)
                
                plt.plot(f, magnitude)
                
                plt.show()


            for i in range(3):
                plt.figure("gyr "+ str(i))
                Y = fft.fft(gyro_before[i])

                N = len(Y)  # 频谱的长度
                Fs = 100

                f = Fs * np.arange(0, N/2) / N  # 计算频率范围（仅取正频率部分）
                magnitude = 2 * np.abs(Y[:N//2]) / N  # 计算幅值谱
                xf = fft.fftfreq(896, 1 / 100)
                
                plt.plot(f, magnitude)
                
                plt.show()

def calculate_cutoff_frequency(data, sample_rate, threshold):
    # 执行傅里叶变换
    spectrum = np.abs(fft(data))
    
    # 计算频率轴
    freq_axis = np.fft.fftfreq(len(data), 1/sample_rate)
    
    # 计算总能量
    total_energy = np.sum(spectrum)
    
    # 计算能量下降的阈值
    threshold_energy = total_energy * threshold
    
    # 找到能量下降到阈值以下的频率点
    cutoff_freq = freq_axis[np.where(np.cumsum(spectrum) <= threshold_energy)][-1]
    
    return cutoff_freq

if __name__ == '__main__':
    rospy.init_node('IMUFFT')
    filter = IMUFFT()
    rospy.spin()
