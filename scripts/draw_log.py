import matplotlib.pyplot as plt
import os
import numpy as np

def draw_data(data):

    # 绘制加速度的原始值和滤波后的值
    plt.figure(1)

    labels = ['Acceleration X', 'Acceleration Y', 'Acceleration Z',
              'Angular Velocity X', 'Angular Velocity Y', 'Angular Velocity Z']

    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.plot(data[-1],
                 data[i], '.', color='red')
        plt.plot(data[-1],
                 data[i+6], '.', color='green')
        # plt.plot(data[-1],
        #          data[i+6], '.', color='yellow')
        plt.xlabel('Time (s)')
        plt.ylabel(labels[i])
        plt.legend(['Raw', 'Butterwooth_Filter', 'zhen_Filter'])

    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.plot(data[-1],
                 data[i+3], '.', color='red')
        plt.plot(data[-1],
                 data[i+9], '.', color='green')
        # plt.plot(data[-1],
        #          data[i+15], '.', color='yellow')
        plt.xlabel('Time (s)')
        plt.ylabel(labels[i+3])
        plt.legend(['Raw', 'Butterwooth_Filtere', 'zhen_Filter'])

    plt.pause(0.0001)


if __name__ == '__main__':
    with open("/home/zhen/imu_ws/src/imu_to_odom/scripts/var.log") as f:
        data = f.read().splitlines()
        for l in data:
            l = l.split(" ")
            l = np.array(l)
            l = l.astype(float)
            draw_data(l)