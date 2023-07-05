#！/usr/bin/env python
# utf-8
import torch
import torch.nn as nn
import numpy as np
from vehicle_state_predict.model import ImuModule

BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
EPOCHS = 3  # 总共训练迭代的次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001  # 设定初始的学习率
drop = 0.6

def run(data):
    data = torch.tensor(data).float()
    net = ImuModule()
    net = torch.load('/home/zhen/imu_ws/src/imu_odom/scripts/vehicle_state_predict/model/imu_cnn.model')
    net.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        output = net(data.reshape(1, 1, 100, 6))
        print("output:{}".format(output))

with open("/home/zhen/imu_ws/src/imu_odom/scripts/datasets.log") as f:
    reader = f.read().splitlines()
    for i in range(len(reader)- 100):
        input = []
        output = []
        data = reader[i:i+100]
        for index, l in enumerate(data):
            l = l.split(" ")
            l = np.array(l)
            l = l.astype(float)

            input.append(l[0:6])
            output.append(l[6:8])

        run(input)
        print("groundtruth:{}".format(output[-1] - output[0]))
        print("#####################################################################")