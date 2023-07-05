#！/usr/bin/env python
# utf-8
import time

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler  # 学习率调整器，在训练过程中合理变动学习率
from torchvision import transforms  # pytorch 视觉库中提供了一些数据变换的接口
from torchvision import datasets  # pytorch 视觉库提供了加载数据集的接口
from imuDatesets import IMUDatasets
from torch.utils.data import Dataset, DataLoader
from model import ImuModule
# 预设网络超参数
BATCH_SIZE = 64  # 由于使用批量训练的方法，需要定义每批的训练的样本数目
EPOCHS = 100  # 总共训练迭代的次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
learning_rate = 0.001  # 设定初始的学习率
# 加载训练集
train_loader = DataLoader(IMUDatasets("train"),batch_size=10,shuffle=True) # 指明批量大小，打乱，这是处于后续训练的需要。

val_loader= DataLoader(IMUDatasets("val"), batch_size=10,shuffle=True)

test_loader = DataLoader(IMUDatasets("test"),batch_size=10,shuffle=True)

def train(_epoch, model, device, train, optimizer, lr_sch):
    model.train()
    lr_sch.step()
    for i, (images, labels) in enumerate(train):
        samples = images.to(device)
        labels = labels.to(device)
        output = model(samples.reshape(0-1, 1, 100, 6))
        loss1 = loss(output, labels)
        # 优化器内部的参数梯度调整为0
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print("Epoch:{}, batch:{}, loss:{:.4f}".format(_epoch, i + 1, loss1.item()))


def test(_test_loader, _model, _device):
    _model.eval()  # 设置模型进入预测模式 evaluation
    loss1 = 0
    correct = 0

    with torch.no_grad():  # 如果不需要 backward更新梯度，那么就要禁用梯度计算，减少内存和计算资源浪费。
        for data, target in _test_loader:
            data, target = data.to(_device), target.to(_device)
            output = module(data.reshape(0-1, 1, 100, 6))
            loss1 += loss(output, target).item()  # 添加损失值

    loss1 /= len(_test_loader.dataset)

    print('Test average loss: {:.4f}\n'.format(loss1))

if __name__ == '__main__':
    start = time.time()
    model_path = "svm_v4.model"
    module = ImuModule()
    module = module.to(DEVICE)
    loss = nn.HuberLoss()
    loss = loss.to(DEVICE)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    for epoch in range(1, EPOCHS + 1):
        train(epoch, module, DEVICE, train_loader, optimizer, exp_lr_scheduler)
        test(val_loader, module, DEVICE)

    test(test_loader, module, DEVICE)

    torch.save(module, '/home/zhen/imu_ws/src/imu_odom/scripts/vehicle_state_predict/model/imu_cnn.model')
    print("Time: " + str(time.time() - start))