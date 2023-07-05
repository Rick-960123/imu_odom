#！/usr/bin/env python
# utf-8
from torch.utils.data import Dataset
import torch
import os
import numpy as np

class IMUDatasets(Dataset):
    def __init__(self,  mode, root="/home/zhen/imu_ws/src/imu_odom/scripts/"):
        super(IMUDatasets, self).__init__()
        self.root = root

        # image, label
        self.inputs, self.outputs = self.load_log("datasets.log")
 
        # 对数据集进行划分
        if mode == "train": # 60%
            self.inputs = self.inputs[:int(0.6*len(self.inputs))]
            self.outputs = self.outputs[:int(0.6*len(self.outputs))]
        elif mode == "val": # 20% = 60%~80%
            self.inputs = self.inputs[int(0.6*len(self.inputs)):int(0.8 * len(self.inputs))]
            self.outputs = self.outputs[int(0.6*len(self.outputs)):int(0.8 * len(self.outputs))]
        else: # 20% = 80%~100%
            self.inputs = self.inputs[int(0.8 * len(self.inputs)):]
            self.outputs = self.outputs[int(0.8 * len(self.outputs)):]
 
 
    def load_log(self, filename):
        """
        :param filename:
        :return:
        """
        inputs, outputs = [], []
        with open(os.path.join(self.root, filename)) as f:
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

                output = output[-1] - output[0]
                inputs.append(input)
                outputs.append(output)

        assert len(inputs) == len(outputs)
        return inputs, outputs
 
 
 
    def __len__(self):
        return len(self.inputs)
 
    def __getitem__(self, idx):
        input, output = torch.tensor(self.inputs[idx]) , torch.tensor(self.outputs[idx]) 
        return  input.float(), output.float()
    
if __name__ == "__main__":
    imu = IMUDatasets("test")