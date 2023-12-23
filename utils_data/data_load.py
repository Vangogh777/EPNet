import os

import pandas as pd
import torch
from torch.utils.data import Dataset


class myDataSet(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        """
        :param data_dir: 数据文件路径
        :param label_dir: 标签文件路径
        :param transform: transform操作
        """
        self.transform = transform
        # 读文件夹下每个数据文件名称
        # os.listdir读取文件夹内的文件名称
        self.file_name = os.listdir(data_dir)
        # 读标签文件夹下的数据名称
        self.label_name = os.listdir(label_dir)

        self.data_path = []
        self.label_path = []

        # 让每一个文件的路径拼接起来
        for index in range(len(self.file_name)):
            self.data_path.append(os.path.join(data_dir, self.file_name[index]))
            self.label_path.append(os.path.join(label_dir, self.label_name[index]))

    def __len__(self):
        # 返回数据集长度
        return len(self.file_name)

    def __getitem__(self, index):
        # 获取每一个数据

        # 读取数据
        data = pd.read_csv(self.data_path[index], header=None)
        # 读取标签
        label = pd.read_csv(self.label_path[index], header=None)

        if self.transform:
            data = self.transform(data)
            label = self.transform(label)

        # 转成张量
        data = torch.tensor(data.values)
        label = torch.tensor(label.values)

        return data, label  # 返回数据和标签
        # return {
        #     'image': data,
        #     'mask': label
        # }
        # return {
        #     'image': torch.as_tensor(data.copy()).float().contiguous(),
        #     'mask': torch.as_tensor(label.copy()).long().contiguous()
        # }