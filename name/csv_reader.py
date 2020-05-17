import csv
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as data


def data_transform(csv_file):
    data_encoded = np.zeros((1731,12), dtype=np.float32)
    reader = np.loadtxt(csv_file, delimiter=';', skiprows=1)
    att = reader[:, :-1].astype(np.float32)
    label = reader[:, -1:].astype(np.int64)

    att = torch.from_numpy(att)
    label = torch.from_numpy(label)
    return att, label

def att_split(csv_file='./wine.csv', train_num=1400):
    input_data, input_label = data_transform(csv_file)
    train_data = input_data[:train_num]
    test_data = input_data[train_num:]
    return train_data, test_data

def label_split(csv_file='./wine.csv', train_num=1400):
    input_data, input_label = data_transform(csv_file)
    train_data = input_label[:train_num]
    test_data = input_label[train_num:]
    return train_data, test_data


class DataFolder(data.Dataset):
    def __init__(self, input_data, input_label):
        super(DataFolder, self).__init__()
        self.input_data = input_data
        self.input_label = input_label

    def __getitem__(self, index):
        data_index = self.input_data[index]
        label_index = self.input_label[index]

        return data_index, label_index
    
    def __len__(self):
        return len(self.input_data)

if __name__ == '__main__':
    # line = get_length()
    # print(line)
    data_transform('./wine.csv')