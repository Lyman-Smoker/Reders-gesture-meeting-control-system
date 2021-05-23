'''

A12文件，
功能：从train_x和train_y对应的npy文件读取并获得用于神经网络训练与预测的向量返回值为 x1，x2，label

'''

path_train_data_x =
path_train_data_y =

import numpy as np



def read_dataset_and_process(path_train_data_x, path_train_data_y):
    train_data_x = np.load(path_train_data_x)
    train_data_y = np.load(path_train_data_y)

    train_data_y = train_data_y.astype(np.float32)

    x1 = train_data_x[:, 0]
    x2 = train_data_x[:, 1]

    return x1, x2, train_data_y


