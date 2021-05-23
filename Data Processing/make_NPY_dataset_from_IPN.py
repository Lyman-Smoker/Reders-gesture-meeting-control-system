'''

此为服创 A12 项目文件，
功能：从IPN结构化光流（即三维list_dataset）文件中提取出 np 数组，存储进入.npy文件， train_x 和 train_y文件分开存放

'''

which_dataset =
path_dataset =
path_train_data_x =
path_train_data_y =



import numpy as np
from get_list_from_IPN_opticalflow import get_IPN_dataset
from get_list_from_diy_opticalflow import get_diy_dataset
from get_list_from_diytest_opticalflow import get_diy_dataset_test








def in_whitch_class(pre_sum, no) -> int:
    for i in range(len(pre_sum)):
        if no < pre_sum[i]:
            return i










# 处理预训练数据集，保存进入指定路径
# 如果which_dataset==0，取ipn数据集
# 如果which_dataset==1，取diy单手数据集
# 如果which_dataset==2，取diy双手数据集
def load_data_to_npy(path_train_data_x, path_train_data_y, path_dataset, which_dataset):
    if which_dataset == 0:
        dataset = get_IPN_dataset(path_dataset)
        pass
    elif which_dataset == 1:
        dataset = get_diy_dataset(path_dataset, 1)
    else:
        dataset = get_diy_dataset(path_dataset, 2)

    len_of_dataset = len(dataset)

    list_train_data_x = []
    for i in range(0, len_of_dataset):
        train_data_x = dataset[i][0:2]
        list_train_data_x.append(train_data_x)
        pass
    list_train_data_x = np.array(list_train_data_x)
    np.save(path_train_data_x, list_train_data_x)

    list_train_data_y = []
    for i in range(0, len_of_dataset):
        if dataset[i][2] == True:
            list_train_data_y.append(1)
            pass
        else:
            list_train_data_y.append(0)
        pass
    list_train_data_y = np.array(list_train_data_y)
    np.save(path_train_data_y, list_train_data_y)
    pass





'''

下面函数是用于得到 diy test数据集的 npy 文件

'''
def load_data_to_test_npy(path_train_data_x, path_train_data_y, path_dataset, which_dataset):
    if which_dataset == 0:
        dataset = get_IPN_dataset(path_dataset)
        pass
    elif which_dataset == 1:
        dataset = get_diy_dataset_test(path_dataset, 1)
    else:
        dataset = get_diy_dataset_test(path_dataset, 2)

    len_of_dataset = len(dataset)

    list_train_data_x = []
    for i in range(0, len_of_dataset):
        train_data_x = dataset[i][0:2]
        list_train_data_x.append(train_data_x)
        pass
    list_train_data_x = np.array(list_train_data_x)
    np.save(path_train_data_x, list_train_data_x)

    list_train_data_y = []
    for i in range(0, len_of_dataset):
        if dataset[i][2] == True:
            list_train_data_y.append(1)
            pass
        else:
            list_train_data_y.append(0)
        pass
    list_train_data_y = np.array(list_train_data_y)
    np.save(path_train_data_y, list_train_data_y)
    pass