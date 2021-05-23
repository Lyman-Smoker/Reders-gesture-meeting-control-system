'''

A12 文件，
功能：用 1LSTM+1FULLY_CONNECTION 的已预训练模型结合参数调优图表用来训练 双手动作 数据集最终模型

'''


pre_train_model = 'D:/CNNbkuniversity/compare_model/1LSTM_1DENSE/1lstm_1dense.h5'
path_train_data_x_diy_2hand = 'D:/CNNbkuniversity/diy_dataset/2_hand_dataset/dataset_2_hand_x.npy'
path_train_data_y_diy_2hand = 'D:/CNNbkuniversity/diy_dataset/2_hand_dataset/dataset_2_hand_y.npy'
transfer_model_path = 'D:/CNNbkuniversity/compare_model/1LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_1lstm_1dense.h5'

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,Lambda,BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from read_npy_file_process import read_dataset_and_process
from get_model_from_h5 import get_model
from read_npy_file_process import read_dataset_and_process
from tensorflow import keras


#模型提取
model_1_1_2hand = get_model(pre_train_model)


#数据提取
x1,x2,train_data_y = read_dataset_and_process(path_train_data_x_diy_2hand,path_train_data_y_diy_2hand)



model_1_1_2hand.fit( [x1,x2] , train_data_y , batch_size=16 , verbose=2,epochs=100)


model_1_1_2hand.save(transfer_model_path)
