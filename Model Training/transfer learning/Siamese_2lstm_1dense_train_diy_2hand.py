'''

A12 文件，用包含 2个lstm+1个dense 的已预训练模型用来训练 双手动作 数据集最终模型


'''

import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,Lambda,BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from read_npy_file_process import read_dataset_and_process
from get_model_from_h5 import get_model
from read_npy_file_process import read_dataset_and_process
from tensorflow import keras


#模型提取
# model_2_1_2hand = get_model('D:/CNNbkuniversity/compare_model/2LSTM_1DENSE/2lstm_1dense.h5')
model_2_1_2hand = get_model('D:/CNNbkuniversity/compare_model/2LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_2lstm_1dense.h5')


#数据提取
path_train_data_x_diy_2hand = 'D:/CNNbkuniversity/diy_dataset/2_hand_dataset/dataset_2_hand_x.npy'
path_train_data_y_diy_2hand = 'D:/CNNbkuniversity/diy_dataset/2_hand_dataset/dataset_2_hand_y.npy'
x1,x2,train_data_y = read_dataset_and_process(path_train_data_x_diy_2hand,path_train_data_y_diy_2hand)



model_2_1_2hand.fit( [x1,x2] , train_data_y , batch_size=32 , verbose=2,epochs=10)


model_2_1_2hand.save('D:/CNNbkuniversity/compare_model/2LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_2lstm_1dense.h5')
