'''

A12文件，
功能：用于绘制IPN数据集参数调优参考图线，便于预训练模型参数调优

'''



import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorflow.keras.layers import Input,Dense,LSTM,Lambda,BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
import os


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用gpu，不注释就是使用cpu




path_train_data_x_IPN = 'A12data/train_data.npy'
path_train_data_y_IPN = 'A12data/train_data_y.npy'
npy_file_path = 'loss22.npy'







def read_dataset_and_process(path_train_data_x, path_train_data_y):
    train_data_x = np.load(path_train_data_x)
    train_data_y = np.load(path_train_data_y)

    train_data_y = train_data_y.astype(np.float32)

    x1 = train_data_x[:, 0]
    x2 = train_data_x[:, 1]

    return x1, x2, train_data_y


x1,x2,train_data_y = read_dataset_and_process(path_train_data_x_IPN,path_train_data_y_IPN)




# 构建孪生网络主结构类
class my_lstm:
    def __init__(self):
        self.block1 = LSTM(units=21, return_sequences=True)
        self.block2 = BatchNormalization()
        self.block3 = LSTM(units=21, return_sequences=False)
        self.block4 = BatchNormalization()

    def call(self, inputs):
        x = inputs
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

    pass


# 构建孪生网络主结构
def siamesenet(input_shape):
    my_modellstm = my_lstm()

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    i1 = my_modellstm.call(input1)
    i2 = my_modellstm.call(input2)

    l1_distance_layer = Lambda(
        lambda tensor: K.abs(tensor[0] - tensor[1])
    )
    l1_distance = l1_distance_layer([i1, i2])

    out1 = Dense(10, activation='relu')(l1_distance)
    out = Dense(1, activation='sigmoid')(out1)

    model = Model([input1, input2], out)
    return model


#得到一个实例化的孪生网络
final = siamesenet([60,21])

def lossf_3(y_true,y_pred):
    return K.mean((1-y_true) * K.square(y_pred) + (y_true)*K.square(K.maximum(1-y_pred,0)))



final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=lossf_3)


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'epoch': []}


    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))


    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(type + '_' + label + '.jpg')

    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        ar = np.array(self.losses['epoch'])

        np.save(npy_file_path, ar)





logs_loss = LossHistory()


final.fit([x1,x2] , train_data_y , batch_size=128 , verbose=1,epochs=450,callbacks=[logs_loss])
final.summary()

logs_loss.end_draw()

