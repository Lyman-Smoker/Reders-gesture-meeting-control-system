'''
作者：刘洪达
功能：模型载入文件，传入模型路径参数path，返回值为可用来训练的模型
'''

import tensorflow.keras.backend as K
from tensorflow import keras

def get_model(path):
    def lossf_3(y_true,y_pred):
        return K.mean((1-y_true) * K.square(y_pred) + (y_true)*K.square(K.maximum(1-y_pred,0)))
    model = keras.models.load_model(path, custom_objects={'lossf_3': lossf_3})
    return model

path='./2hand_2lstm_3dense.h5'
model=get_model(path)
