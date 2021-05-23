'''

A12文件，
功能：计算迁移训练模型最终准确率、平均相似度、平均不相似度

'''


from get_model_from_h5 import get_model
from read_npy_file_process import read_dataset_and_process
import numpy as np

dj_test_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/dj_5_pose.npy'
py_test_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/py_5_pose.npy'
zq_test_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/zq_5_pose.npy'




test_model_111 = get_model('D:/CNNbkuniversity/compare_model/1LSTM_1DENSE/add_diy_dataset_model/1_hand/1hand_1lstm_1dense.h5')
test_model_121 = get_model('D:/CNNbkuniversity/compare_model/1LSTM_2DENSE/add_diy_dataset_model/1_hand/1hand_1lstm_2dense.h5')
test_model_131 = get_model('D:/CNNbkuniversity/compare_model/1LSTM_3DENSE/add_diy_dataset_model/1_hand/1hand_1lstm_3dense.h5')
test_model_211 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_1DENSE/add_diy_dataset_model/1_hand/1hand_2lstm_1dense.h5')
test_model_221 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/1hand_2lstm_2dense.h5')
test_model_231 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_3DENSE/add_diy_dataset_model/1_hand/1hand_2lstm_3dense.h5')


print('model load finished')


dj_test = np.load(dj_test_path)
py_test = np.load(py_test_path)
zq_test = np.load(zq_test_path)



def get_acc(model):

    list_butongshoushi = []
    list_xiangtongshoushi = []
    for i in range(5):
        for j in range(i, 5):
            m = dj_test[i].reshape(1, 60, 21)
            n = dj_test[j].reshape(1, 60, 21)
            p = model.predict([n, m])
            list_xiangtongshoushi.append(p)
            pass
        pass

    for i in range(5):
        for j in range(i, 5):
            m = py_test[i].reshape(1, 60, 21)
            n = py_test[j].reshape(1, 60, 21)
            p = model.predict([n, m])
            list_xiangtongshoushi.append(p)
            pass
        pass

    for i in range(5):
        for j in range(i, 5):
            m = zq_test[i].reshape(1, 60, 21)
            n = zq_test[j].reshape(1, 60, 21)
            p = model.predict([n, m])
            list_xiangtongshoushi.append(p)
            pass
        pass


#不同手势对比：
    for i in range(5):
        for j in range(5):
            m = dj_test[i].reshape(1,60,21)
            n = py_test[j].reshape(1,60,21)
            p = model.predict([m,n])
            list_butongshoushi.append(p)
            pass
        pass

    for i in range(5):
        for j in range(5):
            m = dj_test[i].reshape(1,60,21)
            n = zq_test[j].reshape(1,60,21)
            p = model.predict([m,n])
            list_butongshoushi.append(p)
            pass
        pass

    for i in range(5):
        for j in range(5):
            m = py_test[i].reshape(1,60,21)
            n = zq_test[j].reshape(1,60,21)
            p = model.predict([m,n])
            list_butongshoushi.append(p)
            pass
        pass

    list_xiangtongshoushi_array = np.array(list_xiangtongshoushi)
    average_pro = list_xiangtongshoushi_array.mean()


    list_butongshoushi_array = np.array(list_butongshoushi)
    average_ng_pro = list_butongshoushi_array.mean()


    # list_xiangtongshoushi_array = np.array(list_xiangtongshoushi)
    acc_1 = 0
    for i in list_xiangtongshoushi:
        if i > 0.9:
            acc_1 += 1
            pass


    # list_butongshoushi_array = np.array(list_butongshoushi)
    for i in list_butongshoushi:
        if i < 0.1:
            acc_1 += 1
            pass
        pass
    # print(list_xiangtongshoushi)
    # print(list_butongshoushi)
    #
    # print(average_pro)

    acc = acc_1/120
    # print(acc)
    return acc,average_pro,average_ng_pro

#
accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_111)
print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_121)
print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_131)
print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_211)
print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_221)
print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_231)
print(accuracy,similarity_ave,dissimilarity_ave)
