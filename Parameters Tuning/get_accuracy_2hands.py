'''

A12文件，制表使用，计算双手准确率

'''

from get_model_from_h5 import get_model
from read_npy_file_process import read_dataset_and_process
import numpy as np

fs_test_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/fs_5_pose.npy'
xz_test_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/xz_5_pose.npy'

xz_test = np.load(xz_test_path)
fs_test = np.load(fs_test_path)


# test_model_112 = get_model(('D:/CNNbkuniversity/compare_model/1LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_1lstm_1dense.h5'))
# test_model_232 = get_model(('D:/CNNbkuniversity/compare_model/2LSTM_3DENSE/add_diy_dataset_model/2_hand/2hand_2lstm_3dense.h5'))
# test_model_222 = get_model(('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/2_hand/2hand_2lstm_2dense.h5'))
# test_model_212 = get_model(('D:/CNNbkuniversity/compare_model/2LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_2lstm_1dense.h5'))
# test_model_132 = get_model(('D:/CNNbkuniversity/compare_model/1LSTM_3DENSE/add_diy_dataset_model/2_hand/2hand_1lstm_3dense.h5'))
# test_model_122 = get_model(('D:/CNNbkuniversity/compare_model/1LSTM_2DENSE/add_diy_dataset_model/2_hand/2hand_1lstm_2dense.h5'))
test_model_112 = get_model(('D:/CNNbkuniversity/compare_model/1LSTM_1DENSE/add_diy_dataset_model/2_hand/2hand_1lstm_1dense.h5'))







print('model load finished')


def get_acc(model):

    list_butongshoushi = []
    list_xiangtongshoushi = []
    for i in range(5):
        for j in range(i, 5):
            m = xz_test[i].reshape(1, 60, 21)
            n = xz_test[j].reshape(1, 60, 21)
            p = model.predict([n, m])
            list_xiangtongshoushi.append(p)
            pass
        # print('\n')
        pass

    for i in range(5):
        for j in range(i, 5):
            m = fs_test[i].reshape(1, 60, 21)
            n = fs_test[j].reshape(1, 60, 21)
            p = model.predict([n, m])
            list_xiangtongshoushi.append(p)
            # print(p, end='  ')
            pass
        # print('\n')
        pass




#不同手势对比：
    for i in range(5):
        for j in range(5):
            # print('放缩第' + str(i) + '个和旋转第' + str(j) + '相似度对比：', end='  ')
            m = fs_test[i].reshape(1, 60, 21)
            n = xz_test[j].reshape(1, 60, 21)
            p = model.predict([m, n])
            list_butongshoushi.append(p)
            # print(p)
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

    acc = acc_1/55
    # print(acc)
    return acc,average_pro,average_ng_pro

# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_112)
# print(accuracy,similarity_ave,dissimilarity_ave)


# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_232)
# print(accuracy,similarity_ave,dissimilarity_ave)


# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_222)
# print(accuracy,similarity_ave,dissimilarity_ave)

# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_212)
# print(accuracy,similarity_ave,dissimilarity_ave)

# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_132)
# print(accuracy,similarity_ave,dissimilarity_ave)

# accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_122)
# print(accuracy,similarity_ave,dissimilarity_ave)

accuracy,similarity_ave,dissimilarity_ave = get_acc(test_model_112)
print(accuracy,similarity_ave,dissimilarity_ave)

