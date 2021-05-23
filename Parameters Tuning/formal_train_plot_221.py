'''

A12 文件

训练221模型

'''











from get_model_from_h5 import get_model
from read_npy_file_process import read_dataset_and_process
import numpy as np






# 训练模型221的文件代码
# test_model_221 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/2lstm_2dense.h5')
# test_model_221 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/model_100epoch.h5')
test_model_221 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/model_250epoch.h5')





path_train_data_x_diy_1hand = 'D:/CNNbkuniversity/diy_dataset/1_hand_dataset/dataset_1_hand_x.npy'
path_train_data_y_diy_1hand = 'D:/CNNbkuniversity/diy_dataset/1_hand_dataset/dataset_1_hand_y.npy'
x1,x2,train_data_y = read_dataset_and_process(path_train_data_x_diy_1hand,path_train_data_y_diy_1hand)


dj_test = np.load('D:/CNNbkuniversity/diy_dataset/test_optical_flow/dj_5_pose.npy')
py_test = np.load('D:/CNNbkuniversity/diy_dataset/test_optical_flow/py_5_pose.npy')
zq_test = np.load('D:/CNNbkuniversity/diy_dataset/test_optical_flow/zq_5_pose.npy')












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










list_pro_ave = []
list_acc = []
list_pro_ng_ave = []

#
# test_model_111.fit([x1, x2], train_data_y, batch_size=32, verbose=2, epochs=0)
# acc,average_pro,average_pro_ng_pro = get_acc(test_model_111)
# list_acc.append(acc)
# list_pro_ave.append(average_pro)
# list_pro_ng_ave.append(average_pro_ng_pro)

# pre_epochs = 100
pre_epochs = 150

for i in range(pre_epochs):

    # print("this is the:" + str(i) + " epoch")

    # test_model_231 = get_model('D:/CNNbkuniversity/compare_model/2LSTM_3DENSE/2lstm_3dense.h5')
    test_model_221.fit([x1, x2], train_data_y, batch_size=32, verbose=2, epochs=1)
    acc,average_pro,average_pro_ng_pro = get_acc(test_model_221)
    list_acc.append(acc)
    list_pro_ave.append(average_pro)
    list_pro_ng_ave.append(average_pro_ng_pro)
    pass




print(list_acc)
print(list_pro_ave)
print(list_pro_ng_ave)
list_acc_ar = np.array(list_acc)
list_pro_ave_ar = np.array(list_pro_ave)
list_pro_ng_ave_ar = np.array(list_pro_ng_ave)




#存储221文件的代码
# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/accuracy1.npy',list_acc_ar)
# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity1.npy',list_pro_ave_ar)
# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity_ng1.npy',list_pro_ng_ave_ar)
# test_model_221.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/model_100epoch.h5')

# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/accuracy2.npy',list_acc_ar)
# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity2.npy',list_pro_ave_ar)
# np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity_ng2.npy',list_pro_ng_ave_ar)
# test_model_221.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/model_250epoch.h5')
#
np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/accuracy3.npy',list_acc_ar)
np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity3.npy',list_pro_ave_ar)
np.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/probablity_ng3.npy',list_pro_ng_ave_ar)
test_model_221.save('D:/CNNbkuniversity/compare_model/2LSTM_2DENSE/add_diy_dataset_model/1_hand/plot/model_400epoch.h5')
