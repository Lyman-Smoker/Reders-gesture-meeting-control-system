'''

A12文件，用于读取diy测试数据集的光流，并转化成为npy文件，数据格式(5,60,21)

功能：将自制的测试数据集的光流从载体json文件读取出，并转化成为numpy数组，并存储成为npy文件，便于读取预测


'''



folderpath = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/'
test_npy_folder_path = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/'




import json
import numpy as np



hand_list = ['dj','py','zq','xz','fs']








dj_test_list = []
zq_test_list = []
py_test_list = []
fs_test_list = []
xz_test_list = []



k = 0
for action in hand_list:
    k+=1
    this_action = []
    for i in range(1,6):
        file = open(folderpath + action +'_'+str(i)+'_of.json')
        video = json.load(file)
        data = []
        for t in range(60):
            data.append(video[str(t)])
        file.close()
        this_action.append(data)
        pass
    if k==1:
        dj_test_list.append(this_action)
        pass
    elif k==2:
        py_test_list.append(this_action)
        pass
    elif k==3:
        zq_test_list.append(this_action)
        pass
    elif k==4:
        xz_test_list.append(this_action)
        pass
    elif k==5:
        fs_test_list.append(this_action)
    pass

zq_test_list1 = np.array(zq_test_list)
zq_test_list2 = zq_test_list1.reshape(5,60,21)
np.save((test_npy_folder_path+'zq_5_pose.npy'),zq_test_list2)

dj_test_list1 = np.array(dj_test_list)
dj_test_list2 = dj_test_list1.reshape(5,60,21)
np.save((test_npy_folder_path+'dj_5_pose.npy'),dj_test_list2)

py_test_list1 = np.array(py_test_list)
py_test_list2 = py_test_list1.reshape(5,60,21)
np.save(test_npy_folder_path+('py_5_pose.npy'),py_test_list2)


xz_test_list1 = np.array(xz_test_list)
xz_test_list2 = xz_test_list1.reshape(5,60,21)
np.save((test_npy_folder_path+'xz_5_pose.npy'),xz_test_list2)

fs_test_list1 = np.array(fs_test_list)
fs_test_list2 = fs_test_list1.reshape(5,60,21)
np.save((test_npy_folder_path+'fs_5_pose.npy'),fs_test_list2)
