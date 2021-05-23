'''

A12文件，
功能：将标准手势光流从载体json文件中的读出，并转化存储成为npy文件

'''


path_std = 'D:/CNNbkuniversity/diy_dataset/test_optical_flow/'

import json
import numpy as np


file = open(path_std+'dj_1_of.json')
video = json.load(file)
data = []
for i in range(60):
    data.append(video[str(i)])
file.close()
data1 = np.array(data)
data2 = data1.reshape(1,60,21)
np.save(path_std+'dj_std.npy',data2)


file = open(path_std+'py_1_of.json')
video = json.load(file)
data = []
for i in range(60):
    data.append(video[str(i)])
file.close()
data1 = np.array(data)
data2 = data1.reshape(1,60,21)
np.save(path_std+'py_std.npy',data2)



file = open(path_std+'zq_1_of.json')
video = json.load(file)
data = []
for i in range(60):
    data.append(video[str(i)])
file.close()
data1 = np.array(data)
data2 = data1.reshape(1,60,21)
np.save(path_std+'zq_std.npy',data2)


file = open(path_std+'fs_1_of.json')
video = json.load(file)
data = []
for i in range(60):
    data.append(video[str(i)])
file.close()
data1 = np.array(data)
data2 = data1.reshape(1,60,21)
np.save(path_std+'fs_std.npy',data2)


file = open(path_std+'xz_1_of.json')
video = json.load(file)
data = []
for i in range(60):
    data.append(video[str(i)])
file.close()
data1 = np.array(data)
data2 = data1.reshape(1,60,21)
np.save(path_std+'xz_std.npy',data2)




