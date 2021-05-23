<font size=5>**Siamese_xlstm_ydense.py**</font>

**概述(abstract)**

​		该文件夹内文件命名方式为 `Siamese_xlstm_ydense.py` ，表示搭建一个具有**x**个LSTM层和**y**个FULLY_CONNECTION层的孪生神经网络，使用使用IPN训练数据集文件进行训练，得出对应的预训练神经网络模型。

​		为了使模型兼具高效性和准确性，我们将**x**取值范围为[1,2]，**y**取值范围为[1,2,3]

**使用(Usage)**

​		需要手动传入参数：

​		**path_train_data_x_IPN**：IPN训练数据集npy文件路径

​		**path_train_data_y_IPN**：IPN训练数据集标签npy文件路径

​		**path**：指定预训练模型保存的路径

​		**神经网络模型参数**：参数初始设置是由`Parameters Tuning/plot.py`绘制出的最优化参数，不需要更改。如读者需要用本架构实现其他数据集的训练，可使用不同参数进行调整。

