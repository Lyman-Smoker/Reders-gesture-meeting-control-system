<font size=5>**Siamese_xlstm_ydense_train_diy.py**</font>

**概述(abstract)**

​		该类文件命名方式为`Siamese_xlstm_ydense_train_diy.py`，表示使用具有**x**个LSTM层和**y**个FULLY_CONNECTION层的孪生神经网络，对单手(抓取、点击、平移)手势数据集进行迁移训练，得出最终神经网络模型。

**使用(Usage)**

​		需要手动传入参数：

​		**pretrain_model_path**：预训练模型文件路径，即由IPN数据集训练得出的预训练神经网络模型路径

​		**path_train_data_x_diy_1hand**：自制单手手势训练数据集npy文件路径

​		**path_train_data_y_diy_1hand**：自制单手手势训练数据集标签npy路径

​		**transfer_model_path**：经过自制手势数据集迁移训练后得到的最终模型指定存储路径



<font size=5>**Siamese_xlstm_ydense_train_diy_2hand.py**</font>

**概述(abstract)**

​		该类文件命名方式为`Siamese_xlstm_ydense_train_diy_2hand.py`，表示使用具有**x**个LSTM层和**y**个FULLY_CONNECTION层的孪生神经网络，对双手(旋转、放缩)手势数据集进行迁移训练，得出最终神经网络模型。

**使用(Usage)**

​		需要手动传入参数：

​		**pretrain_model_path**：预训练模型文件路径，即由IPN数据集训练得出的预训练神经网络模型路径

​		**path_train_data_x_diy_2hand**：自制双手手势训练数据集npy文件路径

​		**path_train_data_y_diy_2hand**：自制双手手势训练数据集标签npy路径

​		**transfer_model_path**：经过自制手势数据集迁移训练后得到的最终模型指定存储路径

