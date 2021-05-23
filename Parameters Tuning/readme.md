**<font size=5>plot_it.py</font>**

**概述(abstract)**

​		该代码用于绘制6种模型用IPN数据集预训练的相关指标，生成npy文件至指定目录，并绘制出epochs_loss曲线至指定目录

**使用(Usage)**

​		需要手动指定参数：

​		**path_train_data_x_IPN**：IPN训练数据集npy文件路径

​		**path_train_data_y_IPN**：IPN训练数据集标签npy文件路径

​		**npy_file_path**：epoch_loss指标生成的npy文件存储路径

​	

<font size=5>**formal_train_plot_xyz.py**</font>

**概述(abstract)**

​		该类文件的命名格式为`formal_train_plot_xyz.py`，表示对包含**x**个LSTM层、**y**个FULLY_CONNECTION层的孪生网络进行**z**手势动作的指标进行提取存储。

​		指标包含**loss**(损失)、**accuracy**(准确度)、**similarity**(同种手势平均相似度)、**dissimilarity**(不同手势平均不相似度)

​		为了保证模型的效率和准确度，**x**取值为[1,2]，**y**取值为[1,2,3]，**z**取值为[1,2]

**使用(Usage)**

​		当**z**取值为1时，需要手动传入如下参数：

​		**pretrain_model_path**：经过IPN数据集预训练后得到的预训练模型路径

​		**path_train_data_x_diy_1hand**：自制单手手势数据集npy文件路径

​		**path_train_data_y_diy_1hand**：自制单手数据集标签npy文件路径

​		**dj_test_path**：点击手势测试动作npy文件路径

​		**py_test_path**：平移手势测试动作npy文件路径

​		**zq_test_path**：抓取手势测试动作npy文件路径

​		**accuracy_npy_path**：epoch_accuracy指标生成的npy文件路径

​		**similarity_npy_path**：epoch_similarity指标生成的npy文件路径

​		**dissimilarity_npy_path**：epoch_dissimilarity指标生成的npy文件路径



​		当**z**取值为2时，需要手动传入如下参数：

​		**pretrain_model_path**：经过IPN数据集预训练后得到的预训练模型路径

​		**path_train_data_x_diy_2hand**：自制双手手势数据集npy文件路径

​		**path_train_data_y_diy_2hand**：自制双手手势数据集标签npy文件路径

​		**fs_test_path**：放缩手势测试动作npy文件路径

​		**xz_test_path**：旋转手势测试动作npy文件路径

​		**accuracy_npy_path**：epoch_accuracy指标生成的npy文件路径

​		**similarity_npy_path**：epoch_similarity指标生成的npy文件路径

​		**dissimilarity_npy_path**：epoch_dissimilarity指标生成的npy文件路径



<font size=5>**plotitxyz.ipynb**</font>

**概述(abstract)**

​		该代码用于绘制6种模型经过`formal_train_plot_xyz.py`产生的相关训练指标，包括epoch_loss、epoch_accuracy、epoch_similarity、epoch_dissimilarity曲线。

​		该类文件的命名格式为`plotitxyz.ipynb`，表示对包含**x**个LSTM层、**y**个FULLY_CONNECTION层的孪生网络进行**z**手势动作的指标进行绘制。

​		为了保证模型的效率和准确度，**x**取值为[1,2]，**y**取值为[1,2,3]，**z**取值为[1,2]

**使用(Usage)**

​		使用jupyter notebook打开，将经过`formal_train_plot_xyz.py`产生的相关**训练指标npy文件**放置于与该文件同一根目录文件夹下，即可运行，在同一文件夹下可以得到需要被绘制的图线。



<font size=5>**plot_acc.ipynb、plot_similarity.ipynb、plot_dissimilarity.ipynb、plot_loss.ipynb**</font>

**概述(abstract)**

​		以上四个代码文件用于绘制六种模型在自制数据集上训练后产生的指标对比，包括6条**accuracy**、**similarity**、**dissimilarity**、ipynb曲线在同一图表的对比曲线。

**使用(Usage)**

​		使用jupyter notebook打开，将经过`formal_train_plot_xyz.py`训练产生的6个**epoch_loss**、6个**epoch_accuracy**、6个**epoch_similarity**、6个**epoch_dissimilarity**所对应的npy文件放置于与以上四个ipynb文件相同的根目录下，分别运行四个ipynb文件，即可得到需要被绘制的曲线。



<font size=5>**get_accuracy_xhand.py**</font>

**概述(abstract)**

​		该类文件的命名格式为`get_accuracy_xhand.py`，**x**取值为1表示在单手手势测试集做预测，**x**取值为2表示在双手手势测试集做预测。

​		在自制手势测试集上作预测，规定相似度大于**0.9**即认定为相同手势，相似度小于**0.1**即认定为不同手势。计算得出准确率、同种手势之间平均相似度和不同手势之间的平均不相似度。

**使用(Usage)**

​		当**x**取值为1，需要手动传入参数：

​		**dj_test_path**：点击手势测试集npy文件路径

​		**py_test_path**：平移手势测试集npy文件路径

​		**zq_test_path**： 抓取手势测试集npy文件路径

​		当**x**取值为2，需要手动传入参数：

​		**fs_test_path**：放缩手势测试集npy文件路径

​		**xz_test_path**：旋转手势测试集npy文件路径







