<font size=5>**diy_test_dataset_5_pose_to_npy.py**</font>

**概述(abstract)**

​		将自制的测试数据集的光流从载体json文件读取出，并转化成为numpy数组，并存储成为npy文件，便于读取预测。

**使用(Usage)**

​		手动传递的参数：

​		**folderpath**：给出载体json文件所在文件夹路径，文件夹内的文件以 动作缩写+' '+序号+'_of.json'的方式命名，例如抓取的第一个动作被命名为：zq 1_of.json 

​		**test_npy_folder_path**：测试数据集转化成为npy文件后，存储的文件夹位置





<font size=5>**get_list_from_diy_opticalflow.py**</font>

**概述(abstract)**

​		将自制训练小数据集的光流从载体json文件读出，并且进行两两组合，同类打标签True，不同类打标签False。即生成三维list_dataset。

**使用(Usage)**

​		被其他文件调用

​		函数需要传递的参数：

​		**data_path**：自制小数据量训练集中json数据集的文件夹路径，文件夹内的文件以 动作缩写+' '+序号+'_of.json'的方式命名，例如抓取的第一个动作被命名为：zq 1_of.json 

​		**hand_num**：取值为整数1或者2，当取值为1时，说明采集单手动作数据集，例如平移、点击、抓取；当取值为2时，说明采集双手动作数据集，例如放缩、旋转。





<font size=5>**get_list_from_diytest_opticalflow.py**</font>

**概述(abstract)**

​		将自制测试数据集的光流从载体json文件读出，并且进行两两组合，同类打标签True，不同类打标签False。即生成三维list_dataset。

**使用(Usage)**

​		被其他文件调用

​		函数需要传递的参数：

​		**data_path**：自制测试集中json数据集的文件夹路径，文件夹内的文件以 **动作缩写+' '+序号+'_of.json'** 的方式命名，例如抓取的第一个动作被命名为：zq 1_of.json 

​		**hand_num**：取值为整数1或者2，当取值为1时，说明采集单手动作测试集，例如平移、点击、抓取；当取值为2时，说明采集双手动作测试集，例如放缩、旋转。



<font size=5>**get_list_from_IPN_opticalflow.py**</font>

**概述(abstract)**

​		将预训练IPN数据集的光流从载体json文件读出，并且进行两两组合，同类打标签True，不同类打标签False。即生成三维list_dataset。

**使用(Usage)**

​		被其他文件调用，调用时需要传入参数：

​		**data_path**：传入存储IPN数据集载体json文件的文件夹路径；文件夹内文件以 **动作编号+'_optical_flow.json'** 方式命名，例如第6类动作的命名为 6_optical_flow.json





<font size=5>**make_NPY_dataset_from_IPN.py**</font>

**概述(abstract)**

​		获取`get_list_from_IPN_opticalflow.py`、`get_list_from_diy_opticalflow`、`get_list_from_diytest_opticalflow`产生的list_dataset三维列表，存储进入.npy文件， train_x 和 train_y文件分开存放。

**使用(Usage)**

​		包含两个函数：**load_data_to_npy**、**load_data_to_test_npy**

​		

​		**load_data_to_npy**需要手动传入参数：

​			**which_dataset**： 

​							值为0，说明调用`get_list_from_IPN_opticalflow.py`中的**get_IPN_dataset()**函数处理IPN训练数据集；							值为1，说明调用`get_list_from_diy_opticalflow`中的**get_diy_dataset()**处理自制训练集的单手数据集；							值为2，说明调用`get_list_from_diy_opticalflow`中的**get_diy_dataset()**处理自制训练集的双手数据集

​			**path_dataset**：作为传入**get_diy_dataset()**或者**get_IPN_dataset()**函数的训练数据集文件夹参数

​			**path_train_data_x**：指定处理完成后训练集的存储文件

​			**path_train_data_y**：指定处理完成后训练集对应标签的存储文件



​		**load_data_to_test_npy**需要手动传入参数：

​			**which_dataset**：

​							值为0，说明调用`get_list_from_IPN_opticalflow.py`中的**get_IPN_dataset()**函数处理IPN测试数据集；							值为1，说明调用`get_list_from_diytest_opticalflow`中的**get_diy_dataset_test()**处理自制测试集的单手数据集；							值为2，说明调用`get_list_from_diytest_opticalflow`中的**get_diy_dataset_test()**处理自制测试集的双手数据集

​			**path_dataset**：作为传入**get_diy_dataset_test()**或者**get_IPN_dataset()**函数的测试数据集文件夹参数

​			**path_train_data_x**：指定处理完成后测试集的存储文件

​			**path_train_data_y**：指定处理完成后测试集对应标签的存储文件







<font size=5>**read_npy_file_process.py**</font>

**概述(abstract)**

​		从train_x和train_y对应的npy文件读取并获得用于神经网络训练与预测的向量返回值为 x1，x2，label

**使用(Usage)**

​		手动传入参数：

​		**path_train_data_x**：`make_NPY_dataset_from_IPN.py`调用后产生的训练集对应npy文件路径。

​		**path_train_data_y**：`make_NPY_dataset_from_IPN.py`调用后产生的训练集标签对应npy文件路径。





<font size=5>**read_std_of_5_action.py**</font>

**概述(abstract)**

​		将标准手势光流从载体json文件中的读出，在同文件夹下转化存储成为npy文件。

**使用(Usage)**

​		手动传入参数：

​		**path_std**：标准光流文件夹路径，标准光流文件命名 **手势动作缩写+'_1_of.json'** ，例如抓取手势标准光流：zq_1_of_json