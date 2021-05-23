# 基于Readers的手势会议控制系统

###### Real time dynamic gesture recognizer based on hand pose estimation and siamese LSTM network

## 概述(Abstract)

​		为了解决当前市面上手势识别产品主要停留在静态手势识别阶段的问题，在本项目中，我们提出了**Reders模型**：基于手部姿态估计和孪生LSTM网络的实时动态手势识别器(Real time dynamic gesture recognizer based on hand pose estimation and siamese LSTM network)，并基于该模型开发了基于动态手势识别的会议控制系统。

​		通过对数据集的调研，我们将我们需要解决的问题定义为一个**小样本学习问题**。数据预处理方面，我们对视频进行了手部姿态估计后利用估计得到的时序手部特征点计算出视频中手部动作的光流信息。核心模型方面，我们设计了以**双层LSTM为编码器的孪生网络架构**，将处理得到的两组时序光流数据作为输入，通过计算相似度来间接判断两组光流数据对应的是否为同一手势动作。模型训练方面，我们使用IPN Hand数据集作为预训练数据集对Reders模型进行训练，再用我们拍摄的少量手势视频数据对模型进行进一步训练，最终得到的模型在训练集上准确率达到了**93.8%**，手势检测和识别的在CPU上平均用时为**126ms**，而在GPU上仅为**54ms**。另外，模型在实际应用中也达了很好的效果。

​		在此，我们提供训练使用的Python源代码、训练好的模型样例以及可以用来展示模型效果的demo程序。

​		如果你希望在墙内了解我们的工作，可以前往[这个仓库]([Code · lloong/A12_project - 码云 - 开源中国 (gitee.com)](https://gitee.com/lloong_x/a12_project/tree/master/Code))

## 使用(Usage)

​		若你想了解我们是如何对数据进行预处理的，可以根据    `Data Processing ` 文件夹中的README文件的指导进行试验。

​		若你想尝试复现我们的所有实验，可以根据 `Model Train ` 和 `Parameters Tuning ` 中的指引文件README.md来运行相关的代码分别实现模型的预训练和迁移训练。

​		若你想直接体验我们的demo程序，直接运行  `BetaVersionDemo\main.py `  即可。

​		若你想使用我们的模型进行进一步的开发，可以通过查看  `Model Loader\get_model_from_h5.py `中的代码了解如何对我们的模型进行载入。但注意，在进行进一步开发前，请务必联系我们获取开发许可。



## 数据集(Dataset)

​		我们使用了 [IPN数据集](https://gibranbenitez.github.io/IPN_Hand/)对模型进行预训练，同时，使用了自己拍摄的数据对模型进行迁移训练，使得模型可以很好地应用于现实生活中的会议场景。

​		若你希望获取我们用于迁移训练的数据集，请联系将申请邮件发至我们的邮箱：yuanmingli527@gmail.com。

## 开发环境需求(Requirement)

tensorflow-gpu==>2.0.0
keras-gpu==>2.3.1
matplotlib	==>3.2.1
numpy==>1.19.3
h5py==>2.10.0
opencv-python==>4.4.0.46

## 引用(Citation)

```
# 如果你使用了IPN数据集
@inproceedings{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
  pages={4340--4347},
  year={2021},
  organization={IEEE}
}
```

## 联系我们(Contact)

电话: +86 18927652067

邮箱：yuanmingli527@gmail.com



