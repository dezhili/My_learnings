'''
cifar-10 教程演示了在tensorflow 上构建更大更复杂模型的几个重要内容:
 相关核心数学对象, 如卷积， 修正线性激活  最大池化  局部响应归一化
 训练过程中一些网络行为的可视化， 包括 输入图像 损失情况 网络行为的分布 梯度
 算法学习参数的移动平均值的计算函数， 在评估阶段使用这些平均值来提高预测性能
 实现一种机制，使得学习率随着时间的推移而递减
 为输入数据设计预存取队列，将磁盘延迟和高开销的图像预处理操作与模型分离


模型架构:
    本教程中的模型是一个多层架构，由卷积层和非线性层(nonlinearities)交替多次排列后构成。
    这些层最终通过全连通层对接到softmax分类器上。
    这一模型除了最顶部的几层外，基本跟Alex Krizhevsky提出的模型一致
cifar10_input.py cifar10.py cifar10_train.py cifar10_eval.py

'''

'''
CIFAR-10 模型 （cifar10.py）
1. 模型输入: inputs() distorted_inputs() 分别用于读取cifar图像进行预处理，
            为后续评估和训练
2. 模型预测: inference() 
3. 模型训练: loss() train()
'''

'''
1. 模型输入
    输入模型通过inputs() distorted_inputs() 建立，这两个函数会从CIFAR-10二进制文件
    读取图片文件，由于每个图片的存储字节数是固定的，因此可以使用tf.FixedLengthRecordReader()
    图片文件的处理流程如下:
        图片会被统一裁剪到24x24像素大小，裁剪中央区域用于评估或随机裁剪用于训练
        图片会进行近似白化处理，使得模型对图片的动态范围变化不敏感
    对于训练 ， 我们另外采取了一系列随机变换的方法来人为的增加数据集的大小
    对图像进行随机的左右翻转；
    随机变换图像的亮度；
    随机变换图像的对比度；


2. 模型预测
inference() 计算预测值的logits
conv1 : 实现卷积以及relu activation
pool1 : max pooling
norm1: lrn(local response normalization) 局部响应归一化
conv2: 卷积以及 relu activation
norm2:
pool2:
local3:
local4:
softmax_linear: 进行线性变换以输出logits

3. 模型训练
训练一个可进行N维分类的网络的常用方法是使用softmax回归。
Softmax 回归在网络的输出层上附加了一个softmax nonlinearity，
并且计算归一化的预测值和label的1-hot encoding的交叉熵。
模型的目标函数是求交叉熵损失和所有权重衰减项的和，loss()函数的返回值就是这个值。
'''