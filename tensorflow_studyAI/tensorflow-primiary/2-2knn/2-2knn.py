'''
Tensorflow 实现K近邻分类器
1. inference() - 构建学习器模型的前向预测过程(从输入到预测输出的计算图路径)
2. evalute() - 在测试数据集上对模型的预测性能进行评估， 没有Loss也没有Train(knn没有显示模型参数优化过程)

'''
import numpy as np 
import os
import tensorflow as tf 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

Xtrain, Ytrain = mnist.train.next_batch(5000)  # 5000 用于训练
Xtest, Ytest = mnist.test.next_batch(200)
print("Xtrain.shape: ", Xtrain.shape, "Xtest.shape: ", Xtest.shape)
print("Ytrain.shape: ", Ytrain.shape, "Ytest.shape: ", Ytest.shape)


# 计算图输入占位符
xtrain = tf.placeholder('float', [None, 784])
xtest = tf.placeholder('float', [784])
# L1 距离 进行最近邻计算
distance = tf.reduce_sum(tf.abs(tf.add(xtrain, tf.negative(xtest))), axis=1)
# 预测: 获得最小距离的索引(根据最近邻的类标签进行判断)
pred = tf.argmin(distance, 0)
# 评估: 判断给定的一条测试样本是否预测正确(不再计算图里，这里用的是np)


# 最近邻分类器的准确率
accuracy = 0

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    Ntest = len(Xtest)

    for i in range(Ntest):
        # 获取当前测试样本的最近邻
        nn_index = sess.run(pred, feed_dict={xtrain: Xtrain, xtest: Xtest[i, :]})
        pred_class_label = np.argmax(Ytrain[nn_index])
        true_class_label = np.argmax(Ytest[i])
        print("Test", i, "Predicted Class Label:", pred_class_label, 
            "True Class Label:", true_class_label)

        if pred_class_label == true_class_label:
            accuracy += 1

    print("Done!")
    accuracy /= Ntest
    print("Accuracy:", accuracy)
