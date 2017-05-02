#y=0.1x+0.3(构造神经网络结构)
#tensorflow参数 weights(权重)bias(偏置)

import tensorflow as tf
import numpy as np

###create data(使用NumPy 生成假数据)
x_data=np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

###create tensorflow structure start###
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))##最小化方差

optimizer = tf.train.GradientDescentOptimizer(0.5)#神经网络要减小误差，提升参数的准确度，在下一次误差就更小
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()#在神经网络初始化变量
###create tensorflow structure end###

###启动图
sess = tf.Session()#指针指向图
sess.run(init)#激活init,结构

###拟合平面
for step in range(201):
    sess.run(train)
    if step % 20 ==0:
        print(step,sess.run(Weights),sess.run(biases))
