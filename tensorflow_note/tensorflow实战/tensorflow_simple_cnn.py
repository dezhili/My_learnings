import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True) 
# 先定义好权重和偏置 函数 ，以便重复使用。 权重以截断的正态分布噪声，标准差设为0.1， 同时因为使用
# relu ，给偏置添加一些(0.1)避免死亡节点(dead neurons)
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

#卷积层 池化层 
# tf.nn.conv2d(x, w, stride, padding) x:输入， w:[ksize,ksize,input_channel, output_channel(kernel number)]
# stride: [1,1,1,1]步长 ，不会遗漏掉图片中的每个点。 
# tf.nn,max_pool(x, ksize, stride, padding) ,最大池化会保留原始像素块中灰度值最高的那个像素。即保留最显著的特征
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#因为卷积网络会利用到空间结构的信息， 因此需要将1D 的输入向量转换为 2D的图片结构， 即1x784->28x28
#同时只有一个颜色通道，最终[-1, 28, 28, 1] -1代表样本数目不固定， 最后1代表颜色通道数量
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 开始定义结构

# 第一个卷积层
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)                         #7x7

# 全连接层 28x28->7x7 + 卷积核数量为64  将其转成1D的向量，然后连接一个全连接层，
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, w_fc1)+b_fc1)

# 为了减轻过拟合， 使用一个Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 最后将Dropout 的输出连接一个Softmax层，得到最后的概率输出
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)


# 定义损失函数 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 下面开始训练过程
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
print("test accuracy %g"%sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
