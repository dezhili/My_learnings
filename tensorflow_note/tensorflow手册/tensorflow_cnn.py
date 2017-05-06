'''
构建一个多层卷积神经网络
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True) 

x = tf.placeholder("float", shape=[None, 784]) 
y_ = tf.placeholder("float", shape=[None, 10])


# 权重初始化
# 在创建模型之前，我们先来创建权重和偏置。
# 一般来说，初始化时应加入轻微噪声，来打破对称性，防止零梯度的问题。
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


# 卷积和池化
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是
# 同一个大小。我们的池化用简单传统的2x2大小的模板做max pooling
def conv2d(x, W):
    '''
    x: an input tensor of shape: [batch, in_height, in_width, in_channels] 4D
    W: a filter/kernel tensor of shape: [filter_height,filter_width,in_channels,out_channels]
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    '''
    x: A 4-D 'Tensor' with shape [batch, height, width, channels]
    ksize: a list of ints
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
# 它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
# 权重是一个 [5, 5, 1, 32] 的张量，前两个维度是patch的大小，接着是输入的通道数目，
# 最后是输出的通道数目。输出对应一个同样大小的偏置向量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 为了用这一层，我们把 x 变成一个4d向量，第2、3维对应图片的宽高，最后一维代表颜色通道。
x_image = tf.reshape(x, [-1, 28, 28, 1]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1) # 28 28 32
h_pool1 = max_pool_2x2(h_conv1)     # 14 14 32


# 第二层卷积
# 第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)  # 14 14 64
h_pool2 = max_pool_2x2(h_conv2)    # 7 7 64 


# 密集连接层
# 图片降维到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
# 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，使用ReLU激活
W_fc1 = weight_variable([7*7*64, 1024])  # 特征向量7x7x64
b_fc1 = bias_variable([1024])

h_pool2_flatten = tf.reshape(h_pool2, [-1, 7*7*64])  # batch -1
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flatten, W_fc1)+b_fc1)

# Dropout
# 我们用一个 placeholder 来代表一个神经元在dropout中被保
# 留的概率。这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
keep_prob = tf.placeholder(dtype=tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



# 输出层
# 最后，我们添加一个softmax层，就像前面的单层softmax regression一样
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)


# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 ==0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

    test_result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}) 
    print("test accuracy %g"%test_result)
