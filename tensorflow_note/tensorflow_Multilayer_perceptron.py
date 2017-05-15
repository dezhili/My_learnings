'''
多层神经网络 (多层感知机) 减轻过拟合的Dropout 自适应学习速率的Adagrad  可以解决梯度弥散的激活函数relu
MLP (全连接神经网络FCN)
'''

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

# 在训练和预测时，Dropout的比率keep_prob是不一样的，
# 训练时小于1(制造随机性，防止过拟合)，预测时等于1(使用全部特征来预测样本的类别)
x = tf.placeholder(tf.float32, [None, in_units]) 
keep_prob = tf.placeholder(tf.float32)


# 第一步 定于模型架构(forward)
hidden1 = tf.nn.relu(tf.matmul(x, w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2)+b2)


# 第二步 定于损失函数和选择自适应的优化器Adagrad
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)


# 第三步 训练(keep_prob, 因为加入了隐藏层，需要更多的训练迭代来优化模型参数)
sess=  tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys, keep_prob:1.0})
    if i % 100 == 0:
        print(sess.run([cross_entropy],  {x:batch_xs, y_:batch_ys, keep_prob:1.0}))

# 第四步 模型评估 (对模型进行准确率评测)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))


sess.close()