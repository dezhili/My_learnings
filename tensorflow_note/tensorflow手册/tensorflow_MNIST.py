'''
60000 mnist.train (mnist.train.images+mnist.train.labels)  
10000 mnist.test
在MNIST 训练数据集中， mnist.train.images 形状为 [60000, 784]的张量
                      mnist.train.labels [60000, 10]

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True) 

x = tf.placeholder("float", [None, 784]) # 二维浮点数张量。 [None,784]表示此张量的第一个维度可以是任何长度

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

#  训练模型 loss function - cross-entropy(交叉熵)
y_ = tf.placeholder("float", [None, 10])  # 正确值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # (计算一个batch的)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#  评估我们的模型性能
#  找出那些预测正确的标签。 tf.argmax(y,1)是模型认为每个输入最有可能对应的标签的索引
#  tf.argmax(y_,1)代表正确的标签。 tf.equal() 检测预测是否与真实标签匹配
def compute_accuracy(xs, ys):
    global y
    y_pre = sess.run(y, feed_dict={x: xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(ys,1)) #[True, ..]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    result = sess.run(accuracy, feed_dict={x: xs, y: ys})
    return result



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    if i% 50 ==0 :
        print(compute_accuracy(mnist.test.images, mnist.test.labels))

    
    




