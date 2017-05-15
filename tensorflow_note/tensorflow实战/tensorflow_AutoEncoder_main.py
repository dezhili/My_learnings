import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import AdditiveGaussianNoiseAutoencoder
from AdditiveGaussianNoiseAutoencoder import *

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 先定义一个对训练 测试数据进行标准化处理的函数。
# 标准化 即让数据变成0均值 标准差为1的分布。
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 定义一个获取随机block数据的函数: 取一个从 0 到 len(data)-batch_size 之间的随机整数，
# 再以这个随机数作为block的起始位置，然后顺序取到一个 batch size的数据 不放回抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data)-batch_size)
    return data[start_index: (start_index+batch_size)]

# 使用之前定义的 standard_scale() 对训练集 测试集进行标准化变换
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 总训练样本数， 最大训练的轮数(epoch)为20， batch_size=128, 并设置每隔一轮(epoch)显示一次损失cost
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1




# 穿件一个 AGN 自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200,
                                               transfer_function = tf.nn.softplus,
                                               optimizer = tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale = 0.01)

# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0. 
    totle_batch = int(n_samples / batch_size)

    for i in range(totle_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost/n_samples * batch_size

    if epoch % display_step == 0:
        print("Eplch:", '%04d' % (epoch + 1), "cost=", '{:.9f}'.format(avg_cost))

# 最后对训练完的模型进行性能测试
print("Total cost: ", str(autoencoder.calc_total_cost(X_test)))