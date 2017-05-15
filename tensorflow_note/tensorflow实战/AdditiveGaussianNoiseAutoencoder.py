'''
去躁自编码器
'''
import numpy as np 
import sklearn.preprocessing as prep 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


'''
xavier initialization
Xavier 的初始化器在caffe早期版本被频繁使用，它的特点是会根据某一层网络的输入 输出节点数量字典调整最适合的分布
    
Xavier 初始化器就是让权重被初始化的不大不小，正好合适。
从数学角度，权重满足0均值，同时方差满足2/(nin + nout),分布可以用均匀分布或者高斯分布。
'''
def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                              minval = low, maxval = high,
                              dtype = tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale=0.1):
        '''
        Args:
            n_input : 输入变量数
            n_hidden : 隐藏层节点数
            transfer_function : 隐藏层激活函数，默认为softplus
            optimizer: 优化器, 默认为Adam
            scale : 高斯噪声系数 ， 默认为0.1
            参数初始化使用接下来定义的 _initialize_weights()  这里只使用了一层隐藏层
        '''
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        '''
        开始定义网络结构
        输入层: 输入x创建一个维度为n_input 的placeholder
        隐藏层: 先将输入x加上噪声， 将加了噪声的输入与隐藏层的权重w1相乘....n_input
        输出层: 进行数据复原 重建操作
    '''
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale*tf.random_normal((n_input,)), 
                                                self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 定义自编码器的损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                                               self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)


        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # 参数初始化函数 _initialize_weights()
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights

    # 定义计算cost及执行一步
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                   feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost

    # 训练完毕后使用
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale: self.training_scale})

    # trandform() 返回自编码器隐藏层的输出结果。提供一个接口来获取抽象后的特征，学习出的数据中的高阶特征
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x:X, self.scale: self.training_scale})

    # generate() 将隐藏层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据
    # 将高阶特征复原为原始数据
    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # 整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruct, feed_dict={self.x: X, self.scale: self.training_scale})

    # 获得隐藏层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    def getBiases(self):
        return self.sess.run(self.weights['b1'])



