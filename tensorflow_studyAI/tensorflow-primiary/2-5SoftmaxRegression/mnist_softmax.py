import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#  在main()主函数中组织整个计算图并运行会话
def main(_):
    print("~~~~~~~~~~~~~开始设计计算图~~~~~~~~~~~~~")
    # 声明一个计算图作为默认图, 模型将会建在默认的graph上
    with tf.Graph().as_default():
        # Input: 定义输入节点
        with tf.name_scope('Input'):
            X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
            Y_true = tf.placeholder(tf.float32, shape=[None, 10], name='Y_true')

        # Inference: 前向预测
        with tf.name_scope('Inference'):
            W = tf.Variable(tf.zeros([784, 10]), name='Weight')
            b = tf.Variable(tf.zeros([10]), name='Bias')
            logits = tf.add(tf.matmul(X, W), b)
            # Softmax: 把logits变成预测概率分布
            with tf.name_scope('Softmax'):
                Y_pred = tf.nn.softmax(logits=logits)

        # Loss: 定义损失节点
        with tf.name_scope('Loss'):
            Loss = tf.reduce_mean(-tf.reduce_sum(Y_true * tf.log(Y_pred), axis=1))

        # Train: 定义训练节点
        with tf.name_scope('Train'):
            # Optimizer: 创建优化器
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            # Train: 定义训练节点将梯度下降法应用于Loss
            TrainOp = Optimizer.minimize(Loss)

        # Evaluate: 定义评估节点
        with tf.name_scope('Evaluate'):
            correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Initial: 添加所有Variable类型的变量初始化节点
        InitOp = tf.global_variables_initializer()


        print("把计算图写入事件文件，在TensorBoard里面查看")
        # 保存计算图
        writer = tf.summary.FileWriter(logdir='F:\Anaconda\logs', graph=tf.get_default_graph())
        writer.close()


        print('开始运行计算图')
        # 加载数据
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

        # 声明一个交互式会话
        sess = tf.InteractiveSession()
        # 初始化所有变量: W, b
        sess.run(InitOp)

        # 开始按批次训练，总共训练1000个批次，每个批次100个样本
        for step in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            # 将当前批次的样本feed给计算图中的占位符，启动训练节点开启训练
            _, train_loss = sess.run([TrainOp, Loss],
                                     feed_dict={X:batch_xs, Y_true:batch_ys})
            print('train step: ', step, ", train_loss: ", train_loss)

        accuracy_score = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_true:mnist.test.labels})
        print('模型准确率: ', accuracy_score)

# 调用main()
if __name__ == '__main__':
    # 首先申明一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 为参数解析器添加参数: data_dir(指定数据集存放路径)
    parser.add_argument('--data_dir', type=str,
                        default='MNIST_data/', help='数据集存放路径')
    FLAGS, unparsed = parser.parse_known_args()  # 解析参数
    # 运行Tensorflow 应用
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)









