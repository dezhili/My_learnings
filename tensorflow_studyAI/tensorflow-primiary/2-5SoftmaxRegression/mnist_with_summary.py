''' 一个简单的MNIST 分类器用来演示 Tensorboard 的 summaries 功能...'''
import argparse
import sys
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 全局变量, 用来存放基本的模型超参数
FLAGS = None
NUM_CLASSES = 10
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_PIXELS = IMG_WIDTH * IMG_HEIGHT

# 创建一个代有合适的初始值的权重变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
# 创建一个代有合适的初始值的偏置变量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
# 对一个张量进行全面的汇总(均值、标准差、最大最小值、直方图)，用于tensorboard 可视化
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

# 可重用代码来创建一个简单的nn layer
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name = 'activation')
        tf.summary.histogram('activations', activations)
        return activations


# 构造计算图，启动session 训练过程
def train():
    # 输入占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMG_PIXELS], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y-input')

    # input reshape 便于 显示图片 tensorboard
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, IMG_PIXELS, FLAGS.hidden1_units, 'layer1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        dropped = tf.nn.dropout(hidden1, keep_prob=keep_prob)
        tf.summary.scalar('dropout_keep_probability', keep_prob)

    logits = nn_layer(dropped, FLAGS.hidden1_units, NUM_CLASSES, 'layer2', act=tf.identity)

    # 定义损失节点
    with tf.name_scope('cross_entropy'):
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    # 定义训练节点
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_step = optimizer.minimize(cross_entropy)

    # 定义评估节点
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    # 导入数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True, fake_data=FLAGS.fake_data)

    # 生成一个Tensorflow feed_dict: 将真实的张量数据映射到 Tensor placeholder
    def feed_dict(train):
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}


    # 声明一个交互式会话
    sess = tf.InteractiveSession()

    # 整合所有节点，将汇总事件文件写入日志
    merged = tf.summary.merge_all()
    # 用来写入训练日志(训练日志的event文件 中 有计算图)
    train_writer = tf.summary.FileWriter(FLAGS.log_dir+'/train', sess.graph)
    # 用来写入测试日志
    test_writer = tf.summary.FileWriter(FLAGS.log_dir+'/test')

    # 初始化所有变量
    tf.global_variables_initializer().run()

    # # 写入计算图
    # graph_writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())
    # graph_writer.flush()

    for step in range(FLAGS.max_steps):
        _, summary_str, XentropyLoss = sess.run([train_step, merged, cross_entropy], feed_dict=feed_dict(True))
        train_writer.add_summary(summary_str, step)
        print('step idx: ', step, 'xentropy_loss: ', XentropyLoss)

        if step % 100 == 0:
            test_summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(test_summary, step)
            print('Accuracy at step %s: %s'%(step, acc))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    # 启动训练过程
    train()


# 用 Argparser 把模型的超参数全部解析到FLAGS 中
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                      default=False,
                      help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')
    parser.add_argument('--hidden1_units', type=int, default=100,
                        help = 'Number of units in the first hidden layer')
    parser.add_argument('--data_dir', type=str, default='MNIST_data/',
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='F:/Learnings/All_learnings/Learning/tensorflow_studyAI/2-5SoftmaxRegression/logs',
                      help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)