'''
使用RNN来进行分类的训练，使用MNIST，让RNN从每一张图片的第一行像素读到最后一行，然后再进行分类判断

'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters  =100000
batch_size = 128
n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# weights biases 初始值的定义
weights={
    # shape(28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape(128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases={
    # shape(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    # shape(10,)
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
}


# 定义RNN的主体结构(input_layer, cell, output_layer)
def RNN(X, weights, biases):
    # hidden layer for input
    # 原始的X是3维数据， 我们需要把它变成2维数据才能使用weights的矩阵乘法
    # X ==> (128batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换成3维
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])


    # 接着是 cell 中的计算 tf.nn.dynamic_rnn(cell, inputs)
    # 对lstm来说， state 可分为(c_state, h_state)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # 初始化全零 state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # 第一种: 直接调用final_state 中的 h_state(final_state[1])来进行运算
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    # 第二种: 调用最后一个outputs (在这个例子 和上面的final_state[1]一样)
    # 把outputs 变成列表 [(batch, outputs)..]* steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个output

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 训练RNN
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys}))
        step += 1
