'''
1、使用tf.train.Saver.save()方法保存模型
  参数: sess: 用于保存变量操作的会话
        save_path: String类型， 用于指定训练结果的保存路径
        global_step: 这个数字会添加到save_path后面，用于构建checkpoint文件。
                     这个参数有助于区分不同训练阶段的结果

2. 使用tf.train.Saver.restore
        sess: 用于加载变量操作的会话
        save_path: 同保存模型
'''

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# hyper-parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = './tensorflow_save_restore/model.ckpt'

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer 
n_input = 784
n_classes = 10

# tf Graph input 
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.relu(layer2)

    out_layer = tf.matmul(layer2, weights['out']) + biases['out']
    return out_layer

# Store layers weights and biases
weights={
    'h1':tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
optimizer = tf.train.AdamOptimizer(learning_rate)
trian_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Running first session
print('starting 1st session')
with tf.Session() as sess:
    sess.run(init)

    # training cycle
    for epoch in range(10):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([trian_op, cost], feed_dict={x:batch_x, y:batch_y})
            avg_cost += c/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:",'%04d'%(epoch+1), 'cost=', "{:.9f}".format(avg_cost))
    print("First optimization finished")


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s"% save_path)


# Running a new session
print("starting 2nd session...")
with tf.Session() as sess:
    sess.run(init)

    load_path = saver.restore(sess, model_path)
    print("Model restored from file: %s"%save_path)

    # Resume training
    for epoch in range(10):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([trian_op, cost], feed_dict={x:batch_x,y:batch_y})

            avg_cost += c/total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:",'%04d'%(epoch+1), 'cost=', "{:.9f}".format(avg_cost))
    print("Second optimization finished")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy: ", sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels}))
   