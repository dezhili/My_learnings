'''
The original dataset has 60,000 different digits for training and 10,000 for testing
32x32 --> 6x28x28->6x14x14 --> 16x10x10->16x5x5 -->120 --> 84

LeNet-5 architecture
'''

'''
Dataset description and loading

'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt 

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
# print(mnist)
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[1].reshape((28,28), order='C'), cmap='Greys', interpolation='nearest')
plt.show()


'''
Dataset preprocessing--
    better classification scores can be achieved just by augmenting the dataset examples with
    linearly transformed existing samples,such as translated, rotated, and skewed samples.
'''


'''
Modelling architecture
It begins generating a dictionary of with names:
    'wc1':tf.Variable(tf.random_normal([5, 5, 1, 32]))
    'wc2':tf.Variable(tf.random_normal([5, 5, 32, 64]))
    'wd1':tf.Variable(tf.random_normal([7*7*64, 1024]))
    'out':tf.Variable(tf.random_normal([1024, n_classes]))
Define the connected layers,integrating one after another:
    conv_layer_1= conv2d(x_in, weights['wc1'], biases['bc1'])
    conv_layer_1= subsampling(conv_layer_1, k=2)
    conv_layer_2= conv2d(conv_layer_1, weights['wc2'], biases['bc2'])
    conv_layer_2= subsampling(conv_layer_2, k=2)
    fully_connected_layer= tf.reshape(conv_layer_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fully_connected_layer= tf.add(tf.matmul(fully_connected_layer, weights['wd1']), biases['bd1'])
    fully_connected_layer= tf.nn.relu(fully_connected_layer)
    fully_connected_layer= tf.nn.dropout(fully_connected_layer, dropout)
    prediction_output = tf.add(tf.matmul(fully_connected_layer, weights['out']), biases['out'])

'''


'''
Loss function description
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
'''
'''
Loss function optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
'''

'''
Accuracy test
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
''' 

batch_size = 128
learning_rate = 0.05
number_iterations = 2000
steps = 10

n_input = 784
n_classes = 10
dropout = 0.80

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def subsampling(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')



# Create model
def conv_net(x_in, weights, biases, dropout):
    x_in = tf.reshape(x_in, shape=[-1, 28, 28, 1])

    conv_layer_1 = conv2d(x_in, weights['wc1'], biases['bc1'])
    conv_layer_1 = subsampling(conv_layer_1, k=2)

    conv_layer_2 = conv2d(conv_layer_1, weights['wc2'], biases['bc2'])
    conv_layer_2 = subsampling(conv_layer_2, k=2)

    fully_connected_layer = tf.reshape(conv_layer_2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fully_connected_layer = tf.add(tf.matmul(fully_connected_layer, weights['wd1']), biases['bd1'])
    fully_connected_layer = tf.nn.relu(fully_connected_layer)

    fully_connected_layer = tf.nn.dropout(fully_connected_layer, dropout)

    prediction_output = tf.add(tf.matmul(fully_connected_layer, weights['out']),biases['out'])
    return prediction_output

# Store layers weight and biases
weights={
    'wc1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'wc2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64, 1024])),
    'out':tf.Variable(tf.random_normal([1024, n_classes]))    
}

biases={
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(X, weights, biases, keep_prob)

# Define loss and optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step*batch_size < number_iterations:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        test = batch_x[0]
        fig = plt.figure()
        plt.imshow(test.reshape((28, 28),order='C'), cmap='Greys', interpolation='nearest')
        plt.show()

        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y, keep_prob:dropout})
        if step % steps == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:batch_x, Y:batch_y, keep_prob:1.})
            print('Iter'+str(step*batch_size)+', Minibatch Loss= '+\
                  '{:.6f}'.format(loss) + ', Training Accuracy= '+\
                  '{:.5f}'.format(acc))
        step += 1

    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X:mnist.test.images[:256],
                                      Y:mnist.test.labels[:256],
                                      keep_prob: 1.}





