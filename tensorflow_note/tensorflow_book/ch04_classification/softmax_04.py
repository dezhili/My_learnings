import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

# generate some initial 2D data
learning_rate= 0.01
training_epochs= 1000
num_labels = 3
batch_size = 100

x1_label0 = np.random.normal(1, 1, (100,1))
# print(x1_label0.shape)
x2_label0 = np.random.normal(1, 1, (100,1))

x1_label1 = np.random.normal(5, 1, (100,1))
x2_label1 = np.random.normal(4, 1, (100,1))

x1_label2 = np.random.normal(8, 1, (100,1))
x2_label2 = np.random.normal(0, 1, (100,1))

# plt.scatter(x1_label0, x2_label0, c='r', marker='o', s=60)
# plt.scatter(x1_label1, x2_label1, c='g', marker='x', s=60)
# plt.scatter(x1_label2, x2_label2, c='b', marker='_', s=60)
# plt.show()



# define the labels and shuffle the data
xs_label0 = np.hstack((x1_label0, x2_label0)) # 100x2 label0
# print(xs_label0.shape)
# print(type(xs_label0))
xs_label1 = np.hstack((x1_label1, x2_label1))
xs_label2 = np.hstack((x1_label2, x2_label2))

xs = np.vstack((xs_label0, xs_label1, xs_label2)) # 300x2
print('train samples shape: ', str(xs.shape))
labels = np.matrix([[1., 0., 0.]]*len(x1_label0)+
                   [[0., 1., 0.]]*len(x1_label1)+
                   [[0., 0., 1.]]*len(x1_label2)) # 300x3
print('train labels shape: ', str(labels.shape))
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]



# test inputs
test_x1_label0 = np.random.normal(1, 1, (10,1))
test_x2_label0 = np.random.normal(1, 1, (10,1))
test_x1_label1 = np.random.normal(5, 1, (10,1))
test_x2_label1 = np.random.normal(4, 1, (10,1))
test_x1_label2 = np.random.normal(8, 1, (10,1))
test_x2_label2 = np.random.normal(0, 1, (10,1))

test_xs_label0 = np.hstack((test_x1_label0, test_x2_label0))
test_xs_label1 = np.hstack((test_x1_label1, test_x2_label1))
test_xs_label2 = np.hstack((test_x1_label2, test_x2_label2))

test_xs = np.vstack((test_xs_label0, test_xs_label1, test_xs_label2))
print('test samples shape: ', str(test_xs.shape))

test_labels = np.matrix([[1., 0., 0.]]*10 + 
                        [[0., 1., 0.]]*10 +
                        [[0., 0., 1.]]*10)
print('test labels shape: ', str(test_
    labels.shape))



# define the placeholders variables, model and cost function
train_size, num_features = xs.shape

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_labels])

W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y_model = tf.nn.softmax(tf.matmul(X, W)+b)

cost = -tf.reduce_sum(Y * tf.log(y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# train the softmax classification model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs*train_size // batch_size):
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset: (offset+batch_size), :]
        batch_labels = labels[offset:(offset+batch_size)]

        err,_ = sess.run([cost, train_op], feed_dict={X:batch_xs, Y:batch_labels})
        if step % 100 == 0:
            print(step, err)


    W_val = sess.run(W)
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    print('accuracy', sess.run(accuracy, feed_dict={X:test_xs, Y:test_labels}))

