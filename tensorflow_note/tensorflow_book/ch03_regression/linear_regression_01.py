'''
Linear regression

Import TensorFlow for the learning algorithm. 
We'll need NumPy to set up the initial data. 
And we'll use matplotlib to visualize our data.
'''

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# There are called hyper-parameters.
learning_rate = 0.01
training_epochs = 100

x_train = np.linspace(-1, 1, 101)
# print(x_train)
# print(type(x_train))
# print(x_train.shape[0])
# print(np.random.randn(100))
y_train = 2 * x_train + np.random.randn(x_train.shape[0])* 0.33


# plot the raw data
# plt.scatter(x_train, y_train)


X = tf.placeholder("float")
Y = tf.placeholder("float")

W = tf.Variable(0.0, name='weights')

def model(X, w):
    return tf.multiply(X, w)


# loss
y_model = model(X, W)
cost = tf.reduce_mean(tf.square(Y-y_model))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
init =  tf.global_variables_initializer()
sess.run(init)
count=0
for epoch in range(training_epochs):
    for (x, y) in zip(x_train, y_train):
        sess.run(train_op, feed_dict={X:x, Y:y})
    print(sess.run([cost, W], feed_dict={X:x, Y:y}))
    count += 1
    print("这是第%d次"%count)

W_val = sess.run(W)
print("W_val: ",W_val)

sess.close()


# Visualize the best fit curve
plt.scatter(x_train, y_train)
y_learned = x_train * W_val
plt.plot(x_train, y_learned, 'r')
plt.show()