'''
First project - Non linear synthetic function regression
'''
import tensorflow as tf 
import numpy as np 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 

trainsamples = 200
testsamples = 60

# Here we will represent the model, a simple input, a hidden layer of sigmoid activation
def model(X, hidden_weights1, hidden_bias1, ow):
    hidden_layer = tf.nn.sigmoid(tf.matmul(X, hidden_weights1)+hidden_bias1)
    return tf.matmul(hidden_layer, ow)
dsX = np.linspace(-1, 1, trainsamples+testsamples).transpose()
dsY = 0.4 * pow(dsX, 2) + 2 * dsX + np.random.randn(*dsX.shape)*0.22 + 0.8

plt.figure()
plt.title('Original data')
plt.scatter(dsX, dsY)
# plt.show()

X = tf.placeholder('float')
Y = tf.placeholder('float')

hw1 = tf.Variable(tf.random_normal([1, 10], stddev=0.01))
ow = tf.Variable(tf.random_normal([10, 1], stddev=0.01))
b = tf.Variable(tf.random_normal([10], stddev=0.01))

model_y = model(X, hw1, b, ow)

cost = tf.pow(model_y-Y, 2) / 2
train_op = tf.train.AdamOptimizer(0.0001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, 10):
        trainX, trainY = dsX[0:trainsamples], dsY[0:trainsamples]
        for x1, y1 in zip(trainX, trainY):
            sess.run(train_op, feed_dict={X:[[x1]], Y:y1})
        testX, testY = dsX[trainsamples: trainsamples+testsamples], dsY[trainsamples:trainsamples+testsamples]

        cost1=0
        for x1, y1 in zip(testX, testY):
            cost1 += sess.run(cost, feed_dict={X:[[x1]], Y:y1}) 
        cost1 /= testsamples
        print('Average cost for epoch '+str(i)+":"+str(cost1))
        dsX, dsY = shuffle(dsX, dsY)
