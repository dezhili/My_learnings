import pandas as pd 
import tensorflow as tf 
import tensorflow.contrib.learn as skflow
import numpy as np 
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 

boston = pd.read_csv('./boston.csv', header=0)
# print(boston.head())
print(boston.describe())
# boston.CRIM.plot(kind='hist')
# plt.show()

f, ax1 = plt.subplots()
plt.figure() # Create a new figure

y = boston['MEDV']

for i in range (1,8):
    number = 420 + i
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(boston)[i])
    ax1.scatter(boston[boston.columns[i]],y) #Plot a scatter draw of the  datapoints
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# Model architecture

X = tf.placeholder("float", name="X") # create symbolic variables
Y = tf.placeholder("float", name = "Y")

with tf.name_scope("Model"):

    w = tf.Variable(tf.random_normal([2], stddev=0.01), name="b0") # create a shared variable
    b = tf.Variable(tf.random_normal([2], stddev=0.01), name="b1") # create a shared variable
    
    def model(X, w, b):
        return tf.multiply(X, w) + b # We just define the line as X*w + b0  

    y_model = model(X, w, b)

with tf.name_scope("CostFunction"):
    cost = tf.reduce_mean(tf.pow(Y-y_model, 2)) # use sqr error for cost function

train_op = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()
init = tf.global_variables_initializer()
tf.train.write_graph(sess.graph, './multivariate_LR','graph.pbtxt')
cost_op = tf.summary.scalar("loss", cost)
merged = tf.summary.merge_all()
sess.run(init)
writer = tf.summary.FileWriter('./multivariate_LR', sess.graph)

xvalues = boston[[boston.columns[2], boston.columns[4]]].values.astype(float)
yvalues = boston[boston.columns[12]].values.astype(float)
b0temp=sess.run(b)
b1temp=sess.run(w)


for a in range (1,50):
    cost1=0.0
    for i, j in zip(xvalues, yvalues):   
        sess.run(train_op, feed_dict={X: i, Y: j}) 
        cost1+=sess.run(cost, feed_dict={X: i, Y: i})/506.00
        #writer.add_summary(summary_str, i) 
    xvalues, yvalues = shuffle (xvalues, yvalues)
    print (cost1)
    b0temp=sess.run(b)
    b1temp=sess.run(w)
    print (b0temp)
    print (b1temp)

