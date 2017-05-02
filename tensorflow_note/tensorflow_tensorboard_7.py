#神经网络的框架结构
import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    #add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs



# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
    
# add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)#隐藏层
# add output layer
prediction = add_layer(l1,10,1,activation_function=None)#输出层



# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
writer = tf.summary.FileWriter('F:/tensorflow/site-packages/logs/', sess.graph)  #把框架loading到文件夹


# important step 
sess.run(tf.global_variables_initializer())


#cmd 到指定文件夹，tensorboard --logdir='logs/' F:
