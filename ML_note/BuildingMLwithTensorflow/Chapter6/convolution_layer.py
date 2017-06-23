import tensorflow as tf 

def conv_layer(x_in, weights, bias, strides=1):
    x = tf.nn.conv2d(x_in, weights, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, bias)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
   

