import tensorflow as tf 

a = tf.constant([1.0,2.3],name='a')
b = tf.constant([2.0,3.0],name='b')


print(tf.get_default_graph())
print(a.graph)