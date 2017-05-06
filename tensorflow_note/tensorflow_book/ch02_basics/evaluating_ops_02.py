import tensorflow as tf 

x = tf.constant([[1, 2]])
neg_x = tf.negative(x)
print(neg_x)                    #Tensor("Neg:0", shape=(1, 2), dtype=int32)
print(type(neg_x))              #<class 'tensorflow.python.framework.ops.Tensor'>

with tf.Session() as sess:
    result = sess.run(neg_x)
    print(result)                   #[[-1 -2]]