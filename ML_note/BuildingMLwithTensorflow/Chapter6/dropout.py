import tensorflow as tf 
x = [1.0, .5, .75, .25, .2, .8, .4, .6]
dropout = tf.nn.dropout(x, 0.5)
with tf.Session() as sess:
    print(sess.run(dropout))
    