# Loading variables
import tensorflow as tf 
sess = tf.InteractiveSession()

spikes = tf.Variable([False]*8, name='spikes')

saver = tf.train.Saver()
try:
    saver.restore(sess,'./saver_06/spikes.ckpt')
    print(spikes.eval())                # [False False  True False False  True False  True]
except:
    print("file not found")

sess.close()
