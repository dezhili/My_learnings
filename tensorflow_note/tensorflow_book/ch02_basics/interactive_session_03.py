# Interactive sessions are another way to use a session.
import tensorflow as tf 

sess = tf.InteractiveSession()

x = tf.constant([[1., 2.]])
neg_op = tf.negative(x)

# Since we're using an interactive session, we can just call the eval() method on the op.
result = neg_op.eval()
print(result)                   #[[-1. -2.]]

sess.close()