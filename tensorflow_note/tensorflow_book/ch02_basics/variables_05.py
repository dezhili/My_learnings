# Using variables

import tensorflow as tf 
sess = tf.InteractiveSession()

raw_data = [1., 2., 8., -1., 0., 5.5, 6., 13]

#Create a boolean variable called spike to detect a sudden increase in the values.
spike = tf.Variable(False)

#All variables must be initialized. 
init = tf.global_variables_initializer()
sess.run(init)
# spike.initializer.run()

for i in range(1, len(raw_data)):
    if raw_data[i] - raw_data[i-1] >5:
        updater = tf.assign(spike, tf.constant(True))
        updater.eval()
    else:
        tf.assign(spike, False).eval()
    print("Spike", spike.eval())

sess.close()