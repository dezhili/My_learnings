import matplotlib.pyplot as plt 
import numpy as np 

trX = np.linspace(-1, 1, 101)
print(trX.shape)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.4 + 0.2
plt.figure()
plt.scatter(trX, trY)
plt.plot(trX, .2+2*trX)
# plt.show()

# Least squares
# Iterative methods - gradient descent
# We start with the initial set of coefficents and then move in the opposite direction
# of maximum change in the function. 
# The final step is to optionally test the changes between iteration and see whether the 
# changes are greater than an epsilon or to check whether the iteration number is reached



'''
Optimizer methods in Tensorflow - the train module
tf.train.Optimizer
-- The Optimizer class allows you to calculate gradients for a loss function and apply
    them to different variables of a model. gradient descent , Adam , Adagrad
1. opt = GradientDescentOptimizer(learning_rate=[learning rate])
2. optimization_op = opt.minimize(cost, var_list=[varaibles_list])
tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step,var_list=None)
    loss: This is a tensor that contains the values to be minimized. 
    global_step: This variable will increment by one after the Optimizer works
    var_list: This contains variables to optimize
Tip: optimize() --> compute_gradients()  apply_gradients() 
'''

