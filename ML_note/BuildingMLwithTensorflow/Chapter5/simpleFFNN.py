import tensorflow as tf 

'''
FeedForward Network
    Preliminary concepts of neural networks
    Neural network projects on non-linear synthetic function regression
    Projects on predicting car fuel effiency with nonlinear regression
    Learning to classify wines and multiclass classfication
'''

'''
Preliminary concepts
    Artificial neurons --the perceptron  is one the simplest ways of implementing an neuron
    
    Perceptron algorithm
        1. Initialize the weights with a random distribution(normally low values)
        2. Select an input vector and present it to the network
        3. Compute the output y' of the network for the input vector specified and the values of the weights
        4. The function for a perceptron is f(x)=1 if w*x+b>0 else 0
        5. If y' != y modify all connections wi by adding the changes delta w = yxi
        6. Return to step 2

    Neural network layers
    Neural Network activation functions
        In order to represent nonlinear models,a number of different nonlinear functions can be
        used in the activation function. This implies changes in the way the neurons will react 
        to changes in the input variables. 

    Gradient and the back propagation algorithm
        When we described the learning phase of the perceptron, we described a stage in which the
        weights were adjusted proportionally according to the 'responsibility'of a weight in the 
        final error
        In this complex network of neurons, the responsibility of the error will be distributed among
        all the functions applied to the data in the whole architecture. 
        What we need to know to be able to minimize this error, as the Optimization field has studied,
        is the gradient of the loss function.
        Given that the data goes through many weights and transfer functions, the resulting compound
        functions's gradient will have to be solved by the chain rule. 

    Neural networks problem choice - Classification vs Regression
'''


'''
Tensorflow activation functions:
Tensorflow loss optimization methods
Skleaarn preprocessing utilities:
    preprocessing.StandardScaler()
    StandardScaler.fit_transform():Simply fit the data to the required form. 
    cross_validation.train_test_split()
'''

'''
First project - Non linear synthetic function regression
    In this first example, we will model a simple,noisy quadratic function,and will try to 
    regress it by means of a single hidden layer network and sees how close we can be predicting
    values taken from a test population. 
'''