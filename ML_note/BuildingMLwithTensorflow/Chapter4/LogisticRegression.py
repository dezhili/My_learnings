import tensorflow as tf

'''
linear regression can be imagined as a continum of increasingly growing values
The other is a domain where the output can have just two different values,based on the x value
The key here is to understand that we will obtain probabilities of an item pertaining to a class
  and not a totally discrete value.

As we are trying to build a generalized linear model, we want to start from a linear function,
and from the dependent variable,obtain a mapping to a probability distribution

Logit function'
sigmoid function is the inverse of the logit function
The logistic function

Loss function



Multiclass application- softmax regression
In the case of having more than 2 classes to decide from,there are two main approches
    one versus one and one versus all.
one versus all: the output format of the softmax regresion,a generalizationof the logistic regression
    for n classes.
Cost function:
    the cost function of the softmax function is an adapted cross entropy function,which is
    not linear and thus penalizes the big order function differences much more than the very small ones

Data normalization for iterative methods
    for logistic regression we will be using the gradient descent method for minimizing a cost function
    gd method is very sensitive to the form and the distribution of the feature data.
    For the reson, we will be doing some preprocessing in order to get better and faster converging results
    With normalization, we are smoothing the error surface, allowing the iterative gradient descent to reach
        the minimum error faster.

One hot representation of outputs
    This form of encoding simply transforms the numerical integer value of a variable into an array,
        where a list of values is transformed into a list of arrays,each with a length of as mnay elements
        as the maximum value of the list, and the value of each elements is represented by adding a one
        on the index of the value, and leaving the others at zero.
        [1, 3, 2, 4] --> [[0 1 0 0 0]
                          [0 0 0 1 0]
                          [0 0 1 0 0]
                          [0 0 0 0 1]]

'''

'''
Example 1- univariate logistic regression
    tf.one_hot()
    tf.nn.log_softmax(logits, dim=-1, name=None)
        logits: A tensor must be float32,float64 2D with shape [batch_size, num_classes]

'''

'''
Example 2- Univariate logistic regression with skflow
In the example, we will also see how skflow automatically generates a detailed and very 
organized graph for the regression model, just setting a log director as a parameter. 
'''


