'''
Recurrent Neural Networks and LSTM
--recurrent neural networks predictions depends on the current input vector and also the values
  of previous ones.
'''

'''
Recurrent neural networks
Traditional nn don't include any mechanism translating previous seen eles to the current state
Rnn can be defined as s sequential model of nn,which have the property of reusing infs already given.


Cell  -- a simplified diagram of a RNN basic element
the input(Xt),an state
an output(ht) 
cells have not an independent state,so it stores also state information

Once we define the dynamics of the cell,the next objective would be to investigate the contents
of what makes or defines an RNN cell.
In the most common case of standard RNN,there is simply a neural network layer,which takes 
the input,and the previous state as inputs, applies the tanh operation, and outputs the new state h(t+1)

'''

'''
Exploding and vanishing gradients

bur further experimentation showed that for complex knowledge,the sequence distance makes difficult
to relate some contexts. This also brings the associated issue of exploding and vanishing gradients.

Those phenomena receive the name of vanishing and exploding gradients.
This is one of the reasons for which LSTM architecture was created.
'''

# ---------------------------------------------------------------------------------------

'''
LSTM neural networks
-- a specific RNN architecture whose special architecture allows them to represent long term dependencies.
   they are designed to remember information patterns and information over long periods of time


The gate operation - a fundamental component
the main operational block of the LSTM : the gate operation

The binary vector
In order to implement this function, we take a multivariate control vector,which is connected 
with a neural network layer with a sigmoid activation function.
Applying the control vector and passing through the sigmoid function,we will get a binarylike vector

after defining that binary vector,we will multiply the input function with the vector so we will
filter it,letting only parts of the information to get through.

General LSTM cell structure
It mainly consist of three of the mentioned gate operations,to protect and control the cell state.
This operation will allow both discard(Hopefully not important)low state data, and incorporate
(Hopefully import)new data to the state.

As the inputs we have:
The cell state,which will store long term information,because it carries on the optimized weights
from the starting coming from the origin of the cell training
The short term state,h(t),which will be used directly combined with the current input on each
iteration, and so it will have a much bigger influence from the latest values of the inputs
'''

'''
Operation steps:

Part 1 - set values to forget(input gate)
In this section,we will take the values coming from the short term,combined with the input 
itself,and this values will set the values for a binary function,represented by a multivariable
sigmoid.
Depending on the input and short term memory values,the sigmoid output will allow or restrict
some of the previous Knowledge or weights contained on the cell state.

Part 2 - set values to keep,change state
So in this stage, we will determine how much of the new and semi-new information will be incorporated
in the new cell state.
In order to normalize the new and short term information,we pass the new and short term info
via a neural network with tanh activation,this will allow to feed the new information in a normalized
(-1, 1)range.

Part 3 - output filtered cell state
It will also use the new and previous short term state to allow new information to pass,
but the input will be the long term status,dot multiplied by a tanh function,again to normalize
the input to a (-1, 1) range.

'''

# --------------------------------------------------------------------------------------

'''
Tensorflow LSTM useful classes and methods
class tf.nn.rnn_cell.BasicLSTMCell
class MultiRNNCell(RNNCell)

'''

