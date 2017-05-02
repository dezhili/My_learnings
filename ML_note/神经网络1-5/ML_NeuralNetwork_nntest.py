#coding = gbk
import numpy as np

def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0 - np.tanh(x)* np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))
def logistic_derivative(x):
    return logistic(x) * (1-logistic(x))


class NeuralNetwork:
    def __init__(self,layers,activation = 'tanh'):
        """
        :param layers:A list containing the number of units in each layer
        should be at least 2 values
        :param activation:The activation function to be used
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
            
        self.weights = []
        for i in range(1,len(layers)-1):
            self.weights.append((2*np.random.random((layers[i-1]+1,
                                                    layers[i]+1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i]+1,
                                                    layers[i+1]))-1)*0.25)
            
    #epochs :抽样(从X中抽取实例)的方法对神经网络进行更新，神经网络更新循环最多epochs次 
    #见上一讲，有三个终止条件(这边是最简单的，循环次数)
    def fit(self,X,y,learning_rate=0.2,epochs=10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:,0:-1] = X #adding the bias unit to the input layer
        X = temp
        y = np.array(y)
        
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            for l in range(len(self.weights)): #going forward network for each layer
                a.append(self.activation(np.dot(a[l],self.weights[l])))#computing the node value for each layer(O_i)using activation function
            error = y[i] - a[-1] #computing the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])]#For output layer,Err calculation(delta is updated error)
                                                           #for output layer : Errj = (Oj(1-Oj))(Tj-Oj)
            
            #Staring backpropagation
            for l in range(len(a)-2,0,-1):#we need to begin at the second to last layer
                #computing the updated error(i,e,deltas)for each node going from top layer to input layer
                #for hidden layer : Errj = (Oj(1-Oj))((累和)Errk*Wkj)
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                #权重更新:Wij = Wij + (l)(Errj*Oi)
                self.weights[i] += learning_rate * layer.T.dot(delta)
               
        
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a




from NeuralNetwork import NeuralNetwork


nn = NeuralNetwork([2,2,1],'tanh')
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

nn.fit(X, y)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,nn.predict(i))       
             

            
            





