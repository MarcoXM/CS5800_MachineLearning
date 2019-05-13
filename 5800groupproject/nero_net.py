import numpy as np


# Activation Function
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.max(0,Z)

def sigmoid_backward(dx, sig):
    return dx * sig * (1 - sig)

def relu_backward(dx, Z):
    dZ = Z.copy()
    dZ[dZ > 0] = 1
    return dx *dZ;

#def step_backward(Z)    
def step(Z):
    Z[Z>=0] = 1
    Z[Z<0] = 0
    return Z


def tanh(Z):
    return (np.exp(2*Z) -1)/(np.exp(2*Z) +1)

def tanh_backward(dx,tan):
    return dx * (1 - tan**2)
class NN(object):
    def __init__(self, n_input, n_output,actfun = 'sigmoid'):
        np.random.seed(1224)
        self.weights = np.random.randn(n_input,n_output)*0.1
        self.bias = np.random.randn(n_output,1) * 0.1
        self.activation = actfun
        
        
    # Computing the ouput of the neurons
    def forward(self,inputx):
        if self.activation == 'sigmoid':
            return sigmoid(np.dot(inputx,self.weights) + self.bias)
        
        if self.activation == 'step':
            return step(np.dot(inputx,self.weights)+ self.bias)
        
        if self.activation == 'tanh':
            return tanh(np.dot(inputx,self.weights)+ self.bias)
        
        if self.activation == 'relu':
            return relu(np.dot(inputx,self.weights)+ self.bias) 
        
    def loss(self,X_train,y_train):
        self.output = self.forward(X_train)
        #print(self.output.shape)
        self.error = y_train - self.output
        return np.mean(self.error**2)
    
    def backward(self,X_train, y_train,learnrate):
        if self.activation == 'sigmoid':
            self.weights += np.sum(learnrate * self.error * sigmoid_backward(X_train,self.output),axis =0).reshape(-1,1)
            self.bias += np.sum(learnrate * self.error * sigmoid_backward(1,self.output))
  
       # if self.activation == 'step':
           # self.weights += np.sum(learnrate * self.error * sigmoid_backward(X_train,self.output),axis =0).reshape(-1,1)
          #  self.bias += np.sum(learnrate * self.error * sigmoid_backward(1,self.output))

        if self.activation == 'tanh':
            self.weights += np.sum(learnrate * self.error * tanh_backward(X_train,self.output),axis =0).reshape(-1,1)
            self.bias += np.sum(learnrate * self.error * tanh_backward(1,self.output))
            
        if self.activation == 'relu':
            self.weights += np.sum(learnrate * self.error * relu_backward(X_train,self.output),axis =0).reshape(-1,1)
            self.bias += np.sum(learnrate * self.error * relu_backward(1,self.output))
            
    def train(self,X_train,y_train,learnrate, iteration = 100):
        for i in range(iteration):
            self.loss(X_train,y_train)
            self.backward(X_train,y_train,learnrate)
            
    def preidict(self,X,y):
        y_pre = step(self.forward(X))
        acc = list(y-y_pre)
        return acc.count(0)/len(acc)




def weight_initial(m,n):
    weights= np.random.randn(m, n) / np.sqrt(m)
    bias = np.random.randn(m, 1) / np.sqrt(m)
    return weights,bias
        
  
class MN(object):
    def __init__(self,n_feature, n_input,n_hidden, n_output,afList):
        np.random.seed(224)
        self.input_weight,self.input_bias = weight_initial(n_feature, n_input)
        self.hidden_weight,self.hidden_bias = weight_initial(n_input,n_hidden)
        self.output_weight,self.output_bias = weight_initial(n_hidden, n_output)
        self.activations = afList
        
    def forward(self,inputx):
        for i in self.activations:
            if i == 'sigmoid':
                out = sigmoid(np.dot(inputx,self.input_weights) + self.input_bias)
        
            if self.activation == 'step':
                return step(np.dot(inputx,self.weights)+ self.bias)
            
            if self.activation == 'tanh':
                return tanh(np.dot(inputx,self.weights)+ self.bias)
            
            if self.activation == 'relu':
                return relu(np.dot(inputx,self.weights)+ self.bias)