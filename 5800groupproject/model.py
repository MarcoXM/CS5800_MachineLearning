


import numpy as np 
import math 

def tanh(z):
    """The tanh function."""
    return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))

def softmax(z): # This is good version for 2d input
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def tanh_prime(z):
    """Derivative of the tanh function."""
    return 1-z**2

class NeuralNet(object):
    def __init__(self,size):
        self.weight_dicts = {'bias_{}'.format(k): self._b_initialize(size[k-1],size[k]) for k in range(1,len(size))}  
        self.bias_dicts = {'weights_{}'.format(k): self._w_initialize(size[k-1],size[k]) for k in range(1,len(size))}
        ## Cool we have matrix and bias based on our input size
        self.output = [0] * len(size) # output input is also out put
        self.delta = [0] * (len(size)-1) # How many matrix we have means how many delta we have
        self.num_matrix = len(size) - 1
        self.dimension = size


    def _w_initialize(self,num_in,num_out):
        bound = 1.0 / (56000)
        b = np.random.uniform(-bound, bound, (num_in, num_out))
        return b

    def _b_initialize(self,num_in,num_out):
        bound = 1.0 / (56000)
        b = np.random.uniform(-bound, bound, (1, num_out))
        return b

    def forward(self,X):
        self.output[0] = X
        for n,(w,b) in enumerate(zip(self.weight_dicts.values(),self.bias_dicts.values())): # 0,1
            n += 1 # First time would be 1 
            if n == self.num_matrix: #last layer ! is 2 就是说到了最后一层就softmax
                X = softmax(np.dot(X,w) + b) # when it comes to the last layers 
            else:
                X = tanh(np.dot(X,w) + b) # tanh activation
            self.output[n] = X
        return X

    def loss(self,output,target):
        loss = 0.0
        # target is a array , and output is a matrix.
        self.bs = output.shape[0]
        log_likelihood = -np.log(output[range(self.bs),target])
        loss = np.sum(log_likelihood) / self.bs
        return loss


    def backprop(self,target,learning_rate):

        self.lr = learning_rate
        deltaw = [np.zeros(w.shape) for w in self.weights_dicts.values()] # record the gradient
        deltab = [np.zeros(b.shape) for b in self.bias_dicts.values()]
        y_matrix = np.eye(self.dimension[-1])[target] # One Hot

        
        for i in range(self.num_matrix)[::-1]: # getting the delta term
            if i == self.num_matrix -1:
                self.delta[i] = self.output[i+1] - y_matrix
            else:
                self.delta[i] = self.delta[i+1].dot(self.weight_dicts['weights_{}'.format(i+2)].T)*tanh_prime(self.output[i+1])
        
        for j in range(self.num_matrix): # get delta ~ 0,
            #print(deltaw[j].shape)
            deltaw[j] = np.dot(self.output[j].T,self.delta[j])
            deltab[j] = np.sum(self.delta[j],axis=0)
        
        for k in range(self.num_matrix): # update by gradient decent.
            #print(deltab[k].shape)
            self.weight_dicts['weights_{}'.format(k+1)] -= self.lr * deltaw[k]/self.bs # update hidden-to-output weights with gradient descent step
            self.bias_dicts['bias_{}'.format(k+1)] -= self.lr * deltab[k]/self.bs
    
        return self.weight_dicts,self.bias_dicts


if __name__ == '__main__':
    nn = NeuralNet([784,784,256,10])
    print(nn) 
    print('Initialized ! ! !')   
    

