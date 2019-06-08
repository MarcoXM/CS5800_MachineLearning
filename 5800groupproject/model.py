import numpy as np


# Activation Function
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    z = Z[:]
    z[z<0] = 0
    return z

def sigmoid_backward(sig):
    return (sig * (1 - sig))

def relu_backward(Z):
    dZ = Z.copy()
    dZ[dZ > 0] = 1
    return dZ;

def step(Z):
    Z[Z>=0] = 1
    Z[Z<0] = 0
    return Z

def tanh(Z):
    return (np.exp(2*Z) -1)/(np.exp(2*Z) +1)

def tanh_backward(tan):
    return (1 - tan**2)

def bina(Z):
    dZ = Z.copy()
    dZ[dZ > 0.5] = 1
    dZ[dZ <= 0.5] = 0
    return dZ

##Modeling

def weight_initial(m,n): # xiviar initial
    weights= np.random.randn(m, n) / np.sqrt(m)
    bias = np.random.randn(1, n) / np.sqrt(m)
    return weights,bias

class MN(object):
    def __init__(self,n_feature, n_input,n_hidden, n_output,activation = 'sigmoid'): # 3 matrix
        np.random.seed(224)
        self.input_weights,self.input_bias = weight_initial(n_feature, n_input) #m x l1
        self.hidden_weights,self.hidden_bias = weight_initial(n_input,n_hidden) # l1 x l2
        self.output_weights,self.output_bias = weight_initial(n_hidden, n_output) # l2 x l3
        self.activation = activation
        
    def forward(self,inputx,batch_size):
        self.batch_size = batch_size
        if self.activation == 'sigmoid':
            self.out1 = sigmoid(np.dot(inputx,self.input_weights) + self.input_bias)
            self.out2 = sigmoid(np.dot(self.out1,self.hidden_weights) + self.hidden_bias)
            self.out3 = sigmoid(np.dot(self.out2,self.output_weights) + self.output_bias)
            #print(self.out3)
        
        if self.activation == 'step':
            self.out1 = step(np.dot(inputx,self.input_weights) + self.input_bias)
            self.out2 = step(np.dot(self.out1,self.hidden_weights) + self.hidden_bias)
            self.out3 = step(np.dot(self.out2,self.output_weights) + self.output_bias)
            
        if self.activation == 'tanh':
            self.out1 = tanh(np.dot(inputx,self.input_weights) + self.input_bias)
            self.out2 = tanh(np.dot(self.out1,self.hidden_weights) + self.hidden_bias)
            self.out3 = tanh(np.dot(self.out2,self.output_weights) + self.output_bias)
            
        if self.activation == 'relu':
            self.out1 = relu(np.dot(inputx,self.input_weights) + self.input_bias)
            self.out2 = relu(np.dot(self.out1,self.hidden_weights) + self.hidden_bias)
            self.out3 = relu(np.dot(self.out2,self.output_weights) + self.output_bias)

    def loss(self,X_train,y_train):
        self.forward(X_train)
        self.error3 = y_train - self.out3
        return np.mean(self.error3**2)

    def backward(self,inputx, y_train,lr):
        self.lr = lr

            if self.activation == 'sigmoid':
            self.error3term = self.out3 * (1 - self.out3) * self.error3
            self.error2 = np.dot(self.output_weights,self.error3term)
            self.error2term = self.out2 * (1 - self.out2) * self.error2
            self.error1 = np.dot(self.hidden_weights,self.error2term)
            self.error1term = self.out1 * (1 - self.out1) * self.error1

            ## update !
            self.output_weights += self.lr * error3term * self.out2[:,None] / self.batch_size
            self.output_bias += np.mean(self.lr * error3 * sigmoid_backward(self.out3))
            
            self.hidden_weights += self.lr * self.error2term * self.out1[:,None] / self.batch_size
            self.hidden_bias += np.mean(self.lr * self.error2 * (sigmoid_backward(self.out2)),axis =0)
            
            self.input_weights += self.lr * self.error1term * inputx[:,None] / self.batch_size
            self.input_bias += np.mean(self.lr * self.error1 * sigmoid_backward(self.out1),axis = 0)
    

            
        if self.activation == 'tanh':
            self.error3term = (1 - self.out3**2) * self.error3
            self.error2 = np.dot(self.output_weights,self.error3term)
            self.error2term = (1 - self.out2**2) * self.error2
            self.error1 = np.dot(self.hidden_weights,self.error2term)
            self.error1term = (1 - self.out1**2) * self.error1

            
            self.output_weights += self.lr * error3term * self.out2[:,None] / self.batch_size
            self.output_bias += np.sum(self.lr * error3 * tanh_backward(self.out3))
            
            self.hidden_weights += self.lr * self.error2term * self.out1[:,None] / self.batch_size
            self.hidden_bias += np.sum(self.lr * self.error2 * (tanh_backward(self.out2)),axis =0)
            
            self.input_weights += self.lr * self.error1term * inputx[:,None] / self.batch_size
            self.input_bias += np.sum(self.lr * self.error1 * tanh_backward(self.out1),axis = 0)
            
        
        if self.activation == 'relu':
            self.error3term = relu_backward(self.out3) * self.error3
            self.error2 = np.dot(self.output_weights,self.error3term)
            self.error2term = relu_backward(self.out2) * self.error2
            self.error1 = np.dot(self.hidden_weights,self.error2term)
            self.error1term = relu_backward(self.out1) * self.error1

            self.output_weights += self.lr * error3term * self.out2[:,None] / self.batch_size
            self.output_bias += np.mean(self.lr * error3 * relu_backward(self.out3))
            
            self.hidden_weights += self.lr * self.error2term * self.out1[:,None] / self.batch_size
            self.hidden_bias += np.mean(self.lr * self.error2 * (relu_backward(self.out2)),axis =0)
            
            self.input_weights += self.lr * self.error1term * inputx[:,None] / self.batch_size
            self.input_bias += np.mean(self.lr * self.error)

    def train(self,X_train,y_train,learnrate, iteration = 100):
            self.losses = []
        for i in range(iteration):
            self.loss(X_train,y_train)
            self.backward(X_train,y_train,learnrate)
            
            if i % 100 ==0:
                self.losses.append(self.loss(X_train,y_train))
                print('Epoch: ', str(i) +'| ' +str(iteration), '| train loss: %.4f' % self.losses[-1])

    def predict(self,X,y):
        self.forward(X)
        self.prob =self.out3
        self.pre = bina(self.prob)
        fpr,tpr,thresholds = roc_curve(y, self.pre)
        
        print(classification_report(y,self.pre))
        print(confusion_matrix(y,self.pre))
        print("AUC: {}".format(roc_auc_score(y, self.pre)))
        sns.set_style()
        plt.figure(figsize=(8,6),dpi = 300)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
                 
    
