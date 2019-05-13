#!/usr/bin/env python
# coding: utf-8

#Import packages
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
 
#Read Files
dataFile = 'hw1data.mat'
data = scio.loadmat(dataFile)
data['__header__']

#train and test
train = data['trainData']
test = data['testData']

#getting the different class
y = train[:,0]
labels = list(set(y))

train_1 = train[train[:,0] == labels[0]]
train_2 = train[train[:,0] == labels[1]]
train_3 = train[train[:,0] == labels[2]]
train_4 = train[train[:,0] == labels[3]]

#Class 1
plt.figure(figsize=(8,6), dpi = 300)
plt.hist(train_1[:,1])
plt.show()

#Class 2
plt.figure(figsize=(8,6), dpi = 300)
plt.hist(train_2[:,1])
plt.show()

#Class 3
plt.figure(figsize=(8,6), dpi = 300)
plt.hist(train_3[:,1])
plt.show()

#Class 4
plt.figure(figsize=(8,6), dpi = 300)
plt.hist(train_4[:,1])
plt.show()


#Q2
def learnMean(Data,classNum):
    data = Data[Data[:,0] == classNum]
    sum_ = data[:,1].sum()
    divisor = len(data)
    return sum_/divisor
        
    

#Q3
def labelML(amountAlc, meanVector):
    label = ['M','Y','A','S']
    list_pro = []
    for n in range(len(meanVector)):
        pro = 1/((2*np.pi)**0.5 *2)* np.e**((amountAlc - meanVector[n])**2/(-8))                                    
        list_pro.append(pro)
    
    return label[np.argmax(list_pro)]
    


#Q4 
def labelMP(amountAlc, meansVector, priorVector):
    label = ['M','Y','A','S']
    nprovector = np.append(priorVector, 1-priorVector.sum())
    list_pro = []
    for n in range(len(meansVector)):
        pro = 1/((2*np.pi)**0.5 *2)* np.e**((amountAlc - meansVector[n])**2/(-8)) * nprovector[n]
                                            
        list_pro.append(pro)
    
    return label[np.argmax(list_pro)]
    


#Q5
def evaluateML(testData, meanVector):
    pred = np.zeros(testData.shape[0])
    labels = testData[:,0]
    for instance in range(testData.shape[0]):
        list_pro = []
        for n in range(len(meanVector)):
            pro = 1/((2*np.pi)**0.5 *2)* np.e**((testData[:,1][instance] - meanVector[n])**2/(-8))
            list_pro.append(pro)
        pred[instance] = np.argmax(list_pro) + 1
        
    return list(pred - labels).count(0)/testData.shape[0]
        



#Q6
def evaluateMP(testData, meanVector, priorVector):
    pred = np.zeros(testData.shape[0])
    labels = testData[:,0]
    nprovector = np.append(priorVector, 1-priorVector.sum())
    for instance in range(testData.shape[0]):
        list_pro = []
        for n in range(len(meanVector)):
            pro = 1/((2*np.pi)**0.5 *2)* np.e**((testData[:,1][instance] - meanVector[n])**2/(-8)) * nprovector[n]
            list_pro.append(pro)
        pred[instance] = np.argmax(list_pro) + 1
    return list(pred - labels).count(0)/testData.shape[0]
    



'''
Q7
Report the percent of correctly labeled test data for max likelihood and max posterior
separately when means are learned:
'''


# In[22]:

# for convinence
def get_firstdata(data,row):
    train = data[:row,:]
    train_1 = train[train[:,0] == labels[0]]
    train_2 = train[train[:,0] == labels[1]]
    train_3 = train[train[:,0] == labels[2]]
    train_4 = train[train[:,0] == labels[3]]
    mvector = np.array([learnMean(train,1),learnMean(train,2),learnMean(train,3),learnMean(train,4)])
    return mvector


#on the first 6 data points in the training set,
mvectorf6 = get_firstdata(train,6)
print('max likelihood is', evaluateML(test,mvectorf6))
print('max posterior is', evaluateMP(test,mvectorf6,np.array([.3,.4,.2])))

# on the first 18 data points
mvectorf18 = get_firstdata(train,18)
print('max likelihood is %.2f '%evaluateML(test,mvectorf18))
print('max posterior is', evaluateMP(test,mvectorf18,np.array([.3,.4,.2])))
# on the first 54 data points,
mvectorf54 = get_firstdata(train,54)
print('max likelihood is %.2f '%evaluateML(test,mvectorf54))
print('max posterior is', evaluateMP(test,mvectorf54,np.array([.3,.4,.2])))
# on and the first 162 data points
mvectorf162 = get_firstdata(train,162)
print('max likelihood is %.2f '%evaluateML(test,mvectorf162))
print('max posterior is', evaluateMP(test,mvectorf162,np.array([.3,.4,.2])))

# Aswers

#max likelihood is 0.5433333333333333
#max posterior is 0.5591666666666667

#max likelihood is 0.54 
#max posterior is 0.57

#max likelihood is 0.54 
#max posterior is 0.58

#max likelihood is 0.53 
#max posterior is 0.57

# Q8 
d8 = scio.loadmat('hw1dataQ8.mat')
train8 = d8['trainData']
test8 = d8['testData']


def labelMP2(amountDrinks, meansMatrix, priorVector):
    label = ['M','Y','A','S']
    nprovector = np.append(priorVector, 1-priorVector.sum())
    list_pro_soda = []
    for n in range(len(label)): # this is for soda
        pro = 1/((2*np.pi)**0.5 *2)* np.e**((amountDrinks - meansMatrix[1][n])**2/(-8)) * nprovector[n]                                  
        list_pro_soda.append(pro)
    
    list_pro_alchol = []
    for n in range(len(label)): # this is for soda
        pro = 1/((2*np.pi)**0.5 *2)* np.e**((amountDrinks - meansMatrix[0][n])**2/(-8)) * nprovector[n]                                  
        list_pro_alchol.append(pro)
    
    answer = np.array(list_pro_soda) * np.array(list_pro_alchol)
    return label[np.argmax(answer)]
    




