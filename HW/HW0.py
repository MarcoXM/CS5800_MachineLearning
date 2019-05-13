#!/usr/bin/env python
# coding: utf-8

#Q1
def threshClassify(input_listdata, xthresh):
    answer = [1 if height > xthresh else 0 for height in input_listdata]
    return answer


#Q2
def findAccuracy(classifierOutput, trueLabels):
    answer = [x-y for x,y in zip(classifierOutput,trueLabels)].count(0)/len(classifierOutput)
    return answer


print('Hello World')


#Q3
def getTraining(fulldata):
    x,y = fulldata
    length = int(len(x)/3)
    return [x[:length],
            y[:length]]



