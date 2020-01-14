#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:42:59 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#%%
def displayData(X):
    import matplotlib.image as mpimg
    fig,axis = plt.subplots(10,10,figsize=(8,8))
    for i in np.arange(10):
        for j in np.arange(10):
            axis[i,j].imshow(X[np.random.randint(0,len(X)),:].reshape(20,20,order=("F")),cmap="magma")
            

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def lrcostFunctionReg(theta,X,y,Lambda):
    z = np.dot(X,theta)
    hypothesis = sigmoid(z)
    term1 = np.dot(y.T,np.log(hypothesis))
    term2 = np.dot((1-y).T,np.log(1-hypothesis))
    cost = (np.sum(-term1 - term2) / len(X)) + ((Lambda * np.sum(np.square(theta[1:]))) / (2 * len(X)))
    error = hypothesis - y
    term_1 = np.dot(X.T, error)[0]
    term_2 = np.dot(X.T,error)[1:] + ((Lambda * theta[1:]))
    gradient = np.vstack((term_1,term_2)) / len(X)
    return(cost,gradient)


def gradientdescent(theta,X,y,alpha,Lambda,iteration):
    for iteration in np.arange(iteration):
        cost,gradient = lrcostFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * gradient)
    return(theta)

def OneVsAll(X,y,num_labels,alpha,Lambda,iteration):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    m,n = X.shape
    initial_theta = np.zeros((n,1))
    theta_vector=[]
    for i in np.arange(1,num_labels+1):
        grad = gradientdescent(initial_theta,X,np.where(y==i,1,0),alpha,Lambda,iteration)
        theta_vector.append(grad)
    return(theta_vector)

def predictOneVsAll(X,theta1):
    predict=[]
    for i in np.arange(len(theta1)):
        prediction = np.dot(X,theta1[i])
        predict.append(prediction)  
    return(predict)
#%%

data = loadmat("ex3data1.mat")
X = data["X"]
y = data["y"]
m,n = X.shape
theta = np.zeros((n,1))
#%%
displayData(X)
theta1 = OneVsAll(X,y,10,0.1,3,300)
X = np.hstack((np.ones((X.shape[0],1)),X))
#%%
predict = predictOneVsAll(X,theta1)

#%%
R = np.array(predict).reshape(10,5000).T
R.shape
actual_predictions = np.argmax(R, axis=1) + 1

#%%
for n, i in enumerate(actual_predictions):
    
    if i == 0:
        actual_predictions[n] = 10
#%%
actual_predictions = actual_predictions.reshape(5000,1)
final_accuracy = (np.sum(y==actual_predictions) / len(y) )* 100
print("Final Accuracy : ", final_accuracy, "%")




