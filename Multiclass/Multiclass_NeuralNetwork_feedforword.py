#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:25:22 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("ex3data1.mat")
theta = loadmat("ex3weights.mat")

X = data["X"]
y = data["y"]
theta1 = theta["Theta1"]
theta2 = theta["Theta2"]

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


def displayData(X):
    import matplotlib.image as mpimg
    fig,axis = plt.subplots(10,10,figsize=(8,8))
    for i in np.arange(10):
        for j in np.arange(10):
            axis[i,j].imshow(X[np.random.randint(0,len(X)),:].reshape(20,20,order=("F")),cmap="magma")   
            
            
def predict(theta1,theta2,X):
    a1 = X
    a2 = sigmoid(np.dot(theta1,X.T))
    a2 = a2.T
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = sigmoid(np.dot(theta2,a2.T))
    predict = a3
    R = np.array(predict).reshape(10,5000).T
    actual_predictions = np.argmax(R, axis=1) + 1
    return(actual_predictions)

displayData(X)
X = np.hstack((np.ones((X.shape[0],1)),X))
X.shape

prediction = predict(theta1,theta2,X)
delta = y.reshape(5000) - prediction
final_accuracy = (delta[delta==0].size / y.size) * 100
print("Final Accuracy : ", final_accuracy, "%")