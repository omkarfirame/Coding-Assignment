#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 08:40:12 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex2data1.txt",header = None)
row_len,colomn_len = data.shape
X = data.values[:,:colomn_len-1]
y = data.values[:,colomn_len-1:colomn_len]
add_column = np.ones((row_len,1))
X = np.append(add_column,X,axis=1)
theta = [-24, 0.2, 0.2]
theta = np.array(theta)[np.newaxis] 

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def gradient(theta,X,y):
    theta = np.transpose(theta)    
    z = np.dot(X,theta)
    hypothesis = sigmoid(z)
    error = hypothesis - y
    gradient = np.dot(X.T,error) / len(X)
    theta = gradient
    return(gradient)

def costFunc(theta,X,y):
    cost=0
    theta = np.transpose(theta)    
    z = np.dot(X,theta)
    hypothesis = sigmoid(z)
    term1 = np.dot(y.T,np.log(hypothesis))
    term2 = np.dot((1-y).T,np.log(1-hypothesis))
    cost = np.sum(-term1 - term2) / len(X)
    error = hypothesis - y
    gradient = np.dot(X.T,error) / len(X)
    return(cost,gradient)

gradient(theta,X,y)
costFunc(theta,X,y)    
initial_theta = np.zeros(colomn_len, dtype=int)
import scipy.optimize as opt
result = opt.fmin_tnc(func=costFunc, x0=initial_theta.flatten(), args=(X, y.flatten()))[0]
print("theta",result)