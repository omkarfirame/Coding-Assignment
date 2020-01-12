#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:22:55 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd


def mapFeature(x1,x2):
    degree = 6
    output = np.ones(X.shape[0])
    for i in range(1, degree+1):
        for j in range(i+1):
            output = np.hstack((output, np.multiply(np.power(x1,i-j),np.power(x2,j))))
    return (output)

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))


def costFunctionReg(theta,X,y,Lambda):
    cost=0 
    z = np.dot(X,theta)
    hypothesis = sigmoid(z)
    term1 = np.dot(y.T,np.log(hypothesis))
    term2 = np.dot((1-y).T,np.log(1-hypothesis))
    cost = (np.sum(-term1 - term2) / len(X)) + ((Lambda * np.sum(np.square(theta))) / (2 * len(X)))
    error = hypothesis - y
    gradient = (np.dot(X.T,error) / len(X)) + ((Lambda * theta) / len(X))
    return(cost,gradient)

def Gradient(theta,X,y,Lambda):
        
    z = np.dot(X,theta)
    hypothesis = sigmoid(z)
    error = hypothesis - y
    grad_1 = (np.dot(X.T,error) / len(X))[0]
    grad_2 = (np.dot(X.T,error) / len(X))[1:] + ((Lambda * theta[1:]) / len(X))
    grad = np.vstack((grad_1,grad_2))
    return(grad)

def gradientdescent(theta,X,y,alpha,Lambda,iteration):
    for i in np.arange(iteration):
        cost,gradient = costFunctionReg(theta,X,y,Lambda)
        theta = theta - (alpha * gradient)
    return(theta)


#%%
data = pd.read_csv("ex2data2.txt",header = None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]

X = mapFeature(X.iloc[:,0], X.iloc[:,1])
d = X.reshape(28,118)
X = d.T 
y = np.array(y).reshape(118,1)

m,n = X.shape
theta = np.zeros((n,1))
#%%

cost,gradient = costFunctionReg(theta,X,y,1)
optimum_theta = gradientdescent(theta,X,y,0.01,1,400)[:5]