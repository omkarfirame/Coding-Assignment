#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 15:49:38 2020

@author: in_omkar.firame
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Data

data = pd.read_csv("ex1data1.txt")
X = data.iloc[:,0]
y = data.iloc[:,1]

#y = [4.2,4.8,6.6,7.1]
ones = pd.DataFrame(np.ones([len(y),1]))
#X = [4,5,6,7]

X = pd.DataFrame(X)
new_X = pd.concat([ones,X],axis=1)
theta = [0,0]
theta = pd.DataFrame(theta)
alpha =0.01

#%% Plotting
def plotData(X,y):
    a=plt.scatter(X,y,s=25)
    b=plt.title("Scatter Plot of training data")
    c=plt.xlabel("Population of city in 10,000s")
    d=plt.ylabel("Profit in $10,000s")
    return(a,b,c,d)
    
#%%
def computeCost(theta,X,y):
    fun_J =[]
    for m in np.arange(len(X)):
        J = (np.dot(np.transpose(theta),X.iloc[m,:]) - y[m])**2
        fun_J.append(J)
    cost_fun = 0.5 * np.sum(fun_J) / len(X)      
    return(print("Cost Function : ",cost_fun))
#%%
def gradientDescent(theta,X,y):
    theta_new=[]
    for i in np.arange(len(X)):
        deriv_term=[]
        for j in np.arange(X.shape[1]):
            derivative_J = (np.dot(np.transpose(theta),X.iloc[i,:]) - y[i])[0] * X.iloc[i,j]
            deriv_term.append(derivative_J)
        n_deriv = np.array(deriv_term)
        n_deriv = np.matrix(n_deriv)
        theta_new.append(n_deriv)
    
    theta_new_dataframe = pd.DataFrame(list(map(np.ravel, theta_new)))
 
    for i in np.arange(X.shape[1]):    
        theta2 = theta.iloc[i,0] - (alpha * np.sum(theta_new_dataframe.iloc[i,:])) / len(X)
        theta.iloc[i,0] = theta2
    return(theta)

#%%
for m in np.arange(1500):
    gradientDescent(theta,new_X,y)
computeCost(theta,new_X,y)
print("theta : ",theta)
