#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:20:28 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Feature Scaling - Function

def featureNormalize(x):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(x)
    return(pd.DataFrame(X))

#%% Gradient Descent - function
    
def gradientDescentMulti(theta,x,y,alpha,iteration):
    Cost_fun_vec=[]
    for m in np.arange(iteration):
        hypothesis =[]
        for i in np.arange(len(X)): 
            h = np.dot(theta,X.iloc[i,:])
            hypothesis.append(h)
        derivative_vec=[]
        
        for i in np.arange(len(X)):
            deriv_term=[]
            for j in np.arange(X.shape[1]):
                deriv = (hypothesis[i] - y[i]) * X.iloc[i,j]
                deriv_term.append(deriv)
            derivative_vec.append(deriv_term)
        
        dataframe_dvec = pd.DataFrame(list(map(np.ravel, derivative_vec)))
        for i in np.arange(X.shape[1]):
            theta_J = theta[i] - ((alpha * np.sum(dataframe_dvec.iloc[:,i])) / len(X))
            theta[i] = theta_J        
      # Cost Function
        cost_fun = 0.5 * np.dot(np.transpose(np.dot(X,theta) - y),(np.dot(X,theta) - y))
        Cost_fun_vec.append(cost_fun)
    return(print("Theta : ",theta),Cost_fun_vec,print("Cost : ",cost_fun))

#%% cost function
    
#def computeCostMulti(theta,X,y):
#    cost_fun = 0.5 * np.dot(np.transpose(np.dot(X,theta) - y),(np.dot(X,theta) - y))
#    return(print("Cost Function : ",cost_fun))

#%% Data

data = pd.read_csv("ex1data2.txt")
y = data.iloc[:,2]
x = data.drop(data.columns[2], axis=1)
X = pd.DataFrame(x)

#%% Input Parameters

alpha =0.03
iteration = 400
theta = [0,0,0]
T_theta = np.transpose(theta)

#%% Adding Colomn with all rows are 1

ones = pd.DataFrame(np.ones([len(y),1]))
X = pd.concat([ones,featureNormalize(X)],axis = 1)    

#%% Gradient Descent 

theta,cost_vector,final_cost=gradientDescentMulti(theta,X,y,alpha,iteration)


#%% Number of iteration and cost graph
plt.plot(list(np.arange(1,iteration+1)),cost_vector)
plt.xlabel("number of iteration")
plt.ylabel("Cost J")


