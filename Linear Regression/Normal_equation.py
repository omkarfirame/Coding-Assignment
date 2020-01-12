#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:57:06 2020

@author: in_omkar.firame
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ex1data2.txt",header = None)
y = data.iloc[:,2]
x = data.drop(data.columns[2], axis=1)
X = pd.DataFrame(x)

ones = pd.DataFrame(np.ones([len(y),1]))
X = pd.concat([ones,X],axis = 1)    

def normalEqn(X,y):
    xT = np.transpose(X)
    xTx = np.dot(xT,X)
    inverse_xTx = np.linalg.inv(xTx)
    inverse_xt_prod = np.dot(inverse_xTx,xT)
    theta = np.dot(inverse_xt_prod,y)
    return(print("theta : ",theta))

normalEqn(X,y)