import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder 

data = loadmat("ex3data1.mat")
theta = loadmat("ex3weights.mat")

X = data["X"]
y = data["y"]
""" X, y : Training set """

#%%
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def displayData(X):
    fig,axis = plt.subplots(10,10,figsize=(8,8))
    for i in np.arange(10):
        for j in np.arange(10):
            axis[i,j].imshow(X[np.random.randint(0,len(X)),:].reshape(20,20,order=("F")),cmap="magma")   
            
def sigmoidGradient(z):
    """
    Compute gradient of sigmoid function
    """
    g = sigmoid(z)
    g_dash = g*(1-g)
    return(g_dash)

def randInitializeWeights(L_in,L_out):
    """
    Initialize the random weights.
    L_in : Length of incoming layer 
    L_out : Length of outgoing layer
    """
    eps = (np.sqrt(6) / (np.sqrt(L_in + L_out)))
    random_initial_weights = np.random.rand(L_out,1+L_in) * (2*eps) - eps
    return(random_initial_weights)
    
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    """ 
    compute the cost and gradient
    nn_params : parameters unrolled into vector
    input_layer_size : input layer size
    hidden_layer_size : hidden layer size
    num_labels : number of class in target variable
    X : array(number of examples, number of features)
    y : array(number of examples, 1)
    Lambda : float
    """
   
    theta1 = nn_params[:((input_layer_size + 1) * (hidden_layer_size))].reshape(hidden_layer_size,input_layer_size + 1)
    theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    m,n = X.shape
    X = np.hstack((np.ones((m,1)),X))
    
    a1 = X
    a2 = sigmoid(np.dot(theta1,a1.T))
    a2 = a2.T
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = sigmoid(np.dot(theta2,a2.T))
    a3 = a3.T
    
    onehotencoder = OneHotEncoder(categories='auto')    
    Y = onehotencoder.fit_transform(y).toarray() 
    cost=[]
    for j in np.arange(num_labels):
        J = np.sum(-Y[:,j] * np.log(a3[:,j]) - (1-Y[:,j])*np.log(1-a3[:,j]))
        cost.append(J)
    Cost = np.sum(cost) / m
    regularized_J = Cost + (Lambda / (2*m)) * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))
    
    """ Backpropogation """
    
    theta1_grad = np.zeros((theta1.shape[0],theta1.shape[1]))
    theta2_grad = np.zeros((theta2.shape[0],theta2.shape[1]))
    for example in np.arange(len(X)):
        a_1 = a1[example,:]
        a_2 = a2[example,:]
        a_3 = a3[example,:]
        delta_3 = a_3 - Y[example,:]    
        z = np.dot(a_1,theta1.T)
        g_dash = np.hstack((1,sigmoidGradient(z)))
        delta_2 = np.dot(theta2.T,delta_3) * g_dash
        
        theta1_grad = theta1_grad + np.dot(delta_2[1:].reshape(len(delta_2)-1,1),a_1.reshape(1,len(a_1))) 
        theta2_grad = theta2_grad + np.dot(delta_3.reshape(len(delta_3),1),a_2.reshape(1,len(a_2)))
    
    theta1_grad = theta1_grad / len(X)
    theta2_grad = theta2_grad / len(X)

    regularized_theta1_grad = theta1_grad[:,1:] +((Lambda * theta1[:,1:])) / len(X)
    regularized_theta2_grad = theta2_grad[:,1:] +((Lambda * theta2[:,1:])) / len(X)
    regularized_theta1_grad = np.hstack((theta1[:,1][:,np.newaxis],regularized_theta1_grad))
    regularized_theta2_grad = np.hstack((theta2[:,1][:,np.newaxis],regularized_theta2_grad))
    return(Cost,theta1_grad,theta2_grad,regularized_J,regularized_theta1_grad,regularized_theta2_grad)
   

def gradientdescent(input_layer_size,hidden_layer_size,num_labels,alpha,number_of_iteration,theta1,theta2,Lambda):
    """
    Compute optimum parameter theta
    input_layer_size : input layer size
    hidden_layer_size : hidden layer size
    num_labels : number of class in target variable
    alpha : learning rate value between 0 and 1
    Lambda : regularization parameter
    """

    for iteration in np.arange(number_of_iteration):
        nn_params = np.append(theta1.flatten(),theta2.flatten())
        cost1,gradient1,gradient2 = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)[:3]
        new_theta1 = theta1 - (alpha * gradient1)
        new_theta2 = theta2 - (alpha * gradient2)
        theta1 = new_theta1
        theta2 = new_theta2
    return(theta1,theta2)

def predict(theta1,theta2,X):
    """
    Returns prediction vector
    X : array(number of examples, number of features)
    theta1,theta2 : parameters obtained from gradientdescent function
    """
    X = np.hstack((np.ones((X.shape[0],1)),X))
    a1 = X
    a2 = sigmoid(np.dot(theta1,X.T))
    a2 = a2.T
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = sigmoid(np.dot(theta2,a2.T))
    predict = a3
    R = np.array(predict).reshape(10,5000).T
    actual_predictions = np.argmax(R, axis=1)+1
    return(actual_predictions)
#%%
input_layer_size  = 400     
hidden_layer_size = 25
num_labels = 10
alpha = 0.8
Lambda = 1
number_of_iteration = 100
grad_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
grad_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
param_initial = np.append(grad_theta1.flatten(),grad_theta2.flatten())

#%%
t1,t2 = gradientdescent(input_layer_size,hidden_layer_size,num_labels,alpha,number_of_iteration,grad_theta1,grad_theta2,Lambda)
prediction = predict(t1,t2,X)
    
delta = prediction - y.reshape(5000) 

final_accuracy = (delta[delta==0].size / y.size) * 100
print("Final Accuracy : ", final_accuracy, "%")
