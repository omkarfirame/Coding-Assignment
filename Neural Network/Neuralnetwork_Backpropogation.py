import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("ex3data1.mat")
theta = loadmat("ex3weights.mat")

X = data["X"]
y = data["y"]
#%%
def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def displayData(X):
    fig,axis = plt.subplots(10,10,figsize=(8,8))
    for i in np.arange(10):
        for j in np.arange(10):
            axis[i,j].imshow(X[np.random.randint(0,len(X)),:].reshape(20,20,order=("F")),cmap="magma")   
            
def sigmoidGradient(z):
    g = sigmoid(z)
    g_dash = g*(1-g)
    return(g_dash)

def randInitializeWeights(L_in,L_out):
    eps = (np.sqrt(6) / (np.sqrt(L_in + L_out)))
    random_initial_weights = np.random.rand(L_out,1+L_in) * (2*eps) - eps
    return(random_initial_weights)
    
    
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    """ nn_params is parameter values flatten into a vector """
   
    Theta1 = nn_params[:((input_layer_size + 1) * (hidden_layer_size))].reshape(hidden_layer_size,input_layer_size + 1)
    Theta2 = nn_params[((input_layer_size +1)* hidden_layer_size ):].reshape(num_labels,hidden_layer_size+1)
    m,n = X.shape
    X = np.hstack((np.ones((m,1)),X))
    
    a1 = X
    a2 = sigmoid(np.dot(Theta1,a1.T))
    a2 = a2.T
    a2 = np.hstack((np.ones((a2.shape[0],1)),a2))
    a3 = sigmoid(np.dot(Theta2,a2.T))
    a3 = a3.T
    
    from sklearn.preprocessing import OneHotEncoder 
    onehotencoder = OneHotEncoder(categories='auto')    
    Y = onehotencoder.fit_transform(y).toarray() 
    cost=[]
    for j in np.arange(num_labels):
        J = np.sum(-Y[:,j] * np.log(a3[:,j]) - (1-Y[:,j])*np.log(1-a3[:,j]))
        cost.append(J)
    Cost = np.sum(cost) / m
    regularized_J = Cost + (Lambda / (2*m)) * (np.sum(Theta1[:,1:]**2) + np.sum(Theta2[:,1:]**2))
    
    """ Backpropogation """
    
    Theta1_grad = np.zeros((Theta1.shape[0],Theta1.shape[1]))
    Theta2_grad = np.zeros((Theta2.shape[0],Theta2.shape[1]))
    for example in np.arange(len(X)):
        a_1 = a1[example,:]
        a_2 = a2[example,:]
        a_3 = a3[example,:]
        delta_3 = a_3 - Y[example,:]    
        z = np.dot(a_1,Theta1.T)
        g_dash = np.hstack((1,sigmoidGradient(z)))
        delta_2 = np.dot(Theta2.T,delta_3) * g_dash
        
        Theta1_grad = Theta1_grad + np.dot(delta_2[1:].reshape(len(delta_2)-1,1),a_1.reshape(1,len(a_1))) 
        Theta2_grad = Theta2_grad + np.dot(delta_3.reshape(len(delta_3),1),a_2.reshape(1,len(a_2)))
    
    Theta1_grad = Theta1_grad / len(X)
    Theta2_grad = Theta2_grad / len(X)

    regularized_Theta1_grad = Theta1_grad[:,1:] +((Lambda * Theta1[:,1:])) / len(X)
    regularized_Theta2_grad = Theta2_grad[:,1:] +((Lambda * Theta2[:,1:])) / len(X)
    regularized_Theta1_grad = np.hstack((Theta1[:,1][:,np.newaxis],regularized_Theta1_grad))
    regularized_Theta2_grad = np.hstack((Theta2[:,1][:,np.newaxis],regularized_Theta2_grad))
    return(Cost,Theta1_grad,Theta2_grad,regularized_J,regularized_Theta1_grad,regularized_Theta2_grad)
   

def gradientdescent(input_layer_size,hidden_layer_size,num_labels,alpha,number_of_iteration,Theta1,Theta2,Lambda):

    for iteration in np.arange(number_of_iteration):
        nn_params = np.append(Theta1.flatten(),Theta2.flatten())
        cost1,gradient1,gradient2 = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda)[:3]
        new_theta1 = Theta1 - (alpha * gradient1)
        new_theta2 = Theta2 - (alpha * gradient2)
        Theta1 = new_theta1
        Theta2 = new_theta2
    return(Theta1,Theta2)

def predict(theta1,theta2,X):
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
number_of_iteration = 800
grad_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
grad_theta2 = randInitializeWeights(hidden_layer_size, num_labels)
param_initial = np.append(grad_theta1.flatten(),grad_theta2.flatten())

#%%
t1,t2 = gradientdescent(input_layer_size,hidden_layer_size,num_labels,alpha,number_of_iteration,grad_theta1,grad_theta2,Lambda)
prediction = predict(t1,t2,X)
    
delta = prediction - y.reshape(5000) 

final_accuracy = (delta[delta==0].size / y.size) * 100
print("Final Accuracy : ", final_accuracy, "%")
