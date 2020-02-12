import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

data = loadmat("ex7data1.mat")
X = data["X"]


def featureNormalize(X):
    """
    X : training data
    
    returns normalized data
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return(X)

X_norm = featureNormalize(X)

def pca(X):        
    """
    X : normalized training data
    
    computes eigenvectors of covariance matrix of X
    """
    sigma = np.dot(X.T,X) / len(X)
    u,s,v = np.linalg.svd(sigma)
    return(u,s,v)

u,s,v = pca(X_norm)
mean = np.mean(X,axis=0)

eigen_vector = u[:,0]

def projectData(X,u,K): 
    """
    X : normalized training data
    u : principal componant
    K : desired number of diamensions
    
    projects each example in X onto top K componants in principal componant
    """
    u_reduced = u[:,:K]
    projection_K=[]
    for i in np.arange(len(X)):
        projection=[]
        for j in np.arange(K):
            project = X[i,:].dot(u_reduced[:,j])
            projection.append(project)
        projection_K.append(projection)

    return(np.array(projection_K))
  
K=1
Z = projectData(X_norm, u, K)
    
def recoverData(Z,u,K): 
    """
    Z : projected data onto lower diamensions
    u : principal componants
    K : desired number of diamensions
    
    recovers the data by projecting them back onto the original high diamension space 
    """
    recoverd_J=[]
    for i in np.arange(len(X)):
        recover = Z[i,:].dot(u[:,:K].T)
        recoverd_J.append(recover)
    return(np.array(recoverd_J))
    
X_recoverd = recoverData(Z,u,K)
"""Plotting recovered data and original data"""
plt.scatter(X_recoverd[:,0],X_recoverd[:,1])
plt.scatter(X_norm[:,0],X_norm[:,1])

face_data = loadmat("ex7faces.mat")
X1 = face_data["X"]
def displayData(X,dim):
    """
    X : training data
    dim : dim value represents dim * dim numbers of  images
    
   returns dim * dim random number of images
    """
    import matplotlib.image as mpimg
    fig,axis = plt.subplots(dim,dim,figsize=(8,8))
    for i in np.arange(dim):
        for j in np.arange(dim):
            axis[i,j].imshow(X[np.random.randint(0,len(X)),:].reshape(32,32,order=("F")),cmap="gray")
            
displayData(X1,10)

"""visualization of first 36 eigenfaces"""
X2 = featureNormalize(X1)
U,S,V = pca(X2)
U_reduced = U[:,:36].T
displayData(U_reduced,6)

"""Visualization of reduced diamension data"""
K1 = 100
Z1 = projectData(X2,U,K1)
X1_recoverd = recoverData(Z1,U,K1)
displayData(X1_recoverd,10)
