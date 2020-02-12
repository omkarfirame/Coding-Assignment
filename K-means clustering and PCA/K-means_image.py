import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy
import matplotlib.image as mpimage

img = plt.imread("bird_small.png")
img_size = img.shape
X = img.reshape(img_size[0]*img_size[1],img_size[2])
K = 16
"""
X : image reshaped into N * 3 matrix (N = number of pixel)
K = number of clusters
"""

def kMeansInitCentroids(X, K):
    """
    X : training data
    K : number of cluster
    
    returns random initial centroid values
    """
    centroid=[]
    for i in np.arange(K):
        center = X[np.random.randint(0,X.shape[0]+1),:]
        centroid.append(center)
    return (np.array(centroid))

centroid = kMeansInitCentroids(X, K)

def findClosestCentroids(X,centroid,K):
    """
    X : training data
    centroid : (number of cluster * number of features) array which gives initial centroid values
    
    returns cluster number of each example 
    """
    
    centroid_indx=[]
    for example in np.arange(X.shape[0]):
        centroid_dvector=[]
        for centroid_ in np.arange(K):
            centroid_distance = X[example] - centroid[centroid_]
            distance = np.sum(centroid_distance**2)
            centroid_dvector.append(distance)
            index = np.argmin(centroid_dvector)+1
        centroid_indx.append(index)
        
    return(np.array(centroid_indx))
    
idx = findClosestCentroids(X,centroid,K) 

def computeCentroids(X,idx,K):
    """
    X : training data
    idx : array which contain cluster number of each example
    K : number of Cluster
    
    returns new centroid values according to new clusters
    """   
    ind=[]
    new_centroid=[]
    for i in np.arange(K):
        index = np.where(idx==i+1)
        cluster_eg = X[index]
        ind.append(cluster_eg)
        cluster_mean = np.mean(ind[i],axis=0)
        new_centroid.append(cluster_mean)
    return(new_centroid)

new_centroid = computeCentroids(X,idx,K)

def runKmeans(X,centroid,max_iterations,plot_progress):
    """
    X : training data
    centroid : (number of cluster * number of features) array which gives initial centroid values
    max_iterations : maximum number of iteration (int)
    plot_progress : plot of clusters and centroids for each iteration (True or False)
    
    return optimum centroid values and plot
    """    
    K = centroid.shape[0]
    for iteration in np.arange(max_iterations):
        close_centre = findClosestCentroids(X,centroid,K)
        new_centroid = computeCentroids(X,close_centre,K)
        new_centroid = np.array(new_centroid)
        centroid = new_centroid
        cluster=[]
        if plot_progress == True:         
            for j in np.arange(K):
                cluster1 = (close_centre==j+1).reshape(len(X),1)
                cluster.append(cluster)
                plt.scatter(X[cluster1[:,0],0],X[cluster1[:,0],1],s=50)
                
            new_centroid = np.array(centroid)  
            plt.scatter(new_centroid[:,0],new_centroid[:,1],c = "black",marker="+",s=220)
            plt.show()
    return(centroid)
    
max_iterations = 10
centroids = runKmeans(X,centroid,max_iterations,False)
centroid_points = findClosestCentroids(X,centroids,K)

""""Mapping of pixel to its centroid value"""
X_recovered = copy.copy(X)
for i in np.arange(K):
    X_recovered[(centroid_points==i+1),:] = centroids[i]
    
X_recovered = X_recovered.reshape(img_size[0],img_size[1],img_size[2])

"""Display Image"""
fig,ax = plt.subplots(1,2)
ax[0].imshow(img)
ax[1].imshow(X_recovered)
