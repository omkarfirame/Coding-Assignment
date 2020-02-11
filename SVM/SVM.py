import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    from scipy.io import loadmat
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score,make_scorer


data = loadmat("ex6data1.mat")
X = data["X"]
y = data["y"]
"""
X,y : training set
"""

def visualizeBoundaryLinear(X,y,model):
    """
    X : training set data
    y : training set target
    model : algorithm used to solve the problem
    
    plot the decision boundary
    """
    pos = (y==1).reshape(len(X),1)
    neg = (y==0).reshape(len(X),1)
    plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
    plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50)
    
    X1,X2 = np.meshgrid(np.linspace(X[:,0].min(),X[:,0].max(),num=100),np.linspace(X[:,1].min(),X[:,1].max(),num=100))
    plt.contour(X1,X2,model.predict(np.array([X1.flatten(),X2.flatten()]).T).reshape(X1.shape),1,color = "b")

"""SVM using linear kernel and regularization parameter is 1"""
model1 = svm.SVC(C=1,kernel='linear')
model1.fit(X,y)

"""SVM using linear kernel and regularization parameter is 100"""
model2 = svm.SVC(C=100,kernel='linear')
model2.fit(X,y)

visualizeBoundaryLinear(X,y,model1)
visualizeBoundaryLinear(X,y,model2)

#%%
data2 = loadmat("ex6data2.mat")
X2 = data2["X"]
y2 = data2["y"]
"""
X2,y2 : training set for ex6data2
"""
"""SVM using Gaussian kernel and regularization parameter is 1"""
model3 = svm.SVC(C=1,kernel='rbf',gamma=30)
model3.fit(X2,y2)

visualizeBoundaryLinear(X2,y2,model3)

#%%
data3 = loadmat("ex6data3.mat")
X3 = data3["X"]
y3 = data3["y"]
"""
X3,y3 : training set for ex6data3
"""
Xval = data3["Xval"]
yval = data3["yval"]
"""
Xval,yval : Cross validation set for ex6data3
"""
values = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30)


def dataset3Params(X,y,Xval,yval,values):    
    """
    X : training set data
    y : training set target
    Xval : cross validation set data
    yval : cross validation set target
    values : parameter values for regularization parameter and sigma
    
    compute best values of regularization parameter and sigma
    """
    final_accuracy_vec=[]
    for i in np.arange(len(values)):
        accuracy_vec=[]    
        for j in np.arange(len(values)):          
            gamma = 1 / (values[j])
            model = svm.SVC(C = values[i],kernel = 'rbf', gamma = gamma)
            model.fit(X,y.ravel())
            pred = model.predict(Xval)
            accuracy = accuracy_score(yval,pred)
            accuracy_vec.append(accuracy)
        final_accuracy_vec.append(accuracy_vec)
    
    accuracy_matrix = np.array(final_accuracy_vec)
    max_acc = np.max(accuracy_matrix)
    location = np.argwhere(accuracy_matrix==max_acc)
    best_C = values[location[0,0]]
    best_sigma = values[location[0,1]]
    return(best_C,best_sigma)

C,sigma = dataset3Params(X3,y3,Xval,yval,values)
gamma = 1 / sigma
model4 = svm.SVC(C = C,kernel='rbf',gamma=gamma)
model4.fit(X3,y3)
visualizeBoundaryLinear(X3,y3,model4)
