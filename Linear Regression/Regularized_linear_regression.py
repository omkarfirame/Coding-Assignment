import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat("ex5data1.mat")
X = data["X"]                                       
Xval = data["Xval"]
y = data["y"]
yval = data["yval"]
Xtest = data["Xtest"]
ytest = data["ytest"]
theta = np.ones((X.shape[1]+1,1))

""" X, y : Training set
    Xval, yval : Cross validation set
    Xtest, ytest : Test set for evaluating performance.
"""

def plotData(X,y):
    """
    Function to plot data
    X : array(number of examples, number of features)
    y : array(number of examples, 1)
    """
    a = plt.scatter(X,y)
    b = plt.xlabel("Change in water level (x)")
    c = plt.ylabel("Water flowing out of the dam (y)")
    d=plt.title("data")
    return(a,b,c,d)

plotData(X,y)

def featureNormalize(X):
    """
    X : array(number of examples, number of features)
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return(pd.DataFrame(X))


""" Add one's column in training set """
ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))


def linearRegCostFunction(X,y,theta,Lambda):
    """
    Compute Cost and Gradient for multi variable linear regression with regularization.
    X : array(number of examples, number of features+1)
    y : array(number of examples, 1)
    theta : array(number of features+1,1)
    Lambda : Regularization parameter.
    """    
    theta = theta.reshape(-1,y.shape[1])
    hypothesis = np.dot(X,theta)
    error = hypothesis - y
    cost = (1/(2*len(X))) * np.sum(np.square(error)) +(Lambda / (2*len(X))) * (np.sum(np.square(theta)))
    gradient = (np.dot(X.T,error) / len(X)) + ((Lambda/ len(X)) * theta)
    
    return(cost,gradient.flatten())


cost,grad = linearRegCostFunction(X,y,theta,1)


from scipy.optimize import minimize

def trainLinearReg(X,y,Lambda):
    """
    Trains linear regression using dataset (X,y) and regularization parameter Lambda.
    Returns trained parameter theta.
    X : array(number of examples, number of features+1)
    y : array(number of examples, 1)
    Lambda : Regularization parameter.
    """
    initial_theta = np.zeros((X.shape[1], 1))

    def cost(theta):
        return(linearRegCostFunction(X,y,theta,1))
        
    results = minimize(fun=cost,x0=initial_theta,method='CG',jac=True, options={'maxiter':200})
    theta = results.x
    return(theta)

theta = trainLinearReg(X,y,0)

plt.plot(data['X'],np.dot(X,theta))


""" Learning Curves """

""" Add one's column in cross-validation set """
ones1 = np.ones((Xval.shape[0],1))
Xval = np.hstack((ones1,Xval))

def learningCurve(X,y,Xval,yval,Lambda):
    """ 
    Calculate train and cross validation set errors to plot learning curves.
    X : array(number of examples, number of features+1)
    y : array(number of examples, 1)
    (Xval, yval) : Cross validation set
    Lambda : Regularization parameter.
    """
    error_train = []
    error_val = []
    for i in np.arange(1,len(X)+1):
        theta = trainLinearReg(X[:i],y[:i],Lambda)
        
        train_error = linearRegCostFunction(X[:i],y[:i],theta,Lambda)[0]
        cv_error = linearRegCostFunction(Xval,yval,theta,Lambda)[0]
        error_train.append(train_error) ; error_val.append(cv_error)
    return(error_train,error_val)
        
    
error_train,error_val = learningCurve(X,y,Xval,yval,0)    

""" Plotting Learning Curves """

plt.figure(figsize=(6,4))
plt.plot(error_train,'b',label='train')
plt.plot(error_val,'g',label='Cross Validation')
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.legend()


""" Polynomial Regression """

new_x = X[:,1][:,np.newaxis]

def polyFeatures(X,degree):
    """
    Maps each example of X into its polynomial features.
    X : array(number of examples, number of features+1)
    degree : Polynomial degree 
    """
    poly1=[]
    for i in np.arange(1,degree):
        poly = X**(i+1)
        poly1.append(poly)
            
    poly1 = np.hstack(poly1)
    poly1 = featureNormalize(poly1)
    polynomial = np.hstack((np.ones((X.shape[0],1)),poly1))
    return(polynomial)

x_poly = polyFeatures(new_x,8)
x_poly_test = polyFeatures(Xtest,8)
x_poly_val = polyFeatures(Xval[:,1][:,np.newaxis],8)

poly_theta = trainLinearReg(x_poly_val,y_val,0)

def validationCurve(x_poly,y,x_poly_val,yval):
    """
    calculate train and validation error to plot validation curve for selecting optimum Lambda.
    x_poly : array(number of examples, number of features+1)
    y : array(number of examples, 1)
    x_poly_val : array(number of examples in validation set, number of features+1)
    yval : array(number of examples in validation set, 1)
    """
    
    Lambda = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
    error_train = []
    error_val = []
       
    for i in np.arange(len(Lambda)):
     
        theta = trainLinearReg(x_poly,y,Lambda[i])
        
        train_error = linearRegCostFunction(x_poly,y,theta,Lambda[i])[0]
        cv_error = linearRegCostFunction(x_poly_val,y_val,theta,Lambda[i])[0]
        error_train.append(train_error) ; error_val.append(cv_error)
    
    return(error_train,error_val,Lambda)
        
validation_train_error,validation_error,Lambda = validationCurve(x_poly,y,x_poly_val,y_val)
plt.figure(figsize=(6,4))
plt.plot(Lambda,validation_train_error,'b',label='train')
plt.plot(Lambda,validation_error,'g',label='Cross Validation')
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.legend()

