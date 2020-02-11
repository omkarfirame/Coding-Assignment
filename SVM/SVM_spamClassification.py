import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from scipy.io import loadmat


file_contents = open("emailSample1.txt").read()
vocablist = open("vocab.txt").read()

"""Convert List into dictionary"""
vocablist=vocablist.split("\n")[:-1]
vocabList_d={}
for ea in vocablist:
    value,key = ea.split("\t")[:]
    vocabList_d[key] = value
    
def processEmail(file_contents,vocabList_d):
    """
    file_contents : body of an email (string)
    vocabList_d : vocabulary list and it's index stored as dictionary
    
    returns word indices vector
    """
            
    file_contents = file_contents.lower()
    file_contents = re.sub('\d+','number',file_contents)
    
    file_contents = re.sub("[http|https]://[^\s]*",'paddr',file_contents)
    file_contents = re.sub('[^\s]+@[^\s]+','emailaddr',file_contents)
    file_contents = re.sub('[$]','doller',file_contents)
    file_contents = re.sub("[<.*?,:;>']",'',file_contents)
    
    file_contents = re.sub('\s+', ' ', file_contents).strip()
    
    ps = PorterStemmer()
    file_contents = [ps.stem(token) for token in file_contents.split(" ")]
    file_contents= " ".join(file_contents)
    

    indices = [vocabList_d[t] for t in file_contents.split() if t in vocabList_d]

    return(indices)
indices = processEmail(file_contents,vocabList_d)
indices = [int(i) for i in indices] 
#%%
def emailFeatures(indices,vocabList_d):    
    """
    indices : word indices vector 
    vocabList_d : vocabulary list and it's index stored as dictionary
    
    returns feature vector (gives 1 if word from email present in vocabulary list otherwise 0)
    """
    faeture = np.zeros((len(vocabList_d),1))
    for i in indices:
        faeture[i] = 1
    return(faeture)    
feature = emailFeatures(indices,vocabList_d)
#%%
from sklearn.metrics import accuracy_score
data = loadmat("spamTrain.mat")
testdata = loadmat("spamTest.mat")

X = data["X"]
y = data["y"]
"""
X : training set data
y : training set target
"""
Xtest = testdata["Xtest"]
ytest = testdata["ytest"]
"""
Xtest : test set data
ytest : test set target
"""
"""SVM using linear kernel and regularization parameter is 0.1"""
model = svm.SVC(C=0.1,kernel = 'linear')
model.fit(X,y)
pred = model.predict(Xtest)
accuracy = accuracy_score(ytest,pred) * 100
#%%
"""gives word with most predictive of spam email"""
weights = model.coef_
weight_dframe = pd.DataFrame(np.hstack((np.arange(1,1900).reshape(1899,1),weights.T)))
weight_dframe.sort_values(by=[1],ascending=False,inplace=True)

ind = (weight_dframe.index).tolist()
predictor=[]
for i in np.arange(len(vocablist)):
        
    n = vocablist[ind[i]]
    predictor.append(n)

"""top 15 word with most predictive of spam email"""
top_predictors  = [elem.split('\t')[1] for elem in predictor]
print("top predictors are ",top_predictors[:15])
print("Accuracy is :",accuracy)