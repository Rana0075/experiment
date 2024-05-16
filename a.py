import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
a=pd.read_csv("https://github.com/Rana0075/experiment/blob/main/train.csv")
a['specialisation'] = a['specialisation'].map({'Mkt&HR': 1, 'Mkt&Fin': 0}) 
a['degree_t'] = a['degree_t'].map({'Sci&Tech': 1, 'Comm&Mgmt': 2,'Others':0}) 
a['hsc_s'] = a['hsc_s'].map({'Science': 1, 'Commerce': 2,'Arts':3}) 
a['hsc_b'] = a['hsc_b'].map({'Central': 1, 'Others': 0}) 
a['ssc_b'] = a['ssc_b'].map({'Central': 1, 'Others': 0}) 
a['status'] = a['status'].map({'Placed': 1, 'Not Placed': 0}) 
a['workex'] = a['workex'].map({'Yes': 1, 'No': 0}) 
c=['salary','sl_no','gender']
b=a.drop(c,axis=1)
from sklearn.model_selection import train_test_split
y=b.status
x=b.drop(['status'],axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2 , random_state = 3)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ida= LinearDiscriminantAnalysis()
ida.fit(xtrain,ytrain)
