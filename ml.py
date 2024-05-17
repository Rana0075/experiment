import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data=pd.read_csv("D:/python/git/experiment/train.csv")
c=['salary','sl_no','gender']
data=data.drop(c,axis=1)

data2=pd.get_dummies(data,drop_first=True)

y=data2.status_Placed
x=data2.drop(['status_Placed'],axis=1)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2 , random_state = 3)


lda= LinearDiscriminantAnalysis()
lda.fit(xtrain,ytrain)
pickle.dump(lda,open('model.pkl','wb'))
