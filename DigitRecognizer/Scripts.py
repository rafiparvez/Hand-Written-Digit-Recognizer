# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 22:19:25 2016

@author: Parvez
"""
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('E:/Kaggle/Hand-Written-Digit-Recognizer/Data/train.csv')
X_test=pd.read_csv('E:/Kaggle/Hand-Written-Digit-Recognizer/Data/test.csv')

# Y contains the Training Outputs
Y=df['label'].values

# X contains the training inputs
X = df[df.columns[1:]]

#number of training data
m=len(Y)

#Creating Random forest predictive model
model= RandomForestClassifier(n_estimators=100)
X_train, X_cv, Y_train, Y_cv = cross_validation.train_test_split(X/255.,Y,test_size=0.1,random_state=4)
model.fit(X_train, Y_train)

#Predict using cross-validation set
Y_pred_cv = model.predict(X_cv)
accuracy_rf = accuracy_score(Y_cv, Y_pred_cv)
print("Accuracy using Random Forest: ",accuracy_rf)

#Predict using test set
Y_pred_test = model.predict(X_test)

#Generate csv for submission
s1 = pd.Series(list(range(1,len(Y_pred_test)+1)), name='ImageId')
s2 = pd.Series(Y_pred_test, name='Label')
df_sb=pd.concat([s1, s2], axis=1)
df_sb.to_csv('E:/Kaggle/Hand-Written-Digit-Recognizer/Data/submission.csv',sep=',',index=False)