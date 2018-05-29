from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

prospect_data = pd.read_csv("browsing.csv")

prospect_data.dtypes
# Look at the top records to understand how the data looks like.
prospect_data.head()

#Do summary statistics analysis of the data
prospect_data.describe()

#Perform correlation analysis
prospect_data.corr()['BUY']

#Drop columns with low correlation
predictors = prospect_data[['REVIEWS','BOUGHT_TOGETHER','COMPARE_SIMILAR','WARRANTY','SPONSORED_LINKS']]
targets = prospect_data.BUY
 # the above is an attempt at concentrating the data for better and efficient analysis


#Training and Testing Split
pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

print( "Predictor - Training : ", pred_train.shape, "Predictor - Testing : ", pred_test.shape )

#Building Model and testing accuracy : 
from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

#Analyze accuracy of predictions
sklearn.metrics.confusion_matrix(tar_test,predictions)

sklearn.metrics.accuracy_score(tar_test, predictions)
#the above model gives an accuracy of 71.9999999999%
#considering the few thousand-size data, it is a good figure as real life datasets are millions in size


#Instead of doing a Yes/No prediction, we can instead do a probability computation to show the probability for the prospect to buy the product
pred_prob=classifier.predict_proba(pred_test)
pred_prob[0,1]



#now we dwelve into real time predictions;  probablity increases or decreases dynamically depending on the users actions :

#probablity after user checks in - 4%
browsing_data = np.array([0,0,0,0,0]).reshape(1, -1)
print("New visitor: propensity :",classifier.predict_proba(browsing_data)[:,1] )


#probability after user checks reviews - 57%  - according to results of the project which can be viewed in the Jupyter notebooks 
browsing_data = np.array([0,0,1,0,0]).reshape(1, -1)
print("After checking similar products: propensity :",classifier.predict_proba(browsing_data)[:,1] )


#we can do similar dynamic checks to fluctuate probablity - this marks the end of Module 1 - Propensity


