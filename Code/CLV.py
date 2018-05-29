from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics

raw_data = pd.read_csv("history.csv")

raw_data.dtypes

#output is as :
'''
CUST_ID    int64
MONTH_1    int64
MONTH_2    int64
MONTH_3    int64
MONTH_4    int64
MONTH_5    int64
MONTH_6    int64
CLV        int64
dtype: object
'''

raw_data.head()
#output   :
'''
CUST_ID	MONTH_1	MONTH_2	MONTH_3	MONTH_4	MONTH_5	MONTH_6	CLV
0	1001	150	75	200	100	175	75	13125
1	1002	25	50	150	200	175	200	9375
2	1003	75	150	0	25	75	25	5156
3	1004	200	200	25	100	75	150	11756
4	1005	200	200	125	75	175	200	15525
'''

#performing correlation analysis :

cleaned_data = raw_data.drop("CUST_ID",axis=1)
cleaned_data .corr()['CLV']

'''
MONTH_1    0.734122
MONTH_2    0.250397
MONTH_3    0.371742
MONTH_4    0.297408
MONTH_5    0.376775
MONTH_6    0.327064
CLV        1.000000
Name: CLV, dtype: float64
'''

#Training and Testing Split :

predictors = cleaned_data.drop("CLV",axis=1)
targets = cleaned_data.CLV

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.1)
print( "Predictor - Training : ", pred_train.shape, "Predictor - Testing : ", pred_test.shape )

#output  "

''' Predictor - Training :  (90, 6) Predictor - Testing :  (10, 6) '''

#we now perform linear regression as we have output data to predict CLV and determine accuracy of built model

#Build model on training data
model = LinearRegression()
model.fit(pred_train,tar_train)
print("Coefficients: \n", model.coef_)
print("Intercept:", model.intercept_)

#Test on testing data
predictions = model.predict(pred_test)
predictions

sklearn.metrics.r2_score(tar_test, predictions)

#output :
'''
Coefficients: 
 [ 34.59195048  11.53796271  15.17878598  11.72051702   8.60555913
   5.44443443]
Intercept: -199.535985333
0.91592106093124581
'''

#from the above output we see that an accuracy of 91.59% has been achieved. Not bad!

#Predicting for a new Customer

new_data = np.array([100,0,50,0,0,0]).reshape(1, -1)
new_pred=model.predict(new_data) 
print("The CLV for the new customer is : $",new_pred[0])


#Output :
'''
The CLV for the new customer is : $ 4018.59836236
'''

#This marks the end of Module - CLV 

