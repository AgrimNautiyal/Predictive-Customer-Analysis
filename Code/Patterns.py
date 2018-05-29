from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from apyori import apriori

#Load the prospect dataset
raw_data = pd.read_csv("attrition.csv")

raw_data.head()

'''
output :

LIFETIME	TYPE	REASON	AGE_GROUP	EMP_STATUS	MARITAL_STATUS	RENEWALS	PROBLEMS	OFFERS
0	1 - 3 M	CANCEL	BETTER DEALS	< 20	STUDENT	SINGLE	0	0 to 5	0 to 2
1	1 - 3 M	CANCEL	BETTER DEALS	< 20	STUDENT	SINGLE	0	0 to 5	0 to 2
2	1Y - 2Y	CANCEL	NOT HAPPY	30 - 50	EMPLOYED	MARRIED	1	10 plus	0 to 2
3	1Y - 2Y	EXPIRY	BETTER DEALS	30 - 50	EMPLOYED	MARRIED	1	0 to 5	2 to 5
4	1Y - 2Y	CANCEL	NOT HAPPY	30 - 50	UNEMPLOYED	SINGLE	1	10 plus	0 to 2

'''

'''
The CSV contains information about each customer who have left the business. It contains attributes like LIFETIME of the customer, How the customer left, reasons, problems and demographics.

For doing association rules mining, the data needs to be in a specific format.
Each line should be a transaction with a list of items for that transaction.
We will take the CSV file data convert them into values like "name=value" to create this specific data structure'''

basket_str = ""
for rowNum, row in raw_data.iterrows():
    
    #Break lines
    if (rowNum != 0):
        basket_str = basket_str + "\n"
    #Add the rowid as the first column
    basket_str = basket_str + str(rowNum) 
    #Add columns
    for colName, col in row.iteritems():
        basket_str = basket_str + ",\"" + colName + "=" + str(col) +"\""

#print(basket_str)
basket_file=open("warranty_basket.csv","w")
basket_file.write(basket_str)
basket_file.close()


#We now use the apriori algorithm to build association rules.

#read back
basket_data=pd.read_csv("warranty_basket.csv",header=None)
filt_data = basket_data.drop(basket_data.columns[[0]], axis=1)
results= list(apriori(filt_data.as_matrix()))

rulesList= pd.DataFrame(columns=('LHS', 'RHS', 'COUNT', 'CONFIDENCE','LIFT'))
rowCount=0

#Convert results into a Data Frame
for row in results:
    for affinity in row[2]:
        rulesList.loc[rowCount] = [ ', '.join(affinity.items_base) ,\
                                    affinity.items_add, \
                                    len(affinity.items_base) ,\
                                    affinity.confidence,\
                                    affinity.lift]
        rowCount +=1


rulesList.head()


'''output :

LHS	RHS	COUNT	CONFIDENCE	LIFT
0		(AGE_GROUP=20 - 30)	0.0	0.34	1.0
1		(AGE_GROUP=30 - 50)	0.0	0.32	1.0
2		(AGE_GROUP=50PLUS )	0.0	0.16	1.0
3		(AGE_GROUP=< 20)	0.0	0.18	1.0
4		(EMP_STATUS=EMPLOYED)	0.0	0.54	1.0

'''


#We can also filter rules where the count of elements is 1 and the confidence is > 70%


rulesList[(rulesList.COUNT <= 1) & (rulesList.CONFIDENCE > 0.7)].head(5)

'''
output  :

LHS	RHS	COUNT	CONFIDENCE	LIFT
34	LIFETIME=3M to 1Y	(AGE_GROUP=20 - 30)	1.0	1.000000	2.941176
55	AGE_GROUP=20 - 30	(TYPE=CANCEL)	1.0	0.941176	1.568627
61	AGE_GROUP=30 - 50	(LIFETIME=1Y - 2Y)	1.0	1.000000	3.125000
62	LIFETIME=1Y - 2Y	(AGE_GROUP=30 - 50)	1.0	1.000000	3.125000
64	MARITAL_STATUS=MARRIED	(AGE_GROUP=30 - 50)	1.0	0.833333	2.604167
'''




