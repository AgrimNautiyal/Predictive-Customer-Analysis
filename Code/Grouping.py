from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.cluster import KMeans
import sklearn.metrics

raw_data = pd.read_csv("issues.csv")
raw_data.dtypes


#using K-means Clustering we'll analyse problems and issues by means of grouping 
clust_data = raw_data.drop("PROBLEM_TYPE",axis=1)

#Finding optimal no. of clusters
from scipy.spatial.distance import cdist
clusters=range(1,10)
meanDistortions=[]

for k in clusters:
    model=KMeans(n_clusters=k)
    model.fit(clust_data)
    prediction=model.predict(clust_data)
    meanDistortions.append(sum(np.min(cdist(clust_data, model.cluster_centers_, 'euclidean'), axis=1)) / clust_data.shape[0])

#we will now plot our analysis for a better understanding of the clusters obtained
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')


#output of the above plot is available in the Jupyter Notebooks uploaded on the repo
'''
Looking at the plot, we see that the knee happens at cluster=3. That is the ideal number of clusters. We now perform the actual clustering
for 3. Then we add the cluster ID to the original dataset.
'''


#Optimal clusters is 3
final_model=KMeans(3)
final_model.fit(clust_data)
prediction=final_model.predict(clust_data)

#Join predicted clusters back to raw data
raw_data["GROUP"] = prediction
print("Groups Assigned : \n")
raw_data[["GROUP","PROBLEM_TYPE"]]


#feature a box plot to see how groups differ for various feature attributes

#box plot to visualise Count attribute
plt.boxplot([[raw_data["COUNT"][raw_data.GROUP==0]],
              [raw_data["COUNT"][raw_data.GROUP==1]] ,
                [raw_data["COUNT"][raw_data.GROUP==2]] ],
            labels=('GROUP 1','GROUP 2','GROUP 3'));

#boxplit to visualise Average calls made to resolve the issue-attribute
plt.boxplot([[raw_data["AVG_CALLS_TO_RESOLVE"][raw_data.GROUP==0]],
              [raw_data["AVG_CALLS_TO_RESOLVE"][raw_data.GROUP==1]] ,
                [raw_data["AVG_CALLS_TO_RESOLVE"][raw_data.GROUP==2]] ],
            labels=('GROUP 1','GROUP 2','GROUP 3'))

#boxplot to visualise reoccuring rate of problem
 plt.boxplot([[raw_data["REOCCUR_RATE"][raw_data.GROUP==0]],
              [raw_data["REOCCUR_RATE"][raw_data.GROUP==1]] ,
                [raw_data["REOCCUR_RATE"][raw_data.GROUP==2]] ],
            labels=('GROUP 1','GROUP 2','GROUP 3'))

#box plot to visualise replacement rate
 plt.boxplot([[raw_data["REPLACEMENT_RATE"][raw_data.GROUP==0]],
              [raw_data["REPLACEMENT_RATE"][raw_data.GROUP==1]] ,
                [raw_data["REPLACEMENT_RATE"][raw_data.GROUP==2]] ],
            labels=('GROUP 1','GROUP 2','GROUP 3'))




#the above plots offer us insight into how the issues differ from each other and which are easily
#resolved and which aren't
 #this provides us with a better analysis and direction to work to improve customer satisfaction



 #this marks the end of module- Grouping
 
