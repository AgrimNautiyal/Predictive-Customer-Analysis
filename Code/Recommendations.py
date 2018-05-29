import pandas as pd

userItemData = pd.read_csv('ratings.csv')
userItemData.head()

#now we work to create an affinity score to predict recommendations

#Get list of unique items
itemList=list(set(userItemData["ItemId"].tolist()))

#Get count of users
userCount=len(set(userItemData["ItemId"].tolist()))

#Create an empty data frame to store item affinity scores for items.
itemAffinity= pd.DataFrame(columns=('item1', 'item2', 'score'))
rowCount=0

#For each item in the list, compare with other items.
for ind1 in range(len(itemList)):
    
    #Get list of users who bought this item 1.
    item1Users = userItemData[userItemData.ItemId==itemList[ind1]]["userId"].tolist()
    #print("Item 1 ", item1Users)
    
    #Get item 2 - items that are not item 1 or those that are not analyzed already.
    for ind2 in range(ind1, len(itemList)):
        
        if ( ind1 == ind2):
            continue
       
        #Get list of users who bought item 2
        item2Users=userItemData[userItemData.ItemId==itemList[ind2]]["userId"].tolist()
        #print("Item 2",item2Users)
        
        #Find score. Find the common list of users and divide it by the total users.
        commonUsers= len(set(item1Users).intersection(set(item2Users)))
        score=commonUsers / userCount

        #Add a score for item 1, item 2
        itemAffinity.loc[rowCount] = [itemList[ind1],itemList[ind2],score]
        rowCount +=1
        #Add a score for item2, item 1. The same score would apply irrespective of the sequence.
        itemAffinity.loc[rowCount] = [itemList[ind2],itemList[ind1],score]
        rowCount +=1
        
#Check final result
itemAffinity.head()

#output of above segment is as follows :
'''
item1	item2	score
0	5001.0	5002.0	0.4
1	5002.0	5001.0	0.4
2	5001.0	5003.0	0.0
3	5003.0	5001.0	0.0
4	5001.0	5004.0	0.2
'''


searchItem=5001
recoList=itemAffinity[itemAffinity.item1==searchItem]\
        [["item2","score"]]\
        .sort_values("score", ascending=[0])
        
print("Recommendations for item 5001\n", recoList)

#output of above segemnt is as follows :
'''

Recommendations for item 5001
     item2  score
0  5002.0    0.4
6  5005.0    0.4
4  5004.0    0.2
2  5003.0    0.0

'''
