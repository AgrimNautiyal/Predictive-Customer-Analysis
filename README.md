# Predictive-Customer-Analysis
This  project is based on looking at several aspects of predictive customer analysis and operations. Datasets from the LinkedIn Predictive Customer Analysis course  are used and manipulated on.


This project features the following steps : 

1) Determining Customer propensity - using data like links clicked by the user such as FAQs, bought together, reviews, specs, shipping, etc 

2) Recommending Items - this is accomplished by mainly building an affinity score between one item to another, iterated over a list of such items ; this is done by a simple co-relation that if the user has  purchased an item x and item y then with sufficient number of such correlations a positive affinity score can be established, negative- in the converse case 

3) Predicting Customer Lifetime Value (CLV) - this follows a correlation analysis- finding a correlation between time spent every month  on website by user VS the CLV- with given dataset we see a CLV of score 1.00 which is a high score and shows us our model can be built accurately. The results are demonstrated in the Jupyter Notebook attached in this repo. Final accuracy achieved - 91%


4)Grouping problems - This part deals with customer support and efficient escalation analysis. For this particular case study we make use of K-means clustering as this groups different problems into identifiable clusters and helps us find a direct trend and pattern.


5)Evaluating Patterns - This final module of the project deals with finding patterns among customers who have left the business for any particular reason, classified specifically by age, employment status, grievance, renewals, offers, etc.



PARTICULARS : 

this project contains a folder of jupyter notebooks which contain the code with time exec and visualisations and explanations wherever necessary; 
raw code is also available for each module and the final code is presented in a single file under the name of final_code


This project is the sole property of Agrim Nautiyal, completed in the year 2018, May. It has been supplemented by datasets derieved from the LinkedIn learning course- Predictive Customer Analysis.
