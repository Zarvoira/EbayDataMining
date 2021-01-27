# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 22:59:02 2020

@author: santosa
"""


from ebaysdk.finding import Connection as finding # install with pip
from bs4 import BeautifulSoup # install with pip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

itemDetails=[]

#Step 1 - Business Understanding'

# Ebay is one of the largest online market website in the world
# with this website we can sell or buy/bid listings that available
# in this Case we want to help the retail market smartphone industry
# for the data mining im using EBAY API  

# Problems
# to predict if an item is high value or not
# is high value affected by countries?
# is high value affected by seller rating / feedback
# is high value affected by shipping type available
# is high value affected by condition


# solving this problems we can predict prices of items and take advantage of the prediction
# by beating the market.
# we can bid/but an item whenever the model predict it as high value or when the price that we predict is higher than the actual item price
# and make profit from the difference by reselling the item again 

# the predictor variables are : 'feedback','country','positivefeedbackpercent','toprated','condition','shippingtype'
# since the predictor are universal we can easily change thTois to any types of items (not limited to phones only)
# but for this project im specifically trying to predict iphone 8 64 GB



## STEP 2
## DATA MINING

ebayapi = 'enter your key here'
api = finding(appid = ebayapi, siteid="EBAY-IE", config_file=None) # change country with "siteid="


for i in range (16):
    request = {
                'keywords': ' "Iphone 8" ', 'outputSelector' : 'SellerInfo',
                'itemFilter': [
                    {'name': 'StorageCapacity', 'value': '64GB'}
                ],
                'paginationInput': {
                    'entriesPerPage': 100,
                    'pageNumber': i
                },
                'sortOrder': 'EndTime'
            }
    
    response = api.execute('findItemsByKeywords', request)       
    soup = BeautifulSoup(response.content, "lxml")
    items = soup.find_all("item")
    for item in items:
              #title = item.title.string.lower().strip()
              
              price = item.convertedcurrentprice.string.strip()
              
              #url = item.viewitemurl.string.lower()
              
              feedback=item.feedbackscore.string.strip()
              
              country=item.country.string.strip()
              
              positivefeedbackpercent=item.positivefeedbackpercent.string.strip()
              
              
              toprated= item.topratedseller.string.strip()
              
              shippingtype= item.shippingtype.string.strip()
              
              

              
              topratedBool= False
              if(toprated == "true"):
                  topratedBool = True
              
                    
              condition = ""
              if item.conditiondisplayname:
                condition = item.conditiondisplayname.string.lower()
              else:
                condition = "n/a"    
             
              itemDetails.append( [float(price),float(feedback),country,float(positivefeedbackpercent), toprated,condition,shippingtype ]     )        

## turn the data that we got from the api to panda dataseries
numpy_array = np.array(itemDetails)
data = pd.DataFrame(numpy_array, columns=['price', 
                      'feedback','country','positivefeedbackpercent','toprated','condition','shippingtype'])


#because converting all the data become object type so i need to convert them to correct types to do calculations
data.info()
data.describe()

data['price'] = data['price'].astype('float')
data['feedback'] =data['feedback'].astype('float')
data['positivefeedbackpercent'] =data['positivefeedbackpercent'].astype('float')

data.info()
#   Column                               Non-Null Count  Dtype  
# =============================================================================
# ---  ------                               --------------  -----  
#  0   title                                1500 non-null   object 
#  1   price                                1500 non-null   float64
#  2   feedback                             1500 non-null   float64
#  3   country                              1500 non-null   object 
#  4   positivefeedbackpercent              1500 non-null   float64
#  5   toprated                             1500 non-null   object 
#  6   condition                            1500 non-null   object 
#  7   shippingtype                         1500 non-null   object 
# ## STEP 2
# ## =============================================================================


##Step 3
## DATA CLEANING
# Checking the missing values
print(data.isnull().sum()) 


################# Construct New Variables########################


data['topRatedType']=np.where(data.toprated =="true",1,0)

#since this is a classification project i turn the price into classifications type with median as a condition
data['aboveMedianPrice']=np.where(data.price > 304,1,0)

print(data.country.unique())
# ['IE' 'GB' 'CN' 'HU' 'DE' 'AU' 'HK']
data['IEtype']=np.where(data.country=="IE",1,0)
data['GBtype']=np.where(data.country=="GB",1,0)
data['CNtype']=np.where(data.country=="CN",1,0)
data['DEtype']=np.where(data.country=="DE",1,0)
data['AUtype']=np.where(data.country=="AU",1,0)
data['HKtype']=np.where(data.country=="AU",1,0)




print(data.shippingtype.unique())
#['Free' 'Flat' 'FreePickup' 'Calculated' 'NotSpecified''FlatDomesticCalculatedInternational']
data['Free']=np.where(data.shippingtype=="Free",1,0)
data['Flat']=np.where(data.shippingtype=="Flat",1,0)
data['FreePickup']=np.where(data.shippingtype=="FreePickup",1,0)
data['FlatDomesticCalculatedInternational']=np.where(data.shippingtype=="FlatDomesticCalculatedInternational",1,0)


print(data.condition.unique())
#['seller refurbished' 'used' 'new' 'new other (see details)''for parts or not working' 'manufacturer refurbished']
data['seller refurbished']=np.where(data.condition=="seller refurbished",1,0)
data['new']=np.where(data.condition=="new",1,0)
data['new other']=np.where(data.condition=="new other (see details)",1,0)
data['manufacturer refurbished']=np.where(data.condition=="manufacturer refurbished",1,0)






##### Outliers ########


#feedback
figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.feedback)
plt.show()
#the feedback seems normal with one outlier having 250000 feedback which is still normal
#considering big ebay sellers have that kind of size


#price
figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.price)
plt.show()
#Price seems normal no minus price in the outliers



#positvefeedbackpercent
figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.positivefeedbackpercent)
plt.show()
#feedback seems normal too with positive feedback mostly above 90% and there is one outlier




##Step 4
##Data Exploration

#######Univariate for Categorical variable #########

print(data.country.unique())
numberCountry = data.country.value_counts() #Gives a pandas series

#Plot bar chart with pandas plot method (built on matplotlib)
numberCountry.plot.barh()
plt.title("Country")
plt.show()

numberCountry = data.country.value_counts(normalize=True)
numberCountry.plot.barh()
plt.show()
 #we can see that mnajority of the item came from Great Britain and Australia in the second and Ireland in the third



print(data.toprated.unique())
TopRatedNum = data.toprated.value_counts() #Gives a pandas series

#Plot bar chart with pandas plot method (built on matplotlib)
TopRatedNum.plot.barh()
plt.title("Top Rated")
plt.show()

TopRated = data.toprated.value_counts(normalize=True)
TopRated.plot.barh()
plt.show()
#we can see that most of the item listed is top rated around (70% of the data)




print(data.condition.unique())
ConditionNum = data.condition.value_counts() #Gives a pandas series

#Plot bar chart with pandas plot method (built on matplotlib)
ConditionNum.plot.barh()
plt.title("Condition")
plt.show()


Condition = data.condition.value_counts(normalize=True)
Condition.plot.barh()
plt.show()
#we cam see that the most of the listing have condition as 'seller refurbished' and followed by 'used'
#this can be big indicator to predict the price






print(data.shippingtype.unique())
ShippingNum = data.shippingtype.value_counts() #Gives a pandas series

#Plot bar chart with pandas plot method (built on matplotlib)
ShippingNum.plot.barh()
plt.title("Shipping")
plt.show()


# Let's calculate the percentage of each job status category and plot.
Shipping = data.shippingtype.value_counts(normalize=True)
Shipping.plot.barh()
plt.show()
#there are 3 dominant type of shippings Flat,free,and flatdomesticcalculatedinternational






#######Univariate for Numerical variable #########



#Lets examine Price
data.describe().price
sns.boxplot(x=data.price)
plt.show()

#Histogram
sns.distplot(data.price, kde = False, bins=20)
plt.show()
#if we see the price mostly range from 200-300 euro and we can see some outliers past 400 euro and less than 200 euro range
#this can be sweet spot for us to bid/buy the phone



#Lets examine Feedback
data.describe().feedback
sns.boxplot(x=data.feedback)
plt.show()

#Histogram
sns.distplot(data.feedback, kde = False, bins=20)
plt.show()
#most of the feedback we received is on 1000 ranges there is couple big outliers sitting on 250 000

#Lets examine positivefeedbackpercent
data.describe().positivefeedbackpercent
sns.boxplot(x=data.positivefeedbackpercent)
plt.show()

#Histogram
sns.distplot(data.positivefeedbackpercent, kde = False, bins=20)
plt.show()
#most of the feedback we receive is > 90% with few outliers lower than that






####Numeric - numeric analysis for two variables where both are numeric

#plot the scatter plot of Price and feedback variable in data
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.price,data.feedback)
plt.title("Price vs Feedback")
plt.xlabel("Price")
plt.ylabel("Feedback")
plt.show()
#to some extent the feedback affect the price but not by much if we see the plots its pretty much diverse



#plot the scatter plot of Price and positivefeedbackpercent  variable in data
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.price,data.positivefeedbackpercent)
plt.title("Price vs positivefeedbackpercent")
plt.xlabel("Price")
plt.ylabel("positivefeedbackpercent")
plt.show()
#it applies to feedback percent to because most of the listing we receive is > 90%



#plot the scatter plot of Feedback and positivefeedbackpercent  variable in data
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.feedback,data.positivefeedbackpercent)
plt.title("Feedback vs positivefeedbackpercent")
plt.xlabel("Feedback")
plt.ylabel("positivefeedbackpercent")
plt.show()
#here aswell there is so much variations barely a line

#plot the pair plot of TotalClaims,BMI,Age,Children,YearsHealthInsurance in data dataframe.
sns.pairplot(data = data, vars=['price','feedback','positivefeedbackpercent'])
plt.show()



####Correlation Matrix and Heat Map

# Creating a matrix 
data[['price','feedback','positivefeedbackpercent']].corr()

#plot the correlation TotalClaims,BMI,Age,Children,YearsHealthInsurance in data dataframe.
sns.heatmap(data[['price','feedback','positivefeedbackpercent']].corr(), annot=True, cmap = 'Reds')
plt.show()
#after we seeing the heatmap they all have very poor coorelation to each other



#####Categorical - Categorical Analysis


#try to group the data to see how much has the above median price based on shipping type
shippingprice = data.groupby('shippingtype')['aboveMedianPrice'].mean()

#try to group the data to see how much has the Top rated price based on country origin
countrytoprated = data.groupby('country')['topRatedType'].mean()

#try to group the data to see how much has the above median price based on condition

conditionprice = data.groupby('condition')['aboveMedianPrice'].mean()

#try to group the data to see how much has the above median price based on country origin
countryprice = data.groupby('country')['aboveMedianPrice'].mean()



shippingprice.plot.bar()
plt.title("Shipping and Median Price")
plt.show()
#here we can see that if an item shipping type is not specified its 100% above median price while when its free 
#there is 60% of data thats free and if its flat its about 50% to have above median price
#this can be very vital information for our model



countrytoprated.plot.bar()
plt.title("Country and Top Rated")
plt.show()
#now based on country we can see that CN have the highest top rated sellers followwing by HK,GB,IE,and DE


conditionprice.plot.bar()
plt.title("Condition and Median Price")
plt.show()
#if we see the conditions and the median price this can give us a lot of insight 
#the new conditions show us that it will mostly have above median price wjere used and refurbished have lower
#data with above median price


countryprice.plot.bar()
plt.title("Country and Median Price")
plt.show()
#this is also a good insight to see, most of above median data is from HK followed by CN and IE,GN,DE,then AU



#Categorical - Numerical Analysis

#group by coutry based on price 
data.groupby('country')['price'].mean()
#
# =============================================================================
# country
# AU    189.261706
# CN    394.198667
# DE    386.340000
# GB    314.907498
# HK    358.000000
# HU    253.600000
# IE    324.554079
# =============================================================================
data.groupby('country')['price'].median()
# =============================================================================
# country
# AU    220.10
# CN    368.66
# DE    269.86
# GB    312.80
# HK    358.00
# HU    253.60
# IE    349.00
# =============================================================================
# we can see here there is a few differences between the median and mean of prices grouped from country

#examine with box plot
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.country, data.price)
plt.show()
#here we can see that DE has the biggest IQR and most expsensive listing followed by CN, IE,GB..
#we can see that the cheapest listing come from ireland as outlier..


#group by shipping type based on price 
data.groupby('shippingtype')['price'].mean()
# =============================================================================
# shippingtype
# Flat                                   314.635766
# FlatDomesticCalculatedInternational    189.261706
# Free                                   323.618477
# FreePickup                             280.000000
# NotSpecified                           323.930000
# =============================================================================


data.groupby('shippingtype')['price'].median()
# =============================================================================
# shippingtype
# Flat                                   312.80
# FlatDomesticCalculatedInternational    220.10
# Free                                   330.18
# FreePickup                             280.00
# NotSpecified                           323.93
# =============================================================================
#we see very litte difference apart fom the flatdomestic


#examine with box plot
figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(data.shippingtype, data.price)
plt.show()
#as we can see in the plot we can see a lot of outliers come from flat shipping type 
#this might where we want to focus our search because flat type has the most low outliers 






#group by  condition  based on price 
data.groupby('condition')['price'].mean()

# =============================================================================
# for parts or not working    130.000000
# manufacturer refurbished    264.674118
# new                         289.920000
# new other (see details)     374.218857
# seller refurbished          299.441844
# used                        310.406179
# =============================================================================
data.groupby('condition')['price'].median()
# =============================================================================
# for parts or not working    130.00
# manufacturer refurbished    269.86
# new                         356.95
# new other (see details)     325.09
# seller refurbished          290.46
# used                        315.83
# =============================================================================

#i think the data is very logical because we can see that the broken for part has the lowest median and average price and new condition has the highest price

figure(num=None, figsize=(15,8), dpi=100, facecolor='w', edgecolor='k')
b=sns.boxplot(data.condition, data.price)
plt.show()

#interesting part, even though we see new condition as the highest price but we see that new has very big IQR and it is lower than any other condition type.
 




#group by  toprated  based on price 

data.groupby('toprated')['price'].mean()
data.groupby('toprated')['price'].median()
#both of the data shows that being top rated sellers affect the price greatly (around 40 euro)


 

#examine with box plot
figure(num=None, figsize=(8,8), dpi=100, facecolor='w', edgecolor='k')
b=sns.boxplot(data.toprated, data.price)
plt.show()
#but when we see the boxplot even non top rated seller have some outliers that have higher prices than top rated sellers




###Multivariate



#Pivot table automatically uses the mean value of the price
result = pd.pivot_table(data=data, index='country', columns='condition',values='price')
print(result)


#create heat map of counrty vs condition vs price
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()
#we see that that some of the lowest price come from ireland 
#all the countries dont have all the conditions thats why some of it ha s blanks
#interestingly new phone in ireland mostly cost lower than the  used ones or refurbished
#while in other country its mosly the other way


#Pivot table automatically uses the mean value of the price
result = pd.pivot_table(data=data, index='country', columns='shippingtype',values='price')
print(result)

#create heat map of counrty vs shipping type vs price
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()
#based on the shipping type IE and GB has prices closer to the median while CN,AU have the least prices compared to median



result = pd.pivot_table(data=data, index='country', columns='toprated',values='price')
print(result)


#create heat map of counrty vs top rated vs price
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()

#from previous exploration we know that most of the sellers is top rated, 
#for non top rated ones we have AU with only non top rated sellers with low prices too,
#and then we have IE and HU with lower prices
#and GB and DE with higher prices even if their seller is not top rated





#step 8
#Produce scatter and correlation plots - Pay particular attention to the Response variable
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(data)

#Correlations
figure(num=None, figsize=(20,20), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(data.corr(), annot=True, cmap = 'Reds')
plt.show()


 






#because we will predict based on classification i make the target prediction as aboveMedianPrice 
#and for x i removed the price itself and put everything thats available from the data (except the string values)
x = data[['feedback', 'positivefeedbackpercent',
         'topRatedType',
       'IEtype', 'GBtype', 'CNtype', 'DEtype', 'AUtype', 'HKtype', 'Free', 'Flat',
       'FreePickup', 'FlatDomesticCalculatedInternational',
       'seller refurbished', 'new', 'new other', 'manufacturer refurbished']] #pandas dataframe

y = data[ 'aboveMedianPrice'] #Pandas series



#Splitting the Data Set into Training Data and Test Data
from sklearn.model_selection import train_test_split

#split train 66.7%, test 33.3%. Note that if run this more than once will get different selection which can lead to different model particulalry for small datasets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333)

#Check size of training and test sets
print(len(y_train)) #
print(len(y_test)) #


#########Modelling - Step 2: Model Selection
#Simply apply the model - no need for multiple steps.

#Use Decision Tree Model
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
modelDecisionTree = DecisionTreeClassifier(max_depth=6)

# Train Decision Tree Classifer
modelDecisionTree = modelDecisionTree.fit(x_train,y_train)


#Graph the decision tree to understand how it is predicting the outome
#Examine the Decision Tree
from sklearn import tree
plt.figure(figsize=(100,20))
tree.plot_tree(modelDecisionTree)
tree.plot_tree(modelDecisionTree, 
              feature_names=x_train.columns, 
              class_names=['Did not survive', 'Survived'], 
              filled=True, 
              rounded=True, 
              fontsize=14)
plt.savefig('tree.png')
plt.show()




#########Modelling - Step 3: Model Evaluation Based on TEST set.


#Predict the response for test dataset
predictions = modelDecisionTree.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)
#Pred 0 Pred 1
#[[194  49]
 #[ 52 205]]



#Check numbers
#numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy

precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

#Accuracy of  0.798





#########Modelling - Step 2: Model Selection
#Alernative Model using Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#Select the model using the training data
model.fit(x_train, y_train)

#Can explore coefficients but for classification we apply to test set to measure how good it is
print(model.coef_)
print(model.intercept_)



#########Modelling - Step 3: Model Evaluation Based on TEST set.

#Find the predicted values from the test set
predictions = model.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)
#[[208  35]
#[226  31]]

#Check numbers
numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# =============================================================================
# Accuracy: 0.478
# Error Rate: 0.522
# Precision: 0.4696969696969697
# Recall: 0.12062256809338522 
# =============================================================================


#########Modelling - Step 2: Model Selection
#Alernative Model using Neural Networks
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

#Select the model using the training data
model.fit(x_train, y_train)


#########Modelling - Step 3: Model Evaluation Based on TEST set.

#Find the predicted values from the test set
predictions = model.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)
# =============================================================================
# [[ 23 220]
#  [ 10 247]]  
# =============================================================================
#Check numbers
numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# =============================================================================
# 
# Accuracy: 0.54
# Error Rate: 0.45999999999999996
# Precision: 0.5289079229122056
# Recall: 0.96108949416342
# 
# =============================================================================


# training the model on training set 
from sklearn.naive_bayes import GaussianNB 
model = GaussianNB() 
model.fit(x_train, y_train) 
  
#Find the predicted values from the test set
predictions = model.predict(x_test)
  

#########Modelling - Step 3: Model Evaluation Based on TEST set.

#Find the predicted values from the test set
predictions = model.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)
# =============================================================================
# [[ 92 151]
#  [ 29 228]]
# =============================================================================
# Check numbers
numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))


# =============================================================================
# Accuracy: 0.64
# Error Rate: 0.36
# Precision: 0.6015831134564644
# Recall: 0.8871595330739299
# =============================================================================

#https://benalexkeen.com/support-vector-classifiers-in-python-using-scikit-learn/#:~:text=%20Support%20Vector%20Classifiers%20in%20python%20using%20scikit-learn,SVC%20classifier.%20The%20SVC%20classifier%20comes...%20More%20

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1, gamma=0.001, random_state=1)



#Select the model using the training data
model.fit(x_train, y_train)


#########Modelling - Step 3: Model Evaluation Based on TEST set.

#Find the predicted values from the test set
predictions = model.predict(x_test)

#Calculate performance metrics Accuracy, Error Rate, Precision and Recall from the confusion matrix
from sklearn.metrics import confusion_matrix

confusionMatrix = confusion_matrix(y_test, predictions)
print(confusionMatrix)



# =============================================================================
# [[178  65]
#  [ 24 233]]
# =============================================================================


#Check numbers
numberSurvivedTest = y_test.value_counts()

accuracy = (confusionMatrix[0,0]+confusionMatrix[1,1])/len(predictions)
errorRate = 1- accuracy
precision = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[0,1])
recall = (confusionMatrix[1,1])/(confusionMatrix[1,1] + confusionMatrix[1,0])
print("Accuracy: " + str(accuracy))
print("Error Rate: " + str(errorRate))
print("Precision: " + str(precision))
print("Recall: " + str(recall))

# =============================================================================
# Accuracy: 0.822
# Error Rate: 0.17800000000000005
# Precision: 0.7818791946308725
# Recall: 0.9066147859922179
# =============================================================================
#with svc we has the higest accuracy so we will go with this model,
#n 

plt.clf()
plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(confusionMatrix[i][j]))
plt.show()
