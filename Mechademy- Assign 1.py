#!/usr/bin/env python
# coding: utf-8

# # Assignment 1 - Used Cars Price Prediction and Evaluation(Car_price.csv)

# # Step 1 : Importing Libraries and Understanding Data

# In[70]:


import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import sklearn #This lib contains all the models
import statsmodels.api as sm #will help to display data in the form of statistics

get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import HTML
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression


# In[37]:


#Importing Car_price.csv data
car_price = pd.read_csv('Downloads/cars_price.csv')


# In[38]:


#Look at first five rows
car_price.head(5)


# In[39]:


# Looking at the last five rows
car_price.tail()


# # Lets check the Data type & Summary of the Data

# In[40]:


print(car_price.shape)
print(car_price.describe())


# In[41]:


car_price.info()


# # Step 2: Visualising Data

# In[42]:


# Let's plot a pair plot of all numerical variables in our dataframe
sns.pairplot(car_price)


# In[44]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(car_price, x_vars=['year','mileage(kilometers)','segment'], y_vars='priceUSD',
             size=7, aspect=0.7, kind='scatter')


# In[45]:


sns.boxplot(data=car_price["priceUSD"])


# In[48]:


sns.boxplot(data=car_price["year"])


# In[49]:


sns.boxplot(data=car_price["mileage(kilometers)"])


# In[50]:


#for value in colname:
q1 = car_price['priceUSD'].quantile(0.25) #first quartile value
q3 = car_price['priceUSD'].quantile(0.75) #third quartile value
iqr = q3-q1 #interquartile range
low = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
print(low)
print(high)
print(q1)
print(q3)
print(iqr)


# In[51]:


car_price.loc[car_price.priceUSD > 21100.0,'priceUSD'] = 21100.0
print(car_price.priceUSD)


# In[52]:


sns.boxplot(data=car_price["priceUSD"])


# In[53]:


q1 = car_price['year'].quantile(0.25) #first quartile value
q3 = car_price['year'].quantile(0.75) #third quartile value
iqr = q3-q1 #interquartile range
low = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
print(low)
print(high)
print(q1)
print(q3)
print(iqr)


# In[54]:


car_price.loc[car_price.year < 1980,'year'] = 1980
print(car_price.year)


# In[55]:


sns.boxplot(data=car_price["year"])


# In[56]:


q1 = car_price['mileage(kilometers)'].quantile(0.25) #first quartile value
q3 = car_price['mileage(kilometers)'].quantile(0.75) #third quartile value
iqr = q3-q1 #interquartile range
low = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
print(low)
print(high)
print(q1)
print(q3)
print(iqr)


# In[66]:


car_price.loc[car_price.mileage(kilometers) > 566402,'mileage(kilometers)'] = 566402
print(car_price.mileage(kilometers))


# In[58]:


sns.boxplot(data=car_price["mileage(kilometers)"])


# # Seperate the Numerical and Categorical variables

# In[63]:


car_price_num = car_price[['mileage(kilometers)','year','priceUSD']]


# Categorical Variables: The variables which are having category should be converted to dummy variables.
# 
# The Linear regression will only accept numbers so we will make dummy variables.

# In[64]:


car_price_dummies = pd.get_dummies(car_price[['segment']])
car_price_dummies.head()


# # Combine Numerical and Dummy variables

# In[ ]:


Combine Numerical and Dummy variables


# In[65]:


car_price_combined = pd.concat([car_price_num, car_price_dummies], axis=1)
car_price_combined.head()


# # Step 3 : Splitting the data in Training and Test set

# Using sklearn we split 70% of our data into training set and rest in test set.
# 
# Setting random_state will give the same training and test set everytime on running the code.

# In[68]:


# Putting feature variable to X
X = car_price_combined[['mileage(kilometers)','year','segment_A','segment_B','segment_C','segment_D','segment_E','segment_F','segment_J','segment_M','segment_S']]

# Putting response variable to y
y = car_price['priceUSD']


# In[69]:


#random_state is the seed used by the random number generator. It can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=100)


# # Step 4 : Performing Linear Regression

# In[71]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[72]:


# fit the model to the training data
lm.fit(X_train,y_train)


# # Step 5 : Model Evaluation

# In[73]:


# print the intercept
print(lm.intercept_)


# In[74]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# # Step 6 : Predictions

# In[75]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# # Step 7: Model Performance Metrics

# Coefficient of Determination (R square)

# In[79]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[80]:


from math import sqrt
rmse = sqrt(mse)


# In[81]:


print('Mean_Squared_Error :' ,mse)
print('Root_Mean_Squared_Error :' ,rmse)
print('r_square_value :',r_squared)


# If the value of R2 is 0.7146 then this suggests that 71.4% of the variation in Y can be explained with the help of given explanatory variables in that model. In other words, it explains the proportion of variation in the dependent variable that is explained by the independent variables.
# 
# On addition of a new variable the error is sure to decrease, thus R square always increases whenever a new variable is added to our model. This may not describe the importance of a variable

# # Adjusted R square:

# Adjusted R square will always be less than or equal to R square.
# Thus as a general thumb rule if adjusted R square increases when a new variable is added to the model, the variable should remain in the model.
# 
# If the adjusted R square decreases when the new variable is added then the variable should not remain in the mode

# # Multicollinearity
# If independent valriables Xi’s are highly correlated then |X’X| will be close to 0 and hence inverse of (X’X) will not exist or will be indefinitely large. Mathematically, which will be indefinitely large in presence of multicollinearity. Long story in short, multicollinearity increases the estimate of standard error of regression coefficients which makes some variables statistically insignificant when they should be significant.
# 
# How can you detect multicollinearity!!
# 
# Correlation Method:
# By calculating the correlation coefficients between the variables we can get to know about the extent of multicollinearity in the data.
# 
# VIF (Variance Inflation Factor) Method:
# Firstly we fit a model with all the variables and then calculate the variance inflation factor (VIF) for each variable. VIF measures how much the variance of an estimated regression coefficient increases if your predictors are correlated. The higher the value of VIF for ith regressor, the more it is highly correlated to other variables

# 
# # Checking for P-value Using STATSMODELS

# In[82]:


import statsmodels.api as sm
X_train_sm = X_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)
# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[83]:


print(lm_1.summary())


# In[85]:


plt.figure(figsize = (5,5))
sns.heatmap(car_price_num.corr(),annot = True)


# # Step 8 : Implementing the results

# In[86]:


# Check for the forst 150 observations

c = [i for i in range(1,150,1)]
fig = plt.figure()
plt.plot(c,y_test[1:150], color="blue", linewidth=2.5, linestyle="-")
plt.plot(c,y_pred[1:150], color="red",  linewidth=2.5, linestyle="-")
fig.suptitle('Actual and Predicted', fontsize=20)              # Plot heading 
plt.xlabel('year', fontsize=18)                               # X-label
plt.ylabel('priceUSD', fontsize=16)                               # Y-label


# In[89]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[90]:


X_train_final = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant.
X_train_final = sm.add_constant(X_train_final)

# create a fitted model in one line
lm_final = sm.OLS(y_train,X_train_final).fit()

print(lm_final.summary())


# # Variance Inflation Factor
# Variance inflation factor (VIF) for an explanatory variable is given 1/(1-R^2 ) .
# 
# Here, we take that particular X as response variable and all other explanatory
# 
# variables as independent variables. So, we run a regression between one of
# 
# those explanatory variables with remaining explanatory variables.

# In[91]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[92]:


# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


# In[93]:


vif.round(2)

