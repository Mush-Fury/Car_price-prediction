#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 -Propulsion Plants Decay Evaluation using propulsion.csv

# Data contains compressor & turbine data from the propulsion system,we have to find the predicted columns for GT Compressor decay state coefficient & GT Turbine decay state coefficient.

# # Solution Approach:

# As we need to predict GT Compressor decay state coefficient & GT Turbine decay state coefficient which are continuous in nature,Linear Regression can be used for prediction

# # Step 1 : Importing Libraries and Understanding Data

# In[84]:


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


# In[87]:


#Importing (propulsion.csv) data
propulsion = pd.read_csv('Downloads/propulsion.csv')


# In[88]:


#Look at first five rows
propulsion.head()


# # Lets check the Data type & Summary of the Data

# In[89]:


print(propulsion.shape)
print(propulsion.describe())


# In[90]:


propulsion.info()


# # Step 2: Visualising Data

# In[91]:


# Let's plot a pair plot of all numerical variables in our dataframe
sns.pairplot(propulsion)


# In[92]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(propulsion, x_vars=['GT Compressor outlet air temperature (T2) [C]','GT Compressor outlet air pressure (P2) [bar]','Fuel flow (mf) [kg/s]'], y_vars='GT Compressor decay state coefficient.',
             size=7, aspect=0.7, kind='scatter')


# # Step 3: Exploratory data analysis

# In[96]:


sns.boxplot(data=propulsion["GT Turbine decay state coefficient."])


# In[97]:


sns.boxplot(data=propulsion["GT Compressor decay state coefficient."])


# In[98]:


sns.boxplot(data=propulsion["GT Compressor outlet air temperature (T2) [C]"])


# In[99]:


sns.boxplot(data=propulsion["GT Compressor outlet air pressure (P2) [bar]"])


# In[100]:


sns.boxplot(data=propulsion["Fuel flow (mf) [kg/s]"])


# # Step 4 : Splitting the data in Training and Test set

# In[101]:


#create X and Y
X = propulsion[['GT Compressor outlet air temperature (T2) [C]','GT Compressor outlet air pressure (P2) [bar]','Fuel flow (mf) [kg/s]']]
Y = propulsion['GT Compressor decay state coefficient.']


# In[102]:


sns.distplot(Y,hist=True)


# In[103]:


X.hist(bins=20)
#Chebyshev's principle


# In[104]:


from scipy.stats import skew
propulsion_num_skew = X.apply(lambda x:skew(x.dropna()))
propulsion_num_skewed = propulsion_num_skew[(propulsion_num_skew > .75)| (propulsion_num_skew <-.75)]

print(propulsion_num_skew)
print(propulsion_num_skewed)
#import numpy as np
#apply log + 1 transformation for all numeric features with skewnes over .75
#X[data_num_skewed.index] = np.log1p(X[data_num_skewed.index])


# In[105]:


import seaborn as sns

corr_df=X.corr(method="pearson")
print(corr_df)

sns.heatmap(corr_df,vmax=1.0,vmin=-1.0,annot=True)


# In[106]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vif_df = pd.DataFrame()
vif_df['features']= X.columns
vif_df['VIF Factor'] = [vif(X.values,i)for i in range(X.shape[1])]
vif_df.round(2)


# In[107]:


from sklearn.model_selection import train_test_split

#split the data into test and train
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=10)


# In[108]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)

#print intercept an coefficients
print(lm.intercept_)
print(lm.coef_)


# In[109]:


#pair the feature names with the coefficients
print(list(zip(X.columns,lm.coef_)))


# In[110]:


X1 = 100
X2 = 100
X3 = 100
y_preda = 1.623040680604697+(-0.00145432 *X1) +(0.02910604 *X2)+(-0.09993668 *X3)
print(y_preda)


# In[111]:


print(X_test)


# # Step 6 : Predictions

# In[112]:


Y_pred = lm.predict(X_test)
print(Y_pred)


# In[113]:


new_df = pd.DataFrame()
new_df = X_test 

new_df["Actual GT Compressor decay state coefficient."]=Y_test
new_df['Predicted GT Compressor decay state coefficient.']=Y_pred
print(new_df)


# # Step 6 : Model Performance Metrics

# In[115]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
r_squared = r2_score(Y_test, Y_pred)


# In[116]:


from math import sqrt
rmse = sqrt(mse)


# In[117]:


print('Mean_Squared_Error :' ,mse)
print('Root_Mean_Squared_Error :' ,rmse)
print('r_square_value :',r_squared)


# # Statistical output

# In[119]:


import statsmodels.api as sm
X_train_sm = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)

# create a fitted model in one line
lm_1 = sm.OLS(Y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[120]:


print(lm_1.summary())


# # Now lets find for GT Turbine decay state coefficient.

# # Similar steps from 3,4,5,6

# In[121]:


#create X and Y
X = propulsion[['GT Compressor outlet air temperature (T2) [C]','GT Compressor outlet air pressure (P2) [bar]','Fuel flow (mf) [kg/s]']]
Y = propulsion['GT Turbine decay state coefficient.']


# In[122]:


sns.distplot(Y,hist=True)


# In[123]:


X.hist(bins=20)
#Chebyshev's principle


# In[124]:


import seaborn as sns

corr_df=X.corr(method="pearson")
print(corr_df)

sns.heatmap(corr_df,vmax=1.0,vmin=-1.0,annot=True)


# In[125]:


from sklearn.model_selection import train_test_split

#split the data into test and train
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=10)


# In[126]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)

#print intercept an coefficients
print(lm.intercept_)
print(lm.coef_)


# In[127]:


#pair the feature names with the coefficients
print(list(zip(X.columns,lm.coef_)))


# In[128]:


X1 = 100
X2 = 100
X3 = 100
y_preda = 0.9811028908201898 +(1.49012727e-05 *X1) +(-3.11571242e-04 *X2)+(8.77455198e-04 *X3)
print(y_preda)


# In[129]:


print(X_test)


# In[130]:


Y_pred = lm.predict(X_test)
print(Y_pred)


# In[131]:


new_df = pd.DataFrame()
new_df = X_test 

new_df["Actual GT Turbine decay state coefficient."]=Y_test
new_df['Predicted GT Turbine decay state coefficient.']=Y_pred
print(new_df)


# In[132]:


import statsmodels.api as sm
X_train_sm = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)

# create a fitted model in one line
lm_1 = sm.OLS(Y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[133]:


print(lm_1.summary())

