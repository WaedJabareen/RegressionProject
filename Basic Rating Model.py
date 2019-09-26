# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

""" import libaries""" 

import matplotlib.pyplot  as plt
import numpy as np 
import pandas as pd
import category_encoders as ce
import sklearn as sklrn
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OneHotEncoder

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
"""Non linear model"""
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
""" linear model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
"""

""" 1. Importing the Data Set """
data= pd.read_csv(r"C:\Users\waed Jabareen\Desktop\Machineleraning\RatingForAllState.csv")

""" 2. Lets select some features that we want to use for regression. """
dataset = data[['State','MetalLevel','RatingAreaId' ,'Age' ,'IndividualRate']]


""" Summarize the Dataset """
# descriptions
print (dataset.describe())
# data info to see data types
print (dataset.info())
""" 3. checking missing data """
print (dataset.isnull().sum())

""" 4. pre processing : So we need to encode the categoriacal variables into numbers."""
""" for rating id i can use replace, planid using label encoding """


"""dataset.iloc[:, 1] = dataset.RatingAreaId.str.replace('Rating Area ', '')"""

""" dispaly data""" 
print (dataset.head(500))

"""5. data visualization"""

# histograms
dataset.hist()
plt.show()

# box and whisker plots


# scatter plot matrix
scatter_matrix(dataset)
plt.show()
""" get train data"""
X = dataset.drop(columns='IndividualRate')

Y = dataset['IndividualRate'].copy()
""" ROUND Rate """
Y= np.array(np.round((Y)),  
                   dtype='int')

print (Y)

ohe = ce.OneHotEncoder(handle_unknown='ignore',use_cat_names=True)
X_train_ohe = ohe.fit_transform(X)
print (X_train_ohe)
"""print (list(X_train_ohe.columns.values))"""
"""  Evaluate Some Algorithms"""
"""Segregate Independent Variables and Dependent Variables"""

"""We will split the loaded dataset into two,
 80% of which we will use to train our models 
 and 20% that we will hold back as a validation dataset.
 """
# Split-out validation dataset
validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_train_ohe, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
"""scoring = 'accuracy'"""
scoring = 'neg_mean_squared_error'


# Spot Check Algorithms
models = []
models.append(('KNR', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor(n_estimators=100)))


# evaluate each model in turn
results = []
names = []
for name, model in models: 
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
  
