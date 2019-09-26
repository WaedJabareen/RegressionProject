# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:19:04 2019

@author: waed Jabareen
"""

""" import main libaries""" 
import matplotlib.pyplot  as plt
import numpy as np 
import pandas as pd
import category_encoders as ce
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.model_selection import  GridSearchCV
from sklearn import metrics
import pickle
"""Non linear model"""
from sklearn.ensemble import RandomForestRegressor


""" 1. importing the Data Set """
data= pd.read_csv(r"C:\Users\waed Jabareen\Desktop\Machineleraning\RatingForAllState.csv")
""" 2. select the features that we want to use for regression. """
dataset = data[['State','MetalLevel','RatingAreaId' ,'Age' ,'IndividualRate']]
"""  3. summarize the Dataset """
#shap 
print (dataset.shape)
# descriptions
print (dataset.describe())
# data info to see data types
print (dataset.info())

""" 4. dispaly data""" 
print (dataset.head(500))

"""5. data visualization"""

# histograms
dataset.hist()
plt.show()
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

""" 6. get train data"""
# Remove the labels from the features
X = dataset.drop(columns='IndividualRate')
# Labels are the values we want to predict
Y = dataset['IndividualRate'].copy()
# display data
print (Y)
print (X)
""" 7. perprocssing : encode the categoical values""" 
ohe = ce.OneHotEncoder(handle_unknown='ignore',cols=['State','MetalLevel','RatingAreaId'])
X_train_ohe = ohe.fit_transform(X)
print (X_train_ohe)

"""
 We will split the loaded dataset into two,
75% of which we will use to train our models 
 and 25% that we will hold back as a validation dataset.
 """
# Split-out validation dataset
validation_size = 0.25
seed =42
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_train_ohe, Y, test_size=validation_size, random_state=seed)

scoring = 'neg_mean_squared_error'
results = []

# Perform Grid-Search
gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=10, scoring='neg_mean_squared_error', verbose=0,n_jobs=-1)
    
grid_result = gsc.fit(X_train, Y_train)
best_params = grid_result.best_params_
print(best_params)

rfr = RandomForestRegressor(best_params['n_estimators'],random_state=seed, verbose=False,oob_score=True,n_jobs=-1)
# Perform K-Fold CV
scores = model_selection.cross_val_score(rfr, X_train, Y_train, cv=10, scoring='neg_mean_absolute_error')
results.append(scores)
msg = "%f (%f)" % (scores.mean(), scores.std())
print(msg)
# Use model for predictions
rfr.fit(X_train, Y_train)
print(rfr.score(X_validation,Y_validation))
# Use the forest's predict method on the test data
predictions = rfr.predict(X_validation)
# Calculate the absolute errors
errors = abs(predictions - Y_validation)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / Y_validation)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

df = pd.DataFrame({'Actual': Y_validation.values.flatten(), 'Predicted': predictions.flatten()})
df
# visualize comparison result as a bar graph using the below script :
df1 = df.head(25)
print(df1)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, predictions)))

# save the model to disk
filename = 'rate_finalized_model.sav'
pickle.dump(rfr, open(filename, 'wb'))

from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(rfr, 'rate_model.pkl') 

 