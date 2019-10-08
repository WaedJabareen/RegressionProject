#   Individual Rate  Prediction Using Regression

- A machine learning regression model that is trained on Rate-PUF dataset from the Centers for Medicare & Medicaid Services (CMS) Center for Consumer Information & Insurance Oversight (CCIIO)
- We are doing supervised learning here and our aim is to do prediction
- During our journey we'll understand the important tools needed to develop a powerful ML model
- Our aim is to play with tools like cross validation, GridSearchCV, Random Forests and Pipelines to reach our goal
- We'll evaluate the performance of each of our model using RMSE and also tune hyper parameters to further optimize our model
-  We'll validate our predictions against our test dataset and conclude our learnings


**To do an end-to-end Machine Learning project we need to do the following steps**

1.  Understand the requirements of the business.
2.	Acquire the dataset.
3.	Visualize the data to understand it better and develop our intuition.
4.	Pre-process the data to make it ready to feed to our ML model.
5.	Try various models and train them. Select one that we find best.
6.	Fine-tune our model by tuning hyper-parameters
7.	Save and load model to make prediction.

#   Problem Definition
The problem we will tackle predicting Dollar value for the insurance premium cost applicable to a non-tobacco user for the insurance plan in a rating area for 2019 (Individual Rate). These data collected from the Centers for Medicare & Medicaid Services (CMS) Center for Consumer Information & Insurance Oversight (CCIIO). The PUFs include data from states with Federally-facilitated Exchanges (FFEs). The Rate-PUF contains plan-level data on individual rates based on an eligible subscriber's age, and geographic location, and family-tier rates. 
This is a supervised, regression machine learning problem. It's supervised because we have both the features (data for each state) and the targets (Individual Rate) that we want to predict. During training, we give the model both the features and targets and it must learn how to map the data to a prediction. Moreover, this is a regression task because the target value is continuous. Let's get started!

# Installing and Starting Python SciPy
Get the Python and SciPy platform installed on your system if it is not already. I would recommend installing the free version of Anaconda that includes everything you need.
# Acquire the dataset.
```
# Pandas is used for data manipulation
import pandas as pd
data= pd.read_csv(r"C:\Machineleraning\RatingForAllState.csv")
""" 2. Lets select some features that we want to use for regression. """
dataset = data[['State','MetalLevel','RatingAreaId' ,'Age' ,'IndividualRate']]
```
# Summarize the Dataset
In this step we are going to take a look at the data a few different ways:
- Dimensions of the dataset.
- Peek at the data itself.
- Statistical summary of all attributes.
```
# descriptions
print (dataset.describe())
# data info to see data types
print (dataset.info())
""" 3. checking missing data """
print (dataset.isnull().sum())
```
Display some data
```
""" dispaly data""" 
print (dataset.head(500))
```
![alt text](https://github.com/WaedSaleh/RegressionProject/blob/master/Images/Data.png)
# Data visualization
We now have a basic idea about the data. We need to extend that with some visualizations.

We are going to look at two types of plots:

1. Univariate plots to better understand each attribute.
2. Multivariate plots to better understand the relationships between attributes.
- Univariate plots
We can also create a histogram of each input variable to get an idea of the distribution.
```
# histograms
dataset.hist()
plt.show()
```
![alt text](https://github.com/WaedSaleh/RegressionProject/blob/master/Images/DataVisualization.png)

Great! We are seeing each feature of our data-set as a histogram.
- Multivariate Plots
Now we can look at the interactions between the variables.
First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.
```
# scatter plot matrix
scatter_matrix(dataset)
plt.show()
```
![alt text](https://github.com/WaedSaleh/RegressionProject/blob/master/Images/MultivariatePlots.png)
# Data preparation
Now, we need to separate the data into the features and targets. The target is also known as the label. It is the value we want to predict, in this case, the actual individual rate and the features are all the columns the model uses to make a prediction.
```
""" get train data"""
X = dataset.drop(columns='IndividualRate')

Y = dataset['IndividualRate'].copy()
""" ROUND Rate """
Y= np.array(np.round((Y)),  
                   dtype='int')

print (Y)
```
# Pre Procssing 
Because machine learning makes good performance on numbers, so we have to encode the number values.
If we look to the data set, age is a Continues value and it is numerical, we have State, Metal level, and Rating Area which are categorical values. When we have categorical values we have to convert it to numbers.
You may need to install category_encoders library 
```
pip install category_encoders
```
First, we import the category encoders library.
```
import category_encoders as ce
```
```
"""  pre processing : So we need to encode the categoriacal variables into numbers."""

ohe = ce.OneHotEncoder(handle_unknown='ignore',use_cat_names=True)
X_train_ohe = ohe.fit_transform(X)
print (X_train_ohe)
```
# Train the model on the training data
In scikit-learn a random split into training and test sets can be quickly computed with the train_test_split helper function.
Here is what we are going to cover in this step:
1. Separate out a validation dataset.
2. Set-up the test harness to use 10-fold cross-validation.
3. Build 5 different models to predict Individual Rate
4. Select the best model.
# Separate out a validation Dataset
```
# Split-out validation dataset
validation_size = 0.25
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X_train_ohe, Y, test_size=validation_size, random_state=seed)
```
#  Evaluate Some Algorithms
Now it is time to create some models of the data and estimate their accuracy on unseen data.

We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the data are non linearly separable in some dimensions, so we are expecting generally good results.
And  it seems like non liner regssion , we will evaluaute some non liner  regession algorithms.
Let’s evaluate 3 different regession algorithms:
1. K-Nearest Neighbors Regression (KNR)
2.  Decision Tree Regressor (DTR)
3. Random Forest Regressor (RFR).

```
# Spot Check Algorithms
models = []
models.append(('KNR', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('RFR', RandomForestRegressor(n_estimators=100)))
```
Evaluate each model in turn, we will use 10-fold cross validation to estimate accuracy.

This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

```
results = []
names = []
for name, model in models: 
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
``` 

 # Select Best Model
We now have 3 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.

Running the example above, we get the following raw results:
``` 
KNR: -7717.310699 (199.568970)
DTR: -3262.630863 (189.717362)
RFR: -2056.491810 (96.323356)
``` 
# Hyperparameter Tuning the Random Forest
So based on our evaluation of multiple algorithms, we found Random Forest had a better score. 
But we're not too impressed by the results. And now it is the time to move on to model hyperparameter tuning to get better performance of our model.
In sklearn, hyperparameters are passed in as arguments to the constructor of the model classes.
# Tuning Strategies
We will use Grid Search for optimizing hyperparameters:
To use Grid Search, we make another grid based on the best values provided by random search:
Using sklearn'sGridSearchCV, we first define our grid of parameters to search over and then run the grid search.
``` 
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
``` 
Now we will use the optimized parameters in our model.
``` 
rfr = RandomForestRegressor(best_params['n_estimators'],random_state=seed, verbose=False,oob_score=True,n_jobs=-1)
``` 
To check the accuracy, we will use the cross-validation function.
``` 
# Perform K-Fold CV
scores = model_selection.cross_val_score(rfr, X_train, Y_train, cv=10, scoring='neg_mean_absolute_error')
results.append(scores)
msg = "%f (%f)" % (scores.mean(), scores.std())
print(msg)
``` 
Now the accuracy after optimization: 
-26.059689 (0.366915)
# Use model for predictions
```
rfr.fit(X_train, Y_train)
print(rfr.score(X_validation,Y_validation))
```
 Use the forest's predict method on the test data
```
predictions = rfr.predict(X_validation)
```
**Calculate the absolute errors**
```
errors = abs(predictions - Y_validation)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
```
Mean Absolute Error: 24.62 degrees.
 **Calculate mean absolute percentage error (MAPE)**
 ```
mape = 100 * (errors / Y_validation)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
```
Accuracy: 96.2 %.
Display some data after prediction 
```
df = pd.DataFrame({'Actual': Y_validation.values.flatten(), 'Predicted': predictions.flatten()})
df
```
![alt text](https://github.com/WaedSaleh/RegressionProject/blob/master/Images/PredctionData.PNG)

Visualize comparison result as a bar graph using the below script :
```
df1 = df.head(25)
print(df1)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```

![alt text](https://github.com/WaedSaleh/RegressionProject/blob/master/Images/Real.png)


# Save the model to disk
Finally  we will save the model,we save model’s parameter and coefficients i.e.; model’s weights and biases to file on the disk. We can later load the saved model’s weights and biases to make a prediction for unseen data.
Saving of data is also called as Serialization where we store an object as a stream of bytes to save on a disk
We will discuss two different ways to save and load the scikit-learn models using
- pickle
- joblib

``` 
filename = 'rate_finalized_model.sav'
pickle.dump(rfr, open(filename, 'wb'))

from sklearn.externals import joblib 
  
# Save the model as a pickle in a file 
joblib.dump(rfr, 'rate_model.pkl') 
``` 
