# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataframe = pd.read_csv('C:\\Deeps\\Machine Learning Projects\\Use Case\\Churn Modelling\\Data\\ChurnModelling.csv')
list(dataframe)

# EDA 

dataframe.dtypes
dataframe["Gender"] = dataframe["Gender"].astype('category')
dataframe["Gender"] = dataframe["Gender"].cat.codes
dataframe["Geography"] = dataframe["Geography"].astype('category')
dataframe["Geography"] = dataframe["Geography"].cat.codes

X = dataframe.iloc[:,3:].values
y = dataframe.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

# Predicting the Test set results
y_pred = model.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
score = model.score(X_test, y_test)
print(score)


