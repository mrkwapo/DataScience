
#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the Dataset
dataset = pd.read_csv('Academic_Data.csv')

#Create the matrix of features and Dependent Variables vector
X = dataset.iloc[:, :-1].values
#creating the dependent variable vector
y = dataset.iloc[:, 1].values


#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test Set Results (GPAs)
y_pred = regressor.predict(X_test) #predictions of the test set

#Visualising the Training Set Results
plt.scatter(X_train, y_train, color = 'red') #real values/ observation points of the training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  #predictions trained by X and y train
plt.title('SAT Score vs GPA (Training Set)')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.show()

#Visualising the Test Set Results
plt.scatter(X_test, y_test, color = 'red') #observation points of the test set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')  #same prediction line from training results
plt.title('SAT Score vs GPA (Test Set)')
plt.xlabel('SAT Score')
plt.ylabel('GPA')
plt.show()
