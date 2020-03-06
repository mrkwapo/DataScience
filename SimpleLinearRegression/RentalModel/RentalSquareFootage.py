#Simple Linear Regression of Rent and Square Footage in 02903 zip code

#Importing Libraries
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('SquareFootage_Data.csv')

#Creating the Matrix of Features and Dependent Variable Vector
X = dataset.iloc[:, :-1].values

#Creating the Dependent Variable Vector
y = dataset.iloc[:, 1].values

#Splitting the Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results (creating a vector of prediction values)
y_pred = regressor.predict(X_test)

#Visualizing the Training Set Results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Rent vs Square Footage (Training Set)')
plt.xlabel('Square Footage')
plt.ylabel('Rent')
plt.show()

#Visualizing the Test Set Results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Rent vs Square Footage (Test Set)')
plt.xlabel('Square Footage')
plt.ylabel('Rent')
plt.show()
