#Data Processing

#Importing the libraries
import numpy as np #needed to do math
import matplotlib.pyplot as plt # used to plot charts
import pandas as pd # used to import and manage datasets


#Importing the dataset after setting working directory by saving this file in the same folder as the dataset file
dataset = pd.read_csv('Data.csv')
#creating matrix of features with independent variables
X = dataset.iloc[:, :-1].values #taking the rows and all columns except the last one which is the dependent variable 'purchased' column
#creating the dependent variable vector
y = dataset.iloc[:, 3].values # 3 is the index of the column named purchased in the dataset


#Handling missing data
#sklearn is used to make machine learning models
#.preprocessing is a library that handles any datasets that need preprocessing
#imputer class helps to handle missing data
from sklearn.preprocessing import Imputer
#now that we have imported the class we need to create an object of this class
imputer = Imputer(missing_values = 'NaN', strategy ='mean', axis = 0)
#now we need to use imputer on the columns of X
# colon means all the column and 1 is the second index which contains ages
#and we use three to capture index 2
imputer = imputer.fit(X[:, 1:3])
#we need to replace the missing data of the metric X by the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3]) # this is the method that is going to replace the missing data by the mean of the column

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [0]              # The column(s) to be applied on.
         )
    ]
)

#creating first object of the LabelEncoder class
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #encoding the column holding the countries
#to prevent the computer from thinking one country is greater than the other because
#the new encoded value uses numbers, we have to use dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

