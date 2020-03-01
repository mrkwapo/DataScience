# DataScience

Updates:
The "cross_validation" name is now deprecated and was replaced by "model_selection" inside the new anaconda versions.
Therefore you might get a warning or even an error if you run this line of code above.

To avoid this, you just need to replace:
from sklearn.cross_validation import train_test_split 
by
from sklearn.model_selection import train_test_split 
and you will get no warning ;)

***** Also, for those of you who run into Sklearn related deprecations regarding the imputer class and column transformer you can use the following modifications:

from sklearn.impute import SimpleImputer

missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)

missingvalues = missingvalues.fit(X[:, 1:3])

X[:, 1:3]=missingvalues.transform(X[:, 1:3])


# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values

# Encoding categorical data

# Encoding the Independent Variable

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data

from sklearn.preprocessing import LabelEncoder

y = LabelEncoder().fit_transform(y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train.reshape(-1,1))

