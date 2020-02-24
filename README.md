# DataScience

To handle the FutureWarning and deprecations use the following:

http://datascienceopen.com/how-to-use-the-onehotencoder-directly/

Instead of that we can just use OneHotEncoder as shown below.

1
2
3
4
5
6
7
8
9
10
11
from sklearn.preprocessing import OneHotEncoder
transformer = ColumnTransformer(
    transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = transformer.fit_transform(X.tolist())

https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
