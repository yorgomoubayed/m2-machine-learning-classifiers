#!/usr/bin/env python
# coding: utf-8

##############################
# Importing Python libraries #
##############################

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

######################
# Importing datasets #
######################

data=pd.read_csv('data.csv') # dataset with objects (samples) and relative features (genes)
labels=pd.read_csv('labels.csv') # dataset with objects (samples) and relative labels (tumor class)

####################
# Data exploration #
####################

# display first 5 rows of each dataset
data.head(5)
labels.head(5)

# display total number of rows and columns in each dataset
data.shape
labels.shape

# statistical description of the dataset with objects and relative features
data.describe()

# understand further the variabale to predict 
labels['Class'].value_counts()

# Out of 801 samples, 300 are BRCA tumors, 146 are KIRC tumors, 141 are LUAD tumors, 136 are PRAD tumors and 78 are COAD tumors.

######################
# Data preprocessing #
######################

# store the feature sets into the X variable and the series of corresponding labels into the Y variable
X=data.iloc[:,1:]
Y=labels.iloc[:,1]

### Missing value ratio

# checking and saving missing values in a variable
missing_values = data.isnull().sum()/len(data)*100

# saving column names into a variable
variables = data.columns
variable = [ ]

for i in range(0, 20532):
    if a[i]<=20:   # setting the threshold as 20%
        variable.append(variables[i])

### Converting categorical data
# Categorical data are variables that contain label values rather than numeric values.The number of possible values is often limited to a fixed set. Here, we use LabelEncoder to label the categorical data. LabelEncoder is the part of SciKitLearn library in Python and used to convert categorical data, or text data, into numbers, which the predictive models can better understand.

# convert categorical data (BRCA, KIRC, LUAD, PRAD, COAD) into numbers
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

### Splitting the datasets

# The data is usually split into training data and test data. The training set contains a known output and the model learns on this data in order to be generalized to other data later on. The test dataset is employed to test the model’s prediction on this subset. We will do this using SciKit-Learn library in Python using the train_test_split method.

# split the dataset into 70% train and 30% test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=0)

### Feature scaling

# PCA performs best with a normalized feature set. We will perform standard scalar normalization to normalize the feature set. We will use StandardScaler method from SciKit-Learn Python library.

# standard scalar normalization to normalize the features (PCA performs best with normalized features)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Dimensionality reduction using PCA
# Principal component analysis (PCA), is a statistical technique to convert high dimensional data to low dimensional data by selecting the most important features that capture maximum information about the dataset. PCA depends only upon the feature set and not the label data. Therefore, PCA can be considered as an unsupervised machine learning technique. Performing PCA using Scikit-Learn is a three-step process.

# initialize the PCA class
pca = PCA(n_components=1)

# call the fit method by passing the feature set
X_train = pca.fit_transform(X_train) 

# call the transform method by passing the feature set
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

###################################
# Training and making predictions #
###################################

### Logistic regression

classifier_1= LogisticRegression(C=1e70)

# train the model
classifier_1.fit(X_train, Y_train)

# test the model using the test dataset
Y_pred1 = classifier_1.predict(X_test)


### Nearest neighbor

classifier_2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_2.fit(X_train, Y_train)
Y_pred2 = classifier_2.predict(X_test)

### SVM

classifier_3 = SVC(kernel = 'linear', random_state = 0)
classifier_3.fit(X_train, Y_train)
Y_pred3 = classifier_3.predict(X_test)

### Kernel SVM 

classifier_4 = SVC(kernel = 'rbf', random_state = 0)
classifier_4.fit(X_train, Y_train)
Y_pred4 = classifier_4.predict(X_test)

### Naive Bayes 

classifier_5 = GaussianNB()
classifier_5.fit(X_train, Y_train)
Y_pred5 = classifier_5.predict(X_test)

### Decision tree algorithm 

classifier_6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_6.fit(X_train, Y_train)
Y_pred6 = classifier_6.predict(X_test)


### Random forest classification 

classifier_7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_7.fit(X_train, Y_train)
Y_pred7 = classifier_7.predict(X_test)

################################
# Model performance evaluation #
################################

### Logistic regression 

print (accuracy_score(Y_test, Y_pred1))
confusion_matrix = pd.crosstab(Y_test, Y_pred1, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### Nearest neighbor

print (accuracy_score(Y_test, Y_pred2))
confusion_matrix = pd.crosstab(Y_test, Y_pred2, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### SVM 

print (accuracy_score(Y_test, Y_pred3))
confusion_matrix = pd.crosstab(Y_test, Y_pred3, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### Kernel SVM 

print (accuracy_score(Y_test, Y_pred4))
confusion_matrix = pd.crosstab(Y_test, Y_pred4, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### Naive Bayes

print (accuracy_score(Y_test, Y_pred5))
confusion_matrix = pd.crosstab(Y_test, Y_pred5, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### Decision tree algorithm 

print (accuracy_score(Y_test, Y_pred6))
confusion_matrix = pd.crosstab(Y_test, Y_pred6, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

### Random forest classification 

print (accuracy_score(Y_test, Y_pred7))
confusion_matrix = pd.crosstab(Y_test, Y_pred7, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
