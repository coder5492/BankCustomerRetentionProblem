# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv("Churn_Modelling.csv")

#Dropping columns which are not needed
columns_not_needed = ['RowNumber','CustomerId', 'Surname']
dataset = dataset.drop(columns_not_needed, axis=1)

#Replacing the Categorical Data with Dummy variables
columns_with_categorical_data = ['Geography', 'Gender']
dataset = pd.get_dummies(dataset, columns = columns_with_categorical_data, drop_first = True)

#Splitting it into X and y
X = dataset.iloc[:, dataset.columns != 'Exited'].values
Y = dataset.iloc[:, 8:9].values

#Splitting the data into training and test set
from sklearn.cross_validation import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feauture Scaling
from sklearn.preprocessing import StandardScaler
sc      = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test  = sc.transform(X_Test)

#Importing keras library
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#CompliliNg the ANN
classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics=['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_Train,Y_Train, batch_size = 10, nb_epoch = 100)

#Making the prediction
Y_Pred = classifier.predict(X_Test)

#Changing the probability to true or false
Y_Pred = Y_Pred > 0.5

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(Y_Test, Y_Pred)
print(cf_matrix)

