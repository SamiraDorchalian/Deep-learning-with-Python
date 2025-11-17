# Deep learning with python
## multy layer percaptron - ANN
### please predict the churn of customers
                                                #Part 1 : import libraries
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('C:\\Users\\ziggurat\\Desktop\\deep learning with python\\Churn_Modelling2.csv')
# print(dataset)
five_row = dataset.head()
# print(five_row)
information_data = dataset.info
# print(information_data)
statistical_info = dataset.describe()
# print(statistical_info)
                                                #Part 2 : features and label (X and Y)
y = dataset['Exited']
x = dataset[['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
# x = dataset.drop([['RowNumber', 'CustomerId', 'Surname', 'Geography', 'Exite']])
# print(x.head())
# print(y.head())
                                                #Part 3 : Pre-Processing
#### Encoding Categorical data
male = pd.get_dummies(x['Gender'], drop_first=True)
# print(male)
x = pd.concat([x, male], axis=1)
# print(x)
print(x.drop(['Gender'], axis=1))
                                                #Part 4 : Train the model (model selection)
## train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=101)
print(x_train.shape)
print(x_test)
print(y_train)
print(y_test)
## Model Multy-layer Perceptron
from keras.models import Sequential
mlp = Sequential()

from keras import layers

mlp.add(layers.Dense(units=6, activation='relu'))
mlp.add(layers.Dense(units=6, activation='relu'))
mlp.add(layers.Dense(units=1, activation='sigmoid'))

                                                #Part 5 : Training the model
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(x_train, y_train, batch_size=32, epochs=100)

print(mlp)

