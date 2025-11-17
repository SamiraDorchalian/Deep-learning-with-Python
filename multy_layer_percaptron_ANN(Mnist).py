# Deep Learning
# Multy layer percaptron feed forward ANN
# Please detect the mnist Number
                                                    # import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
                                                    # Part 1: Get the Data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

import matplotlib.pyplot as plt

image_index = 53648
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')
image_index = 45219
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')
image_index = 52
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')
                                                #Part 2: Pre-processing 
                                                # (Dimension Revision)
x_train_final = x_train.reshape(-1, 28*28)/ 255.0
x_train_final.shape
x_test_final = x_test.reshape(-1, 28*28)/ 255
x_test_final.shape
                                                # To categorical (One-Hot Encoding)
from keras.utils import to_categorical
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
y_train_cat
y_train_cat[5]
                                                #Part 3: train the model(model selection): MLP-ANN
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense,Input
model.add(Input(shape=(784,)))
model.add(Dense(units=20, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
epochs= 30
batch_size=128
model.fit(x_train_final, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_test_final, y_test_cat))

