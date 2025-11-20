                                                    # import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.datasets import mnist
                                                    # Part 1: Get the Data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

image_index = 505
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')
image_index = 1100
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')
image_index = 52
plt.imshow(x_train[image_index], cmap='gray')
print('label:', y_train[image_index])
print('---------------')

# x_train_final = x_train / 255
# x_train_final.shape
# x_test_final = x_test / 255
# x_test_final.shape

# first method
# x_train_final = x_train.reshape(-1,28, 28, 1)/ 255
# x_train_final.shape
# x_test_final = x_test.reshape(-1, 28, 28, 1)/ 255
# x_test_final.shape
#second_moethod
x_train_final = np.expand_dims(x_train, axis= -1) / 255
x_train_final.shape
x_test_final = np.expand_dims(x_test, axis= -1) / 255
x_test_final.shape
                                                # To categorical (One-Hot Encoding)
from keras.utils import to_categorical
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)
y_train_cat
y_train_cat[5]
y_train[5] 
                                                # CNN
from  keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

model_css = Sequential()
model_css.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model_css.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu'))
model_css.add(MaxPool2D(pool_size=(2,2)))
model_css.add(Flatten())
model_css.add(Dense(16, activation='relu'))
model_css.add(Dense(10, activation='softmax'))

model_css.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_css.summary()

model_css.fit(x_train_final, y_train_cat, batch_size=128, epochs=2, verbose=1, validation_data=(x_test_final, y_test_cat))
