# Emotion_ANN.ipynb
# import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data

from google.colab import drive

drive.mount("/content/drive/")

# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/drive/MyDrive"

data = pd.read_csv('emotions.csv')

data.head()

data['label'].value_counts()

data.info()

label_maping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}

data['label'] = data['label'].replace(label_maping)

data.head()

X = data.drop('label', axis=1)
y = data['label']

X.head()

y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

print('Shape X_Train:', X_train.shape)
print('Shape y_Train:', y_train.shape)
print('Shape X_Test:', X_test.shape)
print('Shape y_Test:', y_test.shape)

# Define Model: ANN

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model_ann = Sequential()
model_ann.add(Dense(64, activation='relu', input_shape=(2548,)))
model_ann.add(Dense(32, activation='relu'))
model_ann.add(Dense(3, activation='softmax'))

model_ann.summary()

# Compile the Model

model_ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the Model

model_ann.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

# Prediction

y_pred_proba = model_ann.predict(X_test)

y_pred_proba[10]

predict_labels = np.argmax(y_pred_proba, axis=1)

y_pred = predict_labels

y_pred[10]

# Results

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

print(classification_report(y_test, y_pred))

