"""
VGG 16
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, Conv3D, MaxPool3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
import numpy as np
import csv

Y =  np.array([[]])
with open('landmark.csv') as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=' ')

    datalength = 0 # UNKNOWN FOR NOW

    X = np.zeros((datalength, 224,224, 42))
    row_count = -1

    for row in csv_reader:  # for every image

        #if starting row
        if (row.split())[2] == "video":
            if row_count != -1:
                #appendtothing
                X[row_count] = x_new;
            row_count+=1
            x_new = np.zeros((224,224,42))
            z_count = 0;
            y_count = 0;
            Y_value = (row.split())[4];
            Y = np.append(Y, [Y_value])

        #if new slice
        elif (row.split())[2] == "split":
            z_count+=1
            y_count = 0;
            continue

        else:
            for num in range(224):
                x_new[num, y_count, z_count] = row[num]
            y_count+=1;


indexs = list(range(0, datalength, 5)) # Every 5th letter is a validation image (80-20 split)

vallength = len(indexs) # Length of validation set
X_val = np.zeros((vallength, 42)) # Initialise numpy array first
Y_val = np.array([[]])


# Add the correct validation data to X_Val and Y_val
for num in range(vallength):
    index = indexs[num]
    X_coords = X[index,:]

    for coordnum in range(len(X_coords)):
        X_val[num,coordnum] = X_coords[coordnum]

    Y_val = np.append(Y_val, Y[index])


# del indexes from Y and X (removes val data from train data)
Y_train = np.delete(Y, indexs)
X_train = np.delete(X, indexs, axis = 0)


Y_train_cat = to_categorical(Y_train, 29)
Y_val_cat = to_categorical(Y_val, 29)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train_cat)).batch(30)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val,Y_val_cat)).batch(30)



"""
Building a CNN
"""
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience = 2)

model = Sequential([Conv3D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)),
                    MaxPool3D(pool_size = (2,2), strides = 2),
                    Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (224,224,3)), # 224,224 is size of image, 3 since RGB
                    MaxPool2D(pool_size = (2,2), strides = 2),
                    Conv2D(filters=64, kernel_size = (3,3), activation = 'relu', padding = 'same'), # Increase filters
                    MaxPool2D(pool_size = (2,2), strides = 2),
                    Flatten(),
                    Dense(units = 2, activation = 'softmax')  # Softmax gives probability of outcomes
                    ])


model.compile(optimizer = Adam(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(x=train_batches, validation_data = valid_batches,
          epochs = 10, verbose = 2, callbacks = [early_stopping_monitor])

"""
Calling VGG16
"""

vgg16_model = tf.keras.applications.vgg16.VGG16()

model = Sequential()
for layer in vgg16_model.layers[:-1]:  # Keep all but last layer of VGG16 model
    model.add(layer)

for layer in model.layers:
    layer.trainable = False  # Since VGG alr trained

model.add(Dense(units=2, activation='softmax'))  # Since cats/dogs
