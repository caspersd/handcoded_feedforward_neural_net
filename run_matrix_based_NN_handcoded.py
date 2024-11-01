#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:43:36 2024

@author: davidcaspers
"""


from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
ytrn = to_categorical(y_train)

# Save the full-size dataset
np.save("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/train_images_full.npy", x_train)
np.save("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/test_images_full.npy", x_test)
np.save("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/train_labels_vector.npy", ytrn)
np.save("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/train_labels.npy", y_train)
np.save("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/test_labels.npy", y_test)

# Downsize images to 14x14
xtrn = np.zeros((60000, 14, 14), dtype="float32")
xtst = np.zeros((10000, 14, 14), dtype="float32")

for i in range(60000):
    img = Image.fromarray(x_train[i])
    xtrn[i, :, :] = np.array(img.resize((14, 14), Image.BILINEAR))


for i in range(10000):
    img = Image.fromarray(x_test[i])
    xtst[i, :, :] = np.array(img.resize((14, 14), Image.BILINEAR))


# Save the downsampled images
np.save("dataset/train_images_small.npy", xtrn)
np.save("dataset/test_images_small.npy", xtst)

#####
# Simple Test
#####

import tracemalloc

tracemalloc.start()
 
from matrix_based_NN_handcoded import *

# Load, reshape, and scale the data
x_train = np.load("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/train_images_small.npy")
x_test  = np.load("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/test_images_small.npy")
y_train = np.load("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/train_labels_vector.npy")
y_test  = np.load("/Users/davidcaspers/Documents/Personal Projects/Hand_Coded_Neural_Networks/dataset/test_labels.npy")

# Flatten and scale pixel values
x_train = x_train.reshape(x_train.shape[0], 1, 14 * 14) / 255.0
x_test = x_test.reshape(x_test.shape[0], 1, 14 * 14) / 255.0

# Build the network
net = NN(verbose=True)
net.add(fully_connected(14*14, 100))  # Input layer -> Hidden layer 1
net.add(activation_layer())            # Activation for hidden layer 1
net.add(fully_connected(100, 50))      # Hidden layer 1 -> Hidden layer 2
net.add(activation_layer())            # Activation for hidden layer 2
net.add(fully_connected(50, 10))       # Hidden layer 2 -> Output layer
net.add(activation_layer())            # Activation for output layer


# Train the network with suitable parameters
net.fit(x_train, y_train, eta=1.0, epochs=1000, batchsize=5000)

# Build the confusion matrix using the test set predictions
out = net.predict(x_test)
cm = np.zeros((10, 10), dtype="uint32")
for i in range(len(y_test)):
    cm[y_test[i], np.argmax(out[i])] += 1

# Show the results
print("\nConfusion Matrix:\n", cm)
accuracy = np.diag(cm).sum() / cm.sum()
print("\nAccuracy = {:.7f}".format(accuracy))
