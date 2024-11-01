#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:43:36 2024

@author: davidcaspers
"""

from vector_based_NN_handcoded import *
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load MNIST dataset from sklearn
mnist = fetch_openml('mnist_784', version=1)

# Prepare the input data and labels
x = mnist.data.to_numpy()  # Convert pandas DataFrame to NumPy array
y = mnist.target.astype(int)  # Convert target labels to integers

# Normalize the data
x = x / 255.0  # Normalize pixel values to [0, 1]

# Convert labels to one-hot encoding
y_one_hot = np.eye(10)[y]

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.2, random_state=42)

# Reshape input data for the neural network
x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)

# Build the network (assuming you have the same `Network` structure)
net = NN()
net.add(FullyConnectedLayer(28 * 28, 100))  # MNIST images are 28x28 pixels
net.add(ActivateLayer())  # Activation function
net.add(FullyConnectedLayer(100, 50))  # Hidden layer with 50 neurons
net.add(ActivateLayer())  # Activation function
net.add(FullyConnectedLayer(50, 10))  # Output layer with 10 neurons (10 classes)
net.add(ActivateLayer())  # Activation function

# Train the network on the MNIST dataset
net.fit(x_train, y_train, minibatches=140, learning_rate=1.0)

# Test the network and compute the confusion matrix
out = net.predict(x_test)
cm = np.zeros((10, 10), dtype="uint32")

for i in range(len(y_test)):
    cm[np.argmax(y_test[i]), np.argmax(out[i])] += 1  # Compare predicted and true labels

# Print confusion matrix and accuracy
print()
print(np.array2string(cm))
print()
print("accuracy = %0.7f" % (np.diag(cm).sum() / cm.sum()))
