#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:36:53 2024

@author: davidcaspers
"""
import numpy as np

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def sig_prime(x):
    return sig(x)*(1-sig(x))

def mse(Y_true, Y_pred):
    return np.square(np.subtract(Y_true,Y_pred)).mean() 
    
def mse_prime(Y_true, Y_pred):
    return (2*(Y_pred-Y_true)).mean()
    


#Define Activation Layer, with Forward, Backward, and Step . . . step in this won't do anything because there's no weights or biases to consider

class ActivateLayer:
    def forward(self, input_data):
        self.input = input_data  #saves this for the backward step where we'll need x transpose to find the weights derivative
        return sig(input_data)
    def backward(self, previous_error):
        return sig_prime(self.input) * previous_error
    def step(self, eta):
        return




#Define Connected Layer with Forward, Backward, and Step
class FullyConnectedLayer:
    def __init__(self, input_size, output_size): #need to know how many incoming variables there are and how many neurons are next (1 if final)
        #instantiate random weights and bias terms, save variables
        self.weights = np.random.rand(input_size, output_size) - 0.5 #outputs matrix of input_size x output_size dimensions populated with values between -.5 and 0.5
        self.bias = np.random.rand(1,output_size) -0.5 #outputs random bias terms corresponding to each output neuron
        #track number of training examples gradient descent has been run on for this particular layer
        self.passes = 0
        #provide an area to collect the bias terms for each backward pass (will be averaged by passes value)
        self.delta_w = np.zeros((input_size, output_size))
        self.delta_b = np.zeros((1,output_size))
        
    def forward(self,input_data):
        #multiply weights times input data + bias term
        self.input = input_data #saves this for the backward step where we'll need x transpose to find the weights derivative
        #apply Wx+b transformation
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_error): #given error term for this layer, returns error for next layer in backprop sequence 
        input_error = np.dot(output_error,self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.delta_w += np.dot(self.input.T, output_error)
        self.delta_b += output_error
        self.passes += 1 #tracks the number of training examples this has been run on (to accumulate the errors and avg them)
        return input_error

    def step(self, eta): #moves gradient for this layer, resets delta and passes variables
        self.weights -= eta * (self.delta_w/self.passes)
        self.bias -= eta * (self.delta_b/self.passes)
        self.delta_w= np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)
        self.passes = 0
        

#Define Network Layer that has function to add layers, predict, fit the model (using gradient descent) 
class NN:
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.layers = [] #tracks the layers that get added in the 'add' method
        
    def add(self, layer):
        self.layers.append(layer)
        
    def predict(self, input_data):
        result = []
        for i in range(input_data.shape[0]):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)
        return result
            
        
    def fit(self, x_train, y_train, minibatches, learning_rate, batch_size=64):
        #define values to use in minibatch
        for i in range(minibatches):
            err = 0
            idx = np.argsort(np.random.random(x_train.shape[0]))[:batch_size]
            x_batch = x_train[idx]
            y_batch = y_train[idx]
            for j in range(batch_size):
                output = x_batch[j]
                for layer in self.layers:
                    output = layer.forward(output)
                err += mse(y_batch[j], output)
                
                error = mse_prime(y_batch[j],output)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
            
            for layer in self.layers:
                layer.step(learning_rate)
            if (self.verbose) and ((i%10) == 0):
                err /= batch_size
                print('minibatch %5d/%d     error=%0.9f' % (i, minibatches, err))
            
        
        
        #feed forward to get various values of xT need for back propogation
        
        for layer in self.layers:
            output = layer.forward(output)
            
        #iterate over epochs
            #iterate over minimbatches to accumilate errors
            #take a step
        
        