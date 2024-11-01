#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:40:42 2024

@author: davidcaspers
"""

import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_prime(x):
    return sig(x)*(1-sig(x))

def mse(y, yhat):
    return np.square(np.subtract(y,yhat)).mean()

def mse_prime(y,yhat):
    return (yhat-y)


# class for activation layer
##Forward
##Backwards
##Step (which is just return because there's no weights or biases to return)

###Input is from previous layer which is this layers error
class activation_layer:
    def forward(self, input):
        self.input = input # saves this for later processing
        return sig(input)
    def backwards(self, input):
        return sig_prime(self.input) * input
    def step(self, eta):
        return 
        



# class for fully connected layer
class fully_connected:
    def __init__(self, input_length, output_length):
        self.weights = np.random.rand(input_length, output_length) - 0.5 
        self.bias = np.zeros((1,output_length)) 
        self.training_examples = 0
        self.delta_w = np.zeros(np.shape(self.weights))
        self.delta_b = np.zeros(np.shape(self.bias))
    
##Forward
    def forward(self,input_data):
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

##backwards
    def backwards(self, input_error):
        output_error = np.dot(input_error, self.weights.T)
        self.delta_w += np.dot(self.input.T, input_error)
        self.delta_b += input_error
        self.training_examples += 1
        return output_error
    
##Step
    def step(self, eta):
        np.divide(self.delta_w, self.training_examples, out=self.delta_w)
        self.weights -= eta * self.delta_w
        np.divide(self.delta_b, self.training_examples, out=self.delta_b)
        self.bias -= eta * self.delta_b
        self.training_examples = 0
        self.delta_w.fill(0)
        self.delta_b.fill(0)
        self.input= None

        
        
#Network class to layer them all together
##Define layer = []
import objgraph
from pympler import summary, muppy

class NN:
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.layers = []
        
    def add(self, layer):
        self.layers.append(layer)
    
    def predict(self, x_test):
        y_pred = []
        for i in range(np.shape(x_test)[0]):
            input = x_test[i]
            for layer in self.layers:
                output = layer.forward(input)
                input = output
            y_pred.append(input)
        return y_pred    
        
    #fit should run forward predictions to get yhat, then optimize
    def fit(self, x_train, y_train, eta, epochs, batchsize=64):
        error = np.zeros_like(y_train[0])  # Pre-allocate error array
        indices = np.arange(x_train.shape[0])
        current_epoch = 1
        o1 = muppy.get_objects()
        for epoch in range(epochs):
            # Memory checkpoint before the epoch

            print(f"starting epoch: {current_epoch}")
            # Memory checkpoint before the epoch

            #shuffle training set before each epoch.  Epoch ends when model is trained on all possible training values 
            #number of mini batches per epoch:
            np.random.shuffle(indices)
            current_minibatch = 1
            for start in range(0, x_train.shape[0], batchsize):
                sqrd_error = 0.0
                
                end = min(start + batchsize, x_train.shape[0])
                batch_indices = indices[start:end]
                
                x_batch = x_train[batch_indices]
                y_batch= y_train[batch_indices]
                
                #Generate y predictions for each value in the mini batch

                for i in range(np.shape(x_batch)[0]):
                    output = x_batch[i]
                    for layer in self.layers:
                        output = layer.forward(output)
                    
                    #run back proposition . . . accumilated errors recorded automatically be layer class
                    error = mse_prime(y_batch[i], output)
                    for layer in reversed(self.layers):
                        error = layer.backwards(error)
                        sqrd_error += mse(y_batch[i],output)
                        
                #After all the minibatch errors are accumilated, take a step in each layer
                for layer in self.layers:
                    layer.step(eta)
                #print(f"starting minibatch: {current_minibatch}      Loss: {sqrd_error}")
                current_minibatch +=1
        
                
            
            
            current_epoch +=1 

          
        # Use objgraph to inspect the top memory-consuming objects
    

        # Memory checkpoint after the epoch
        o2 = muppy.get_objects()
        # Print summary of new objects
        print(f"Memory usage after epoch {epoch + 1}:")
        summary.print_(summary.get_diff(summary.summarize(o1), summary.summarize(o2)), limit=5)
                       
        # Calculate memory differences
        diff_summary = summary.get_diff(summary.summarize(o1), summary.summarize(o2))
        summary.print_(diff_summary, limit=5)
            
        # Filter `diff` to include only specific types and top contributors
        o1_ids = {id(obj) for obj in o1}
        o2_ids = {id(obj): obj for obj in o2}
        diff = [obj for obj_id, obj in o2_ids.items() if obj_id not in o1_ids and isinstance(obj, (list, np.ndarray))]
        
        # Reference graphs only for top memory-growing objects
        for obj in diff[:5]:
            print("\nReference graph for a large new object:")
            objgraph.show_backrefs([obj], max_depth=2, too_many=5)
            
        # View growing object types overall
        objgraph.show_growth(limit=5)
                        
                        
                   
                    
                    
                
            
        
        
        
        