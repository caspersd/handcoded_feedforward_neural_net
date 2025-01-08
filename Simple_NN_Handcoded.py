#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:22:33 2024

@author: davidcaspers
"""

import numpy as np
import random




def sig(x):
    return 1/(1+np.exp(-x))


#Code to Evaluate NN in pass forward
def Forward(wb, x):
    #instantiate an array that equals the length of x
    y = np.zeros(x.shape[0])
    for i in range(x.shape[0]): #loop through each training observation to calculate output y
        z0 = wb["w0"]*x[i,0] + wb["w2"]*x[i,1] + wb["b0"]
        a0 = sig(z0)
        z1 = wb["w1"]*x[i,0] + wb["w3"]*x[i,1] + wb["b1"]
        a1 = sig(z1)
        y[i]= wb["w4"]*a0 + wb["w5"]*a1 + wb["b2"]
    return y

#define function to calculate tp, tn, fp, fn 
def Evaluate(wb, x, y):
    tp = tn = fp = fn = 0
    pred = Forward(wb, x)
    for i in range(pred.shape[0]):
        if pred[i] > 0.5:
            pred[i] = 1
        else: pred[i] = 0
    for i in range(len(y)):
        if (y[i] == 0) and (pred[i]==0):
            tn += 1
        if (y[i] == 1) and (pred[i]==1):
            tp += 1
        if (y[i] == 1) and (pred[i]==0):
            fn +=1
        if (y[i] == 0) and (pred[i]==1):
            fp +=1
    
    return tp, tn, fp, fn, pred


#define gradient descent
def gradient_descent(wb, x, y, epoch, eta):
    #loop through each epoch
    for i in range(epoch):
        #set the initial derivative values to 0
        dw0 = dw1 = dw2= dw3= dw4=dw5=db0 =db1 = db2 =0.0
        #loop through each training value and calculate output 
        for k in range(x.shape[0]):
            #pass forward training value to get intermediate values
            z0 = wb["w0"]*x[k,0] + wb["w2"]*x[k,1] + wb["b0"]
            a0 = sig(z0)
            z1 = wb["w1"]*x[k,0] + wb["w3"]*x[k,1] + wb["b1"]
            a1 = sig(z1)
            a2 = wb["w4"]*a0 + wb["w5"]*a1 + wb["b2"]
            
            #calculate the partial derivative values, and accumlated contribution to loss
            dw0 += (a2-y[k])*wb["w4"]*a0*(1-a0)*x[k,0]
            dw1 += (a2-y[k]) * wb["w4"] * a0 * (1-a0) * x[k,1]
            dw2 += (a2 - y[k]) * wb["w4"]*a0*(1-a0)*x[k,1]
            dw3 += (a2 - y[k])*wb["w5"]*a1 * (1-a1) * x[k,0]
            dw4 += (a2-y[k])*a0
            dw5 += (a2-y[k])*a1
            db0 += (a2-y[k])*wb["w4"]*a0*(1-a0)
            db1 += (a2-y[k]) * wb["w5"] * a1 * (1-a1)
            db2 += (a2 - y[k])
            
        #update the weights and biases based on the accumlated loss for this epoch
        m = x.shape[0]
        wb["b0"] = wb["b0"] - eta * db0 / m
        wb["b1"] = wb["b1"] - eta * db1 / m
        wb["b2"] = wb["b2"] - eta * db2 / m
        wb["w0"] = wb["w0"] - eta * dw0 / m
        wb["w1"] = wb["w1"] - eta * dw1 / m
        wb["w2"] = wb["w2"] - eta * dw2 / m
        wb["w3"] = wb["w3"] - eta * dw3 / m
        wb["w4"] = wb["w4"] - eta * dw4 / m
        wb["w5"] = wb["w5"] - eta * dw5 / m
    return wb 



#load the Iris Dataset
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
x = iris.data[:,:2]
y = iris.target
mask = (y==0) | (y==1)
x = x[mask]
y = y[mask]

#Evaluate NN and record TP, FP, TN, FN

##Set initial weights and biases for Neural Net
net = {}
net["b0"] = 0
net["b1"] = 0
net["b2"] = 0
net["w0"] = (np.random.random()-0.5)*0.0001
net["w1"] = (np.random.random()-0.5)*0.0001
net["w2"] = (np.random.random()-0.5)*0.0001
net["w3"] = (np.random.random()-0.5)*0.0001
net["w4"] = (np.random.random()-0.5)*0.0001
net["w5"] = (np.random.random()-0.5)*0.0001

#evaluate untrained random NN
tp0, tn0, fp0, fn0, pred = Evaluate(net, x, y) 

epoch = 2000
eta = 0.05
#train NN
net = gradient_descent(net, x, y, epoch, eta)

#evaluate trained  NN
tp, tn, fp, fn, pred = Evaluate(net, x, y) 

print(f"Training for {epoch} epochs, learning rate {eta}% \n")
print("Before training:")  
print(f"TN:{tn0} FP:{fp0}") 
print(f" FN:{fn0} TP:{tp0}\n")
print("After training:")  
print(f"TN:{tn} FP:{fp}") 
print(f" FN:{fn} TP:{tp}\n")






