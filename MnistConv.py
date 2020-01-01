# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:56:49 2019

@author: potyraj
"""

import numpy as np
from scipy import signal as sp
from Softmax import *
from ReLU import *
from Convv import *
from Convv2 import *
from Pool import *

from timeit import default_timer as timer

def MnistConv(W1, W5, Wo, x_train, ans_train):

   #This will cause the numberOfBatches to update weights after every batch
    batchSize = 100
    totalToTest = x_train.shape[0]
    numberOfBatches = int(totalToTest/batchSize)
    learningRate = .01 #alpha
    beta = .95 #momentum
    
    
    
   # batchList = np.linspace(0, ans_train.shape[0], numberOfBatches+1, endpoint=True).astype('int32')
    batchList2 = np.arange(0, totalToTest, batchSize)
    

    
    momentum1 = np.zeros_like(W1)
    momentum5 = np.zeros_like(W5)
    momentumo = np.zeros_like(Wo)
    
    for batch in range(len(batchList2)):
            dW1 = np.zeros_like(W1)
            dW5 = np.zeros_like(W5)
            dWo = np.zeros_like(Wo)
            
            begin = batchList2[batch]
           
            for k in range(begin, begin+batchSize):
                        
                        x = x_train[k, :, :] #input 28x28    
                        y1 = Convv(x,W1) #in: 10000x28x28, 9x9x20 out: 20x20x20// input * weights 
                        y2 = ReLU(y1) # in: 20x20x20 out: 20x20x20
                        y3 = Pool(y2) # in: 20x20x20 out: 10x10x20
                        y4 = np.reshape(y3, (-1, 1)) # in: 10x10x20 out: 2000x1 (10*10*20=2000) "Flatten"
                        v5 = np.matmul(W5, y4) # in: 100x2000, 2000x1 out: 100x1 "Dense"
                        y5 = ReLU(v5) # in: 100x1 out: 100x1
                        v  = np.matmul(Wo, y5) # in: 100x1, 10x100 out: 10x1
                        ans  = Softmax(v);
                           
                        # one-hot encoding
                        output = np.zeros((10, 1))
                        output[ans_train[k]][0] = 1 
                        
                #####__Back-Prop__#######
                        
                        # calcs error by subtracting 10x1 ans array (ans_train) by 
                        # the 10x1 output array that hold the guesses the network made 
                        e = output - ans
                        delta = e
                        e5 = np.matmul(Wo.T, delta)    # the goal is to get back to the dims of y5
                                                       # in: Wo 10x100 transposed to 100x10, 
                                                       # delta 10x1
                                                       # out: 100x1 =100x10 * 10x1 (y5 = Wo*T * y) 
                                                       
                        delta5 = (y5 > 0) * e5        # Turns value "on" or "off"
                                                      # (y5 > 0) = if value of pixel (x,y) is > 0 then that
                                                      # value will be represented as a 1 else 0.
                                                      # y5 is a vector of all nodes that were active upon  
                                                      # the outputs "decision" 
                                                      # e5 represents the amount of error 
                                                      # y5 and e5 are multiplyed to come up with the 
                                                      # adjustments to be made for the next layer
                        e4 = np.matmul(W5.T, delta5)
                       
                        e3 = np.reshape(e4, y3.shape) # undoes the reshape made after the initial pooling layer
                        
                        e2 = np.zeros_like(y2)             # shape of what came out of our Convv layer
                        W3 = np.ones_like(y2) / (2*2)      # 20x20x20 Mean tensor
                        
                        for c in range(e2.shape[2]):
                            # takes a 1x1 in e3 and copies it into a 2x2 
                            # thereby expanding a 10x10x20 back into a 20x20x20 
                            # then multiplies each pixel by 1/4 (W3) 20x20x20 * 20x20x20
                            e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]
                            
                        delta2 = (y2 > 0) * e2
                        
                        delta1_x = np.zeros_like(W1)
                       
                        for c in range(20):
                                delta1_x[:, :, c] = sp.convolve2d(x[:, :], np.rot90(delta2[:, :, c], 2), 'valid')
                                
                                
                        dW1 = dW1 + delta1_x
                        dW5 = dW5 + np.matmul(delta5, y4.T)
                        dWo = dWo + np.matmul(delta, y5.T)       
                        
                        dW1 = dW1 / batchSize
                        dW5 = dW5 / batchSize
                        dWo = dWo / batchSize
                        
                        momentum1 = learningRate*dW1 + beta*momentum1
                        W1        = W1 + momentum1
                        
                        momentum5 = learningRate*dW5 + beta*momentum5
                        W5        = W5 + momentum5
                        
                        momentumo = learningRate*dWo + beta*momentumo 
                        Wo        = Wo + momentumo
                
    return W1, W5, Wo
                    
                    
                    
                    
                    