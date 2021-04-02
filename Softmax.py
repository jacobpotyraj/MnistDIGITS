# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:19:55 2019

@author: potyraj
"""
# the last activation function of the CNN to normalize the output of the network
# to a probability distribution over predicted output classes
import numpy as np


def Softmax(x):
    x  = np.subtract(x, np.max(x))        # prevent overflow
    ex = np.exp(x)
    
    return ex / np.sum(ex)
