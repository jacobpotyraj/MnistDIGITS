# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:45:42 2019

@author: potyraj
"""

import numpy as np
from scipy import signal as sp



def Convv(x, W):
    (Wrow, Wcol, numFilters) = W.shape[0:3]
    #x = x.reshape(x.shape[0:2])
    (xrow, xcol) = x.shape[0:2]
    yrow = xrow - Wrow + 1
    ycol = xcol - Wcol + 1
    y = np.zeros((yrow, ycol, numFilters))
   
    for k in range(0,numFilters):
        weights = W[:,:,k]
        weights = np.rot90(np.squeeze(weights),2)



        y[:,:,k] = sp.convolve2d(x,weights,'valid')

     
        
    return y