# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:45:42 2019

@author: potyraj
"""

import numpy as np
from scipy import signal as sp
from timeit import default_timer as timer

def Convv2(x, W):
    Wrow, Wcol, numFilters = W.shape
    #x = x.reshape(x.shape[0:2])
    xrow, xcol = x.shape
    yrow = xrow - Wrow + 1
    ycol = xcol - Wcol + 1
    manualConvOutput = np.zeros((yrow, ycol, numFilters))
    #manualConvOutput4 = np.zeros((yrow, ycol, numFilters))
    for k in range(0,numFilters):
       
        weights = W[:,:,k]
        
        
        #weights = np.rot90(np.squeeze(weights),2)

        for j in prange(yrow):
           
            for i in prange(ycol):
                
#                for m in range(weights.shape[1]):
#                    mm = Wrow -1 -m
#                    
#                    for n in range(weights.shape[0]):
#                        nn = Wrow -1 -n 
#                        
#                        ii = int(i + (kCenterY - mm))
#                        jj = int(j + (kCenterX - nn))
               
                        #if (ii >= 0 and ii < x.shape[1] and jj >= 0 and jj < x.shape[0]):
                    jj = j+9 
                    ii = i+9
                
                    manualConvOutput[j,i,k] = (x[j:jj,i:ii]*weights).sum()
                                #manualConvOutput3 = np.sum(manualConvOutput2)
                                #manualConvOutput[yy,xx,k] = manualConvOutput3
                                #manualConvOutput4[xx,yy,k] = (weights*x[yy:yy+9,xx:xx+9]).sum()
                    
    return manualConvOutput
