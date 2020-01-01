# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:48:24 2019

@author: potyraj
"""
import numpy as np
#import cv2 as cv
#import matplotlib.pyplot as plt
#import tensorflow as tf
import skimage as ski 
#import scipy.signal as sp
#2x2 mean pooling

def Pool(x):
    
    xrow, xcol, numFilters = x.shape[0:3]
   
    yrow = xrow//2
    ycol = xcol//2
    y = np.zeros((int(yrow), int(ycol), numFilters))
    #y2 = np.zeros((int(yrow), int(ycol), numFilters))

    #manualPoolOutput = np.zeros((int(yrow), int(ycol), numFilters))
  
    for k in range(numFilters): 
        #z = 2 #size of filter 2x2 ZxZ
        #kernel = np.ones([z,z])/(z**2)
        #image  = sp.convolve2d(x[:, :, k], kernel, 'valid')
        #y[:,:,k] = tf.nn.avg_pool(x,1,(1,1),1,"VALID")
        y[:,:,k] = ski.measure.block_reduce(x[:,:,k], (2,2), np.nanmean)
        #y2[:, :, k] = image[::2, 0::2]
        #plt.imshow(y, cmap='Greys')
        #testing manual method
            #image_padded = np.zeros((x.shape[0] + 2, x.shape[1] + 2,2))   
            #image_padded[1:-1, 1:-1, 0:-1] = x
        #the 2x2 kernal is supposed to start at [0,0], average them together then place that average
        #in the new output. it then steps by 2 so to not overlap on pooling
        #range(start, stop, step) = range(0, 28, 2)
        #28 is not included
        #MPO xx and yy are the coordinance for the pixel that will be placed in manualPoolOutput
#        MPOxx=-1        
#        print(range(x.shape[1]-3))
#        for xx in range(0,x.shape[1],2):
#            MPOyy=0
#            MPOxx+=1      
#            for yy in range(0,x.shape[0],2):
#                manualPoolOutput[yy,xx,k] = ((kernel*x[yy:yy+2,xx:xx+2,k]).sum())
#                MPOyy+=1
#        plt.imshow(manualPoolOutput[:,:,k], cmap='Greys') 
#          

   
                
  
    return y