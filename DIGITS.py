'''
CNN to recognize handwritten digits from the mnist dataset

Using 60 minibatches of size 1000. 
'''



import keras 
import scipy.signal as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2 as cv
import numpy as np

from Convv import *
from ReLU import *
from Pool import *
from MnistConv import *
from timeit import default_timer as timer
from Softmax import *

(x_train, ans_train), (x_test, ans_test) = tf.keras.datasets.mnist.load_data()


#reshape
x_train = x_train.reshape(x_train.shape[0], 28, 28)
x_test = x_test.reshape(x_test.shape[0], 28, 28)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalize
x_train /= 255
x_test /= 255

x_train = x_train[0:8000,:,:]
ans_train = ans_train[0:8000]

x_test = x_test[8000:10000,:,:]
ans_test = ans_test[8000:10000]

############################_Weights_#########################################

#20 inititial 9x9 filters filled with random numbers      
W1 = 1e-2 * np.random.randn(9,9,20)  

# tensors after being pooled are fed into the W5 weights layer
# 100 is used because after reshaping a 10x10 pooled layer its 100x1 
# and 2000 is used because 10x10x20 of the pooling layer = 2000?
W5 = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)

# 10 is used because this is the last set of weight before the output and theres only 10 answers
# 100 is used because 100x1 is the input from the previous layer
Wo = np.random.uniform(-1, 1, ( 10,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)
  
#####################___TRAINING___###########################################

epochs = 3

print('Train size: ', x_train.shape[0])
print('Validation size: ', x_test.shape[0])
print('Number of epochs: ', epochs)

for _epoch in range(epochs):
 
    print('starting epoch: ',_epoch)
    W1, W5, Wo = MnistConv(W1, W5, Wo, x_train, ans_train)  

#####################___TESTING___############################################
    
acc = 0
N   = len(ans_test)

for k in range(N):
    
    x = x_test[k,:,:] #input 28x28
    
    y1 = Convv(x,W1) #in: 10000x28x28 (10000: 28x28 images), 9x9x20 (20: 9x9 weights) out: 20x20x20
    y2 = ReLU(y1) # in: 20x20x20 out: 20x20x20
    y3 = Pool(y2) # in: 20x20x20 out: 10x10x20
    y4 = np.reshape(y3, (-1, 1)) # in: 10x10x20 out: 2000x1 (10*10*20=2000)
    v5 = np.matmul(W5, y4) # in: 100x2000, 2000x1 out: 100x1
    y5 = ReLU(v5) # in: 100x1 out: 100x1
    v  = np.matmul(Wo, y5) # in: 100x1, 10x100 out: 10x1
    y  = Softmax(v);
    
    i = np.argmax(y)
    if i == ans_test[k]:
        acc = acc + 1

#####################___Accuracy___###########################################
   
acc = acc / N
print("Accuracy is : ", acc)   
