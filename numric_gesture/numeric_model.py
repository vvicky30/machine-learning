import tensorflow as tf
import cv2
from collections import deque
import numpy as np


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
 #using standard datasets(mnist  *npz-file) from keras using tensorflow in backend

x_train.shape

#Reshaping
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)  #configure input 

x_train = x_train.astype('float32') #typecasting into float32 datatype
x_test = x_test.astype('float32') #typecasting into float32 datatype

x_train /= 255
x_test /= 255


#now displaying all configured test 7 train data
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


from keras.models import Sequential  #here The sequential API allows you to create models layer-by-layer for most problems.
                                     #It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs
from keras.layers import *   #importing All
model = Sequential()         #calling sequential-function as in variable model
model.add(Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
   # here the number of output filters in the convolution=28----- here the kernel_size=(3,3)specifying the length of the 1D convolution window
   #here using activation function we going to use is 'relu'----If we don't specify anything, no activation-function is applied (ie. "linear" activation: a(x) = x)
   #here the input_shape is 3-dimensional i.e. 3D tensor with shape: (batch-size, steps-size, no. ofchannels) #syntax
model.add(MaxPooling2D(pool_size=(2, 2)))#here the  tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
             #(2, 2) will halve the input in both spatial dimension. If only one integer is specified, the same window length will be used for both dimensions
             #its using concept of 'recycling of vectors' in this case its 'tuples'
model.add(Flatten())  #If we wanted to use a Dense(a fully connected layer) after your convolution layers, 
                     #we would need to ‘unstack’ all this multidimensional tensor into a very long 1D tensor. You can achieve this using Flatten.
model.add(Dense(128, activation='relu'))#as we make dense CNN
model.add(Dropout(0.2))#for avoiding overfitting in CNN-model
model.add(Dense(10, activation='softmax'))  #here using softmax activation for multinomial logistic regression and interpreting neural network outputs are probabilities as 'yes' or 'no'

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #as here if our targets are integers, so we preffred sparse_categorical_crossentropy
#Examples of integer encodings (for the sake of completion): 1,2,3 and so on
#but if our targets are one-hot-encoded numbers then we use categorical_crossentropy only
#and here we also use 'adam' optimizer as 'adam'can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum


model.fit(x=x_train,y=y_train,epochs=10)  #now starts to train model ##for epoch-size=10 to be generate


model.save('cnn_num_gesture_opencv.h5')  #saving model