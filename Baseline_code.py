% This is code to generate baseline results for DCASE 2021 task 1a baseline model.

% Input: Pre-trained weights or model, DCASE 2021 task1 a validation dataset.
% Output: Accuracy and log-loss.



import h5py    
import numpy as np    
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from scipy.stats import mode
from keras import backend as K
from random import shuffle
from keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
#import scipy.io
#import pickle
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import log_loss


#%%  baseline model architecture 

input_shape=(40,500,1)
##model building
model = Sequential()
#convolutional layer with rectified linear unit activation
model.add(Conv2D(16, kernel_size=(7, 7),padding='same',input_shape=input_shape))

model.add(BatchNormalization(axis=-1)) #layer2
convout1= Activation('relu')
model.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

model.add(Conv2D(16, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(5, 5)))

model.add(Dropout(0.30))



#''''''''''''''''''''''''''''''''''''''''''''''

model.add(Conv2D(32, kernel_size=(7, 7),padding='same'))

model.add(BatchNormalization()) #layer2
convout2= Activation('relu')
model.add(convout2) #laye

model.add(MaxPooling2D(pool_size=(4, 100)))

model.add(Dropout(0.30))


model.add(Flatten())

model.add(Dense(100,activation='relu'))
model.add(Dropout(0.30))

model.add(Dense(10, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#%%

model.load_weights("~/DCASE2021/baseline_model_weight.tf") # load pre-trained weights from "Data downloading link"

W_dcas=model.get_weights()


#%%  DATA load from "Data downloading link"


x_train=np.load('~/DCASE2021/X_train.npy')
x_test=np.load('~/DCASE2021/X_test.npy')
labels_test=np.load('~/DCASE2021/Y_test.npy')
labels_train=np.load('~/DCASE2021/Y_train.npy')


y_test = keras.utils.to_categorical(labels_test, 10)

# y_train = keras.utils.to_categorical(labels_train, 10)

pred_label=model.predict(x_test)


pred=np.argmax(pred_label,1)

asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;


logloss_overall = log_loss(y_true=labels_test, y_pred=pred_label, normalize=True)

print('accuracy is: ', accu, '  logloss is: ', logloss_overall)
