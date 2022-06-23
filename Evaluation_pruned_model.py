# Load a model (pruned) and evaluate it using test dataset.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:23:54 2022

@author: arshdeep
"""

import h5py    
import numpy as np    
from sklearn.metrics import classification_report,confusion_matrix
import keras
import scipy.io
import pickle
import os
from tensorflow.keras.models import load_model
#%% load model from Pruned_model directory..........................................
model= load_model('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/Fine_tuning_cosine_sim/C1/best_model_dcase2021_L1.h5') 
#%%  DATA load
#x_train=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021/X_train.npy')
x_test=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021/X_test.npy')
labels_test=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021/Y_test.npy')
#labels_train=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021/Y_train.npy')
#%% evaluate model ..................................................................
y_test = keras.utils.to_categorical(labels_test, 10)
pred_label=model.predict(x_test)
pred=np.argmax(pred_label,1)
asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')
