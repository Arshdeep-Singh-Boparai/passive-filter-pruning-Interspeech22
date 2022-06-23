# Quantize trained model (LAYER WISE PRUNED MODEL) to float16 or unit8 (default)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:20:49 2022

@author: arshdeep
"""


import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pathlib
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.layers.core import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from tensorflow.keras.models import load_model


#%%
x_test=np.load('~/X_test.npy')
labels_test=np.load('~/Y_test.npy')
# labels_train=np.load('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021/Y_train.npy')

y_test = keras.utils.to_categorical(labels_test, 10)

#%%


# saved_odel_dir='/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning_condor/30_epochs/C123/5'

model_name='C3_best_model_dcase2021.h5'
os.chdir('~/Qunatized_model/C3')


model = tf.keras.models.load_model(model_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()#saving converted model in "converted_model.tflite" file
open("converted_model.tflite", "wb").write(tflite_model)


model = tf.keras.models.load_model(model_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()#saving converted model in "converted_quant_model.tflite" file
open("converted_quant_model_default.tflite", "wb").write(tflite_quant_model)


model = tf.keras.models.load_model(model_name)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()#saving converted model in "converted_quant_model.tflite" file
open("converted_quant_model_16.tflite", "wb").write(tflite_quant_model)

#%%

#%%
interpreter = tf.lite.Interpreter(model_path="converted_quant_model_default.tflite")


all_tensor_details=interpreter.get_tensor_details()
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
# print(input_shape)


#%%

x_test=np.array(x_test,dtype=np.float32)
acc=0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc_def = acc/len(x_test)
# print('default compressed acc:',acc_def*100)


#%%

interpreter = tf.lite.Interpreter(model_path="converted_quant_model_16.tflite")


all_tensor_details=interpreter.get_tensor_details()
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
# print(input_shape)

x_test=np.array(x_test,dtype=np.float32)
acc=0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc_16 = acc/len(x_test)
# print('float 16 accuracy',acc_16*100)


interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")


all_tensor_details=interpreter.get_tensor_details()
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
# print(input_shape)

x_test=np.array(x_test,dtype=np.float32)
acc=0
for i in range(len(x_test)):
    input_data = x_test[i].reshape(input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc_WT = acc/len(x_test)
# print('without compression accuracy',acc_WT*100)

#%%

print('..............................')
print('without compression accuracy   ',acc_WT*100)
print("Float model in Mb:  ", os.path.getsize('converted_model.tflite') / float(2**20))

print('..............................')

print('default compressed acc: ',acc_16*100)
print("Quantized model float16 in Mb: ", os.path.getsize('converted_quant_model_16.tflite') / float(2**20))
print("Compression ratio 16: ", os.path.getsize('converted_model.tflite')/os.path.getsize('converted_quant_model_16.tflite'))

print('..............................')
print('default compressed acc: ',acc_def*100)
print("Quantized model default in Mb: ", os.path.getsize('converted_quant_model_default.tflite') / float(2**20))
print("Compression ratio default: ", os.path.getsize('converted_model.tflite')/os.path.getsize('converted_quant_model_default.tflite'))

print('..............................')
