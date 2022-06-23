% This is a code to fine-tune the pruned network.
% Input: Pre-trained CNN, layer-wise important filter indexes, training data, test data, training and test labels.
% Output: Fine-tuned pruned model.

# load packages such as keras, tensorflow etc. from baseline_code.py

# load data 
x_train=np.load('~/DCASE2021/X_train.npy')
x_test=np.load('~/DCASE2021/X_test.npy')
labels_test=np.load('~/DCASE2021/Y_test.npy')
labels_train=np.load('~/DCASE2021/Y_train.npy')
y_test = keras.utils.to_categorical(labels_test, 10)



#%% load important indexes
L1=sort(np.load('~/Important_index_layerwise /C1.npy'))
L2=sort(np.load('~/Important_index_layerwise /C2.npy'))
L3=sort(np.load('~/Important_index_layerwise /C3.npy'))

L3_n=L3*2
L3_n1=L3_n+1
w_f=[]
for i in range(len(L3)):
	w_f.append(L3_n[i])
	w_f.append(L3_n1[i])
	

Total_filter=len(L1)+len(L2)+len(L3)#+len(L4)+len(L5)+len(L6)+len(L7)+len(L8)+len(L9)+len(L10)+len(L11)+len(L12)+len(L13)


W=W_dcas 

# extract import filters from pre-trained weights and ignore others.
W_pruned=[W[0][:,:,:,L1],W[1][L1],W[2][L1],W[3][L1],W[4][L1],W[5][L1],W[6][:,:,L1,:][:,:,:,L2],W[7][L2],W[8][L2],W[9][L2],W[10][L2],W[11][L2],	W[12][:,:,L2,:][:,:,:,L3],W[13][L3],W[14][L3],W[15][L3],W[16][L3],W[17][L3],W[18][w_f,:],W[19],W[20],W[21]]

						
#%% Pruned model 


input_shape=(40,500,1)
##model building
model1 = Sequential()
#convolutional layer with rectified linear unit activation
layer_C1= Conv2D(len(L1), kernel_size=(7, 7),padding='same',input_shape=input_shape)
layer_C1.trainable =True
model1.add(layer_C1)
layer_BN1 = BatchNormalization(axis=-1)
layer_BN1.trainable = True
model1.add(layer_BN1) #layer2


convout1= Activation('relu')
# convout1.trainable= True
model1.add(convout1) #laye



#''''''''''''''''''''''''''''''''''''''''''''''''

layer_C2=Conv2D(len(L2), kernel_size=(7, 7),padding='same')
layer_C2.trainable= True
model1.add(layer_C2)

layer_BN2= BatchNormalization()#layer2
layer_BN2.trainable=  True
model1.add(layer_BN2)


convout2= Activation('relu')
model1.add(convout2) #laye

model1.add(MaxPooling2D(pool_size=(5, 5)))

model1.add(Dropout(0.30))



layer_C3=Conv2D(len(L3), kernel_size=(7, 7),padding='same')
layer_C3.trainable= True
model1.add(layer_C3)

layer_BN3 =  BatchNormalization() #layer2
layer_BN3.trainable= True
model1.add(layer_BN3)


convout3= Activation('relu')
model1.add(convout3) #laye

model1.add(MaxPooling2D(pool_size=(4, 100)))

model1.add(Dropout(0.30))

# model1.set_weights(W_dcas[0:18])

model1.add(Flatten())

model1.add(Dense(100,activation='relu'))
model1.add(Dropout(0.30))

model1.add(Dense(10, activation='softmax'))


model1.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


 model1.set_weights(W_pruned) # set important weights obtained from unpruned model to pruned model.


model1.summary()

#%% pruned model performance

# model1=load_model('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/Pruned_finetuned_weights/iter/level_1_model.h5')

pred_label=model1.predict(x_test)  # load test dataset 
pred=np.argmax(pred_label,1)
asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')

#os.chdir('/home/arshdeep/Pruning/DCASE_2021_NETWORK/DCASE2021_pruning/random_experminets_11_11_22/50/2/')
#training_percen = 0.50

#p = 1 - training_percen

#X_train, X_test_50, label_train, y_test_50 = train_test_split(x_train, labels_train, test_size=p, random_state=1)


checkpointer = ModelCheckpoint(filepath='best_weights_dcase2021_test.h5py',monitor='val_accuracy',verbose=1, save_best_only=True,save_weights_only=True)
#start = datetime.datetime.now()
hist=model1.fit(X_train, label_train,batch_size=32,epochs=30,verbose=1,validation_data=(x_test, y_test),callbacks=[checkpointer])
# 
#end = datetime.datetime.now()
#diff = (end - start)

#datetime.timedelta(seconds=10, microseconds=885206)
model1.load_weights('best_weights_dcase2021_test.h5py')

model1.save('C123_best_model_dcase2021.h5')
#model1=load_model('/home/arshdeep/DCASE2021_task1a/sorted_inexed/Pruned_model_1.h5py')

#%% predictions from pruned model.


pred_label=model1.predict(x_test)
pred=np.argmax(pred_label,1)
asd=confusion_matrix(labels_test,pred);
accu=(np.trace(asd)/np.size(labels_test))*100;
print(accu,'accuracy')
model1.summary()

#np.save('~/history1.npy',hist.history)
 
#history1=np.load('history1.npy',allow_pickle='TRUE').item()

