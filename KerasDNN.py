from __future__ import print_function
import keras as kr
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate,BatchNormalization
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
from functools import reduce
from keras import regularizers
from keras.models import load_model

from myactivation import cosAct
from myactivation import sinAct
import tarfile
import numpy as np
import re
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
from keras import backend as K


def relativeError(y_true, y_pred):
    differencenorm = tf.norm(tf.subtract(y_pred,y_true))
    correct_prediction = tf.divide(differencenorm,tf.norm(y_true))
    return K.mean(correct_prediction)

modelinfo ="DNN info:"+"\n";

print(colinputcol);
n_input = 7
n_output = 3

numepoch = 10;
batchsize = 1024;
regularWeight = 0.0000;
learningrate = 0.001;

modelinfo+= "n_input " + str(n_input)+"\n";
modelinfo+= "n_output " + str(n_output)+"\n";

nodetypt = 0;

target = np.load('data/target_train_'+str(nodetypt)+'.npy');
origin = np.load('data/origin_train_'+str(nodetypt)+'.npy');
targettest = np.load('data/target_test_'+str(nodetypt)+'.npy');
origintest = np.load('data/origin_test_'+str(nodetypt)+'.npy');

#target = target[:,range(n_input)];
origin = origin[:,range(n_input)];
#targettest = targettest[:,range(n_input)];
origintest = origintest[:,range(n_input)];
 
data_weights = None;


N = origin.shape[0];
Nt = origintest.shape[0];

modelinfo+="training size "+ str(N)+"\n";
modelinfo+="test size "+str(Nt)+"\n";

print(modelinfo);

index = np.arange(0,N-1,20);
indexTest = np.arange(0,Nt-1,36);

#target = target[index]
#origin = origin[index]
#targettest = targettest[indexTest]
#origintest = origintest[indexTest]




model = Sequential()

loadmodel = False;

activationname = 'tanh'
modelinfo += "activation +" + activationname+"\n"

useMyActivation = False

numNodes = 16;
initializer = kr.initializers.RandomNormal(mean= 0.01,stddev = 0.01,seed = None);
initializer = "normal"
model.add(Dense(numNodes, input_dim=n_input,kernel_regularizer = regularizers.l2(regularWeight),kernel_initializer = initializer,bias_initializer = initializer))
modelinfo+="Dense "+str(numNodes)+"\n";

#model.add(BatchNormalization());
if useMyActivation:
    model.add(cosAct());
else:
    model.add(Activation(activationname))
#model.add(Dropout(0.5));
numNodes = 16;
numLayers = 1 ;
modelinfo+="Dense "+str(numNodes)+"X"+str(numLayers)+"\n";
for i in range(numLayers):
    model.add(Dense(numNodes,kernel_regularizer = regularizers.l2(regularWeight),kernel_initializer = initializer,bias_initializer = initializer))
   # model.add(BatchNormalization());
    if useMyActivation:
        model.add(cosAct());
    else:
        model.add(Activation(activationname))
    #model.add(Dropout(0.5));

numNodes = 32;
numLayers = 0;
modelinfo+="Dense "+str(numNodes)+"X"+str(numLayers)+"\n";
for i in range(numLayers):
    model.add(Dense(numNodes,kernel_regularizer = regularizers.l2(regularWeight),kernel_initializer = initializer,bias_initializer = initializer))

#    model.add(BatchNormalization());
    if useMyActivation:
        model.add(cosAct());
    else:
        model.add(Activation(activationname))
    #model.add(Dropout(0.5));


model.add(Dense(n_output,kernel_initializer = initializer,bias_initializer = initializer))

class LossHistory(kr.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('relativeError'))


adam = kr.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd =  kr.optimizers.SGD(lr=0.99, decay=1e-6, momentum=0.9, nesterov=True);
earlystop = kr.callbacks.EarlyStopping(monitor='loss', min_delta = 1e-5,patience = 20);
history = LossHistory();

model.compile(optimizer=adam,
               loss="mean_squared_error",
		#loss=relativeError,
                metrics=[relativeError])

print(N);

if loadmodel:
    model.load_weights('myweights.h5');
else:
    start = timeit.default_timer()
    hist = model.fit(origin, target,epochs=numepoch,batch_size =batchsize,validation_split = 0.01,sample_weight=data_weights,shuffle=True,callbacks = [history])  # starts training
    stop = timeit.default_timer()

    print('time elapsed => ', stop - start)
    model.save_weights('myweights.h5');
    #print(history.losses);
    np.savetxt('loss.txt',history.losses,delimiter = ' ');
  
print("test loss ", model.test_on_batch(origintest,targettest)); # test error
#testLoss = model.test_on_batch(origintest,targettest);
#print("test loss ",model.evaluate(origintest,targettest,batch_size = None,verbose = 1));

filepath = 'type'+str(nodetypt)+'.txt'
with open(filepath,'wb') as f:
    lent = len(model.layers)
    f.write((str(lent)+"\n").encode());
    for layer in model.layers:
        weights = layer.get_weights();
        parameter_len = len(weights)
        f.write((str(parameter_len)+"\n").encode());
        for weight in weights:
            header_ = "";
            for s in weight.shape:
                header_ = header_ + str(s)+ " "
            np.savetxt(f,weight,header = header_,comments="");

N = origintest.shape[0];
p = model.predict(origintest[N-2:N-1], batch_size=1, verbose=1)
print(origintest[N-2:N-1])
print(p)
print(targettest[N-2:N-1]);

import smtplib



#model.save('my_model.h5');

#plot_model(model, to_file='model.png')
