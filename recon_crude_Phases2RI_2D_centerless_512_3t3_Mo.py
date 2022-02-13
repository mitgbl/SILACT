from __future__ import print_function,division
from tensorflow import keras
# from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv3D,MaxPooling2D, MaxPooling3D,Flatten,Reshape,Conv2DTranspose,Conv3DTranspose
from tensorflow.keras.layers import Input,UpSampling2D,Concatenate,BatchNormalization,Input,Activation,Add,Lambda,RepeatVector
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import optimizers,regularizers 
from tensorflow.keras.optimizers import Adam,Adadelta,Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import os
import os.path

import matplotlib.pyplot as plt
import scipy.io as sio
import tensorflow as tf
import numpy as np
import sys 
import h5py as hp
import math
import cmath
import argparse

from tensorflow.keras import backend as K

tf.compat.v1.disable_eager_execution()

## ---------------------------------------Optimizers----------------------------------------------------------------------
# optrms=optimizers.rmsprop(lr=0.0001,decay=1e-6)
optadam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
optsgd=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
optadadelta=Adadelta()
optadagrad=Adagrad()

## ---------------------------     Global Parameters---------------------------------


def npcc_3d(generated_image,true_image):
    
    fsp=generated_image-K.mean(generated_image,axis=(1,2,3,4),keepdims=True)
    fst=true_image-K.mean(true_image,axis=(1,2,3,4),keepdims=True)
    
    devP=K.std(generated_image,axis=(1,2,3,4))
    devT=K.std(true_image,axis=(1,2,3,4))
    
    npcc_loss=(-1)*K.mean(fsp*fst,axis=(1,2,3,4))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return npcc_loss

def pcc_3d(generated_image,true_image):
    
    fsp=generated_image-K.mean(generated_image,axis=(1,2,3,4),keepdims=True)
    fst=true_image-K.mean(true_image,axis=(1,2,3,4),keepdims=True)
    
    devP=K.std(generated_image,axis=(1,2,3,4))
    devT=K.std(true_image,axis=(1,2,3,4))
    
    pcc_loss=K.mean(fsp*fst,axis=(1,2,3,4))/K.clip(devP*devT,K.epsilon(),None)    ## (BL,1)
    return pcc_loss

## ----------------------------------- load and pack data ----------------

str1 = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\Inputs_3_939\\'
str2 = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\GT_BPM_3_939\\'
path_1= r'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\Inputs_3_939'
path_2= r'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\GT_BPM_3_939'
path_list_1=os.listdir(path_1)
path_list_2 = os.listdir(path_2)

print(len(path_list_1))
print(len(path_list_2))
tr_idx=range(900)
test_idx=range(900,939)
# ####
tr_in=np.zeros((len(tr_idx),256,256,4))
tr_out=np.zeros((len(tr_idx),256,256,100,1))
for idx in tr_idx:
    input_path= str1+path_list_1[idx]
    output_path=str2+path_list_2[idx]
    tr_in_item= sio.loadmat(input_path)
    tr_in_item=tr_in_item['Phi_crude']
    tr_in[idx,]=tr_in_item

    tr_out_item= sio.loadmat(output_path)
    tr_out_item= tr_out_item['n_pred']
    tr_out_item= (tr_out_item-1.33)*100
    tr_out_item=np.expand_dims(tr_out_item, axis=-1) 
    tr_out[idx,]=tr_out_item
print(tr_in.shape,tr_out.shape)

test_in=np.zeros((len(test_idx),256,256,4))
test_out=np.zeros((len(test_idx),256,256,100,1))

for idx in test_idx:
    input_path= str1+path_list_1[idx]
    output_path=str2+path_list_2[idx]
    
    test_in_item= sio.loadmat(input_path)
    test_in_item=test_in_item['Phi_crude']
    test_in[idx-900,]= test_in_item

    test_out_item= sio.loadmat(output_path)
    test_out_item= test_out_item['n_pred']
    test_out_item= (test_out_item-1.33)*100
    test_out_item=np.expand_dims(test_out_item, axis=-1)
    test_out[idx-900,]=test_out_item

print(test_in.shape,test_out.shape)

##
### #### ------------------------------------  Define model ------------------------------
### ##
G_in=Input(shape=(256,256,4))
G_1_1_bn=BatchNormalization()(G_in)
G_1_1_relu=Activation('relu')(G_1_1_bn)
G_1_1_c=Conv2D(8,kernel_size=(3,3),strides=(2,2), padding='same')(G_1_1_relu)
G_1_c_bn= BatchNormalization()(G_1_1_c)
G_1_c_bn_relu= Activation('relu')(G_1_c_bn)
G_1_01=Conv2D(8,kernel_size=(3,3),strides=(1,1), padding='same')(G_1_c_bn_relu)
G_1_02=Conv2D(8,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_in)
G_1_0=Add()([G_1_01,G_1_02])
G_1_0_out=Dropout(0.02)(G_1_0)                        

G_1r_bn=BatchNormalization()(G_1_0_out)
G_1r_relu=Activation('relu')(G_1r_bn)
G_1r_c=Conv2D(8,kernel_size=(3,3),strides=(1,1), padding='same')(G_1r_relu)
G_1r_c_bn=BatchNormalization()(G_1r_c)
G_1r_c_bn_relu=Activation('relu')(G_1r_c_bn)
G_1r_c_out=Conv2D(8,(3,3),strides=(1,1),padding='same')(G_1r_c_bn_relu)

G_1_out=Add()([G_1r_c_out,G_1_0_out])
G_1_out=Dropout(0.02)(G_1_out)                        ## 128x128

G_2_1_bn=BatchNormalization()(G_1_out)
G_2_1_relu=Activation('relu')(G_2_1_bn)
G_2_1_c=Conv2D(16,kernel_size=(3,3),strides=(2,2), padding='same')(G_2_1_relu)
G_2_c_bn=BatchNormalization()(G_2_1_c)
G_2_c_bn_relu= Activation('relu')(G_2_c_bn)
G_2_01=Conv2D(16,kernel_size=(3,3),strides=(1,1), dilation_rate=(2,2),padding='same')(G_2_c_bn_relu)
G_2_02=Conv2D(16,kernel_size=(3,3),strides=(2,2), padding='same')(G_1_out)
G_2_0_out=Add()([G_2_01,G_2_02])

G_2r_bn=BatchNormalization()(G_2_0_out)
G_2r_relu=Activation('relu')(G_2r_bn)
G_2r_c=Conv2D(16,kernel_size=(3,3),strides=(1,1), padding='same')(G_2r_relu)
G_2r_c_bn=BatchNormalization()(G_2r_c)
G_2r_c_bn_relu=Activation('relu')(G_2r_c_bn)
G_2r_c_out=Conv2D(16,(3,3),strides=(1,1),padding='same')(G_2r_c_bn_relu)
G_2_out=Add()([G_2r_c_out,G_2_0_out])
G_2_out=Dropout(0.02)(G_2_out)                      ## 64x64


G_3_1_bn=BatchNormalization()(G_2_out)
G_3_1_relu=Activation('relu')(G_3_1_bn)
G_3_1_c=Conv2D(32,kernel_size=(3,3),strides=(2,2), padding='same')(G_3_1_relu)
G_3_c_bn=BatchNormalization()(G_3_1_c)
G_3_c_bn_relu= Activation('relu')(G_3_c_bn)
G_3_01=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same')(G_3_c_bn_relu)
G_3_02=Conv2D(32,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_2_out)
G_3_0_out=Add()([G_3_01,G_3_02])

G_3r_bn=BatchNormalization()(G_3_0_out)
G_3r_relu=Activation('relu')(G_3r_bn)
G_3r_c=Conv2D(32,kernel_size=(3,3),strides=(1,1), padding='same')(G_3r_relu)
G_3r_c_bn=BatchNormalization()(G_3r_c)
G_3r_c_bn_relu=Activation('relu')(G_3r_c_bn)
G_3r_c_out=Conv2D(32,(3,3),strides=(1,1),padding='same')(G_3r_c_bn_relu)
G_3_out=Add()([G_3r_c_out,G_3_0_out])
G_3_out=Dropout(0.02)(G_3_out)                         ## 32x32


G_4_1_bn=BatchNormalization()(G_3_out)
G_4_1_relu=Activation('relu')(G_4_1_bn)
G_4_1_c=Conv2D(64,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_4_1_relu)
G_4_c_bn=BatchNormalization()(G_4_1_c)
G_4_c_bn_relu= Activation('relu')(G_4_c_bn)
G_4_01=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same')(G_4_c_bn_relu)
G_4_02=Conv2D(64,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_3_out)
G_4_0_out=Add()([G_4_01,G_4_02])

G_4r_bn=BatchNormalization()(G_4_0_out)
G_4r_relu=Activation('relu')(G_4r_bn)
G_4r_c=Conv2D(64,kernel_size=(3,3),strides=(1,1), padding='same')(G_4r_relu)
G_4r_c_bn=BatchNormalization()(G_4r_c)
G_4r_c_bn_relu=Activation('relu')(G_4r_c_bn)
G_4r_c_out=Conv2D(64,(3,3),strides=(1,1),padding='same')(G_4r_c_bn_relu)
G_4_out=Add()([G_4r_c_out,G_4_0_out])
G_4_out=Dropout(0.02)(G_4_out)                         ## 16x16

G_5_1_bn=BatchNormalization()(G_4_out)
G_5_1_relu=Activation('relu')(G_5_1_bn)
G_5_1_c=Conv2D(128,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_5_1_relu)
G_5_c_bn=BatchNormalization()(G_5_1_c)
G_5_c_bn_relu= Activation('relu')(G_5_c_bn)
G_5_01=Conv2D(128,kernel_size=(3,3),strides=(1,1), activation='relu', padding='same')(G_5_c_bn_relu)
G_5_02=Conv2D(128,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_4_out)
G_5_0_out=Add()([G_5_01,G_5_02])

G_5r_bn=BatchNormalization()(G_5_0_out)
G_5r_relu=Activation('relu')(G_5r_bn)
G_5r_c=Conv2D(128,kernel_size=(3,3),strides=(1,1), padding='same')(G_5r_relu)
G_5r_c_bn=BatchNormalization()(G_5r_c)
G_5r_c_bn_relu=Activation('relu')(G_5r_c_bn)
G_5r_c_out=Conv2D(128,(3,3),strides=(1,1),padding='same')(G_5r_c_bn_relu)
G_5_out=Add()([G_5r_c_out,G_5_0_out])
G_5_out=Dropout(0.02)(G_5_out)                          ## 8x8 



G_6_up_bn=BatchNormalization()(G_5_out)
G_6_up_relu= Activation('relu')(G_6_up_bn)
G_6_up_ct= Conv2DTranspose(128, kernel_size=(3,3), strides=(2,2),padding='same')(G_6_up_relu)
G_6_up_ct_bn=BatchNormalization()(G_6_up_ct)
G_6_up_ct_relu= Activation('relu')(G_6_up_ct_bn)
G_6_up_c_1_out=Conv2D(200,kernel_size=(3,3),strides=(1,1), padding='same')(G_6_up_ct_relu)
G_6_up_c_2_out=Conv2DTranspose(200,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_5_out)
G_6_up_0=Add()([G_6_up_c_1_out,G_6_up_c_2_out])
G_6_up_out=Concatenate()([G_6_up_0,G_4_out])
G_6_up_out=Dropout(0.02)(G_6_up_out)                 ## 16x16 

G_5_up_bn=BatchNormalization()(G_6_up_out)
G_5_up_relu= Activation('relu')(G_5_up_bn)
G_5_up_ct= Conv2DTranspose(100, kernel_size=(3,3), strides=(2,2),padding='same')(G_5_up_relu)
G_5_up_ct_bn=BatchNormalization()(G_5_up_ct)
G_5_up_ct_relu= Activation('relu')(G_5_up_ct_bn)
G_5_up_c_1_out=Conv2D(180,kernel_size=(3,3),strides=(1,1), padding='same')(G_5_up_ct_relu)
G_5_up_c_2_out=Conv2DTranspose(180,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_6_up_out)
G_5_up_0=Add()([G_5_up_c_1_out,G_5_up_c_2_out])
G_5_up_out=Concatenate()([G_5_up_0,G_3_out])
G_5_up_out=Dropout(0.02)(G_5_up_out)                ## 32x32 


G_4_up_bn=BatchNormalization()(G_5_up_out)
G_4_up_relu= Activation('relu')(G_4_up_bn)
G_4_up_ct= Conv2DTranspose(90, kernel_size=(3,3), strides=(2,2),padding='same')(G_4_up_relu)
G_4_up_ct_bn=BatchNormalization()(G_4_up_ct)
G_4_up_ct_relu= Activation('relu')(G_4_up_ct_bn)
G_4_up_c_1_out=Conv2D(160,kernel_size=(3,3),strides=(1,1), padding='same')(G_4_up_ct_relu)
G_4_up_c_2_out=Conv2DTranspose(160,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_5_up_out)
G_4_up_0=Add()([G_4_up_c_1_out,G_4_up_c_2_out])
G_4_up_out=Concatenate()([G_4_up_0,G_2_out])
G_4_up_out=Dropout(0.02)(G_4_up_out)                 ## 64x64 


G_3_up_bn=BatchNormalization()(G_4_up_out)
G_3_up_relu= Activation('relu')(G_3_up_bn)
G_3_up_ct= Conv2DTranspose(80, kernel_size=(3,3), strides=(2,2),padding='same')(G_3_up_relu)
G_3_up_ct_bn=BatchNormalization()(G_3_up_ct)
G_3_up_ct_relu= Activation('relu')(G_3_up_ct_bn)
G_3_up_c_1_out=Conv2D(140,kernel_size=(3,3),strides=(1,1), padding='same')(G_3_up_ct_relu)
G_3_up_c_2_out=Conv2DTranspose(140,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_4_up_out)
G_3_up_0=Add()([G_3_up_c_1_out,G_3_up_c_2_out])
G_3_up_out=Concatenate()([G_3_up_0,G_1_out])
G_3_up_out=Dropout(0.02)(G_3_up_out)                ## 128x128 



G_2_up_bn=BatchNormalization()(G_3_up_out)
G_2_up_relu= Activation('relu')(G_2_up_bn)
G_2_up_ct= Conv2DTranspose(70, kernel_size=(3,3), strides=(2,2),padding='same')(G_2_up_relu)
G_2_up_ct_bn=BatchNormalization()(G_2_up_ct)
G_2_up_ct_relu= Activation('relu')(G_2_up_ct_bn)
G_2_up_c_1_out=Conv2D(120,kernel_size=(3,3),strides=(1,1), padding='same')(G_2_up_ct_relu)
G_2_up_c_2_out=Conv2DTranspose(120,kernel_size=(3,3),strides=(2,2), activation='relu', padding='same')(G_3_up_out)
G_2_up_0=Add()([G_2_up_c_1_out,G_2_up_c_2_out])
G_2_up_out=Concatenate()([G_2_up_0,G_in])
G_2_up_out=Dropout(0.02)(G_2_up_out)                ## 256x256 

G_out_0_bn=BatchNormalization()(G_2_up_out)
G_out_0_relu=Activation('relu')(G_out_0_bn)
G_out_0_c=Conv2D(60,(3,3),strides=(1,1),padding='same')(G_out_0_relu)
G_out_0_c_bn=BatchNormalization()(G_out_0_c)
G_out_0_bn_relu=Activation('relu')(G_out_0_c_bn)
G_out_0_1_1=Conv2D(100,(3,3),strides=(1,1),padding='same')(G_out_0_bn_relu)
G_out_0_1_2=Conv2D(100,(3,3),strides=(1,1),padding='same')(G_2_up_out)
G_out_0_1=Add()([G_out_0_1_1,G_out_0_1_2])
G_out_0_bn_2=BatchNormalization()(G_out_0_1)
G_out_0_c_2=Conv2D(100,(3,3),strides=(1,1),activation='relu',padding='same')(G_out_0_bn_2)
G_out_0=Dropout(0.02)(G_out_0_c_2)

G_out=Reshape((256,256,100,1))(G_out_0)
G=Model(inputs=G_in, outputs=G_out)
G.summary()
G.compile(optimizer=optadam,loss=npcc_3d)
G_history=G.fit(tr_in,tr_out,batch_size=2,epochs=500,verbose=1,validation_split=0.05)
# ##
train_loss=G_history.history['loss']
validation_loss=G_history.history['val_loss']
G.save('model_3t3_4angles_crudephase2tomo_500epochs_33kernel.h5')
rec=G.predict(test_in,batch_size=5,verbose=1)
sio.savemat('rec_3t3_4angles_crudephase2tomo_3t3_500epochs_33kernel.mat',mdict={'rec':rec})
#sio.savemat('test_out_3t3.mat',mdict={'test_out':test_out})
plt.plot(train_loss)
plt.plot(validation_loss)
plt.legend(['loss', 'val_loss'])
plt.show()
