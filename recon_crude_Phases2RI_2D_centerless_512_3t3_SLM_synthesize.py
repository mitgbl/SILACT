from __future__ import print_function,division
from tensorflow import keras
#from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv3D,MaxPooling2D, MaxPooling3D,Flatten,Reshape,Conv2DTranspose,Conv3DTranspose
from tensorflow.keras.layers import Input,UpSampling2D,Concatenate,BatchNormalization,Input,Activation,Add,Lambda,RepeatVector
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras import optimizers,regularizers 
from tensorflow.keras.optimizers import Adam,Adadelta,Adagrad
from tensorflow.keras.callbacks import EarlyStopping
import os
import os.path
import h5py
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

## ---------------------------------------Optimizers----------------------------------------------------------------------

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

rec_train_low_1=sio.loadmat('rec_3t3_lf_train_33kernel_1.mat',squeeze_me=True) 
rec_train_low_1=rec_train_low_1['rec_lf_train']

rec_train_low_2=sio.loadmat('rec_3t3_lf_train_33kernel_2.mat',squeeze_me=True) 
rec_train_low_2=rec_train_low_2['rec_lf_train']

rec_train_low_3=sio.loadmat('rec_3t3_lf_train_33kernel_3.mat',squeeze_me=True) 
rec_train_low_3=rec_train_low_3['rec_lf_train']

rec_train_low_4=sio.loadmat('rec_3t3_lf_train_33kernel_4.mat',squeeze_me=True) 
rec_train_low_4=rec_train_low_4['rec_lf_train']

rec_train_low_5=sio.loadmat('rec_3t3_lf_train_33kernel_5.mat',squeeze_me=True) 
rec_train_low_5=rec_train_low_5['rec_lf_train']

rec_train_low_6=sio.loadmat('rec_3t3_lf_train_33kernel_6.mat',squeeze_me=True) 
rec_train_low_6=rec_train_low_6['rec_lf_train']
#
# rec_train_low_7=sio.loadmat('rec_3t3_lf_train_33kernel_7.mat',squeeze_me=True) 
# rec_train_low_7=rec_train_low_7['rec_lf_train']

# rec_train_low_8=sio.loadmat('rec_3t3_lf_train_33kernel_8.mat',squeeze_me=True) 
# rec_train_low_8=rec_train_low_8['rec_lf_train']

# rec_train_low_9=sio.loadmat('rec_3t3_lf_train_33kernel_9.mat',squeeze_me=True) 
# rec_train_low_9=rec_train_low_9['rec_lf_train']

# rec_train_low=np.concatenate((rec_train_low_1,rec_train_low_2,rec_train_low_3,rec_train_low_4,rec_train_low_5,rec_train_low_6,rec_train_low_7,rec_train_low_8,rec_train_low_9),axis=0)
# print(rec_train_low.shape)

rec_train_low=np.concatenate((rec_train_low_1,rec_train_low_2,rec_train_low_3,rec_train_low_4,rec_train_low_5,rec_train_low_6),axis=0)
print(rec_train_low.shape)

rec_train_high_1=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_1.mat',squeeze_me=True) 
rec_train_high_1=rec_train_high_1['rec_hf_train']

rec_train_high_2=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_2.mat',squeeze_me=True) 
rec_train_high_2=rec_train_high_2['rec_hf_train']

rec_train_high_3=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_3.mat',squeeze_me=True) 
rec_train_high_3=rec_train_high_3['rec_hf_train']

rec_train_high_4=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_4.mat',squeeze_me=True) 
rec_train_high_4=rec_train_high_4['rec_hf_train']

rec_train_high_5=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_5.mat',squeeze_me=True) 
rec_train_high_5=rec_train_high_5['rec_hf_train']

rec_train_high_6=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_6.mat',squeeze_me=True) 
rec_train_high_6=rec_train_high_6['rec_hf_train']
#
# rec_train_high_7=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_7.mat',squeeze_me=True) 
# rec_train_high_7=rec_train_high_7['rec_hf_train']

# rec_train_high_8=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_8.mat',squeeze_me=True) 
# rec_train_high_8=rec_train_high_8['rec_hf_train']

# rec_train_high_9=sio.loadmat('rec_3t3_hf_train_spectral_scheme2_9.mat',squeeze_me=True) 
# rec_train_high_9=rec_train_high_9['rec_hf_train']

# rec_train_high=np.concatenate((rec_train_high_1,rec_train_high_2,rec_train_high_3,rec_train_high_4,rec_train_high_5,rec_train_high_6,rec_train_high_7,rec_train_high_8,rec_train_high_9),axis=0)
# print(rec_train_high.shape)

rec_train_high=np.concatenate((rec_train_high_1,rec_train_high_2,rec_train_high_3,rec_train_high_4,rec_train_high_5,rec_train_high_6),axis=0)
print(rec_train_high.shape)
rec_train_low=np.expand_dims(rec_train_low,axis=-1)
rec_train_high=np.expand_dims(rec_train_high,axis=-1)

print(rec_train_low.shape,rec_train_high.shape)
str2 = 'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\GT_BPM_3\\'

path_2= r'C:\\Users\\Quadro\\MIT 3D Optics Group Dropbox\\HEK-global\\3t3\\GT_BPM_3'

path_list_2 = os.listdir(path_2)

print(len(path_list_2))
tr_idx=range(600)
test_idx=range(900,966)

tr_out=np.zeros((len(tr_idx),256,256,100,1))
for idx in tr_idx:
    output_path=str2+path_list_2[idx]
    tr_out_item= sio.loadmat(output_path)
    tr_out_item= tr_out_item['n_pred']
    tr_out_item= (tr_out_item-1.33)*100
    tr_out_item=np.expand_dims(tr_out_item, axis=-1) 
    tr_out[idx,]=tr_out_item
    
print(tr_out.shape)

rec_test_low=sio.loadmat('rec_3t3_4angles_crudephase2tomo_3t3_500epochs_33kernel.mat',squeeze_me=True) 
rec_test_low=rec_test_low['rec']

rec_test_high=sio.loadmat('rec_3t3_4angles_crudephase2tomo_high_spectral_scheme2.mat',squeeze_me=True) 
rec_test_high=rec_test_high['rec']

rec_test_low=np.expand_dims(rec_test_low,axis=-1)
rec_test_high=np.expand_dims(rec_test_high,axis=-1)
### #### ------------------------------------  Define model ------------------------------
### ##
G_in_1=Input(shape=(256,256,100,1))
G_in_2=Input(shape=(256,256,100,1))

G_1_1_bn=BatchNormalization()(G_in_1)
G_1_1_relu=Activation('relu')(G_1_1_bn)
G_1_1_c=Conv3D(4,kernel_size=(3,3,3),strides=(2,2,2), padding='same')(G_1_1_relu)
G_1_c_bn= BatchNormalization()(G_1_1_c)
G_1_c_bn_relu= Activation('relu')(G_1_c_bn)
G_1_01=Conv3D(4,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_1_c_bn_relu)
G_1_02=Conv3D(4,kernel_size=(3,3,3),strides=(2,2,2), activation='relu', padding='same')(G_in_1)
G_1_0=Add()([G_1_01,G_1_02])
G_1_out=Dropout(0.02)(G_1_0)                        

G_2_1_bn=BatchNormalization()(G_1_out)
G_2_1_relu=Activation('relu')(G_2_1_bn)
G_2_1_c=Conv3D(8,kernel_size=(3,3,3),strides=(2,2,2), padding='same')(G_2_1_relu)
G_2_c_bn=BatchNormalization()(G_2_1_c)
G_2_c_bn_relu= Activation('relu')(G_2_c_bn)
G_2_01=Conv3D(8,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_2_c_bn_relu)
G_2_02=Conv3D(8,kernel_size=(3,3,3),strides=(2,2,2), padding='same')(G_1_out)
G_2_0=Add()([G_2_01,G_2_02])
G_2_out=Dropout(0.02)(G_2_0)   

G_3_1_bn=BatchNormalization()(G_2_out)
G_3_1_relu=Activation('relu')(G_3_1_bn)
G_3_1_c=Conv3D(16,kernel_size=(3,3,3),strides=(2,2,1), padding='same')(G_3_1_relu)
G_3_c_bn=BatchNormalization()(G_3_1_c)
G_3_c_bn_relu= Activation('relu')(G_3_c_bn)
G_3_01=Conv3D(16,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_3_c_bn_relu)
G_3_02=Conv3D(16,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_2_out)
G_3_0_out=Add()([G_3_01,G_3_02])
G_3_out=Dropout(0.02)(G_3_0_out)  

G_4_1_bn=BatchNormalization()(G_3_out)
G_4_1_relu=Activation('relu')(G_4_1_bn)
G_4_1_c=Conv3D(32,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_4_1_relu)
G_4_c_bn=BatchNormalization()(G_4_1_c)
G_4_c_bn_relu= Activation('relu')(G_4_c_bn)
G_4_01=Conv3D(32,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_4_c_bn_relu)
G_4_02=Conv3D(32,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_3_out)
G_4_0_out=Add()([G_4_01,G_4_02])
G_4_out=Dropout(0.02)(G_4_0_out)  

G_5_1_bn=BatchNormalization()(G_4_out)
G_5_1_relu=Activation('relu')(G_5_1_bn)
G_5_1_c=Conv3D(64,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_5_1_relu)
G_5_c_bn=BatchNormalization()(G_5_1_c)
G_5_c_bn_relu= Activation('relu')(G_5_c_bn)
G_5_01=Conv3D(64,kernel_size=(3,3,3),strides=(1,1,1), activation='relu', padding='same')(G_5_c_bn_relu)
G_5_02=Conv3D(64,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_4_out)
G_5_0_out=Add()([G_5_01,G_5_02])
G_5_out=Dropout(0.02)(G_5_0_out)  
##
##
G_6_up_bn=BatchNormalization()(G_5_out)
G_6_up_relu= Activation('relu')(G_6_up_bn)
G_6_up_ct= Conv3DTranspose(32, kernel_size=(3,3,3), strides=(2,2,1),padding='same')(G_6_up_relu)
G_6_up_ct_bn=BatchNormalization()(G_6_up_ct)
G_6_up_ct_relu= Activation('relu')(G_6_up_ct_bn)
G_6_up_c_1_out=Conv3D(32,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_6_up_ct_relu)
G_6_up_c_2_out=Conv3DTranspose(32,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_5_out)
G_6_up_0=Add()([G_6_up_c_1_out,G_6_up_c_2_out])
G_6_up_out=Concatenate()([G_6_up_0,G_4_out])
G_6_up_out=Dropout(0.02)(G_6_up_out)                 ## 16x16 
##
G_5_up_bn=BatchNormalization()(G_6_up_out)
G_5_up_relu= Activation('relu')(G_5_up_bn)
G_5_up_ct= Conv3DTranspose(16, kernel_size=(3,3,3), strides=(2,2,1),padding='same')(G_5_up_relu)
G_5_up_ct_bn=BatchNormalization()(G_5_up_ct)
G_5_up_ct_relu= Activation('relu')(G_5_up_ct_bn)
G_5_up_c_1_out=Conv3D(16,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_5_up_ct_relu)
G_5_up_c_2_out=Conv3DTranspose(16,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_6_up_out)
G_5_up_0=Add()([G_5_up_c_1_out,G_5_up_c_2_out])
G_5_up_out=Concatenate()([G_5_up_0,G_3_out])
G_5_up_out=Dropout(0.02)(G_5_up_out)                ## 32x32 

##
G_4_up_bn=BatchNormalization()(G_5_up_out)
G_4_up_relu= Activation('relu')(G_4_up_bn)
G_4_up_ct= Conv3DTranspose(8, kernel_size=(3,3,3), strides=(2,2,1),padding='same')(G_4_up_relu)
G_4_up_ct_bn=BatchNormalization()(G_4_up_ct)
G_4_up_ct_relu= Activation('relu')(G_4_up_ct_bn)
G_4_up_c_1_out=Conv3D(8,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_4_up_ct_relu)
G_4_up_c_2_out=Conv3DTranspose(8,kernel_size=(3,3,3),strides=(2,2,1), activation='relu', padding='same')(G_5_up_out)
G_4_up_0=Add()([G_4_up_c_1_out,G_4_up_c_2_out])
G_4_up_out=Concatenate()([G_4_up_0,G_2_out])
G_4_up_out=Dropout(0.02)(G_4_up_out)                 ## 64x64 
##
##
G_3_up_bn=BatchNormalization()(G_4_up_out)
G_3_up_relu= Activation('relu')(G_3_up_bn)
G_3_up_ct= Conv3DTranspose(4, kernel_size=(3,3,3), strides=(2,2,2),padding='same')(G_3_up_relu)
G_3_up_ct_bn=BatchNormalization()(G_3_up_ct)
G_3_up_ct_relu= Activation('relu')(G_3_up_ct_bn)
G_3_up_c_1_out=Conv3D(4,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_3_up_ct_relu)
G_3_up_c_2_out=Conv3DTranspose(4,kernel_size=(3,3,3),strides=(2,2,2), activation='relu', padding='same')(G_4_up_out)
G_3_up_0=Add()([G_3_up_c_1_out,G_3_up_c_2_out])
G_3_up_out=Concatenate()([G_3_up_0,G_1_out])
G_3_up_out=Dropout(0.02)(G_3_up_out)                ## 128x128 
##
##
G_2_up_bn=BatchNormalization()(G_3_up_out)
G_2_up_relu= Activation('relu')(G_2_up_bn)
G_2_up_ct= Conv3DTranspose(1, kernel_size=(3,3,3), strides=(2,2,2),padding='same')(G_2_up_relu)
G_2_up_ct_bn=BatchNormalization()(G_2_up_ct)
G_2_up_ct_relu= Activation('relu')(G_2_up_ct_bn)
G_2_up_c_1_out=Conv3D(1,kernel_size=(3,3,3),strides=(1,1,1), padding='same')(G_2_up_ct_relu)
G_2_up_c_2_out=Conv3DTranspose(1,kernel_size=(3,3,3),strides=(2,2,2), activation='relu', padding='same')(G_3_up_out)
G_2_up_0=Add()([G_2_up_c_1_out,G_2_up_c_2_out])
G_2_up_out=Concatenate()([G_2_up_0,G_in_1])
G_2_up_out=Dropout(0.02)(G_2_up_out)                ## 256x256 
##
G_out_0_bn=BatchNormalization()(G_2_up_out)
G_out_0_relu=Activation('relu')(G_out_0_bn)
G_out_0_1=Add()([G_out_0_relu,G_in_2])
G_out_0_bn_2=BatchNormalization()(G_out_0_1)
G_out_0_c_2=Conv3D(1,(3,3,3),strides=(1,1,1),activation='relu',padding='same')(G_out_0_bn_2)
G_out=Dropout(0.02)(G_out_0_c_2)
###
##G_out=Reshape((256,256,100,1))(G_out_0)
G=Model(inputs=[G_in_1,G_in_2], outputs=G_out)
G.summary()
G.compile(optimizer=optadam,loss=npcc_3d)

G.fit([rec_train_low,rec_train_high],tr_out,batch_size=2,epochs=500,verbose=1,validation_split=0.05)
### ##
# train_loss=G_history.history['loss']
# validation_loss=G_history.history['val_loss']
G.save('model_3t3_scheme2_synthesizer.h5')
rec_final=G.predict([rec_test_low,rec_test_high],batch_size=2,verbose=1)
sio.savemat('rec_3t3_scheme2_final.mat',mdict={'rec_final':rec_final})

