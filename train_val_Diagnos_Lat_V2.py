## Final result is saving the Authors' model trained on Gigantix.
## Re-written by: Waziha Kabir
## Started: January 4, 2021
## Last Modification Date: January 26, 2021

######################################### Import Libraries [START]
##### Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
import cv2
import gc
import copy
import tensorflow as tf
import keras

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,Adam
from keras.models import load_model
######################################### Import Libraries [END]

######################################### Import Diagnos Training Data [START]
## Training Data
import h5py
with h5py.File("./prepro_42K_plus_data/Xy_Train.h5", "r") as f:
#     print(f.file)
    X_train = np.array(f['X'])
    y_train = np.array(f["y"])

#print('** total number of training Label:', sum(y_train[:]))

print('** Shape of X_train: ', X_train.shape)
print('** Shape of y_train: ', y_train.shape)
######################################## Import Diagnos Training Data [END]

######################################### Import Diagnos Validation Data [START]
## Training Data
import h5py
with h5py.File("./prepro_8K_plus_data/Xy_Val.h5", "r") as f_v:
#     print(f.file)
    X_val = np.array(f_v['X'])
    y_val = np.array(f_v["y"])

#print('** total number of validation Label:', sum(y_val[:]))

print('** Shape of X_val: ', X_val.shape)
print('** Shape of y_val: ', y_val.shape)
######################################## Import Diagnos Validation Data [END]


######################################## Function: Build Base Model [START]
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

base_model = InceptionV3(include_top = False, 
                               weights = None)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers:
    layer.trainable = True

print('** Model is built')
######################################## Function: Build Base Model [END]

######################################## Function: Train Model with Generator [START]
def train_model_big(model,data,label,batch_size,epoch,lr,model_save_path,log_dir):
    """Train model
    Args: 
        model: a keras model
        data: image data array
        label: label array (without one-hot encoder)
        batch_size
        lr: learning rate
    Return:
        a keras History object
    """
    
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    from keras.utils.np_utils import to_categorical
    label = to_categorical(label, num_classes=2)
    index = int(data.shape[0] * 0.8)
    
    train_X = data[0:index]
    train_y = label[0:index]
    
    valid_X = data[index:]
    valid_y = label[index:]
    
#     flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')
    train_gen = ImageDataGenerator(rotation_range=30, 
                         width_shift_range= 10.0, 
                         height_shift_range= 10.0, 
                         rescale=1/255)
    valid_gen = ImageDataGenerator(rescale=1/255)
    
    train_generator = train_gen.flow(train_X, 
                     train_y, 
                     batch_size=batch_size, 
                     shuffle=True)
    valid_generator = valid_gen.flow(valid_X, valid_y, batch_size = batch_size)
    
    
#     keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    tf_record = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_check = keras.callbacks.ModelCheckpoint(filepath = model_save_path, 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, mode='auto', period=1)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

    his = model.fit_generator(train_generator, 
                        steps_per_epoch = train_X.shape[0] // batch_size, 
                        epochs=epoch,
                        validation_data=valid_generator, 
                        validation_steps= valid_X.shape[0] // batch_size, 
                        callbacks = [tf_record, model_check, reduce_lr])
    
    return model,his

print('** Function to train the model is working')
######################################## Function: Train Model with Generator [END]

######################################## Import CUDA [START]
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
######################################## Import CUDA [START]

######################################## Function: Train Model with Generator [START]
#model_save_path = 'weights.{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5'
#model, his= train_model_big(model,X_train,y_train,40,1000,0.1,
 #                           model_save_path = model_save_path,
  #                          log_dir = '.')
######################################## Function: Train Model with Generator [END]

######################################## Preparing Data for prediction [START]
train_X = X_train[:]
train_y = y_train[:]
valid_X = X_val[:]
valid_y = y_val[:]
#train_X = train_X / 255
valid_X = valid_X / 255

print('** Training data is splitted into Train and Validation Sets')
######################################## Preparing Data for prediction [END]

print('** Training has started....')

######################################## Function: Train Model without Generator [START]
def train_model_big_no_gen(model,train_X,train_y,valid_X,valid_y,batch_size,epoch,lr,model_save_path,log_dir):
    """Train model
    Args: 
        model: a keras model
        data: image data array
        label: label array (without one-hot encoder)
        batch_size
        lr: learning rate
    Return:
        a keras History object
    """
    
    optimizer = keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    from keras.utils.np_utils import to_categorical
    train_y = to_categorical(train_y, num_classes=2) 
    valid_y = to_categorical(valid_y, num_classes=2)
#     flow(self, X, y, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png')
    train_gen = ImageDataGenerator(rotation_range=30, 
                         width_shift_range= 10.0, 
                         height_shift_range= 10.0, 
                         rescale=1/255)
    
    train_generator = train_gen.flow(train_X, 
                     train_y, 
                     batch_size=batch_size, 
                     shuffle=True)
#     keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    tf_record = keras.callbacks.TensorBoard(log_dir=log_dir)
    model_check = keras.callbacks.ModelCheckpoint(filepath = model_save_path, 
                                                  monitor='val_loss', 
                                                  verbose=0, 
                                                  save_best_only=True, 
                                                  save_weights_only=False, mode='auto', period=1)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = 5)

    his = model.fit_generator(train_generator, 
                        steps_per_epoch = train_X.shape[0] // batch_size, 
                        epochs=epoch,
                        validation_data=(valid_X,valid_y),
                        callbacks = [tf_record, model_check, reduce_lr])
#                        callbacks = [tf_record, model_check])
    
    return model,his
######################################## Function: Train Model without Generator [END]

######################################## Function: Train Model without Generator [START]
model, his = train_model_big_no_gen(model,
                                    train_X,train_y,
                                    valid_X,valid_y,
                                    84,20,0.1,
                                   './model_diagnos_final_v2/inception_no/weights.{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5','./logs_diagnos_final_v2/inception_no')

print('** Training has completed successfully.')
######################################## Function: Train Model without Generator [END]

######################################## Save Model [START]
#model.save("./model/model_2_ep.h5")

print("** Model is saved in the location 'model_diagnos_final_v2'. ")
######################################## Save Model [END]

######################################## Show accuracy and loss of Model [START]
#his.history['val_accuracy'][1]
#his.history['val_loss'][1]
######################################## Show accuracy and loss of Model [END]

