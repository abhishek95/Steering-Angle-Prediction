# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 11:20:38 2018

@author: abhis
"""


# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D, Dense, Flatten
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


batch_size = 64
epochs = 30
pool_size = (2, 2)


def create_model(input_shape):
        
    ### Here is the actual neural network ###
    model = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    model.add(BatchNormalization(input_shape=input_shape))
    
    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))
    
    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))
    
    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    model.add(Dropout(0.2))
    
    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    model.add(Dropout(0.2))
    
    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    model.add(Dropout(0.2))
    
    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    model.add(Dropout(0.2))
    
    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    model.add(Dropout(0.2))
    
    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size)) #(6l,16L,64)
    
    model.add(Flatten())  #(N,6*16*64)
    
    model.add(Dense(2048))
    model.add(Activation(activation='relu'))
    
    model.add(Dense(100))
    model.add(Activation(activation='relu'))
    
    model.add(Dense(10))
    model.add(Activation(activation='relu'))
    
    model.add(Dense(1))
    
    
    ### End of network ###
    return model


def train_model(X_train,y_train,X_val,y_val,filename):
    
    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)
    
    input_shape = X_train.shape[1:]
    model = create_model(input_shape)
    # Compiling and training the model
    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs, verbose=1, validation_data=(X_val, y_val))
    # Freeze layers since training is done
    model.trainable = True
    model.compile(optimizer='Adam', loss='mean_squared_error')    
    # Save model architecture and weights
    model.save(str(filename))    
    # Show summary of model
    model.summary()
    return model
    
def load_trained_model(input_shape,weights_path):
    model = create_model(input_shape)
    model.load_weights(weights_path)
    return model