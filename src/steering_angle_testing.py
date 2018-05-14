# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:17:19 2018

@author: abhis
"""

import numpy as np
import pickle
import lane_detection_layer
#import steering_vgg_model
import steering_nvidia_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from scipy.misc import imresize
from PIL import Image

train_path = 'X_center.npy'
label_path = 'Y_center.npy'
test_path = 'X_test.npy'
test_label_path = 'Y_test.npy'
lane_weights_path = 'full_CNN_model_30.h5'
steering_weights_path = 'steering_nvidia_model.h5'

#test_images = np.load(train_path)
#test_labels = np.load(label_path)
test_images = np.load(test_path)
test_labels = np.load(test_label_path)


def resize(train_images):
    original_size = train_images.shape[1:]
    print ('image original size',original_size)
    small_images = np.array(list(map(lambda image: imresize(image,(80,160,3)), train_images[:])))
    input_shape = small_images.shape[1:]
    print ('resize',input_shape)
    return small_images

small_images = resize(test_images)
#predict lanes
print ('loading lane detection model..')
input_shape = small_images.shape[1:]
lane_model = lane_detection_layer.load_trained_model(input_shape,lane_weights_path)
print ('predicting lanes..')
lane_images = lane_detection_layer.lanes(small_images,lane_model)
print (lane_images.shape)

#see sample preformance of lane detector on steering angle images
sample_image = small_images[80]
sample_image = Image.fromarray(sample_image)
sample_image
sample_image = lane_images[80]
sample_image = Image.fromarray(sample_image)
sample_image

#concatonate
X_test = np.concatenate((small_images,lane_images),axis=2)
y_test = test_labels


#train steering angle CNN
input_shape = X_test.shape[1:]
#angle_model = steering_angle_layer.train_model(X_train,y_train,X_val,y_val)
print ('loading steering angle model')
angle_model = steering_nvidia_model.load_trained_model(input_shape,steering_weights_path)
print (angle_model.summary())

#bins
bins = np.linspace(-1,1,50)

#testing
print ('testing accuracy..')
test_result = angle_model.predict(X_test[:])
test_result= test_result.reshape(test_result.shape[0])
test_score  = np.sqrt(np.mean(np.square(test_result- y_test)))
print (test_score)
test_result_digitize = np.digitize(test_result,bins)
y_test_digitize = np.digitize(y_test,bins)
test_accuracy = np.mean(test_result_digitize==y_test_digitize)
print (test_accuracy)


