# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:56:54 2018

@author: abhis
"""

import numpy as np
import pickle
import lane_detection_layer
import steering_angle_layer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
from scipy.misc import imresize
from PIL import Image
import matplotlib.pyplot as plt

train_path = 'X_center.npy'
label_path = 'Y_center.npy'
test_path = 'X_test.npy'
test_label_path = 'Y_test.npy'
lane_weights_path = 'full_CNN_model_30.h5'
steering_weights_path = 'steering_angle_model.h5'
output_path = '../datasets/steering angle/'

train_images = np.load(train_path)
labels = np.load(label_path)
mask = np.zeros_like(labels.shape)

plt.hist(labels, bins=50)  # arguments are passed to np.histogram
plt.title("Histogram with 20 bins")
plt.show()

st_angle_threshold = 0.010
neutral_drop_pct = 0.4
mask = []
for idx in range(labels.shape[0]):
    st_angle = labels[idx]
    if abs(st_angle) < st_angle_threshold and np.random.random_sample() <= neutral_drop_pct :
        mask.append(idx)

labels = np.delete(labels,mask)

train_images = np.delete(train_images,mask,axis=0)

plt.hist(labels, bins=50)  # arguments are passed to np.histogram
plt.title("Histogram with 20 bins")
plt.show()

np.save('X_center_new',train_images)
np.save('Y_center_new',labels)


