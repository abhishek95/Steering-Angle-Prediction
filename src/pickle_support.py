import numpy as np
import pickle

train_path = '../datasets/lane detection/mvirgo/full_CNN_train.p'
label_path = '../datasets/lane detection/mvirgo/full_CNN_labels.p'

train_images = pickle.load(open(train_path,'rb'))
labels = pickle.load(open(label_path,'rb'))

train_dest = open('../datasets/lane detection/mvirgo/full_CNN_train3.p','wb')
label_dest = open('../datasets/lane detection/mvirgo/full_CNN_labels3.p','wb')
pickle.dump(train_images, train_dest, protocol=2)
pickle.dump(labels, label_dest, protocol=2)
