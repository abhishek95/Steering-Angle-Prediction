# Steering Angle Prediction using Lane detection

Please see [full report](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/reports/NN_Project.pdf)

## Getting Started

We have built lane detection module and steering angle module. Though steering angle module uses inputs of lane detection, it can be tweaked to use only input images only.

### Prerequisites

All of code is written in Python (2.7)
* Install Keras, with tensorflow as backend
* Install cv2
* Install PIL for python


## Lane Detection

The objective of this module is to detect the lane on the road on which the vehicle is travelling given an image facing towards the road taken while driving. we have used a deep learning architecture inspired from : [Project](https://github.com/mvirgo/mlnd-capstone)
```
The architecture uses a combination of convolutional layers with RELu, batch normalization, pooling, upsampling and dropout layer. The network used a filter size of 3x3 and max pooling layer of size 2x2 with a total parameter count of 181, 693.
```
### Dataset
  We have used datset from [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) for input images, and [here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0). Again thanks to [github](https://github.com/mvirgo/mlnd-capstone)

### Files

* [Lane detection layer](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/lane_detection_layer.py) contains the architecture of CNN layers, loading saved model etc.
* [Lane detection](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/lane_detection.py) uses all the methods in above file to train and save lane detection module. 
* [Full CNN model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/full_CNN_model_30.h5) has been trained for 30 epochs. We can directly load this trained network.



## Steering Angle Predictor

### Dataset

We use [Udacity datset](https://github.com/udacity/self-driving-car/tree/master/datasets/CH2) for our training and testing. [This code](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/read_udacity.py) contains methods to convert training (and testing) data to npy files for easier use.
Both the labels and images saved as 2 separate files.

```
For ex: Training labels  are saved as [Y_center.npy](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/datasets/Y_center.npy)
```

 We have used only centre camera from dataset as evident from file name. 
 
[Dataset overview](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/dataset_overview.py) file visualises the hisotgram of dataset and is used for data augmentation. For now, we have removed some neutral angles with 40% probability. We are saving the new npy files with added keyword 'new' in filename.
 
 ```
 For ex: Y_center.npy becomes Y_center_new.npy
 We are augmenting training files only. We are not touching testing files at all.
 ```
 We will use theese 'new' files our training purposes.

### Vanilla Training

This does not use the lane detector module. It simply trains on input images.
* This [code](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/vanilla_steering_angle.py) - Vanilla steering angle

* There are 2 types of architectures:-
** [VGG model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/steering_vgg_model.py) All details in report.
** [Nvidia model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/steering_nvidia_model.py) All details in report.

* models are stored in 'vanilla' keyword in h5 files.

### Lane detector + Steering angle

First predict, lanes on the road. Then use those images to predict steering anlgle.

* This [code](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/steering_angle.py) - Steering angle

* There are 2 types of architectures:-
** [VGG model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/steering_vgg_model.py) All details in report.
** [Nvidia model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/steering_nvidia_model.py) All details in report.



## Authors

* **Abhishek Singhal** - [abhishek95](https://github.com/abhishek95)
* **Dilip Chakravrthy** - [kavarthapudilip](https://github.com/kavarthapudilip)

## Acknowledgments

* [Udacity](https://github.com/udacity/self-driving-car) for steering angle dataset.
* [Mvirgo](https://github.com/mvirgo/mlnd-capstone) for lane detection module and dataset.


