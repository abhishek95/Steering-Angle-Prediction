# Steering angle prediction using Lane detection

Steering angle prediction using Lane detection

## Getting Started

We have built lane detection module and steering angle module. Though steering angle module uses inputs of lane detection, it can be tweaked to use only input images only.

### Prerequisites

All of code is written in Python (2.7)
* Install Keras, with tensorflow as backend
* Install cv2
* Install PIL for python


### Lane detection

The objective of this module is to detect the lane on the road on which the vehicle is travelling given an image facing towards the road taken while driving. we have used a deep learning architecture inspired from : [Project](https://github.com/mvirgo/mlnd-capstone)
```
The architecture uses a combination of convolutional layers with RELu, batch normalization, pooling, upsampling and dropout layer. The network used a filter size of 3x3 and max pooling layer of size 2x2 with a total parameter count of 181, 693.
```
 ## Dataset
  We have used datset from [here](https://www.dropbox.com/s/rrh8lrdclzlnxzv/full_CNN_train.p?dl=0) for input images, and [here](https://www.dropbox.com/s/ak850zqqfy6ily0/full_CNN_labels.p?dl=0). Again thanks to [github](https://github.com/mvirgo/mlnd-capstone)

## Files

* [Lane detection layer](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/lane_detection_layer.py) contains the architecture of CNN layers, loading saved model etc.
* [Lane detection](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/lane_detection.py) uses all the methods in above file to train and save lane detection module. 
* [Full CNN model](https://github.com/abhishek95/Steering-Angle-Prediction/blob/master/src/full_CNN_model_30.h5) has been trained for 30 epochs. We can directly load this trained network.



End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

