# DroneCV

Running a binary classifier on images of CX-OF drones to assist in a real-time object tracker. Runs Keras with VGG features on a tensorflow backend.


## Methods

Before training a classifier could begin, a labeled dataset must be created, where one contains images of our CX-OF drone and the other doesn't. A color thresholding OpenCV script created by Brian Liao cut bounding boxes around videos that were either found online or recorded in the office. The procedure for recording our own video involved flying the drone at multiple heights while rotating it about the vertical axis to expose all sides of the drone. Our dataset of images containing our drone were then extracted from the video frames.

The dataset of images that do not contain the drone were collected from ImageNet.

20% of the dataset is split into a validation set by selecting images at random.

The above methods produced thousands of images, which is not enough to for our model. Therefore real-time data augmentation is applied to the images that randomly rotate, scale, and flip horizontally the images of our drone.

With the dataset, it was decided to use a pre-trained model, specifically VGG16, as a feature extractor. An example by [JGuillaumin](https://github.com/keras-team/keras/issues/4465) was the starting point. After overnight training it was found the training loss does not converge even for the training set. Another attempt supplied only 10 images for each the drone and not-drone classes and the model was still unable to overfit. This strange behavior was fixed by removing both fully connected ReLU layers in the architecture and freezing the VGG layers from being trained. With this modification, the model effectively performs binary softmax regression on VGG features. Unexpectedly, after 10 minutes of training on the Nvidia GTX 1080 both the training and validation accuracy reach 100%. This behavior is likely due to the high similarity between all frames in the videos, so our validation set is not distinct enough from our training set despite being selected at random.

A new labeled dataset containing 100 images of drones was created by Brian Liao, and it is found that the model only gets 80% accuracy on this dataset. Failure cases involve images with significant occlusion, top-down or bottom-up views of the drone, and views that include the top of the drone when the drone color scheme is red (our training set drones are blue).


## Authors

* **Chandler Chen** - *Initial work* - [SquareMouse](https://github.com/SquareMouse/)
