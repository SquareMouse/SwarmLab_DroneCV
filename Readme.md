# DroneCV

Running a binary classifier on images of CX-OF drones to assist in a real-time object tracker. Runs Keras with VGG features on a tensorflow backend. This jupyter notebook was run on the swarm lab windows machine with a Nvidia GTX1080, and the models (which are too large for github) are found inside ~/Documents/DroneCV in that machine and are also with Brian Liao.


## Methods

### Creating the Dataset
A labeled dataset of images that either contain the CX-OF drone (drone) or don't (not_drone) are created. Color thresholding on the drone's blue color in Brian Liao's OpenCV script cut bounding boxes around drones in frames of videos that were either found online or recorded in the office. Recording our video in office involved flying the drone at multiple heights while rotating it about its vertical axis to expose all sides of the drone. Our dataset was then extracted from the frames.

The "not-drone" dataset was collected from ImageNet classes for trees and model airplanes, and in office. Recording in office involved recording a video of the office where no drone was in sight, and running Brian Liao's region proposal script to generate the same distribution of cutouts that the classifier will encounter durring operation.

The above methods produced thousands of images, which is not enough to for our model. Therefore real-time data augmentation is applied. During training, each loaded batch of images receives random gamma correction to simulate exposure variance, temperature change to simulate lighting variance, gaussian blurring to simulate scale variance, along with random rotation and vertical mirroring. 

## Training
VGG16, a pre-trained CNN model originally used for ImageNet was modified by freezing the convolutional layers so that they serve as feature extractors, then replacing the final fully connected layers with our own architecture to train over our dataset, inspiration taken from [JGuillaumin](https://github.com/keras-team/keras/issues/4465). 

## Debugging Architecture
Initial architecture flattened VGG features -> Dense 4096 node hidden layer with ReLU activation -> Dense 4096 node hidden layer with ReLu activation -> 2 output softmax. After overnight training with the ADAM optimizer the training loss does not converge even for the training set. The next attempt used only 10 images for each the drone and not-drone classes and the model was still unable to overfit. This indicates a problem in the model. This strange behavior was then fixed by removing both fully connected ReLU layers in the architecture and freezing the VGG layers from being trained. With this modification the model effectively performs binary softmax regression on VGG features. Unexpectedly, after 10 minutes of training on the Nvidia GTX 1080, both the training and validation accuracy reach 100%. This behavior is likely due to the high similarity between all frames in the videos, so our validation set is too similar to our training set despite being selected at random.

After more data was included in the dataset, it was shown to not perform as well as we had hoped. Thus the same architecture was re-trained on the new data to make `vgg_softmax.hdf5`. This model was trained over ~10 epochs

As an experimenet, one dense hidden layer consisting of 4096 nodes with ReLU activation and 50% dropout regularization was added before the softmax layer. This model appears to perform well but has significantly increased model complexity. This trained model is saved as `vgg_reul_softmax.hdf5`. This model was trained over 72 epochs

Some analysis on these two models can be found in [this google doc](https://docs.google.com/document/d/1a3qfoPPhjjoKoTa-_sjnhL4BDINTQPrOar8-0oLa0bA/edit?usp=sharing)

The vgg_softmax architecture was then also trained on 72 epochs but no analysis has yet to be done on that model.

## Thanks!
* **Brian Liao**  [bCom5](https://github.com/bCom5/drone-cv)
* **Brian Kilberg** 
