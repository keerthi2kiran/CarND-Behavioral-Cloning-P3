#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.jpg "Model Visualization"
[image2]: ./examples/center.jpg "Center Driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/original.jpg "Normal Image"
[image7]: ./examples/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model implements the model described in Nvidia paper [https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf]

The model includes RELU layers to introduce nonlinearity. </br>
The images are cropped with Keras Cropping2D function. The data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains one dropout layer just before the final layer in order to reduce overfitting. 

The model was trained and validated on Udacity data set. Data augmentation was applied as follows. </br>
1. Use images from all three cameras. Apply a correction of +-0.15 to the left and right camera images </br>
2. Add additional data by flipping selected images (with bigger steering angle than +-0.15) and flipping the sign of corresponding measurements. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the Lenet model... I thought this model might be appropriate because it is simple and I could easlily test the pipeline...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model like the Nvidia model... I thought this model might be appropriate because Nvidia has used it successfully with outstanding result on real world driving data......

Then I trained the model... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track near the bridge... to improve the driving behavior in these cases, I modified data augmentation (flip the images only for bigger steering angles), added a dropout layer....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used Udacity sample data. Here is an example image of center lane driving:

![alt text][image2]

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 39013 number of data points. I then preprocessed this data by Cropping and normalizing. 
I finally randomly shuffled the data set and put 20% of the data into a validation set. 

Train on 31210 samples, validate on 7803 samples
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 72, 318, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 72, 318, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________



I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by plateauing of validation loss... I used an adam optimizer so that manually training the learning rate wasn't necessary.

31210/31210 [==============================] - 85s - loss: 0.0253 - val_loss: 0.0203
Epoch 2/5
31210/31210 [==============================] - 61s - loss: 0.0208 - val_loss: 0.0166
Epoch 3/5
31210/31210 [==============================] - 55s - loss: 0.0192 - val_loss: 0.0163
Epoch 4/5
31210/31210 [==============================] - 54s - loss: 0.0172 - val_loss: 0.0149
Epoch 5/5
31210/31210 [==============================] - 53s - loss: 0.0165 - val_loss: 0.0143