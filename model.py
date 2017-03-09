import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

#Model Architecture: Nvidia paper for end-to-end learning
model = Sequential()
#Crop the sky and bonnet of the car
model.add(Cropping2D(cropping=((65, 23), (1, 1)), input_shape=(160, 320, 3), dim_ordering='tf'))
#Preprocessing
model.add(Lambda(lambda x: x/255.0 - 0.5))
#____________________________________________________________________________________________________
#Layer (type)                     Output Shape          Param #     Connected to
#====================================================================================================
#cropping2d_1 (Cropping2D)        (None, 72, 318, 3)    0           cropping2d_input_1[0][0]
#____________________________________________________________________________________________________
#lambda_1 (Lambda)                (None, 72, 318, 3)    0           cropping2d_1[0][0]
#____________________________________________________________________________________________________
#convolution2d_1 (Convolution2D)  (None, 34, 157, 24)   1824        lambda_1[0][0]
#____________________________________________________________________________________________________
#convolution2d_2 (Convolution2D)  (None, 15, 77, 36)    21636       convolution2d_1[0][0]
#____________________________________________________________________________________________________
#convolution2d_3 (Convolution2D)  (None, 6, 37, 48)     43248       convolution2d_2[0][0]
#____________________________________________________________________________________________________
#convolution2d_4 (Convolution2D)  (None, 4, 35, 64)     27712       convolution2d_3[0][0]
#____________________________________________________________________________________________________
#convolution2d_5 (Convolution2D)  (None, 2, 33, 64)     36928       convolution2d_4[0][0]
#____________________________________________________________________________________________________
#flatten_1 (Flatten)              (None, 4224)          0           convolution2d_5[0][0]
#____________________________________________________________________________________________________
#dense_1 (Dense)                  (None, 100)           422500      flatten_1[0][0]
#____________________________________________________________________________________________________
#dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
#____________________________________________________________________________________________________
#dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
#____________________________________________________________________________________________________
#dropout_1 (Dropout)              (None, 10)            0           dense_3[0][0]
#____________________________________________________________________________________________________
#dense_4 (Dense)                  (None, 1)             11          dropout_1[0][0]
#====================================================================================================
#Total params: 559,419
#Trainable params: 559,419
#Non-trainable params: 0
#____________________________________________________________________________________________________
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))
model.summary()
#Read the images and measurements
lines = []
with open("./Data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
images = []
measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i]
        tokens = source_path.split("/")
        filename = tokens[-1]
        local_path = "./data/IMG/"+filename
        image = cv2.imread(local_path)
        images.append(image)
    # For left and right camera images, apply a correction of 0.15
    correction = 0.15
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement+correction)
    measurements.append(measurement-correction)

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    #Data augmentation: Flip the images and negate the measurements
    #Augment data only if the steering angle is >+-0.1
    if(measurement > -0.10 and measurement < 0.10):
        #measurement = 0
        augmented_images.append(image)
        augmented_measurements.append(measurement)
    else:
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image,1)
        flipped_measurement = measurement*-1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)
X_train = np.array(augmented_images)
print(X_train.shape)
y_train = np.array(augmented_measurements)
print(y_train.shape)
#Use Adam optimizer
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')