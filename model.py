import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn

lines = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    [lines.append(line) for line in reader]

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                
                source_path_center = batch_sample[0]
                source_path_left = batch_sample[1]
                source_path_right = batch_sample[2]
                
                filename_center = source_path_center.split('/')[-1]
                filename_left = source_path_left.split('/')[-1]
                filename_right = source_path_right.split('/')[-1]
                
                current_path_center = './data/IMG/' + filename_center
                current_path_left = './data/IMG/' + filename_left
                current_path_right = './data/IMG/' + filename_right
                
                center_image = cv2.imread(current_path_center)
                left_image = cv2.imread(current_path_left)
                right_image = cv2.imread(current_path_right)
                
                angle = float(batch_sample[3])
                flipped_angle = angle * (-1)

                flip_center_image = cv2.flip(center_image,1)       
            
                images.append(center_image)
                angles.append(angle)

                # Help the car to steering properly
                # by adding the images from left and right cameras
                # and adding or substracting them 0.1rad
                images.append(left_image)
                angles.append(angle+0.1)
                
                images.append(right_image)
                angles.append(angle-0.1)
                
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24,5,5, subsample = (2, 2), activation="elu"))
model.add(Conv2D(36,5,5, subsample = (2, 2), activation="elu"))
model.add(Conv2D(48,5,5, subsample = (2, 2), activation="elu"))
model.add(Conv2D(64,3,3, subsample = (1, 1), activation="elu"))
model.add(Conv2D(64,3,3, subsample = (1, 1), activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))
model.summary()

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator,
                    samples_per_epoch=3*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=3*len(validation_samples),
                    nb_epoch=14, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
