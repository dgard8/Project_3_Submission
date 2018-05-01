import os
import csv
import argparse
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential, load_model
from keras.layers import Flatten, Lambda, Cropping2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def find_all_folders(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root))
    return result

def addImagePaths(samples, directory):
    with open(directory + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if line[0] != 'center':
                line[4] = directory
                samples.append(line)

def addFlippedImage(image, angle, images, angles):
    image_flipped = np.fliplr(image)
    angle_flipped = -angle
    images.append(image_flipped)
    angles.append(angle_flipped)
    return

def addImage(image, angle, images, angles):
    images.append(image)
    angles.append(angle)

    addFlippedImage(image, angle, images, angles)
    return

def getImage(imageFilePath, imageDirectory):
    image_name = imageDirectory + imageFilePath.split('/')[-1]
    image = cv2.imread(image_name)
    # drive.py passes RGB images, cv2 reads BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:

        shuffle(samples)
        angle_correction = 0.05

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []

            for batch_sample in batch_samples:
                imageDirectory = batch_sample[4] + '/IMG/'
                center_image = getImage(batch_sample[0], imageDirectory)
                left_image = getImage(batch_sample[1], imageDirectory)
                right_image = getImage(batch_sample[2], imageDirectory)
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + angle_correction
                right_angle = center_angle - angle_correction
                
                addImage(center_image, center_angle, images, angles)
                addImage(left_image, left_angle, images, angles)
                addImage(right_image, right_angle, images, angles)
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def defineModel():
    model = Sequential()
        
    # pre-process: crop and normalize
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(160, 320, 3)))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    model.add(Dropout(0.5))
    
    model.add(Dense(50))
    model.add(Dropout(0.5))
    
    model.add(Dense(10))
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('-load',
        dest='load_model',
        action='store_true',
        help='use -load to load model instead of retraining from the start'
        )
    parser.add_argument('-num',
        dest='num_epochs',
        type=int,
        default=2,
        help='number of epochs to use to train'
        )
    parser.add_argument(
        'data_folders',
        type=str,
        nargs='*',
        default='',
        help='Path(s) to data folder(s). If empty we will attempt to use them all'
        )
    parser.set_defaults(load_model=False)
        
    args = parser.parse_args()
    

    if len(args.data_folders) == 0:
        folders = find_all_folders('driving_log.csv', './')
    else:
        folders = args.data_folders

    samples = []
    for folder in folders:
        print(folder)
        addImagePaths(samples, folder)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    if args.load_model:
        print('loading model')
        model = load_model('model.h5')
    
    else:
        print('creating model')
        model = defineModel()

    model.fit_generator(train_generator,
                   samples_per_epoch=len(train_samples)*6,
                   validation_data=validation_generator,
                   nb_val_samples=len(validation_samples)*6,
                   nb_epoch=args.num_epochs)

    model.save('model.h5')

