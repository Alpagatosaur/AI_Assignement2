# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
from keras.applications.vgg16 import VGG16

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
import cv2
import glob
import splitfolders

# PATH
path_file = os.getcwd()
os.path.dirname(os.path.abspath(path_file))

output_dir = os.path.join(path_file, 'train_val_test')
img_dir = os.path.join(path_file, 'data_img')
model_dir = os.path.join(path_file, 'model.h5')

BATCH_SIZE = 100
EPOCHS = 5
learning_rate = 0.0001
MIN_IMGS_IN_CLASS=500;
image_size = 50;
nb_classes = 11

list_all_img = glob.glob(img_dir + "/*/*.png")

# Resize img
def preprocess(my_img):
    height, width = my_img.shape[:2]
    scale = image_size / max(height, width)
    dx = (image_size - scale * width) / 2
    dy = (image_size - scale * height) / 2
    trans = np.array([[scale, 0, dx], [0, scale, dy]], dtype=np.float32)
    my_img = cv2.warpAffine(my_img, trans, (image_size, image_size), flags=cv2.INTER_AREA)
    my_img = cv2.resize(my_img, (image_size, image_size))
    return my_img

# Shuffle
def mixing(images, labels):
    images = np.array(images)
    labels = np.array(labels)
    s = np.arange(images.shape[0])
    np.random.seed(1337)
    np.random.shuffle(s)
    images=images[s]
    labels=labels[s]
    return images, labels

# Read all img in the folder
def load_img(list_path):
    list_images = []
    list_labels = []
    for img_path in list_path:
        path_split = img_path.split("\\")
        img = cv2.imread(img_path)
        list_images.append(preprocess(img))
        list_labels.append(int(path_split[-2]))
    return mixing(list_images, list_labels)

# Train model
def train_model(model):
    adam = Adam(lr=learning_rate)
    # SparseCategoricalCrossentropy == to provide labels as integers
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_img, train_labels,
                        batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle = True, verbose=1)
    #loss: 2.1762 - accuracy: 0.1870 - val_loss: 2.6660 - val_accuracy: 0.1656
    #loss: 1.6501 - accuracy: 0.3529 - val_loss: 3.1875 - val_accuracy: 0.1885
    #loss: 1.0064 - accuracy: 0.5773 - val_loss: 1.9096 - val_accuracy: 0.3829
    #loss: 0.4988 - accuracy: 0.8185 - val_loss: 6.1963 - val_accuracy: 0.4919
    #loss: 0.2163 - accuracy: 0.9337 - val_loss: 1.0803 - val_accuracy: 0.6834
    model.summary()
    model.save('model.h5')

    # Show evolution of accurracy/ loss depending on epoch
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 2, 1)
    plt.plot(history.history['accuracy'], label = 'train_accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(history.history['loss'], label = 'train_loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    x = input("Do you want to re-create the model? (yes or no) ->  ")
    if x == "yes":
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            # To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
            splitfolders.ratio(img_dir, output=output_dir, seed=1337, ratio=(.8, .1, .1))

        # Load test dataset
        test_dir = os.path.join(path_file, 'train_val_test', 'test')
        list_train_img = glob.glob(test_dir + "/*/*.png")
        test_img, test_labels = load_img(list_train_img)
        print(test_img.shape)
        print(test_labels.shape)

        # Load train dataset
        train_dir = os.path.join(path_file, 'train_val_test', 'train')
        list_train_img = glob.glob(train_dir + "/*/*.png")
        train_img, train_labels = load_img(list_train_img)
        
        # Create model
        model = Sequential()
        model.add(VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size,3)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))
        
        # Train model
        train_model(model)

        # Test model
        test_loss, test_acc = model.evaluate(test_img, test_labels)
        print('Test accuracy:', test_acc) # 0.68
        print('Test loss:', test_loss) # 1.07
        
    x = input("Do you want to test the model (just test accuracy)? (yes or no) ->  ")
    if x == "yes":
        # Load test dataset
        test_dir = os.path.join(path_file, 'train_val_test', 'test')
        list_train_img = glob.glob(test_dir + "/*/*.png")
        test_img, test_labels = load_img(list_train_img)
        print(test_img.shape)
        print(test_labels.shape)
        
        model = load_model(model_dir)
        
        test_loss, test_acc = model.evaluate(test_img, test_labels)
        print('Test accuracy:', test_acc) # 0.68
        print('Test loss:', test_loss) # 1.07