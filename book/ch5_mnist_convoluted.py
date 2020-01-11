"""
Deep Learning with Python
Chapter 5: MNIST with a convolutional NN.
"""
from __future__ import print_function
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

def create_model():
    network = models.Sequential()

    # Convnets layers
    # Parameter
    #  - output_depth: number of filters to be used.
    #  - (window_hight, window_width): filter dimensions
    network.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))

    # MaxPooling is used to aggressively reduce the umber of features maps.
    network.add(layers.MaxPooling2D((2, 2)))

    network.add(layers.Conv2D(64, (3,3), activation='relu'))
    network.add(layers.MaxPooling2D((2, 2)))

    network.add(layers.Conv2D(64, (3,3), activation='relu'))

    #  Flatten output
    network.add(layers.Flatten())

    # Classify into digits (0-9)
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(10, activation='softmax'))

    return network

if __name__ == "__main__":

    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # reshape and normalize images
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32')/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Create model
    network = create_model()

    # Output summary of the model
    print(network.summary())

    # Train model
    network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    network.fit(train_images, train_labels, epochs=5, batch_size=64)

    # Evaluate the model
    test_loss, test_acc = network.evaluate(test_images, test_labels)

    # print results
    print('test_acc:', test_acc)