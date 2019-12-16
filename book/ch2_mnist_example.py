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
    network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
    network.add(layers.Dense(10, activation='softmax'))

    return network

if __name__ == "__main__":

    
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # reshape and normalize images
    train_images = train_images.reshape((60000, 28*28))
    train_images = train_images.astype('float32')/255

    test_images = test_images.reshape((10000, 28*28))
    test_images = test_images.astype('float32')/255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # Create model
    network = create_model()

    # Train model
    network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    # Evaluate the model
    test_loss, test_acc = network.evaluate(test_images, test_labels)

    # print results
    print('test_acc:', test_acc)