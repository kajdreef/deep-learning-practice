"""
Deep Learning with Python
Chapter 3.5: Classifying newswires
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import reuters
from keras import optimizers
from keras import losses
from keras import metrics


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1

    return results

def create_model(input_size, hidden_layers, output_nodes):
    network = models.Sequential()
    network.add(layers.Dense(64, activation='relu', input_shape=(input_size,)))

    for layer in hidden_layers:
        network.add(layers.Dense(layer, activation='relu'))

    network.add(layers.Dense(output_nodes, activation='softmax'))

    return network


if __name__ == "__main__":
    # Get data set
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data( num_words=10000)

    # Vecotrize the data
    train_x = vectorize_sequences(train_data)
    test_x = vectorize_sequences(test_data)


    #  to_categorical one-hot encodes the assignments.
    train_y = to_categorical(train_labels)
    test_y = to_categorical(test_labels)


    # Split training data into training and validation data set
    x_val = train_x[:1000]
    x_train = train_x[1000:]

    y_val = train_y[:1000]
    y_train = train_y[1000:]

    # Create model and train it
    network = create_model(10000, [64], 46)

    network.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = network.fit(
        x_train,
        y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val)
    )

    # Print accuracy
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc') 
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
