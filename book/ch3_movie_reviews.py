from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.datasets import imdb
from keras import optimizers
from keras import losses
from keras import metrics

def vectorize_data(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def create_model():
    network = models.Sequential()
    network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dense(1, activation='sigmoid'))

    return network

if __name__ == "__main__":
    # Load IMDB data set, but limit to only the 10.000 most often used words.
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
 
    train_x = vectorize_data(train_data)
    test_x = vectorize_data(test_data)

    train_y = np.asarray(train_labels).astype('float32')
    test_y = np.asarray(test_labels).astype('float32')


    model = create_model()
    model.compile(
        optimizer=optimizers.RMSprop(lr=0.001),
        loss=losses.binary_crossentropy,
        metrics=['binary_accuracy']
    )

    x_val = train_x[:10000]
    x_train_partial = train_x[10000:]

    y_val = train_y[:10000]
    y_train_partial = train_y[10000:]

    print(x_val.shape)
    print(y_val.shape)
    print(np.unique(y_val))

    history = model.fit(
        x_train_partial,
        y_train_partial,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val)
    )

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(history_dict['binary_accuracy']) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()