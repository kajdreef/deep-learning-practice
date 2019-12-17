"""
Deep Learning with Python
Chapter 3.6: Predicting house prices
- Still need to add cross validation (Sklearn should make this easy.)
"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.datasets import boston_housing
from keras import optimizers
from keras import losses
from keras import metrics

def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mae']
    )

    return model


if __name__ == "__main__":
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    # Normalize the data
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean
    test_data /= std

    # Build and train model
    model = build_model(train_data.shape[1])

    model.fit(
        train_data,
        train_targets,
        batch_size=1,
        validation_data=(test_data, test_targets),
        epochs=20
    )

