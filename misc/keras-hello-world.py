import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(test_images.shape)

# Create a network
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

# Compiling the model
network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Preparing data
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32')/255

train_labels = to_categorical(train_labels)

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32')/255

test_labels = to_categorical(test_labels)

# Train model
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# Evalutate model
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)