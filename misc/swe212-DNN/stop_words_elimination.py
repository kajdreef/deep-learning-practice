# -*- coding: utf-8 -*-
'''
# An implementation of deep learning for counting symbols
Input:  [10, 12, 10, 11, 2, 2, 2, 1, 1]
Output: words=[2, 10, 1, 12, 11] counts=[3, 2, 2, 1, 1] (Not necessarily in this order)

'''  # noqa

from __future__ import print_function
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from pprint import pprint
from keras.models import Sequential, Model
from keras import layers, metrics
from keras import backend as K
from keras.utils import plot_model
from keras.utils import to_categorical
import math
import random
from six.moves import range
import string, re, collections, os, sys, operator


def build_model(VOCAB_SIZE, reduced_layer_size):
    print('Build model...')
    
    model = Sequential()

    model.add(layers.Dense(VOCAB_SIZE, activation='relu', input_shape=(VOCAB_SIZE, )))
    model.add(layers.Dense(reduced_layer_size, activation='relu'))
    model.add(layers.Dense(VOCAB_SIZE, activation='softmax'))

    return model

class DataGenerator():
    def __init__(self, batch_size=100):
        
        self.stopwords = set(open('./stop_words.txt').read().split(','))
        self.all_words = re.findall('[a-z]{2,}', open(sys.argv[1]).read().lower())
        self.words = [w for w in self.all_words if w not in self.stopwords]

        self.uniqs_in = [''] + list(set(self.all_words).union(self.stopwords))
        self.uniqs_out = [''] + list(set(self.words))

        self.uniqs_indices = dict((w, i) for i, w in enumerate(self.uniqs_in))
        self.indices_uniqs = dict((i, w) for i, w in enumerate(self.uniqs_in))

        self.BATCH_SIZE = batch_size
        self.VOCAB_SW_SIZE = len(self.uniqs_in)
        self.VOCAB_SIZE = len(self.uniqs_out)
        self.BIN_SIZE = math.ceil(math.log(self.VOCAB_SIZE, 2))

    def test_data_generator(self, threshold=.9):

        while True:
            train_word_list = []
            train_word_label = []

            for word in self.all_words:
                train_word_list.append(self.uniqs_indices[word])
                if word in self.stopwords:
                    train_word_label.append(self.uniqs_indices[''])
                else:
                    train_word_label.append(self.uniqs_indices[word])
                
                if len(train_word_label) >= self.BATCH_SIZE:
                    train_x = self.one_hot_encode(train_word_list)
                    train_y = self.one_hot_encode(train_word_label)

                    yield train_x, train_y
                    train_word_list = []
                    train_word_label = []

    def one_hot_encode(self, W):
        x = np.zeros((len(W), self.VOCAB_SW_SIZE))
        
        for i, w in enumerate(W):
            x[i, w] = 1
        return x

if __name__ == '__main__':    

    # Prepare training data
    gen = DataGenerator(batch_size=400)

    #  Build and train model
    model = build_model(gen.VOCAB_SW_SIZE, gen.VOCAB_SIZE)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    model.fit_generator(gen.test_data_generator(), epochs=300, steps_per_epoch=10)

    # Check the first 200 words and show that they are not stopwords
    no_stopwords = []
    for i, w in enumerate(gen.one_hot_encode([gen.uniqs_indices[w] for w in gen.all_words[:200]])):
        y = model.predict(w.reshape((1, gen.VOCAB_SW_SIZE)))
        index = np.argmax(y)
        if gen.indices_uniqs[index] == '':
            continue
        else:
            no_stopwords.append(gen.indices_uniqs[np.argmax(w)])
    
    for w in no_stopwords:
        if w in gen.stopwords:
            print("apparently we missed one... :()")

    pprint(no_stopwords)