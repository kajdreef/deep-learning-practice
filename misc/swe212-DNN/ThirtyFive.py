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

from keras.models import Sequential, Model
from keras import layers, metrics
from keras import backend as K
from keras.utils import plot_model
from keras.utils import to_categorical
import math
from six.moves import range
import string, re, collections, os, sys, operator

stopwords = set(open('../stop_words.txt').read().split(','))
all_words = re.findall('[a-z]{2,}', open(sys.argv[1]).read().lower())
words = [w for w in all_words if w not in stopwords]

uniqs = [''] + list(set(words))
uniqs_indices = dict((w, i) for i, w in enumerate(uniqs))
indices_uniqs = dict((i, w) for i, w in enumerate(uniqs))

indices = [uniqs_indices[w] for w in words]

BATCH_SIZE = 1000 # batch size
VOCAB_SIZE = len(uniqs)
BIN_SIZE = math.ceil(math.log(VOCAB_SIZE, 2))

def encode_binary(W):
    x = np.zeros((1, BATCH_SIZE, BIN_SIZE, 1))
    for i, w in enumerate(W):
        for n in range(BIN_SIZE): 
            n2 = pow(2, n)
            x[0, i, n, 0] = 1 if (w & n2) == n2 else 0
    return x

print(f'Batch size {BATCH_SIZE}, vocab size {VOCAB_SIZE}, bin size {BIN_SIZE}')
#print(f'Words={words}')
#print(f'Uniqs={uniqs}')
#print(f'Indices={indices}')

def set_weights(clayer):
    wb = []
    b = np.zeros((VOCAB_SIZE), dtype=np.float32)
    w = np.zeros((1, BIN_SIZE, 1, VOCAB_SIZE), dtype=np.float32)
    for i in range(VOCAB_SIZE):
        for n in range(BIN_SIZE):
            n2 = pow(2, n)
            w[0][n][0][i] = 1 if (i & n2) == n2 else -1 #-(BIN_SIZE-1)
    for i in range(VOCAB_SIZE):
        slice_1 = w[0, :, 0, i]
        n_ones = len(slice_1[ slice_1 == 1 ])
        if n_ones > 0: slice_1[ slice_1 == 1 ] = 1./n_ones 
        n_ones = len(slice_1[ slice_1 == -1 ])
        if n_ones > 0: slice_1[ slice_1 == -1 ] = -1./n_ones 
    # Scale the whole thing down one order of magnitude
    #w = w * 0.1
    wb.append(w)
    wb.append(b)
    clayer.set_weights(wb)

def Max(x):
    zeros = K.zeros_like(x)
    return K.switch(K.less(x, 0.9), zeros, x)

def sigmoid_steep(x):
    base = K.ones_like(x) * pow(10, 20)
    return 1. / (1. + K.pow(base, -x))

def Max2(x):
    return sigmoid_steep(x - (1-1/BIN_SIZE))  * x

def Reduce(x):
    return K.pow(x, 15)

def SumPooling2D(x):
    return K.sum(x, axis = 1) 

def model_convnet2D():
    print('Build model...')
    model = Sequential()
    model.add(layers.Conv2D(VOCAB_SIZE, (1, BIN_SIZE),  input_shape=(BATCH_SIZE, BIN_SIZE, 1)))
    set_weights(model.layers[0])
    model.add(layers.ReLU(threshold=1-1/BIN_SIZE))
#    model.add(layers.Lambda(Max))
#    model.add(layers.Lambda(Max2))
#    model.add(layers.Lambda(Reduce))
    model.add(layers.Lambda(SumPooling2D))
    model.add(layers.Reshape((VOCAB_SIZE,)))

    return model, "words-nolearning-{}v-{}f".format(VOCAB_SIZE, BIN_SIZE)

# Split the input in chunks based on the batch size
def chunk(input, batch_size):
    for i in range(0, len(input), batch_size):
        yield input[i:i+batch_size]

model, name = model_convnet2D()
model.summary()
plot_model(model, to_file=name + '.png', show_shapes=True)

intermediate_model = Model(inputs=model.input, outputs=[l.output for l in model.layers])

# Split the prediction up in multiple batches. The size of batches is given above.
final_predictions = np.zeros(VOCAB_SIZE, dtype=np.float32)
for indice_batch in chunk(indices, BATCH_SIZE):
    # Appy binary encoding on the batch
    batch = encode_binary(indice_batch)

    #  Predict the output
    preds = intermediate_model.predict(batch) # outputs a list of 4 arrays

    # Combine all the predictions to the final prediction.
    final_predictions = np.add(final_predictions, preds[-1][0])
 

#  Output the final results.
for w, c in sorted(list(zip(uniqs, final_predictions)), key = operator.itemgetter(1), reverse=True)[:25]:
    print(w + "  -  " + str(c))

