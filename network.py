from __future__ import print_function
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import glob
import sys
import io
import pylab as plt

from keras.callbacks import History

def findFiles(path): return glob.glob(path)

# Settings
# TensorBoard logdir
tb_log_dir = './logs'

# Filename for out file
out_filename = 'out_generated.txt'

# Setting up epoch and iterations
# We will print generated text %iterations% times after every %epochs%
epochs = 10
iterations = 3

layers_count = 5

# Count of chars to generate every iteration
count_of_chars = 800

# History callback for matplotlib
history = History()

# TensorBoard Callback
tbCallBack = callbacks.TensorBoard(log_dir=tb_log_dir, histogram_freq=1, batch_size=128, write_graph=True,
                                   write_grads=True, write_images=False, embeddings_freq=1, embeddings_layer_names=None,
                                   embeddings_metadata=None)

# Workaround to solve problem saving history after every epoch
All_history = {
    'acc': list(),
    'val_acc': list(),
    'loss': list(),
    'val_loss': list()
}

sentence = ''

# Write generated passwords
out = io.open(out_filename, 'a')

# Read data
path = 'data/*.*'
text = ''

print('Proceed files')
for filename in findFiles(path):
    print('Filename: ', filename)
    old_len = len(text)
    text += io.open(filename, encoding='cp1251').read()
    print('corpus lenght: ', len(text) - old_len)
    print()
print('All files lenght is:', len(text))

# Getting set of chars for vectors
chars = sorted(list(set(text)))
print('total num of chars set:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 24
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Build the model
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
for i in range(layers_count - 1):
    model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
for iteration in range(iterations):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    out.write('\n')
    out.write('-' * 50)
    out.write("\nIteration " + str(iteration))

    model.fit(x, y,
              batch_size=128,
              epochs=epochs, validation_data=(x, y), callbacks=[history])

    All_history['acc'] += history.history['acc']
    All_history['loss'] += history.history['loss']

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)
        out.write("----- diversity: " + str(diversity))

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence.replace('\n', '\\n') + '"')
        out.write(str('----- Generating with seed: "' + sentence.replace('\n', '\\n') + '"'))
        # sys.stdout.write(generated)

        for i in range(count_of_chars):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            out.write(next_char)
            sys.stdout.flush()
            out.flush()
        print()

plt.ion()
fig = plt.figure()

plt.plot(All_history['acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.show()

plt.plot(All_history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
