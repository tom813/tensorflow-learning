import json
import tensorflow.keras as tfk
import csv
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dim = 100
maxlength = 16
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_size = 160000
test_portion = .1

corpus = []
num_sentences = 0

with open('../data/training_cleaned.csv', errors='ignore') as f:
    reader = csv.reader(f)
    for row in reader:
        list_item = []
        list_item.append(row[5])
        if row[0] == '0':
            list_item.append(0)
        else:
            list_item.append(1)
        num_sentences += 1
        corpus.append(list_item)


sentences = []
labels = []
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
vocab_size = len(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=maxlength, padding=padding_type, truncating=trunc_type)

split = int(test_portion * training_size)

test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = labels[0:split]
training_labels = labels[split:training_size]


embeddings_index = {}
with open('../data/glove.6B.100d.txt', errors='ignore', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;


model = tfk.Sequential([
    tfk.layers.Embedding(vocab_size+1, embedding_dim, input_length=maxlength, weights=[embeddings_matrix], trainable=False),
    tfk.layers.Conv1D(64, 5, activation='relu'),
    tfk.layers.Bidirectional(tfk.layers.LSTM(64)),
    tfk.layers.Dense(32, activation='relu'),
    tfk.layers.Dense(1, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences)
testing_labels = np.array(test_labels)
model.summary()
model.fit(training_padded, training_labels, epochs=50, validation_data=(testing_padded, testing_labels), verbose=1)