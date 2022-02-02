import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.utils as ku
import numpy as np

tokenizer = Tokenizer()

data = open('../data/sonnets.txt').read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in corpus:
    #print(line)
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print(token_list)
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        #print(n_gram_sequence)
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])
print(max_sequence_len)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

predictors, label = input_sequences[:,:-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=total_words)

model = tfk.Sequential([
    tfk.layers.Embedding(total_words, 100, input_length=max_sequence_len-1),
    tfk.layers.LSTM(64, return_sequences=True),
    tfk.layers.Dropout(0.2),
    tfk.layers.LSTM(64),
    tfk.layers.Dense(total_words/2, activation='relu', kernel_regularizer=tfk.regularizers.l2(0.01)),
    tfk.layers.Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    predictors,
    label,
    epochs=50,
    verbose=1
)