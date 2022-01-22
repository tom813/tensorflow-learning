import json
import tensorflow.keras as tfk
import csv
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

embedding_dim = 120
maxlength = 16
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>'
training_size = 160000
test_portion = .1

corpus = []