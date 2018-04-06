import json
import os

import numpy as np

import anago
from anago.reader import load_data_and_labels

VOCAB_PATH = 'embedding/vocabs_2.json'
EMBEDDING_PATH = 'embedding/word_embeddings_2.npy'

train_path = "data/train/Doi_song.muc"
valid_path = "data/dev/Doi_song.muc"
test_path = "data/test/Doi_song.muc"

print('Loading data...')
x_train, y_train = load_data_and_labels(train_path)
x_valid, y_valid = load_data_and_labels(valid_path)
x_test, y_test = load_data_and_labels(test_path)
print(len(x_train), 'train sequences')
print(len(x_valid), 'valid sequences')

embeddings = np.load(EMBEDDING_PATH)
vocabs = json.load(open(VOCAB_PATH, "r", encoding="utf8"))

# Use pre-trained word embeddings
model = anago.Sequence(max_epoch=50, embeddings=embeddings, batch_size=100, early_stopping=False, vocab_init=vocabs)
model.train(x_train, y_train, x_valid, y_valid)
model.eval(x_test, y_test)