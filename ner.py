import json
import os
import numpy as np
import anago
from glob import glob
from anago.reader import load_data_and_labels

VOCAB_PATH = 'embedding/vocabs.json'
EMBEDDING_PATH = 'embedding/word_embeddings.npy'
KB_PATH = 'embedding/kb_words.json'
#
# ignore = "data/train/Doi_song.muc"
# train_paths = "data/train/*.muc"
# valid_path = "data/dev/Doi_song.muc"
# test_path = "data/test/Doi_song.muc"
def main(train_dir, dev_dir, test_dir):
    print('Loading data...')
    x_valid, y_valid = load_data_and_labels(dev_dir)
    x_test, y_test = load_data_and_labels(test_dir)
    print(len(x_valid), 'valid sequences')
    print(len(x_test), 'test sequences')

    embeddings = np.load(EMBEDDING_PATH)
    vocabs = json.load(open(VOCAB_PATH, "r", encoding="utf8"))
    kb_words = json.load(open(KB_PATH, "r", encoding='utf8'))
    for k, v in kb_words.items():
        print(k, len(v))

    # Use pre-trained word embeddings
    model = anago.Sequence(max_epoch=20, embeddings=embeddings, vocab_init=vocabs, log_dir="log")

    # for train_path in glob(train_dir):
    #     if train_path != ignore:
    x_train, y_train = load_data_and_labels(train_dir)
    print(len(x_train), 'train sequences')
    model.train(x_train, kb_words, y_train, x_valid, y_valid)
    model.eval(x_test, kb_words, y_test)