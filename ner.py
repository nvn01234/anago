import json
import os
import numpy as np
import anago
from glob import glob
from anago.reader import load_data_and_labels

VOCAB_PATH = 'embedding/vocabs.json'
EMBEDDING_PATH = 'embedding/word_embeddings.npy'

train_paths = ["data/train/%s.muc" % s for s in ["Doi_song", "Giai_tri", "Giao_duc", "KH-CN", "Kinh_te","Phap_luat", "The_gioi", "The_thao","Xa_hoi"]]
valid_path = "data/dev/Van_hoa.muc"
test_path = "data/test/Van_hoa.muc"

print('Loading data...')
x_valid, y_valid = load_data_and_labels(valid_path)
x_test, y_test = load_data_and_labels(test_path)
print(len(x_valid), 'valid sequences')
print(len(x_test), 'test sequences')

embeddings = np.load(EMBEDDING_PATH)
vocabs = json.load(open(VOCAB_PATH, "r", encoding="utf8"))

# Use pre-trained word embeddings
model = anago.Sequence(max_epoch=20, embeddings=embeddings, vocab_init=vocabs, patience=4, log_dir="log")

for train_path in train_paths:
    x_train, y_train = load_data_and_labels(train_path)
    print(len(x_train), 'train sequences')
    model.train(x_train, y_train, x_valid, y_valid)
model.eval(x_test, y_test)