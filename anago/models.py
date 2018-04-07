import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda, RepeatVector, Subtract
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from anago.layers import ChainCRF


class BaseModel(object):

    def __init__(self, config, embeddings, ntags):
        self.config = config
        self.embeddings = embeddings
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)


class SeqLabeling(BaseModel):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, embeddings=None, ntags=None):
        # build word embedding
        word_ids = Input(batch_shape=(None, None), dtype='int32')
        if embeddings is None:
            word_embed = Embedding(input_dim=config.vocab_size,
                                        output_dim=config.word_embedding_size,
                                        mask_zero=True)
        else:
            word_embed = Embedding(input_dim=embeddings.shape[0],
                                        output_dim=embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[embeddings])
        word_embeddings = word_embed(word_ids)

        pre_word_ids = Input(batch_shape=(None, None), dtype='int32')
        pre_word_embeddings = word_embed(pre_word_ids) # batch_size, max_sen_len, word_embed_size
        kb_words_input = Input(batch_shape=(None, None, ntags, embeddings.shape[1]), dtype='int32')
        s = K.shape(kb_words_input)
        kb_words = Lambda(lambda x: K.reshape(x, shape=(-1, ntags*embeddings.shape[1])))(kb_words_input) # batch_size, max_sen_len, ntags*word_embed_size
        pre_word_feature = Concatenate()([pre_word_embeddings, kb_words])
        pre_word_feature = Dense(config.pre_word_feature_size)(pre_word_feature)


        pos_embed_weights = [
            np.zeros([1, config.pos_vocab_size-1]),  # padding
            np.identity(config.pos_vocab_size-1)
        ]
        pos_embed_weights = np.concatenate(pos_embed_weights)
        pos_ids = Input(batch_shape=(None, None), dtype='int32')
        pos_embeddings = Embedding(input_dim=pos_embed_weights.shape[0],
                                    output_dim=pos_embed_weights.shape[1],
                                    mask_zero=True,
                                    weights=[pos_embed_weights])(pos_ids)

        # build character based word embedding
        char_ids = Input(batch_shape=(None, None, None), dtype='int32')
        char_embeddings = Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True
                                    )(char_ids)
        s = K.shape(char_embeddings) # batch_size, max_sen_len, max_word_len, char_embed_size
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], config.char_embedding_size)))(char_embeddings)

        fwd_state = LSTM(config.num_char_lstm_units, return_state=True)(char_embeddings)[-2]
        bwd_state = LSTM(config.num_char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
        char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
        # shape = (batch size, max sentence length, char hidden size)
        char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * config.num_char_lstm_units]))(char_embeddings)

        # combine characters and word
        x = Concatenate(axis=-1)([word_embeddings, pos_embeddings, char_embeddings, pre_word_feature])
        x = Dropout(config.dropout)(x)

        x = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True))(x)
        x = Dropout(config.dropout)(x)
        x = Dense(config.num_word_lstm_units, activation='tanh')(x)
        x = Dense(ntags)(x)
        self.crf = ChainCRF()
        pred = self.crf(x)

        sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        self.model = Model(inputs=[word_ids, pos_ids, char_ids, pre_word_ids, kb_words_input, sequence_lengths], outputs=[pred])
        self.config = config