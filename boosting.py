#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from amused.emotions import Emotions

from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

import numpy as np

from sklearn.model_selection import train_test_split


class BNDReader(object):
    """Reader class for BND file"""

    def __init__(self, filename):
        self._pars = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f.read().splitlines()):
                if i == 0:
                    self.fieldnames = line.strip('# ').split()
                    current_par = []
                elif not line:
                    self._pars.append(current_par)
                    current_par = []
                else:
                    current_par.append(dict(zip(self.fieldnames,
                                                line.split('\t'))))

    def pars(self):
        return self._pars

    def rows(self):
        return (row for par in self._pars for row in par)


def fit_model_on_dialogs(bnd):
    """Process parsed dialogs"""
    emotions = Emotions(aggregation_function=np.max)
    vocabulary = set()
    max_length = 0
    lemmatized_utterances = []
    emotion_coords = []
    reader = BNDReader(bnd)
    for par in reader.pars():
        lemmas = []
        postags = []
        manner = []
        for row in par:
            if row['dip'] == 'utt':
                lemmas.append(row['lemma'])
            elif row['dip'] == 'manner':
                manner.append(row['lemma'])
                postags.append(row['pos'])
        if lemmas and manner:
            if len(lemmas) > max_length:
                max_length = len(lemmas)
            vocabulary.update(lemmas)
            lemmatized_utterances.append(lemmas)
            emotion_coords.append(
                emotions.get_coords_from_text(manner, postags=postags))

    vocab_size = len(vocabulary)
    encoded_utterances = [one_hot(' '.join(lemmas), vocab_size)
                          for lemmas in lemmatized_utterances]
    padded_utterances = pad_sequences(encoded_utterances, maxlen=max_length, padding='post')

    X = padded_utterances
    y = np.array(emotion_coords)

    embedding_dim = 100

    print('Vocabulary size:', vocab_size)
    print('Embedding dimension:', embedding_dim)
    print('Max. sentence length:', max_length)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(4, activation='sigmoid'))
    print(model.summary())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1)

    return model, {'vocab_size': vocab_size, 'max_length': max_length}


def predict(model, params, text):
    """Predict emotions on text from trained model"""
    preprocessed_text = [one_hot(text, params['vocab_size'])]
    padded_text = pad_sequences(preprocessed_text, maxlen=params['max_length'], padding='post')
    return model.predict(padded_text)


if __name__ == '__main__':
    emotion_model, model_params = fit_model_on_dialogs('corpora/wl-lalka.bnd')

    sample_text = 'Ale z niego wariat!'

    coords = predict(emotion_model, model_params, sample_text)
    emotion = Emotions.coords_to_name(coords[0])
    print(sample_text, coords, emotion)
