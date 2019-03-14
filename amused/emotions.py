#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
import requests
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split

from amused.bnd_reader import BNDReader


class Emotions(object):
    """Emotion analysing class"""

    def __init__(self, aggregation_function=np.mean):
        """Constructor.
        Parameter: aggregation_function - function that operates on list,
        for aggregating emotion coordinates
        """
        self.aggregation_function = aggregation_function
        self._dict = {}
        self._dict2 = {}
        dir_name = os.path.dirname(__file__)
        data_file_name = os.path.join(dir_name, 'plwordnet-emo/emo-dict.csv')
        self._from_csv(data_file_name)

    def _from_csv(self, path):
        """Read emotion databese from plWordNet-emo CSV file"""
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self._dict.setdefault(row['lemat'], {})[row['wariant']] = (
                    row['emocje'].split(';'))
                self._dict2.setdefault(
                    (row['lemat'], row['czesc_mowy']), {})[row['wariant']] = (
                    row['emocje'].split(';'))

    @staticmethod
    def coords_to_basic_name(coords):
        """Return emotion name (one of 8 basic emotions)
        for a given emotion coordinates
        """
        indices = sorted(range(4), key=lambda i: abs(coords[i]), reverse=True)
        value = indices[0] + 1
        if coords[indices[0]] < 0:
            value = -value
        if abs(coords[indices[0]]) > 0.1:
            return {
                -1: 'sadness',
                -2: 'surprise',
                -3: 'fear',
                -4: 'disgust',
                1: 'joy',
                2: 'anticipation',
                3: 'anger',
                4: 'trust',
            }[value]
        return 'neutral'


    @staticmethod
    def coords_to_name(coords):
        """Return emotion name for a given emotion coordinates"""
        indices = sorted(range(4), key=lambda i: abs(coords[i]), reverse=True)
        value = indices[0] + 1
        if coords[indices[0]] < 0:
            value = -value
        value2 = indices[1] + 1
        if coords[indices[1]] < 0:
            value2 = -value2
        if abs(coords[indices[0]]) > 0.5:
            if abs(coords[indices[1]]) > 0.5:
                values = tuple(sorted([value, value2]))
                return {
                    (-1, -2): 'disappointment',
                    (-1, 2): 'pessimism',  # or frustration
                    (1, -2): 'delight',  # or frivolity
                    (1, 2): 'optimism',
                    (-1, -3): 'despair',
                    (-1, 3): 'envy',
                    (1, -3): 'guilt',
                    (1, 3): 'pride',
                    (-1, -4): 'remorse',
                    (-1, 4): 'sentimentality',  # or envy
                    (1, -4): 'morbidness',  # or gloat
                    (1, 4): 'love',
                    (-2, -3): 'awe',
                    (-2, 3): 'outrage',  # or rejection
                    (2, -3): 'anxiety',
                    (2, 3): 'aggressiveness',
                    (-2, -4): 'unbelief',
                    (-2, 4): 'curiosity',
                    (2, -4): 'cynicism',
                    (2, 4): 'hope',
                    (-3, -4): 'shame',  # or coercion
                    (-3, 4): 'submission',
                    (3, -4): 'contempt',
                    (3, 4): 'dominance',  # or rivalry
                }[values]
            else:
                return {
                    -1: 'sadness',
                    -2: 'surprise',
                    -3: 'fear',
                    -4: 'disgust',
                    1: 'joy',
                    2: 'anticipation',
                    3: 'anger',
                    4: 'trust',
                }[value]
        elif abs(coords[indices[0]]) > 0.0:
            return {
                -1: 'pensiveness',
                -2: 'distraction',
                -3: 'apprehension',
                -4: 'boredom',
                1: 'serenity',
                2: 'interest',
                3: 'annoyance',
                4: 'acceptance',
            }[value]
        else:
            return 'neutral'


    @staticmethod
    def _localize_name(self, name, lang='en'):
        """Return localized version of a given emotion name"""
        dictionary = {
            'pl': {
                'serenity': 'błogość',
                'joy': 'radość',
                'ecstasy': 'ekstaza',
                'pensiveness': 'zaduma',
                'sadness': 'smutek',
                'grief': 'cierpienie',
                'acceptance': 'akceptacja',
                'trust': 'zaufanie',
                'admiration': 'podziw',
                'boredom': 'znudzenie',
                'disgust': 'wstręt',
                'loathing': 'nienawiść',
                'apprehension': 'obawa',
                'fear': 'strach',
                'terror': 'przerażenie',
                'annoyance': 'irytacja',
                'anger': 'gniew',
                'rage': 'wściekłość',
                'distraction': 'roztargnienie',
                'surprise': 'zaskoczenie',
                'amazement': 'zdumienie',
                'interest': 'ciekawość',
                'anticipation': 'przeczuwanie',
                'vigilance': 'czujność',
                'optimism': 'optymizm',
                'hope': 'nadzieja',
                'anxiety': 'lęk',
                'love': 'miłość',
                'guilt': 'poczucie winy',
                'delight': 'zachwyt',
                'submission': 'uległość',
                'curiosity': 'ciekawość',
                'sentimentality': 'sentymentalizm',
                'awe': 'poruszenie',
                'despair': 'rozpacz',
                'shame': 'wstyd',
                'disappointment': 'rozczarowanie',
                'unbelief': 'szok',
                'outrage': 'oburzenie',
                'remorse': 'żal',
                'envy': 'zazdrość',
                'pessimism': 'pesymizm',
                'contempt': 'pogarda',
                'cynicism': 'cynizm',
                'morbidness': 'makabryczność',
                'aggressiveness': 'agresja',
                'pride': 'duma',
                'dominance': 'dominacja',
                'neutral': 'neutralność',
            }
        }
        return dictionary[lang][name]

    @staticmethod
    def coords_to_basic_localized_name(coords, lang='en'):
        """Return emotion name for a given emotion coordinates
        in given language
        """
        return Emotions._localize_name(
            Emotions.coords_to_basic_name(coords), lang)

    @staticmethod
    def coords_to_localized_name(coords, lang='en'):
        """Return emotion name for a given emotion coordinates
        in given language
        """
        return Emotions._localize_name(
            Emotions.coords_to_name(coords), lang)

    def aggregate(self, coords_list):
        """Aggregate a list of emotion coords"""
        if coords_list:
            return tuple([self.aggregation_function(coord_list)
                          for coord_list in zip(*coords_list)])
        return 0.0, 0.0, 0.0, 0.0

    @staticmethod
    def name_to_coords(emotion_name):
        """Return coordinates in emotion space for given emotion name"""
        return {
            'radość':       ( 1.0,  0.0,  0.0,  0.0),
            'smutek':       (-1.0,  0.0,  0.0,  0.0),
            'cieszenie się na coś oczekiwanego':
                ( 0.0,  1.0,  0.0,  0.0),
            'zaskoczenie czymś nieprzewidywanym':
                ( 0.0, -1.0,  0.0,  0.0),
            'strach':       ( 0.0,  0.0, -1.0,  0.0),
            'złość':        ( 0.0,  0.0,  1.0,  0.0),
            'zaufanie':     ( 0.0,  0.0,  0.0,  1.0),
            'wstręt':       ( 0.0,  0.0,  0.0, -1.0),
            '':             ( 0.0,  0.0,  0.0,  0.0),
            '-':            ( 0.0,  0.0,  0.0,  0.0),
            'NULL':         ( 0.0,  0.0,  0.0,  0.0),
        }[emotion_name]

    @staticmethod
    def convert_postag(postag):
        return {
            'adj': 'przymiotnik',
            'adja': 'przymiotnik',
            'adjc': 'przymiotnik',
            'adjp': 'przymiotnik',
            'adv': 'przyslowek',
            'bedzie': 'czasownik',
            'burk': 'rzeczownik',
            'fin': 'czasownik',
            'ger': 'czasownik',
            'imps': 'czasownik',
            'impt': 'czasownik',
            'inf': 'czasownik',
            'pact': 'czasownik',
            'pant': 'czasownik',
            'pcon': 'czasownik',
            'ppas': 'czasownik',
            'praet': 'czasownik',
            'subst': 'rzeczownik',
        }.get(postag, None)

    @staticmethod
    def search_online(lexeme):
        """Return emotion analysis results as a 4-tuple of coordinates
        in emotion space: (+joy/-sadness, +trust/-disgust, +fear/-terror,
        +surprise/-anticipation)

        Relevant curl command:
            curl --header "Content-Type: application/json" --request POST
            --data '{"task": "all", "tool": "plwordnet", "lexeme": "$lexeme"}'
            http://ws.clarin-pl.eu/lexrest/lex | jq
            '.results.synsets[].units[].emotion_names[]' | sort | uniq -c
        """
        request = requests.post(
            'http://ws.clarin-pl.eu/lexrest/lex',
            headers={
                'Content-Type': 'application/json'},
            data=u'''
                {{
                    "task": "all",
                    "tool": "plwordnet",
                    "lexeme": {}
                }}
            '''.format(lexeme).encode('utf-8'))
        response = request.json()
        return [unit['emotion_names']
                for synset in response['results']['synsets']
                for unit in synset['units']]

    def search_offline(self, lexeme, postag=None):
        """Get emotions offline"""
        if lexeme in self._dict:
            if postag:
                postag = Emotions.convert_postag(postag)
                return self._dict2.get((lexeme, postag),
                                       self._dict[lexeme]).values()
            return self._dict[lexeme].values()
        return []

    def get_coords(self, lexeme, postag=None, online=False):
        """Get emotions offline or online"""
        if online:
            emotions = Emotions.search_online(lexeme)
        else:
            emotions = self.search_offline(lexeme, postag=postag)
        if not emotions:
            return 0.0, 0.0, 0.0, 0.0
        return self.aggregate([
            self.aggregate([
                list(Emotions.name_to_coords(emotion))
                for emotion in emotion_list])
            for emotion_list in emotions])

    def get_coords_from_text(self, lexemes, postags=None, online=False):
        """Return the aggregated emotions from given text.
        The input text should be in lemmatized form,
        as a list of lexemes.
        If an additional argument `postags` is given,
        emotions are calculated using given POS-tag information.
        """
        if postags:
            return self.aggregate(self.get_coords(token, postag=postag, online=online)
                                  for token, postag in zip(lexemes, postags))
        else:
            return self.aggregate(self.get_coords(token, online=online)
                                  for token in lexemes)


class EmotionsModel(object):
    """Model of emotions trained on a corpus of unannotated dialogs"""

    def __init__(self, bnd_file_name, verbose=False, train_on='manners', epochs=10):
        """Constructor.
        Parameters:
            bnd_file_name – corpus file in BND format
            verbose – verbosity (True/False)
        """
        self.vocabulary = set()
        self.vocab_size = 0
        self.max_length = 0
        self.verbose = verbose
        self.emotions = Emotions(aggregation_function=np.max)
        self.lemmatized_utterances = []
        self.emotion_coords = []
        if train_on == 'manners':
            self._gather_data_from_manners(bnd_file_name)
        elif train_on == 'reporting_clauses':
            self._gather_data_from_reporting_clauses(bnd_file_name)
        else:
            raise Exception('You can train on *manners* or *reporting_clauses* only')
        self._train(epochs=epochs)

    def _gather_data_from_manners(self, bnd):
        with BNDReader(bnd) as reader:
            for par in reader:
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
                    if len(lemmas) > self.max_length:
                        self.max_length = len(lemmas)
                    self.vocabulary.update(lemmas)
                    self.lemmatized_utterances.append(lemmas)
                    self.emotion_coords.append(
                        self.emotions.get_coords_from_text(manner, postags=postags))

    def _gather_data_from_reporting_clauses(self, bnd):
        with BNDReader(bnd) as reader:
            for par in reader:
                lemmas = []
                postags = []
                rc = []
                for row in par:
                    if row['dip'] == 'utt':
                        lemmas.append(row['lemma'])
                    elif row['dip'] != 'utt':
                        rc.append(row['lemma'])
                        postags.append(row['pos'])
                if lemmas and rc:
                    if len(lemmas) > self.max_length:
                        self.max_length = len(lemmas)
                    self.vocabulary.update(lemmas)
                    self.lemmatized_utterances.append(lemmas)
                    self.emotion_coords.append(
                        self.emotions.get_coords_from_text(rc, postags=postags))
        pass

    def _train(self, epochs=10):
        """Process parsed dialogs"""
        self.vocab_size = len(self.vocabulary)
        encoded_utterances = [one_hot(' '.join(lemmas), self.vocab_size)
                              for lemmas in self.lemmatized_utterances]
        padded_utterances = pad_sequences(encoded_utterances, maxlen=self.max_length, padding='post')

        X = padded_utterances
        y = np.array(self.emotion_coords)

        embedding_dim = 100

        if self.verbose:
            print('Data shape:', X.shape)
            print('Vocabulary size:', self.vocab_size)
            print('Embedding dimension:', embedding_dim)
            print('Max. sentence length:', self.max_length)

        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, embedding_dim, input_length=self.max_length))
        self.model.add(Flatten())
        self.model.add(Dense(4, activation='sigmoid'))

        if self.verbose:
            print(self.model.summary())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs)

    def get_coords_from_text(self, text):
        """Predict emotions on text from trained model"""
        preprocessed_text = [one_hot(text, self.vocab_size)]
        padded_text = pad_sequences(preprocessed_text, maxlen=self.max_length, padding='post')
        return tuple(self.model.predict(padded_text)[0].tolist())


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 1:
        print('Usage: python3 ./emotions.py <LEXEME>')
    else:
        my_emotions = Emotions()
        my_emotion_coords = my_emotions.get_coords(sys.argv[1])
        print(my_emotion_coords, Emotions.coords_to_name(my_emotion_coords))
