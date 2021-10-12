#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import math
import os
import pickle
import re

import numpy as np
import requests
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from amused.bnd_reader import BNDReader
from amused.freqlist import build_frequency_list, get_freq
from amused.lemmatizer import MorfeuszLemmatizer

__version__ = '0.11.4'


RE_PUNCT = re.compile(r'([!,.:;?])')


def identity_check(a, b):
    """Returns 1.0 if a equals b, 0.0 otherwise"""
    return 1.0 if a == b else 0.0


class Emotions(object):
    """Emotion analysing class"""

    def __init__(self,
                 aggregation_function=np.mean,
                 lemmatizer=None,
                 wsd_method='none'):
        """Constructor.
        Parameters:
        aggregation_function - function that operates on list,
            for aggregating emotion coordinates
        lemmatizer - lemmatizer, a new instance of MorfeuszLemmatizer by default
        wsd_method - method for word sense disambiguation:
            * none (default)
            * simplified_lesk
            * freq_weighted_lesk
            * idf_weighted_lesk
            * lesk_with_bootstrapping
        """
        self.aggregation_function = aggregation_function
        if lemmatizer:
            self._lemmatizer = lemmatizer
        else:
            self._lemmatizer = MorfeuszLemmatizer()
        self.wsd_method = wsd_method

        self._dict = {}
        self._dict2 = {}
        self._wsdict = {}
        self._wsdict2 = {}
        self._freqlist = {}

        dir_name = os.path.dirname(__file__)
        freqlist_file_name = os.path.join(dir_name, 'freqlist.pickle')
        data_file_name = os.path.join(dir_name, 'emo-dict.csv')

        with open(freqlist_file_name, 'rb') as freqlist_file:
            self._freqlist = pickle.load(freqlist_file)
        self._log_freqlist_total = math.log(self._freqlist['__TOTAL__'])
        self._from_csv(data_file_name)

    def _lemmatize(self, utterance):
        tokens = RE_PUNCT.sub(' \1', utterance.lower()).split()
        return [self._lemmatizer.lemmatize(token) for token in tokens]

    def _from_csv(self, path):
        """Read emotion databese from plWordNet-emo CSV file"""
        with open(path) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lemma = row['lemat']
                variant = row['wariant']
                pos = row['czesc_mowy']
                emotions = set(row['emocje'].split(';'))
                examples = [row['przyklad1'], row['przyklad2']]
                lemmatized_examples = [
                    self._lemmatize(e) for e in examples if e and e != 'NULL']
                cowords = set(
                    lemma for example in lemmatized_examples
                    for lemma in example)
                cowords_freq = {
                    coword: self._freqlist[coword] for coword in cowords}
                self._dict.setdefault(lemma, {}).setdefault(
                    variant, set()).update(emotions)
                self._dict2.setdefault((lemma, pos), {}).setdefault(
                    variant, set()).update(emotions)
                self._wsdict.setdefault(lemma, {}).setdefault(
                    variant, {}).update(cowords_freq)
                self._wsdict2.setdefault((lemma, pos), {}).setdefault(
                    variant, {}).update(cowords_freq)

    def _idf(self, word, counter):
        """Returns the inverse document frequency of a word"""
        try:
            return self._log_freqlist_total - math.log(self._freqlist[word])
        except ValueError:
            return 0

    @staticmethod
    def coords_to_basic_name(coords, threshold=0.1):
        """Return emotion name (one of 8 basic emotions)
        for a given emotion coordinates
        """
        indices = sorted(range(4), key=lambda i: abs(coords[i]), reverse=True)
        value = indices[0] + 1
        if coords[indices[0]] < 0:
            value = -value
        if abs(coords[indices[0]]) > threshold:
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

            'joy':          ( 1.0,  0.0,  0.0,  0.0),
            'sadness':      (-1.0,  0.0,  0.0,  0.0),
            'anticipation': ( 0.0,  1.0,  0.0,  0.0),
            'surprise':     ( 0.0, -1.0,  0.0,  0.0),
            'fear':         ( 0.0,  0.0, -1.0,  0.0),
            'anger':        ( 0.0,  0.0,  1.0,  0.0),
            'trust':        ( 0.0,  0.0,  0.0,  1.0),
            'disgust':      ( 0.0,  0.0,  0.0, -1.0),
            'neutral':      ( 0.0,  0.0,  0.0,  0.0),
        }.get(emotion_name, ( 0.0,  0.0,  0.0,  0.0))

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

    def _word_overlap(self, word1, word2):
        """Check how similar are two words"""
        return 1.0 if word1 == word2 else 0.0

    def _similarity(self, words1, words2, function=identity_check):
        """Check how similar are two lists of words"""
        try:
            return sum(function(w1, w2)
                       for w1 in words1
                       for w2 in words2) / (len(words1) * len(words2))
        except ZeroDivisionError:
            return 0.5

    def search_offline(self, lexeme, postag=None, context=[]):
        """Get emotions offline"""
        emotions = []
        if lexeme in self._dict:
            emo_items = {}.items()
            wsd_items = {}.items()
            if postag:
                postag = Emotions.convert_postag(postag)
                emod = self._dict2.get((lexeme, postag),
                                       self._dict[lexeme])
                wsdd = self._wsdict2.get((lexeme, postag),
                                         self._wsdict[lexeme])
            else:
                emod = self._dict[lexeme]
                wsdd = self._wsdict[lexeme]
            if self.wsd_method.startswith('simplified_lesk'):
                variant_rating = {
                    variant: self._similarity(
                        wcounter.keys(), context,
                        function=self._word_overlap)
                    for variant, wcounter in wsdd.items()}
            elif self.wsd_method.startswith('freq_weighted_lesk'):
                variant_rating = {
                    variant: sum(get_freq(word, wcounter)
                                 for word in context)
                    if wcounter else 0.1
                    for variant, wcounter in wsdd.items()}
            elif self.wsd_method.startswith('idf_weighted_lesk'):
                variant_rating = {
                    variant: sum(self._idf(word, wcounter)
                                 for word in context if word in wcounter)
                    if wcounter else 0.1
                    for variant, wcounter in wsdd.items()}
            else:
                variant_rating = {variant: 0.0 for variant in wsdd.keys()}
            max_rating = max(variant_rating.values())
            best_variants = [variant
                             for variant, rating in variant_rating.items()
                             if rating == max_rating]
            if self.wsd_method.endswith('with_bootstrapping'):
                for variant in best_variants:
                    self._wsdict[lexeme][variant].update({
                        word: self._freqlist[word]
                        for word in context})
            emotions = [emod[variant] for variant in best_variants]
        return emotions

    def get_coords(self, lexeme, postag=None, context=None, online=False):
        """Get emotions offline or online"""
        if online:
            emotions = Emotions.search_online(lexeme)
        else:
            emotions = self.search_offline(
                lexeme, postag=postag, context=context)
        if not emotions:
            return 0.0, 0.0, 0.0, 0.0
        return self.aggregate([
            self.aggregate([
                list(Emotions.name_to_coords(emotion))
                for emotion in emotion_list])
            for emotion_list in emotions])

    def mark_word(self, word, lexeme, postag=None, online=False, context=[]):
        """Mark word with emotion markup"""
        if online:
            emotions = Emotions.search_online(lexeme)
        else:
            emotions = self.search_offline(
                lexeme, postag=postag, context=context)
        if not emotions:
            return word
        emotion_vector = self.aggregate([
            self.aggregate([
                list(Emotions.name_to_coords(emotion))
                for emotion in emotion_list])
            for emotion_list in emotions])
        emotion = self.coords_to_basic_name(emotion_vector)
        return '<{emo}>{word}</{emo}>'.format(emo=emotion, word=word)

    def get_coords_from_text(self, lexemes, postags=None, online=False):
        """Return the aggregated emotions from given text.
        The input text should be in lemmatized form,
        as a list of lexemes.
        If an additional argument `postags` is given,
        emotions are calculated using given POS-tag information.
        """
        if postags:
            return self.aggregate(
                self.get_coords(
                    token, postag=postag, context=lexemes, online=online)
                for token, postag in zip(lexemes, postags))
        else:
            return self.aggregate(
                self.get_coords(
                    token, context=lexemes, online=online)
                for token in lexemes)

    def mark_text(self, words, lexemes, postags=None, online=False):
        """Return text with words marked with emotions"""
        if postags:
            return ' '.join(
                self.mark_word(
                    word, token, postag=postag, online=online, context=lexemes)
                for word, token, postag in zip(words, lexemes, postags))
        else:
            return ' '.join(
                self.mark_word(
                    word, token, online=online, context=lexemes)
                for word, token in zip(words, lexemes))


class EmotionsModel(object):
    """Model of emotions trained on a corpus of unannotated dialogs"""

    def __init__(
            self,
            bnd_file_name,
            verbose=False,
            coords_or_labels='coords',
            use_transformer=False,
            train_on='manners',
            epochs=10,
            dim=100,
            dropout=0.5,
            recurrent_dropout=0.0,
            lstm_layers=1,
            dense_layers=2,
            early_stopping=True):
        """Constructor.
        Parameters:
            bnd_file_name – corpus file in BND format
            verbose – verbosity (default False)
            coords_or_labels - train `coords` (default) or `labels`?
            use_transformer - should we start with pretrained Transformer (HerBERT) (default False)?
            train_on - should the model be trained on `manners` (default) or `reporting_clauses`?
            epochs - number of training epochs
            dim - embeddings dimension
            dropout - dropout
            recurrent_dropout - recurrent dropout
            lstm_layers - number of LSTM layers
            dense_layers - number of dense layers
            early_stopping - use early stopping
        """
        if coords_or_labels not in ['coords', 'labels']:
            raise Exception(
                "You can train emotions' *coords* or *labels* only")
        self.coords_or_labels = coords_or_labels

        self.use_transformer = use_transformer

        self.vocabulary = set()
        self.vocab_size = 0
        self.max_length = 0
        self.verbose = verbose
        self.emotions = Emotions(aggregation_function=np.max)
        self.lemmatized_utterances = []
        self.emotion_labels = []
        self.emotion_coords = []

        if train_on == 'manners':
            self._gather_data_from_manners(bnd_file_name)
        elif train_on == 'reporting_clauses':
            self._gather_data_from_reporting_clauses(bnd_file_name)
        else:
            raise Exception(
                'You can train on *manners* or *reporting_clauses* only')

        self._train(
            epochs=epochs,
            dim=dim,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            lstm_layers=lstm_layers,
            dense_layers=dense_layers,
            early_stopping=early_stopping)

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
                    emotion_coords = self.emotions.get_coords_from_text(
                        manner, postags=postags)
                    self.emotion_coords.append(emotion_coords)
                    self.emotion_labels.append(
                        Emotions.coords_to_basic_name(emotion_coords))

    def _gather_data_from_reporting_clauses(self, bnd):
        with BNDReader(bnd) as reader:
            for i, par in enumerate(reader):
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
                    emotion_coords = self.emotions.get_coords_from_text(
                        rc, postags=postags)
                    self.emotion_coords.append(emotion_coords)
                    self.emotion_labels.append(
                        Emotions.coords_to_basic_name(emotion_coords))

    def _train(
            self,
            epochs=10,
            dim=100,
            dropout=0.5,
            recurrent_dropout=0.0,
            lstm_layers=1,
            dense_layers=2,
            early_stopping=True):
        """Process parsed dialogs"""
        self.vocab_size = len(self.vocabulary)
        encoded_utterances = [
            one_hot(' '.join(lemmas), self.vocab_size)
            for lemmas in self.lemmatized_utterances]
        padded_utterances = pad_sequences(
            encoded_utterances, maxlen=self.max_length, padding='post')

        X = padded_utterances

        if self.coords_or_labels == 'labels':
            self.label_encoder = LabelEncoder()
            one_hot_encoder = OneHotEncoder(sparse=False)
            emotion_labels = np.array(self.emotion_labels)
            encoded_labels = self.label_encoder.fit_transform(
                emotion_labels)
            y = one_hot_encoder.fit_transform(encoded_labels.reshape(-1, 1))
        else:
            y = np.array(self.emotion_coords)

        embedding_dim = dim
        output_dim = y.shape[1]

        X = X[:1000]
        y = y[:1000]

        if self.verbose:
            print('Training set size: ', len(self.lemmatized_utterances))
            print('Data shape:', X.shape)
            print('Output dimension:', output_dim)
            print('Vocabulary size:', self.vocab_size)
            print('Embedding dimension:', embedding_dim)
            print('Max. sentence length:', self.max_length)

        if self.use_transformer:
            self.tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
            self.model = AutoModelForSequenceClassification.from_pretrained("allegro/herbert-base-cased", num_labels=output_dim)
        else:
            self.model = Sequential()
            self.model.add(Embedding(
                self.vocab_size, embedding_dim, input_length=self.max_length))

            if lstm_layers >= 2:
                self.model.add(LSTM(
                    dim, dropout=dropout, recurrent_dropout=recurrent_dropout,
                    return_sequences=True))
            if lstm_layers >= 1:
                self.model.add(LSTM(
                    dim, dropout=dropout, recurrent_dropout=recurrent_dropout))
            if lstm_layers == 0:
                self.model.add(Flatten())

            if dense_layers >= 3:
                self.model.add(Dense(dim, activation='tanh'))

            if dense_layers >= 2:
                self.model.add(Dense(dim, activation='tanh'))

            self.model.add(Dense(output_dim, activation='tanh'))

            if self.verbose:
                print(self.model.summary())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)

            if self.coords_or_labels == 'labels':
                self.model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
            else:
                self.model.compile(
                    optimizer='adam',
                    loss='cosine_similarity',
                    metrics=['cosine_similarity'])

            callback = EarlyStopping(monitor='loss', patience=3)
            self.model.fit(X_train, y_train, epochs=epochs, callbacks=[callback])

    def get_coords_from_text(self, text):
        """Predict emotions on text from trained model"""
        if self.use_transformer:
            output = self.model(
                **self.tokenizer.batch_encode_plus(
                    [(text)],
                    padding='longest',
                    add_special_tokens=True,
                    return_tensors='pt'
                )
            )
            prediction = output['logits'].detach().numpy()
        else:
            preprocessed_text = [one_hot(text, self.vocab_size)]
            padded_text = pad_sequences(
                preprocessed_text, maxlen=self.max_length, padding='post')
            prediction = self.model.predict(padded_text)

        if self.coords_or_labels == 'labels':
            predicted_id = np.argmax(prediction, axis=-1)
            predicted_label = self.label_encoder.inverse_transform(predicted_id)[0]
            predicted_coords = Emotions.name_to_coords(predicted_label)
            return list(predicted_coords)

        return tuple(prediction[0].tolist())


if __name__ == '__main__':
    import sys

    if len(sys.argv) <= 1:
        print('Usage: python3 ./emotions.py <LEXEME>')
    else:
        my_emotions = Emotions()
        my_emotion_coords = my_emotions.get_coords(sys.argv[1])
        print(my_emotion_coords, Emotions.coords_to_name(my_emotion_coords))
