#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys

from amused.emotions import Emotions
from amused.lemmatizer import SGJPLemmatizer
from amused.tokenizer import TreebankWordTokenizer

from gensim.models import Word2Vec


RE_DIALOG = re.compile(r'\s—\s(?P<dl>[^—]+)\s—\s(?P<rc>[^—]+).*')


def tokenize_if_possible(text, tokenizer):
    if tokenizer:
        print(text)
        return tokenizer.tokenize(text)
    return [token.strip(',.!?') for token in text.split()]


def lemmatize_if_possible(tokens, lemmatizer):
    if lemmatizer:
        return [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def annotate_dialogs(datafile_path,
                     tokenizer=None,
                     lemmatizer=None,
                     emotion_analyser=None):
    dllems = []
    data = []
    with open(datafile_path, 'r') as datafile:
        for line in datafile:
            m_dialog = RE_DIALOG.match(line)
            if m_dialog:
                dialog_line = m_dialog.group('dl')
                reporting_clause = m_dialog.group('rc')
                dltok = tokenize_if_possible(dialog_line, tokenizer)
                rctok = tokenize_if_possible(reporting_clause, tokenizer)
                dllem = lemmatize_if_possible(dltok, lemmatizer)
                rclem = lemmatize_if_possible(rctok, lemmatizer)
                emotion_coords = emotion_analyser.get_coords_from_text(rclem)
                dllems.append(dllem)
                data.append((dialog_line, reporting_clause, emotion_coords))
    print(dllems)
    wvmodel = Word2Vec(dllems, size=100, window=5, min_count=1, workers=4)
    # TODO


if __name__ == '__main__':
    tokenizer = TreebankWordTokenizer()
    lemmatizer = SGJPLemmatizer()
    emotions = Emotions()
    annotate_dialogs('corpora/Lalka_(Prus)_całość.txt',
                     tokenizer=tokenizer,
                     lemmatizer=lemmatizer,
                     emotion_analyser=emotions)
    # tokens = sys.argv[1:]
    # morphs = [lemmatizer.get_morph(token) for token in tokens]
    # print(morphs)
    # emotion_coords = emotions.get_coords_from_text(lemmas)
    # print(emotion_coords, Emotions.coords_to_name(emotion_coords))
