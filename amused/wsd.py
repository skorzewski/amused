#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os
import re

from amused.lemmatizer import SGJPLemmatizer
from amused.freqlist import build_frequency_list


RE_PUNCT = re.compile(r'([!,.:;?])')


class WSD(object):
    def __init__(self):
        self.lemmatizer = SGJPLemmatizer()
        self.dict = {}
        self.dict2 = {}

    def lemmatize(self, utterance):
        tokens = RE_PUNCT.sub(' \1', utterance.lower()).split()
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def train(self):
        freqlist = build_frequency_list('../corpora/wl-20190209-all.bnd')
        with open('emo-dict.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lemma = row['lemat']
                pos = row['czesc_mowy']
                variant = row['wariant']
                examples = [row['przyklad1'], row['przyklad2']]
                lemmatized_examples = [
                    self.lemmatize(e) for e in examples if e and e != 'NULL']
                cowords = set(
                    lemma for example in lemmatized_examples
                    for lemma in example)
                cowords_freq = {
                    coword: freqlist[coword] for coword in cowords}
                self._dict.setdefault(lemma, {}).setdefault(
                    variant, {}).update(cowords_freq)
                self._dict2.setdefault((lemma, pos), {}).setdefault(
                    variant, {}).update(cowords_freq)


if __name__ == '__main__':
    wsd = WSD()
    wsd.train()
