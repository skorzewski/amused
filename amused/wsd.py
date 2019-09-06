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

    def lemmatize(self, utterance):
        tokens = RE_PUNCT.sub(' \1', utterance.lower()).split()
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def run(self):
        freqlist = build_frequency_list('../corpora/wl-20190209-all.bnd')
        with open('emo-dict.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                lemma = row['lemat']
                examples = [row['przyklad1'], row['przyklad2']]
                lemmatized_examples = [self.lemmatize(e) for e in examples if e and e != 'NULL']
                cowords = set(lemma for example in lemmatized_examples for lemma in example)
                cowords_freq = {coword: freqlist[coword] for coword in cowords}
                if lemmatized_examples:
                    print(lemma, lemmatized_examples)


if __name__ == '__main__':
    wsd = WSD()
    wsd.run()