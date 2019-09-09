#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pointwise mutual information"""

import math
import os
import pickle

from collections import Counter

from amused.bnd_reader import BNDReader


class PMI(object):
    def __init__(self, logarithmic=False):
        """Prepare data for calculating PMI"""
        self.logarithmic = logarithmic

        dir_name = os.path.dirname(__file__)
        pickle_file_name = os.path.join(dir_name, 'pmi.pickle')
        corpus = os.path.join(dir_name, '../corpora/wl-20190209-all.bnd')

        try:
            with open(pickle_file_name, 'rb') as data_file:
                self.occurrences, self.cooccurrences = pickle.load(data_file)
        except (EOFError, FileNotFoundError):
            self.occurrences = Counter()
            self.cooccurrences = Counter()
            step = 10000
            print('Calculating pointwise mutual information', end='')
            with BNDReader(corpus) as reader:
                for i, par in enumerate(reader):
                    if i % step == 0:
                        print('.', end='', flush=True)
                    if i > 500000:
                        break
                    lemmas = [row['lemma'] for row in par]
                    lemma_pairs = list(set((l1, l2)
                                           for l1 in lemmas
                                           for l2 in lemmas
                                           if l1 < l2))
                    try:
                        self.occurrences.update(lemmas)
                        self.cooccurrences.update(lemma_pairs)
                    except MemoryError:
                        break
            print('DONE')
            if self.logarithmic:
                self.occurrences = {
                    word: math.log(count)
                    for word, count in self.occurrences.items()}
                self.cooccurrences = {
                    word_pair: math.log(count)
                    for word_pair, count in self.cooccurrences.items()}
            else:
                self.occurrences = dict(self.occurrences)
                self.cooccurrences = dict(self.cooccurrences)
            with open('pmi.pickle', 'wb') as data_file:
                pickle.dump((self.occurrences, self.cooccurrences), data_file)

    def get(self, word1, word2):
        """Get pointwise mutual information of two words"""
        if word1 > word2:
            word1, word2 = word2, word1
        if self.logarithmic:
            return self.cooccurrences[(word1, word2)] - (
                self.occurrences[word1] + self.occurrences[word2])
        try:
            return self.cooccurrences[(word1, word2)] / (
                self.occurrences[word1] * self.occurrences[word2])
        except ZeroDivisionError:
            return 0.0


if __name__ == '__main__':
    pmi = PMI()
    word1 = input('First word: ')
    word2 = input('Second word: ')
    print('PMI =', pmi.get(word1, word2))
    print('log PMI =', math.log(pmi.get(word1, word2)))
