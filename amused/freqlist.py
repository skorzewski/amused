#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle

from collections import Counter

from amused.bnd_reader import BNDReader


def get_freq(word, counter):
    try:
        return 1 / counter[word]
    except (KeyError, ZeroDivisionError):
        return 0.0


def build_frequency_list(bnd):
    """Build frequency list from BND file
    """
    step = 10000
    print('Building frequency list', end='')
    with BNDReader(bnd) as reader:
        counter = Counter()
        for i, par in enumerate(reader):
            if i % step == 0:
                print('.', end='', flush=True)
            for row in par:
                counter.update([row['lemma']])
    print('DONE')
    with open('freqlist.pickle', 'wb') as data_file:
        pickle.dump(counter, data_file)
    return counter


if __name__ == '__main__':
    freqlist = build_frequency_list('../corpora/wl-20190209-all.bnd')
