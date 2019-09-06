#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter

from amused.bnd_reader import BNDReader


def build_frequency_list(bnd, limit=1000000):
    """Build frequency list from BND file
    """
    step = limit / 100
    print('Building frequency list...')
    with BNDReader(bnd) as reader:
        counter = Counter()
        for i, par in enumerate(reader):
            if i % step == 0:
                print('{}%'.format(int( i // step)), flush=True)
            if i > limit:
                print('DONE')
                break
            for row in par:
                counter.update([row['lemma']])
    return counter


if __name__ == '__main__':
    freqlist = build_frequency_list('../corpora/wl-20190209-all.bnd')
    print(dict(freqlist))