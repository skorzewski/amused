#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Word2vec model"""

import os

from gensim.models import Word2Vec

from amused.bnd_reader import BNDLemmaReader


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    corpus = os.path.join(dir_name, '../corpora/wl-20190209-all.bnd')
    model_file_path = os.path.join(dir_name, 'wv.model')

    with BNDLemmaReader(corpus) as reader:
        model = Word2Vec(reader)
        model.save(model_file_path)

        print(model.wv['dom'])
