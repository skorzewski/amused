#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os


class SGJPLemmatizer(object):
    """Simple SGJP-based lemmatizer"""

    def __init__(self):
        """Constructor"""
        self._lemmas = {}
        dir_name = os.path.dirname(__file__)
        data_file_name = os.path.join(dir_name, 'sgjp/sgjp-20181216.tab')
        with open(data_file_name) as tsvfile:
            reader = csv.DictReader(
                tsvfile,
                delimiter='\t',
                fieldnames=['form', 'lemma', 'morph', 'ner', 'qualif'])
            print('Reading morphological dictionary', end='', flush=True)
            self._lemmas = {}
            self._morphs = {}
            for i, row in enumerate(reader):
                self._lemmas.setdefault(row['form'], []).append(row['lemma'])
                self._morphs.setdefault(row['form'], []).append(row['morph'])
                if i % 200000 == 0:
                    print('.', end='', flush=True)
            print('DONE')

    def lemmatize(self, form):
        return self._lemmas.get(form, [form])[0]

    def get_morph(self, form):
        """Get morphological forms"""
        return self._morphs[form]
