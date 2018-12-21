#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import os


class SGJPLemmatizer(object):
    """Simple SGJP-based lemmatizer"""

    def __init__(self):
        """Constructor"""
        self._dict = {}
        dir_name = os.path.dirname(__file__)
        data_file_name = os.path.join(dir_name, 'sgjp/sgjp-20181216.tab')
        with open(data_file_name) as tsvfile:
            reader = csv.DictReader(
                tsvfile,
                delimiter='\t',
                fieldnames=['form', 'lemma', 'morph', 'ner', 'qualif'])
            print('Reading morphological dictionary...')
            self._dict = {row['form']: row['lemma'] for row in reader}
            print('Done.')

    def lemmatize(self, form):
        return self._dict[form]
