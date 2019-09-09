#! /usr/bin/env python3
# -*- coding: utf-8 -*-


class BNDReader(object):
    """Reader class for BND file"""

    def __init__(self, file):
        """Constructor.
        Parameter: file – a BND file or a path to it
        """
        try:
            self._file = open(file, 'r')
        except TypeError:
            self._file = file
        first_line = next(self._file)
        self.fieldnames = first_line.strip('# ').split()

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        paragraph = []
        while True:
            line = next(self._file).strip()
            if not line:
                break
            paragraph.append(dict(zip(self.fieldnames,
                                      line.split('\t'))))
        return paragraph

    def __exit__(self, exc_type, exc_value, traceback):
        """Close file"""
        try:
            self._file.close()
        except TypeError:
            pass


class BNDLemmaReader(BNDReader):
    """Reader class for BND file; reads only lemmas"""

    def __next__(self):
        paragraph = []
        while True:
            line = next(self._file).strip()
            if not line:
                break
            paragraph.append(dict(zip(self.fieldnames,
                                      line.split('\t')))['lemma'])
        return paragraph
