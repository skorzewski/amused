#! /usr/bin/env python3
# -*- coding: utf-8 -*-


class BNDReader(object):
    """Reader class for BND file"""

    def __init__(self, file_name):
        """Constructor.
        Parameter: filename – a path to BND file
        """
        self._file = open(file_name, 'r')
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
        self._file.close()
