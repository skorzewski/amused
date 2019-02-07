#! /usr/bin/env python3
# -*- coding: utf-8 -*-


class BNDReader(object):
    """Reader class for BND file"""

    def __init__(self, filename):
        """Constructor.
        Parameter: filename – a path to BND file
        """
        self._pars = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f.read().splitlines()):
                if i == 0:
                    self.fieldnames = line.strip('# ').split()
                    current_par = []
                elif not line:
                    self._pars.append(current_par)
                    current_par = []
                else:
                    current_par.append(dict(zip(self.fieldnames,
                                                line.split('\t'))))

    def pars(self):
        """Return paragraphs list"""
        return self._pars

    def rows(self):
        """Return rows iterator"""
        return (row for par in self._pars for row in par)

