#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from random import choices


def train_test_split(dataset_path, trainset_path, testset_path,
                     ratio=0.2):
    weights = [ratio, 1 - ratio]
    with open(dataset_path, 'r') as data, \
            open(trainset_path, 'w') as train, \
            open(testset_path, 'w') as test:
        file_choices = [test, train]
        outfile = choices(file_choices, weights=weights)[0]
        for i, line in enumerate(data):
            if not line.strip():
                outfile = choices(file_choices, weights=weights)[0]
            if i == 0:
                train.write(line)
                test.write(line)
            else:
                outfile.write(line)


if __name__ == '__main__':
    train_test_split('corpora/wl-20190209-all.bnd',
                     'corpora/wl-20190209-train.bnd',
                     'corpora/wl-20190209-test.bnd')
