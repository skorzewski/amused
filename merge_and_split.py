#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect annotated utterances to create the emotion corpus"""
import numpy
import os
import random
import re

RE_OTHER = re.compile(r'[^GLNORSUWZ]')


def letter_to_vector(letter):
    return {
        'G': (0, 0, 1, 0),  # gniew - anger
        'L': (0, 0, -1, 0), # lęk - fear
        'N': (0, 0, 0, 0),  # neutralność - neutral
        'O': (0, 1, 0, 0),  # oczekiwanie - anticipation
        'R': (1, 0, 0, 0),  # radość - joy
        'S': (-1, 0, 0, 0), # smutek - sadness
        'U': (0, 0, 0, 1),  # ufność - trust
        'W': (0, 0, 0, -1), # wstręt - disgust
        'Z': (0, -1, 0, 0), # zaskoczenie - surprise
    }[letter]


def letters_to_vector(letters):
    vectors_list = [letter_to_vector(letter) for letter in letters]
    return tuple([numpy.mean(coords) for coords in zip(*vectors_list)])


def main(input_dir, output_dir):
    data = {}
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                # print(line)
                # print(line.split('\t'))
                try:
                    annotation, utterance = line.split('\t', maxsplit=2)
                except ValueError:
                    continue
                data.setdefault(utterance, []).append(annotation)
    with open(os.path.join(output_dir, 'train/train.tsv'), 'w') as trainset, \
            open(os.path.join(output_dir, 'dev-0/in.tsv'), 'w') as devset_in, \
            open(os.path.join(output_dir, 'dev-0/out.tsv'), 'w') as devset_out, \
            open(os.path.join(output_dir, 'test-A/in.tsv'), 'w') as testset_in, \
            open(os.path.join(output_dir, 'test-A/out.tsv'), 'w') as testset_out:
        for utterance, annotations in data.items():
            if len(annotations) > 3:
                r = random.random()
                if r < 0.2:
                    print('{}\t{}'.format(annotations, utterance),
                          file=trainset)
                else:
                    sentic_vectors = [letters_to_vector(annotation) for annotation in annotations]
                    sentic_vector = tuple([numpy.mean(coords) for coords in zip(*sentic_vectors)])
                    sentic_vector_str = '\t'.join(str(coord) for coord in sentic_vector)
                    if r < 0.4:
                        print(utterance,
                              file=devset_in)
                        print(sentic_vector_str,
                              file=devset_out)
                    else:
                        print(utterance,
                              file=testset_in)
                        print(sentic_vector_str,
                              file=testset_out)


if __name__ == '__main__':
    main('corpora/crowd', 'challenge')
