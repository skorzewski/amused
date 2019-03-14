#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect annotated utterances to create the emotion corpus"""
import numpy
import os
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


def main(input_dir):
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
    for utterance, annotations in data.items():
        if len(annotations) > 2:
            sentic_vectors = [letters_to_vector(annotation) for annotation in annotations]
            sentic_vector = tuple([numpy.mean(coords) for coords in zip(*sentic_vectors)])
            sentic_vector_str = '\t'.join(str(coord) for coord in sentic_vector)
            print('{}\t{}'.format(sentic_vector_str, utterance))


if __name__ == '__main__':
    main('corpora/crowd')
