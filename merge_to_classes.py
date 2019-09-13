#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""Collect annotated utterances to create the emotion corpus"""
import numpy
import os
import re

RE_OTHER = re.compile(r'[^GLNORSUWZ]')


def letter_to_name(letter):
    return {
        'G': 'anger',  # gniew - anger
        'L': 'fear',  # lęk - fear
        'N': 'neutral',  # neutralność - neutral
        'O': 'anticipation',  # oczekiwanie - anticipation
        'R': 'joy',  # radość - joy
        'S': 'sadness',  # smutek - sadness
        'U': 'trust',  # ufność - trust
        'W': 'disgust',  # wstręt - disgust
        'Z': 'surprise',  # zaskoczenie - surprise
    }[letter]


def annotations_to_summary(annotations, ignore_neutral=False):
    summary = {}
    for letters in annotations:
        for letter in letters:
            name = letter_to_name(letter)
            if ignore_neutral and name == 'neutral':
                continue
            if name not in summary:
                summary[name] = 0
            summary[name] += 1
    return summary


def main(input_dir, threshold=3, ignore_neutral=False):
    data = {}
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), 'r',
                  encoding='utf-8') as file:
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
    # totals = {}
    for utterance, annotations in data.items():
        summary = annotations_to_summary(
            annotations, ignore_neutral=ignore_neutral)
        if not summary:
            summary = {'neutral': threshold + 1}
        emotion = max(summary.keys(), key=lambda key: summary[key])
        # print(utterance, annotations, summary, best_letter)
        if summary[emotion] > threshold:
            print('{}\t{}'.format(emotion, utterance))
            # if emotion not in totals:
            #     totals[emotion] = 0
            # totals[emotion] += 1
    # print(totals)
    # print(sum(totals.values()))


if __name__ == '__main__':
    main('corpora/crowd', threshold=3, ignore_neutral=True)
