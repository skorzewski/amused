#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from amused.emotions import Emotions
from amused.lemmatizer import SGJPLemmatizer


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python3 ./main.py <TEXT>')
    else:
        tokens = sys.argv[1:]
        lemmatizer = SGJPLemmatizer()
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        emotions = Emotions()
        emotion_coords = emotions.get_coords_from_text(lemmas)
        print(emotion_coords, Emotions.coords_to_name(emotion_coords))
