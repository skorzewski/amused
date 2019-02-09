#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from amused.emotions import Emotions, EmotionsModel
from amused.lemmatizer import SGJPLemmatizer

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 ./main.py "<TEXT1>" "<TEXT2>" ...')
    else:
        model = EmotionsModel('corpora/wl-20190209-all.bnd',
                              verbose=True,
                              train_on='reporting clauses',
                              epochs=1)
        for text in sys.argv[1:]:
            # tokens = text.split()
            # lemmatizer = SGJPLemmatizer()
            # lemmas = [lemmatizer.lemmatize(token) for token in tokens]
            # emotions = Emotions()
            # emotion_coords = emotions.get_coords_from_text(lemmas)
            # print(emotion_coords, Emotions.coords_to_name(emotion_coords))

            emotion_coords = model.get_coords_from_text(text)
            print('"{}"\n\t{}\t{}'.format(
                text,
                emotion_coords,
                Emotions.coords_to_name(emotion_coords)))
