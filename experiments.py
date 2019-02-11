#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# from amused.emotions import Emotions
from nltk import word_tokenize
from sacred import Experiment

from amused.bnd_reader import BNDReader
from amused.emotions import EmotionsModel, Emotions
from amused.lemmatizer import SGJPLemmatizer

ex = Experiment()


@ex.config
def config():
    trainset_path = 'corpora/wl-20190209-train.bnd'
    testset_path = 'corpora/wl-20190209-test.bnd'
    verbose = True
    method = 'manners'
    epochs = 20

    if method in ['manners', 'reporting_clauses']:
        model = 'neural'
    elif method in ['mean', 'max']:
        model = 'handmade'


@ex.automain
def run(trainset_path, testset_path, verbose, method, epochs, model):
    results_path = 'experiment_results/{}_{}.tsv'.format(
        method, epochs)
    with open(results_path, 'w') as results:
        with BNDReader(testset_path) as reader:
            emotions_model = None
            emotions = None

            if model == 'neural':
                emotions_model = EmotionsModel(trainset_path,
                                      verbose=verbose,
                                      train_on=method,
                                      epochs=epochs)
            elif model == 'handmade':
                emotions = Emotions()

            for i, par in enumerate(reader):
                if i % 1000 == 0:
                    print('.', end='')
                lemmas = []
                postags = []
                for row in par:
                    lemmas.append(row['lemma'])
                    postags.append(row['pos'])

                emotion_coords = [0.0, 0.0, 0.0, 0.0]
                if emotions_model:
                    emotion_coords = emotions_model.get_coords_from_text(
                        ' '.join(lemmas))
                elif emotions:
                    emotion_coords = emotions.get_coords_from_text(
                        lemmas, postags=postags)

                print('\t'.join([str(coord) for coord in emotion_coords]),
                      file=results)

            print('Done')
