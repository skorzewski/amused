#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re

import numpy as np

from sacred import Experiment

from amused.emotions import EmotionsModel, Emotions
from amused.lemmatizer import SGJPLemmatizer

RE_PUNCT = re.compile(r'([!,.:;?])')


ex = Experiment()


@ex.config
def config():
    trainset_path = 'corpora/wl-20190209-train.bnd'
    testset_path = 'corpora/gold.tsv'
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
        with open(testset_path, 'r') as testset:
            lemmatizer = SGJPLemmatizer()

            emotions_model = None
            emotions = None

            if model == 'neural':
                emotions_model = EmotionsModel(trainset_path,
                                      verbose=verbose,
                                      train_on=method,
                                      epochs=epochs)
            elif model == 'handmade':
                emotions = Emotions()

            distances = []

            reader = csv.DictReader(testset, delimiter='\t', fieldnames=['P', 'At', 'S', 'Ap', 'utt'])
            for row in reader:
                utterance = row['utt']
                tokens = RE_PUNCT.sub(' \1', utterance).split()
                lemmas = [lemmatizer.lemmatize(token) for token in tokens]

                sentic_vector = [0.0, 0.0, 0.0, 0.0]
                if emotions_model:
                    sentic_vector = emotions_model.get_coords_from_text(utterance)
                elif emotions:
                    sentic_vector = emotions.get_coords_from_text(lemmas)

                sentic_vector_str = '\t'.join([str(coord) for coord in sentic_vector])
                print('{}\t{}'.format(sentic_vector_str, utterance),
                      file=results)

                reference = np.array([float(row['P']),
                                      float(row['At']),
                                      float(row['S']),
                                      float(row['Ap'])])
                sentic_vector = np.asarray(sentic_vector)
                distance = np.linalg.norm(sentic_vector - reference)
                distances.append(distance)

            distances = np.asarray(distances)
            rmse = np.sqrt(np.mean(distances**2))

            print('Done.')
            print('RMSE: ' + rmse)
            print('RMSE: ' + rmse, file=results)
