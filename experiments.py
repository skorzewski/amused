#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re

import numpy as np
from sacred import Experiment
from scipy.spatial.distance import cosine

from amused.emotions import EmotionsModel, Emotions
from amused.lemmatizer import SGJPLemmatizer

RE_PUNCT = re.compile(r'([!,.:;?])')


ex = Experiment()


@ex.config
def config():
    trainset_path = 'corpora/wl-20190209-all.bnd'
    testset_path = 'corpora/gold3.tsv'
    verbose = True
    method = 'manners'
    epochs = 1
    dim = 100
    dropout = 0.5
    recurrent_dropout = 0.0
    lstm_layers = 1
    dense_layers = 2

    if method in ['manners', 'reporting_clauses']:
        model = 'neural'
    elif method in ['mean', 'max', 'maxabs', 'zero']:
        model = 'handmade'


def maxabs(l):
    """Return maximum absolute value"""
    max_abs = 0.0
    for e in l:
        if np.abs(e) > np.abs(max_abs):
            max_abs = e
    return max_abs


def zero(l):
    """Return all-zero vector"""
    return 0.0


@ex.automain
def run(trainset_path, testset_path, verbose,
        method, model, epochs,
        dim, dropout, recurrent_dropout,
        lstm_layers, dense_layers):
    results_path = 'new_experiment_results/{}_dl{}_ll{}_e{}_dim{}_do{}_rdo{}.tsv'.format(
        method, dense_layers, lstm_layers, epochs,
        dim, int(10*dropout), int(10*recurrent_dropout))
    with open(results_path, 'w') as results:
        with open(testset_path, 'r') as testset:
            lemmatizer = SGJPLemmatizer()

            emotions_model = None
            emotions = None

            if model == 'neural':
                emotions_model = EmotionsModel(
                    trainset_path,
                    verbose=verbose,
                    train_on=method,
                    epochs=epochs,
                    dim=dim,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    lstm_layers=lstm_layers,
                    dense_layers=dense_layers)
            elif model == 'handmade':
                if method == 'mean':
                    emotions = Emotions(aggregation_function=np.mean)
                elif method == 'max':
                    emotions = Emotions(aggregation_function=np.max)
                elif method == 'maxabs':
                    emotions = Emotions(aggregation_function=maxabs)
                elif method == 'zero':
                    emotions = Emotions(aggregation_function=zero)

            distances = []
            cos_dists = []

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

                sentic_vector_str = '\t'.join(['{:.6}'.format(coord) for coord in sentic_vector])
                print('{}\t{}'.format(sentic_vector_str, utterance),
                      file=results)

                reference = np.array([float(row['P']),
                                      float(row['At']),
                                      float(row['S']),
                                      float(row['Ap'])])
                sentic_vector = np.asarray(sentic_vector)
                distance = np.linalg.norm(sentic_vector - reference)
                distances.append(distance)

                if reference.any():
                    cosine_distance = (cosine(sentic_vector, reference)
                                       if sentic_vector.any()
                                       else 1.0)
                    cos_dists.append(cosine_distance)

            distances = np.asarray(distances)
            mcosd = np.mean(cos_dists)
            mae = np.mean(np.abs(distances))
            rmse = np.sqrt(np.mean(distances**2))

            print('Done.')
            print('MCosD: {}'.format(mcosd))
            print('MAE: {}'.format(mae))
            print('RMSE: {}'.format(rmse))
            print('MCosD: {}'.format(mcosd), file=results)
            print('MAE: {}'.format(mae), file=results)
            print('RMSE: {}'.format(rmse), file=results)
