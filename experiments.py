#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import re

import numpy as np
from sacred import Experiment
from scipy.spatial.distance import cosine
from sklearn.metrics import precision_recall_fscore_support
import transformers

from amused.emotions import EmotionsModel, Emotions
from amused.lemmatizer import MorfeuszLemmatizer

RE_PUNCT = re.compile(r'([!,.:;?])')


ex = Experiment()


@ex.config
def config():
    trainset_path = 'corpora/wl-20190209-all.bnd'
    testset_path = 'corpora/gold_classes.tsv'
    verbose = True
    coords_or_labels = 'coords'
    use_transformer = False
    method = 'manners'
    wsd_method = 'simplified_lesk'
    epochs = 1
    dim = 100
    dropout = 0.5
    recurrent_dropout = 0.0
    lstm_layers = 1
    dense_layers = 2
    early_stopping = True

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
        coords_or_labels, use_transformer,
        method, wsd_method, model, epochs,
        dim, dropout, recurrent_dropout,
        lstm_layers, dense_layers,
        early_stopping):
    # results_path = 'new_experiment_results/{}_dl{}_ll{}_e{}_dim{}_do{}_rdo{}.tsv'.format(
    #     method, dense_layers, lstm_layers, epochs,
    #     dim, int(10*dropout), int(10*recurrent_dropout))
    transformer_str = "_transformer" if use_transformer else ""
    experiment_name = (
        f"{method}_{wsd_method}"
        if model == 'handmade'
        else f"{method}_{coords_or_labels}{transformer_str}")
    results_path = f"experiment_results/{experiment_name}.tsv"

    with open(results_path, 'w') as results:
        with open(testset_path, 'r') as testset:
            lemmatizer = MorfeuszLemmatizer()

            emotions_model = None
            emotions = None

            if model == 'neural':
                emotions_model = EmotionsModel(
                    trainset_path,
                    verbose=verbose,
                    coords_or_labels=coords_or_labels,
                    use_transformer=use_transformer,
                    train_on=method,
                    epochs=epochs,
                    dim=dim,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    lstm_layers=lstm_layers,
                    dense_layers=dense_layers,
                    early_stopping=early_stopping)
            elif model == 'handmade':
                if method == 'mean':
                    aggregation_function = np.mean
                elif method == 'max':
                    aggregation_function = np.max
                elif method == 'maxabs':
                    aggregation_function = maxabs
                elif method == 'zero':
                    aggregation_function = zero
                emotions = Emotions(
                    aggregation_function=aggregation_function,
                    lemmatizer=lemmatizer,
                    wsd_method=wsd_method)

            distances = []
            cos_dists = []

            predictions = []
            references = []

            reader = csv.DictReader(testset, delimiter='\t', fieldnames=['emo', 'utt'])
            for row in reader:
                utterance = row['utt']
                tokens = RE_PUNCT.sub(' \1', utterance).split()
                lemmas = [lemmatizer.lemmatize(token).lower() for token in tokens]

                sentic_vector = [0.0, 0.0, 0.0, 0.0]
                if emotions_model:
                    sentic_vector = emotions_model.get_coords_from_text(utterance)
                    marked_utterance = utterance
                elif emotions:
                    sentic_vector = emotions.get_coords_from_text(lemmas)
                    marked_utterance = emotions.mark_text(tokens, lemmas)

                sentic_vector_str = '\t'.join(['{:.6}'.format(coord) for coord in sentic_vector])
                print('{}\t{}'.format(sentic_vector_str, utterance),
                      file=results)

                reference_class = row['emo']
                reference = Emotions.name_to_coords(reference_class)
                sentic_vector = np.asarray(sentic_vector)
                distance = np.linalg.norm(sentic_vector - reference)
                distances.append(distance)
                predicted_class = Emotions.coords_to_basic_name(sentic_vector, threshold=0.0)

                # if reference_class != predicted_class:
                #     print('{} != {} {}: "{}"'.format(reference_class, predicted_class, sentic_vector, marked_utterance))
                # else:
                #     print('{} == {} {}: "{}"'.format(reference_class, predicted_class, sentic_vector, marked_utterance))

                predictions.append(predicted_class)
                references.append(reference_class)

                if any(reference):
                    cosine_distance = (cosine(sentic_vector, reference)
                                       if sentic_vector.any()
                                       else 1.0)
                    cos_dists.append(cosine_distance)

            distances = np.asarray(distances)
            mcosd = np.mean(cos_dists)
            mae = np.mean(np.abs(distances))
            rmse = np.sqrt(np.mean(distances**2))

            predictions = np.array(predictions)
            references = np.array(references)

            precision, recall, f_score, support = precision_recall_fscore_support(
                references, predictions, average='micro')

            print('Done.')

            print('MCosD: {}'.format(mcosd))
            print('MAE: {}'.format(mae))
            print('RMSE: {}'.format(rmse))
            print('PREC: {}'.format(precision))
            print('RECALL: {}'.format(recall))
            print('FSCORE: {}'.format(f_score))

            print('MCosD: {}'.format(mcosd), file=results)
            print('MAE: {}'.format(mae), file=results)
            print('RMSE: {}'.format(rmse), file=results)
            print('PREC: {}'.format(precision), file=results)
            print('RECALL: {}'.format(recall), file=results)
            print('FSCORE: {}'.format(f_score), file=results)
