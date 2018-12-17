#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import sys


def emotion_name(coords):
    """Return emotion name for a given emotion coordinates"""
    indices = sorted(range(4), key=lambda i: abs(coords[i]), reverse=True)
    value = indices[0] + 1
    if coords[indices[0]] < 0:
        value = -value
    value2 = indices[1] + 1
    if coords[indices[1]] < 0:
        value2 = -value2
    if abs(coords[indices[0]]) > 0.5:
        if abs(coords[indices[1]]) > 0.5:
            values = tuple(sorted([value, value2]))
            return {
                (-4, -3): 'aggressiveness',
                (-4, -2): 'cynicism',
                (-4, -1): 'pessimism',
                (-4, 1): 'optimism',
                (-4, 2): 'hope',
                (-4, 3): 'anxiety',
                (-3, -2): 'contempt',
                (-3, -1): 'envy',
                (-3, 1): 'pride',
                (-3, 2): 'dominance',
                (-3, 4): 'outrage',
                (-2, -1): 'remorse',
                (-2, 1): 'morbidness',
                (-2, 3): 'shame',
                (-2, 4): 'unbelief',
                (-1, 2): 'sentimentality',
                (-1, 3): 'despair',
                (-1, 4): 'disapproval',
                (1, 2): 'love',
                (1, 3): 'guilt',
                (1, 4): 'delight',
                (2, 3): 'submission',
                (2, 4): 'curiosity',
                (3, 4): 'awe',
            }[(value, value2)]
        else:
            return {
                -4: 'anticipation',
                -3: 'anger',
                -2: 'disgust',
                -1: 'sadness',
                1: 'joy',
                2: 'trust',
                3: 'fear',
                4: 'surprise',
            }[value]
    elif abs(coords[indices[0]]) > 0.0:
        return {
            -4: 'interest',
            -3: 'annoyance',
            -2: 'boredom',
            -1: 'pensiveness',
            1: 'serenity',
            2: 'acceptance',
            3: 'apprehension',
            4: 'distraction',
        }[value]
    else:
        return 'none'



def aggregate(coords_list):
    """Aggregate alist of emotion coords"""
    if coords_list:
        return tuple([sum(coord_list) / len(coord_list)
                      for coord_list in zip(*coords_list)])
    return (0.0, 0.0, 0.0, 0.0)


def emotion_coords(emotion_name):
    """Return coordinates in emotion space for given emotion name"""
    return {
        'radość':       ( 1.0,  0.0,  0.0,  0.0),
        'smutek':       (-1.0,  0.0,  0.0,  0.0),
        'zaufanie':     ( 0.0,  1.0,  0.0,  0.0),
        'wstręt':       ( 0.0, -1.0,  0.0,  0.0),
        'strach':       ( 0.0,  0.0,  1.0,  0.0),
        'złość':        ( 0.0,  0.0, -1.0,  0.0),
        'zaskoczenie':  ( 0.0,  0.0,  0.0,  1.0),
        'cieszenie się na coś oczekiwanego':
                        ( 0.0,  0.0,  0.0, -1.0),
    }[emotion_name]


def get_emotions(lexeme):
    """Return emotion analysis results
    as a 4-tuple of coordinates in emotion space:
    (+joy/-sadness, +trust/-disgust, +fear/-terror, +surprise/-anticipation)

    Relevant curl command:
        curl --header "Content-Type: application/json" --request POST --data
        '{"task": "all", "tool": "plwordnet", "lexeme": "$lexeme"}'
        http://ws.clarin-pl.eu/lexrest/lex | jq
        '.results.synsets[].units[].emotion_names[]' | sort | uniq -c
    """
    request = requests.post(
        'http://ws.clarin-pl.eu/lexrest/lex',
        headers={
            'Content-Type': 'application/json'},
        data=u'''
            {{
                "task": "all",
                "tool": "plwordnet",
                "lexeme": {}
            }}
        '''.format(lexeme).encode('utf-8'))
    response = request.json()
    emotions = [unit['emotion_names']
                for synset in response['results']['synsets']
                for unit in synset['units']
                ]
    if not emotions:
        return (0.0, 0.0, 0.0, 0.0)
    return aggregate([
        aggregate([
            list(emotion_coords(emotion))
            for emotion in emotion_list])
        for emotion_list in emotions])


def get_emotions_from_text(text):
    """Return the aggregated emotions from given text"""
    return aggregate(get_emotions(token) for token in text.split())


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python3 ./emotions.py <LEXEME>')
    else:
        ec = get_emotions(sys.argv[1])
        print(ec, emotion_name(ec))
