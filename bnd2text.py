#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import math
import random
import re
import sys

from amused.bnd_reader import BNDReader


RE_PUNCT = re.compile(r' ([!,.:;?])')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert BND file to list of utterances')
    parser.add_argument(
        'infile',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='Input BND file',
    )
    parser.add_argument(
        'outfile',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='Output text file',
    )
    parser.add_argument(
        '-m', '--max',
        type=int,
        default=math.inf,
        help='Maximum number of utterances converted',
    )
    parser.add_argument(
        '-s', '--sample',
        type=float,
        default=1.0,
        help='Sample randomly a given fraction of utterances',
    )
    return parser.parse_args()


def main(bnd, txt, max_utt=math.inf, sample=1.0):
    with BNDReader(bnd) as reader:
        utterances_collected = 0
        for par in reader:
            if utterances_collected > max_utt:
                break
            if random.random() > sample:
                continue
            utterance = RE_PUNCT.sub(
                r'\1', ' '.join(row['word'] for row in par))
            print(utterance, file=txt)
            utterances_collected += 1


if __name__ == '__main__':
    args = parse_arguments()
    main(args.infile, args.outfile,
         max_utt=args.max, sample=args.sample)
