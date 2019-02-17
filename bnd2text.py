#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import math
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
    return parser.parse_args()


def main(bnd, txt, max_utt=math.inf):
    with BNDReader(bnd) as reader:
        for i, par in enumerate(reader):
            if i > max_utt:
                break
            utterance = RE_PUNCT.sub(
                r'\1', ' '.join(row['word'] for row in par))
            print(utterance, file=txt)


if __name__ == '__main__':
    args = parse_arguments()
    main(args.infile, args.outfile, max_utt=args.max)
