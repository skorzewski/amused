#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Annotate utterances with emotions')
    parser.add_argument(
        'infile',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='Input file with utterances',
    )
    parser.add_argument(
        'outfile',
        nargs='?',
        type=argparse.FileType('w'),
        default=sys.stdout,
        help='Output TSV file',
    )
    return parser.parse_args()


def main(infile, outfile):
    for utt in infile:
        print('Która emocja z poniższych najlepiej opisuje zdanie:')
        print()
        print(utt)
        print()
        print('R – Radość   U – Ufność   L – Lęk     Z – Zaskoczenie')
        print('S – Smutek   W – Wstręt   G – Gniew   O – Oczekiwanie')
        print('N – wypowiedź jest Neutralna')
        print()
        letter = input().lower()
        print('{}\t{}'.format(letter.upper(), utt), file=outfile)
        print()


if __name__ == '__main__':
    args = parse_arguments()
    main(args.infile, args.outfile)
