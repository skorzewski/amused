#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import random
import re
import sys


RE_OTHER = re.compile(r'[^GLNORSUWZ]')


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
        default=None,
        help='Output TSV file',
    )
    return parser.parse_args()


def main(infile, outfile):
    utterances = infile.readlines()
    total = len(utterances)
    for i, utt in enumerate(utterances):
        progress = i * 20 // total
        progress_bar = progress * '#' + (20 - progress) * ' '
        print('[{}] Zdanie {} z {}:'.format(progress_bar, i + 1, total))
        print()
        print('"{}"'.format(utt.strip()))
        print()
        print('Która z poniższych emocji opisuje powyższe zdanie? Wybierz jedną lub więcej i zatwierdź klawiszem ENTER:')
        print('R – Radość   U – Ufność   L – Lęk     Z – Zaskoczenie')
        print('S – Smutek   W – Wstręt   G – Gniew   O – Oczekiwanie')
        print('N – wypowiedź jest Neutralna')
        print()
        letters = ''.join(sorted(c for c in RE_OTHER.sub('', input().upper())))
        print('{}\t{}'.format(letters, utt), file=outfile)
        print()


if __name__ == '__main__':
    args = parse_arguments()
    if not args.outfile:
        outfile_id = random.randint(0, 10000)
        outfile_name = '{}.{:04}.tsv'.format(args.infile.name, outfile_id)
        with open(outfile_name, 'w') as outfile:
            main(args.infile, outfile)
    else:
        main(args.infile, args.outfile)
