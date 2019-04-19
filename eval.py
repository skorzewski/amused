#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from scipy.spatial.distance import cosine


def evaluate(results_file_name, reference_file_name):
    """Evaluate results"""
    cos_dists = []
    with open(results_file_name, 'r') as results_file:
        with open(reference_file_name, 'r') as reference_file:
            results = results_file.readlines()
            reference = reference_file.readlines()
            for res, ref in zip(results, reference):
                p, at, s, ap, utt = res.split('\t')
                rp, rat, rs, rap, rutt = ref.split('\t')
                result = np.asarray(
                    [float(p), float(at), float(s), float(ap)])
                reference = np.asarray(
                    [float(rp), float(rat), float(rs), float(rap)])
                if reference.any():
                    cosine_distance = (cosine(result, reference)
                                       if result.any()
                                       else 1.0)
                # print(cosine_distance, utt)
                cos_dists.append(cosine_distance)
    mcosd = np.mean(cos_dists)
    with open(results_file_name, 'a') as results_file:
        print('MCosD: {}'.format(mcosd))
        print('MCosD: {}'.format(mcosd), file=results_file)


def main():
    """Main function"""
    results_file = sys.argv[1]
    reference_file = 'corpora/gold2.tsv'
    evaluate(results_file, reference_file)


if __name__ == '__main__':
    main()
