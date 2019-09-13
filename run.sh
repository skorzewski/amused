#!/bin/bash -ex

for method in mean zero max maxabs
do
    for wsd_method in none simplified_lesk freq_weighted_lesk idf_weighted_lesk simplified_lesk_with_bootstrapping freq_weighted_lesk_with_bootstrapping idf_weighted_lesk_with_bootstrapping
    do
        ./experiments.py with "method=${method}" "wsd_method=${wsd_method}"
    done
done

