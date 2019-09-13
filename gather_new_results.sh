#!/bin/bash

grep $1 new_experiment_results/* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -n | sed 's|new_experiment_results/||;s|[.]tsv$||;s|_|\t|' | column -nts$'\t'
