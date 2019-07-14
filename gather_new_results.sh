#!/bin/bash

echo -e '\t\t\tm\tdl\tll\te\tdim\tdo\trdo'

grep $1 new_experiment_results/* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -n | tee /dev/stderr | sed 's/manners/0/g' | sed 's/reporting_clauses/1/g' | sed 's/[^0-9.]\+/\t/g' | sed 's/^\t//g' | sed 's/[.]\t$//g' | awk 'NF==8{print}{}' >metadata.tsv
