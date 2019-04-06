#!/usr/bin/bash

grep RMSE experiment_results/*reporting_clauses* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -n