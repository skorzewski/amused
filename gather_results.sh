#!/bin/bash

grep COS experiment_results/* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -nr
