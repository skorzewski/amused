#!/bin/bash

grep $1 experiment_results/* | awk 'BEGIN {FS=OFS=":"} {print $3 "\t" $1}' | sort -n
