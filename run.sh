#!/bin/bash -ex

for method in manners reporting_clauses mean max
do
    ./experiments.py with "method=${method}" "epochs=100"
done