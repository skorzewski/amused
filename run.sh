#!/bin/bash -ex

for method in manners reporting_clauses
do
    for epochs in 20
    do
        for dim in 50 100
        do
            for dropout in 0.5
            do
                for recurrent_dropout in 0.0 0.2
                do
                    ./experiments.py with "method=${method}" "epochs=${epochs}" "dim=${dim}" "dropout=${dropout}" "recurrent_dropout=${recurrent_dropout}" ""
                done
            done
        done
    done
done

