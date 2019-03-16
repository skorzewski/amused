#!/bin/bash -ex

for method in manners reporting_clauses
do
    for epochs in 10
    do
        for dim in 50 100
        do
            for dropout in 0.5
            do
                for recurrent_dropout in 0.0 0.2
                do
                    for lstm_layers in 1 2
                    do
                        for dense_layers in 1
                        do
                            ./experiments.py with "method=${method}" "epochs=${epochs}" "dim=${dim}" "dropout=${dropout}" "recurrent_dropout=${recurrent_dropout}" "lstm_layers=${lstm_layers}" "dense_layers=${dense_layers}"
                        done
                    done
                done
            done
        done
    done
done

