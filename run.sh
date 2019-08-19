#!/bin/bash -ex

for method in sentences
do
    for epochs in 1
    do
        for dim in 100
        do
            for dropout in 0.5
            do
                for recurrent_dropout in 0.0
                do
                    for lstm_layers in 0
                    do
                        for dense_layers in 1
                        do
                            for attention in True
                            do
                                ./experiments.py with "method=${method}" "epochs=${epochs}" "dim=${dim}" "dropout=${dropout}" "recurrent_dropout=${recurrent_dropout}" "lstm_layers=${lstm_layers}" "dense_layers=${dense_layers}" "attention=${attention}"
                            done
                        done
                    done
                done
            done
        done
    done
done

