#!/bin/bash

# set -x

TRAIN_SCRIPT="scripts/ODM/run_ella_odm_detroit.sh"
while true
do
    sleep 5
    echo "I will run 'bash $TRAIN_SCRIPT'"
    salloc -G 1 -c 6 --mem 128G -t 8:0:0 -p gpu-preempt --constraint="a40|l40s" srun bash $TRAIN_SCRIPT
done