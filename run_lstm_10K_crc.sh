#!/bin/bash

#$ -pe smp 4
#$ -N lstm_branch
#$ -o crc_logs/lstm/
#$ -e crc_logs/lstm/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-10K.txt" \
    -o "$BASE_PATH/results/lstm/preds_10k_lstm-w2.csv"
    --window_size 2 \
    --units 1
