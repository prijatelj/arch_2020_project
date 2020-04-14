#!/bin/bash

#$ -pe smp 4
#$ -N lstm_8M_8w8u
#$ -o crc_logs/lstm/
#$ -e crc_logs/lstm/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-8M.txt" \
    -o "$BASE_PATH/results/lstm/preds_8M_lstm-w8-u8.csv" \
    --window_size 1 \
    --units 4 \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
