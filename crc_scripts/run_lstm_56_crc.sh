#!/bin/bash

#$ -pe smp 4
#$ -N lstm_4w_4u
#$ -q gpu
#$ -l gpu_card=1
#$ -o crc_logs/lstm/10K/
#$ -e crc_logs/lstm/10K/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-10K.txt" \
    -o "$BASE_PATH/results/lstm/10k/preds_lstm-4w-4u.csv" \
    --window_size 4 \
    --units 4 \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
