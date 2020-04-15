#!/bin/bash

#$ -pe smp 4
#$ -N L_1w_2u_32b_0h
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/56/
#$ -e $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/56/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-56.txt" \
    -o "$BASE_PATH/results/lstm/56/preds_lstm-1w-2u-32b-0h.csv" \
    --window_size 1 \
    --units 2 \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
    --batch_size 32 \
    #--history 100 \
