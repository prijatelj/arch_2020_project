#!/bin/bash

#$ -pe smp 4
#$ -N L8M
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/8M/logs/
#$ -e $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/8M/logs/
BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-8M.txt" \
    -o "$BASE_PATH/results/lstm/8M/" \
    --units 8 \
    --batch_size 512 \
    --epochs 1 \
    --history 10 \
    --batch_history \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
    --gru \
    --log_file "$BASE_PATH/crc_scripts/crc_logs/lstm/8M/" \
