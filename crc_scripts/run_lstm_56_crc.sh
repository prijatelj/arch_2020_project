#!/bin/bash

#$ -pe smp 4
#$ -N L56
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/56/logs/
#$ -e $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/56/logs/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-56.txt" \
    -o "$BASE_PATH/results/lstm/56/" \
    --units 4 \
    --batch_size 4 \
    --epochs 1 \
    --history 0 \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
    --gru \
    --log_file "$BASE_PATH/crc_scripts/crc_logs/lstm/56/" \
