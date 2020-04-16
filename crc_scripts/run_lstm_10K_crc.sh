#!/bin/bash

#$ -pe smp 4
#$ -N L10k
#$ -q gpu
#$ -l gpu_card=1
#$ -o $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/10k/logs/
#$ -e $HOME/Public/arch_2020_project/crc_scripts/crc_logs/lstm/10k/logs/

BASE_PATH="$HOME/Public/arch_2020_project"

# set up the environment
module add conda
conda activate tf-1.15

python3 "$BASE_PATH/lstm.py" \
    "$BASE_PATH/trace_files/gcc-10K.txt" \
    -o "$BASE_PATH/results/lstm/10k/" \
    --units 8 \
    --batch_size 32 \
    --epochs 2 \
    --history 10 \
    --batch_history \
    --cpu_cores 4 \
    --cpu 1 \
    --gpu 1 \
    --cudnn \
    --log_file "$BASE_PATH/crc_scripts/crc_logs/lstm/10k/" \
