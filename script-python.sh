#!/bin/sh
for f in ~/repos/arch_2020_project/trace_files/*
do
    for m in {0..6}
    do
        for n in {1..2}
        do
            (python3 baselines/python/branchsim.py -f $f -m $m -n $n -k 8)
            (echo '------------------------------')
        done
    done
done
