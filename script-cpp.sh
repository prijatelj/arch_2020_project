#!/bin/sh
for input in ~/repos/arch_2020_project/trace_files/*
do
    (output="results/results-cpp-"$(basename $input)
    echo $output
    ~/repos/arch_2020_project/baselines/cpp/predictors $input $output)
done
