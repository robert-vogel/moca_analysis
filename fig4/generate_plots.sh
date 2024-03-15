#!/usr/bin/env bash


random_seed=3241345

for fname in $(ls data/*.csv); do
    printf "%s\n" "Cross-validation AUC computation: $fname"

    python statistical_analyses.py \
        --seed  "$random_seed" \
        -o "out" \
        --kfolds 5 \
        "$fname"

    random_seed=$((random_seed + 10))
done


# ls "stat_analysis/*.csv" | xargs python mkplots.py -o plots
