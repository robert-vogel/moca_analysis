#!/usr/bin/env bash


random_seed=3241345
cv_result_dir="stats_out"

for fname in $(ls data/*.csv); do
    printf "%s\n" "Cross-validation AUC computation: $fname"

    python statistical_analyses.py \
        --seed  "$random_seed" \
        -o "$cv_result_dir" \
        --kfolds 5 \
        "$fname"

    random_seed=$((random_seed + 10))
done

sleep 2

ls "${cv_result_dir}/*.csv" | xargs \
    python mkplots.py \
    --challenge_name_map stats_out/challenge_names.tsv \
    -o plots \
    --statistic "AUC"
