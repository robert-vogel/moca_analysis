#!/usr/bin/env bash


# Run Umoca analysis on DREAM data
#

random_seed=3241345

for fname in $(ls data/*.csv); do
    echo "$fname"
    echo""
    python mkplots.py \
        --seed  "$random_seed" \
        --stratified_cv \
        "$fname"
    echo ""

    random_seed=$((random_seed + 10))
done
