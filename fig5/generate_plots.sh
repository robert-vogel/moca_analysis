#!/usr/bin/env bash
# 
# By: Robert Vogel
#
#


python transfer_learning.py train \
    --train_dir imgs_train \
    --ckpt_dir checkpoints \
    --l1_reg_constant 0.01 \
    --learning_rate 0.0001 \
    --epochs 10 \
    --seed 786283475


python transfer_learning.py predict \
    --validate_dir imgs_validate \
    --ckpt_dir checkpoints \
    --out_dir stats_out \
    --validate_set_metadata data/validate_set.tsv


python statistical_analyses.py \
    -o stats_out \
    --kfolds 10 \
    --seed 134651 \
    stats_out/sample_scores.csv

python mkplots.py \
    -o plots \
    stats_out/sample_scores_auc.csv \
    stats_out/sample_scores_ba.csv \
    stats_out/sample_scores_f1score.csv
