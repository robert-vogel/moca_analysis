#!/usr/bin/env bash


# Run Umoca analysis on DREAM data
#
python mkplots.py \
    --seed 3241345 \
    --stratified_cv \
    "data/dream_2_bcl6_challenge.csv"


python mkplots.py \
    --seed 454326 \
    --stratified_cv \
    "data/dream_9_5-prostate_cancer.csv"

# Run data cond independent simulation data
# and perform Umoca analysis
#
python mk_sim_data.py \
    --m_classifiers 15 \
    --n_samples 1000 \
    --n_positives 300 \
    --auc_lims 0.4 0.8 \
    --cond_independent \
    "data/simulate_cond_independent.csv"

python mkplots.py \
    --seed 324132315 \
    --stratified_cv \
    "data/simulate_cond_independent.csv"


# Run data cond dependent simulation data
# and perform Umoca analysis
#
python mk_sim_data.py \
    --m_classifiers 15 \
    --n_samples 1000 \
    --n_positives 300 \
    --auc_lims 0.4 0.8 \
    "data/simulate_cond_dependent.csv"

python mkplots.py \
    --seed 320798562315 \
    --stratified_cv \
    "data/simulate_cond_dependent.csv"
