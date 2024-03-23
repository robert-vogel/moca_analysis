"""Statistical analyses for figure 5


By: Robert Vogel


Apply Umoca, wisdom-of-crowds, best individual, 
and Smoca, to data that
conforms to the data 
"""

import sys
import os
import argparse
import re

import numpy as np

from moca import classifiers as cls
from moca import cross_validate as cv
from moca import stats

sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.pardir))
import utils


METHOD_REGEX = "^m?M?odel_[0-9]$"

# Set global parameters

FONTSIZE=11
FIGSIZE=(2.25, 1.87)
POSITION=[0.325, 0.25, 0.625, 0.65]
CAPSIZE=3
MS=5

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        type=str,
                        help=("Name of file which contains data"
                              " table"))
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help=("Seed for random number generator"
                              " default is None"))
    parser.add_argument("--kfolds",
                        type=int,
                        default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("-o",
                        type=str,
                        default=None,
                        help="Directory to write auc data.")
    return parser.parse_args()


def main(filename, seed, kfolds, out_dir):

    # instantiate random number generator
    rng = np.random.default_rng(seed=seed)

    data, labels, team_names = utils.read_data(filename,
                                               METHOD_REGEX)

    # Perform statistical analyses
    moca_cls = [cls.Smoca(),
                cls.Smoca(subset_select=None),
                cls.Woc()]
    moca_cls[2].is_supervised = True

    for i, _ in enumerate(team_names):
        moca_cls.append(cls.BaseClassifier(i))


    auc, auc_labels = utils.performance_by_stratified_cv(data, labels,
                                                moca_cls,
                                                utils.compute_auc,
                                                kfolds=kfolds,
                                                seed=rng)
    f1score, f1_labels = utils.performance_by_stratified_cv(data, labels,
                                                moca_cls,
                                                utils.compute_f1_score,
                                                kfolds=kfolds,
                                                seed=rng)
    ba, ba_labels = utils.performance_by_stratified_cv(data, labels,
                                               moca_cls,
                                               utils.compute_ba,
                                               kfolds=kfolds,
                                               seed=rng)

    # make plots directory exists, otherwise create it
    if out_dir is None:
        out_dir = "."

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    savename = os.path.basename(filename).split('.')

    # write auc
    with open(os.path.join(out_dir, 
                           f"{savename[0]}_auc.csv"),
              "w") as fid:
        #header
        fid.write(','.join(["k_folds",*auc_labels]))

        for i in range(kfolds):
            tmp = [str(w) for w in auc[i,:]]
            fid.write(f"\n{i+1},{','.join(tmp)}")

    # write f1 score
    with open(os.path.join(out_dir, 
                           f"{savename[0]}_f1score.csv"),
              "w") as fid:
        #header
        fid.write(','.join(["k_folds",*f1_labels]))

        for i in range(kfolds):
            tmp = [str(w) for w in f1score[i,:]]
            fid.write(f"\n{i+1},{','.join(tmp)}")

    # write balanced accuracy
    with open(os.path.join(out_dir, 
                           f"{savename[0]}_ba.csv"),
              "w") as fid:
        #header
        fid.write(','.join(["k_folds",*ba_labels]))

        for i in range(kfolds):
            tmp = [str(w) for w in ba[i,:]]
            fid.write(f"\n{i+1},{','.join(tmp)}")

    return 0


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    main(args.filename, args.seed, args.kfolds, args.o)
