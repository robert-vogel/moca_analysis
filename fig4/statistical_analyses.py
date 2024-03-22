"""Statistical analyses for figure 4


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


METHOD_REGEX = "^t?T?eam_[0-9]{2}$"

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
    # make plots directory exists, otherwise create it
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # instantiate random number generator
    rng = np.random.default_rng(seed=seed)

    data, labels, team_names = utils.read_data(filename,
                                               METHOD_REGEX)

    
    # Perform AUC analysis
    moca_cls = [cls.Umoca(max_iter=utils.UMOCA_MAX_ITER,
                          tol=utils.UMOCA_TOL),
                cls.Smoca(),
                cls.BestBC(),
                cls.Woc()]

    auc, cl_labels = utils.auc_by_stratified_cv(data, labels,
                                                moca_cls,
                                                kfolds=kfolds,
                                                seed=rng)
    
    savename = os.path.basename(filename).split('.')

    if out_dir is None:
        out_dir = ""

    with open(os.path.join(out_dir, 
                           f"{savename[0]}.csv"),
              "w") as fid:
        #header
        fid.write(','.join(["k_folds",*cl_labels]))

        for i in range(kfolds):
            tmp = [str(w) for w in auc[i,:]]
            fid.write(f"\n{i+1},{','.join(tmp)}")

    return 0


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    main(args.filename, args.seed, args.kfolds, args.o)
