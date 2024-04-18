"""Generate simulation data.

By: Robert Vogel

Generate simulation data with and without conditional
correlation.  Write both tables to specified
directory.  This code will overwrite pre-existing files.
"""

import os
import sys
import argparse
import numpy as np

from moca import simulate


def _write_to_file(filename, data, labels):
    m_classifiers, n_samples = data.shape

    with open(filename, "w") as fid:
    
        header = [f"Team_{i:02d}" for i in range(1, m_classifiers+1)]
        fid.write(','.join(["sample_id", *header, "class"]))
        
        for n in range(n_samples):
            tmp = [str(w) for w in data[:, n]]
            fid.write(",".join([f"\nsimulated_{n+1:03d}",
                                *tmp,
                                f"{labels[n]}"]))


def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--m_classifiers",
                        type=int,
                        default=10,
                        help="Number of classifiers to simulate")
    parser.add_argument("--n_samples",
                        type=int,
                        default=1000,
                        help="Number of samples to simulate")
    parser.add_argument("--n_positives",
                        type=int,
                        default=300,
                        help="number of positive class samples")
    parser.add_argument("--auc_lims",
                        type=float,
                        nargs=2,
                        help="Classifier AUC limits")
    parser.add_argument("--seed",
                        type=int,
                        help="Random number generator seed")
    parser.add_argument("--cond_independent",
                        action="store_true",
                        help="Simulate conditionally independent samples")

    parser.add_argument("filename",
                        type=str,
                        help="Name and path of file to save sim data.")


    args = parser.parse_args()

    if args.m_classifiers <= 0:
        raise ValueError("Insufficient number of base classifiers")

    if args.n_samples <= 0:
        raise ValueError("Insufficient number of samples")

    if args.n_positives <= 0:
        raise ValueError("Insufficient number of positive class samples")

    if args.n_positives >= args.n_samples:
        raise ValueError("n_positives must be less than n_samples")

    if args.auc_lims[0] <= 0 or args.auc_lims[0] >= 1:
        raise ValueError("AUC out of bounds, must be (0,1)")

    if args.auc_lims[1] <= 0 or args.auc_lims[1] >= 1:
        raise ValueError("AUC out of bounds, must be (0,1)")


    # make (lower, upper) bounds format
    if args.auc_lims[0] > args.auc_lims[1]:
        tmp = args.auc_lims[0]
        args.auc_lims[0] = args.auc_lims[1]
        args.auc_lims[1] = tmp

    return args



def main(m_classifiers, n_samples, n_positives,
         auc_lims, seed, cond_independent, filename):

    auc = np.linspace(auc_lims[0], auc_lims[1], m_classifiers)

    # setting pseudo-random number generator objects
    rng = np.random.default_rng(seed=seed)


    # Run conditional independent simulation and Umoca analysis
    corr_matrix = simulate.make_corr_matrix(m_classifiers,
                                            independent=cond_independent,
                                            seed=rng)
    
    data, labels = simulate.gaussian_scores(n_samples,
                                        n_positives,
                                        auc,
                                        corr_matrix=corr_matrix,
                                        seed=rng)
    
    _write_to_file(filename, data, labels)



if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    main(args.m_classifiers, args.n_samples,
         args.n_positives, args.auc_lims,
         args.seed, args.cond_independent,
         args.filename)
