"""Generate plots for Figure 3


By: Robert Vogel


Apply Umoca, Wisdom-of-crowds, best base classifier
(Best_BC), Gaussian mixture model with spherical,
diagonal, and full covariance to our data sets.
Plot the results in a directory named plots.
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from moca import classifiers as cls
from moca import cross_validate as cv
from moca import stats

sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.path.pardir))
import utils


# Set global parameters

FONTSIZE=11
FIGSIZE=(2.25, 1.87)
FIGSIZE_BARCHART=(2.75, 1.87)

POSITION=[0.325, 0.25, 0.625, 0.65]
POSITION_BARCHART=[0.425, 0.25, 0.525, 0.65]

CAPSIZE=3
MS=5
UMOCA_MAX_ITER=20000
UMOCA_TOL=1e-4

METHOD_REGEX = "^t?T?eam_[0-9]{2}$"

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
    return parser.parse_args()


def scale_lims(lim, scale):
    return (lim[1]-lim[0]) * scale + lim[0]


def text_lims(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    return [scale_lims(xlim, 0.4), scale_lims(ylim, 0.05)]


def weights_by_stratified_cv(data, labels, kfolds=5, seed=None):
    """Compute weight statistics by cross-validation
    
    Args:
        data : ((m classifiers, n samples) np.ndarray)
            sample rank predictions per base classifier
        labels : ((n samples,) np.ndarray)
            sample class labels, 0 and 1, representing the
            negative and positive class labels, respectively. 
        kfolds: (int)
            number of folds for cross validations, > 0,
            default 5
        seed: (np.random.default_rng compatible seed)
            default None

    Returns:
        inferred_weights : ((kfolds, m classifiers) np.ndarray)
        empirical_weights : ((kfolds, m classifiers) np.ndarray)
    """
    rng = np.random.default_rng(seed=seed)

    cv_generator = cv.stratified_kfold(data, labels, kfolds, seed=rng)

    m_classifiers = data.shape[0]

    empirical_weights = np.zeros(shape=(kfolds, m_classifiers))
    umoca_weights = np.zeros(shape=(kfolds, m_classifiers))

    # note the umoca weights are computed using the 
    # training data, and the empirical weights using 
    # the test data
    i = 0
    for train, test in cv_generator:
        cl_umoca = cls.Umoca()
        cl_umoca.fit(train["data"])

        umoca_weights[i, :] = cl_umoca.weights

        cl_smoca = cls.Smoca(subset_select=None)

        cl_smoca.fit(train["data"], train["labels"])
        empirical_weights[i, :] = cl_smoca.weights

        i += 1

    return umoca_weights, empirical_weights


def main(filename, seed, kfolds):
    # make plots directory exists, otherwise create it
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # instantiate random number generator
    rng = np.random.default_rng(seed=seed)

    data, labels, team_names = utils.read_data(filename,
                                            METHOD_REGEX)

    # weights analysis
    inf_weights, emp_weights = weights_by_stratified_cv(data, labels,
                                                    kfolds=kfolds,
                                                    seed=rng)

    # weights plots
    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)

    mean_emp_weights = np.mean(emp_weights, 0)
    mean_inf_weights = np.mean(inf_weights, 0)
    ax.errorbar(mean_emp_weights,
                mean_inf_weights,
                xerr=np.std(emp_weights, 0) / np.sqrt(kfolds),
                yerr=np.std(inf_weights, 0) / np.sqrt(kfolds),
                fmt='o', capsize=CAPSIZE,
                ms=MS,
                mew=1, linewidth=1,
                mfc='none')
    ax.axline((0,0), slope=1, linestyle=":",
              color="k", alpha=0.25)

    ax.ticklabel_format(style='sci', axis='both',
                            scilimits=(-2,2),
                            useMathText=True)

    ax.set_xlabel(r"$w_\text{empirical}$", fontsize=FONTSIZE)
    ax.set_ylabel(r"$w_\text{Umoca}$", fontsize=FONTSIZE)
    ax.set_position(POSITION)

    lims = text_lims(ax)

    c = np.corrcoef(mean_emp_weights, mean_inf_weights)[0,1]
    ax.text(lims[0], lims[1],
            f'Corr : {c:0.2f}', fontsize=11)

    plot_filename = os.path.basename(filename)
    plot_filename = plot_filename.split('.')

    if len(plot_filename) > 2:
        plot_filename = ["_".join(plot_filename[:-1]),
                         plot_filename[-1]]
    
    fig.savefig(os.path.join("plots",
                             f"{plot_filename[0]}_weights.pdf"))
    
    # Perform AUC analysis
    cls_compare = [cls.Umoca(max_iter=UMOCA_MAX_ITER,
                             tol=UMOCA_TOL),
                   cls.Woc(),
                   utils.Gmm("spherical"),
                   utils.Gmm("diag"),
                   utils.Gmm("full"),
                   cls.BestBC()]

    auc, cl_labels = utils.performance_by_stratified_cv(data, labels,
                                          cls_compare,
                                          utils.compute_auc,
                                          kfolds=kfolds,
                                          seed=rng)

    fig, ax = plt.subplots(1,1,figsize=FIGSIZE_BARCHART)
    ax.barh(np.arange(len(cl_labels)),
            np.mean(auc,0),
            xerr=np.std(auc,0) / np.sqrt(kfolds),
            tick_label=cl_labels,
            capsize=CAPSIZE)

    ax.set_xlim(0.9*np.min(auc), 1.01*np.max([1., np.max(auc)]))

    ax.set_xlabel("AUC", fontsize=FONTSIZE)

    ax.set_position(POSITION_BARCHART)

    fig.savefig(os.path.join("plots",
                             f"{plot_filename[0]}_auc.pdf"))

    
    # Plot conditional correlation plot

    C = stats.moca_cov(stats.rank_transform(data), labels)

    fig = sns.clustermap(utils.cov2corr(C),
                         cmap='bwr',
                         figsize=FIGSIZE, vmin=-1, vmax=1,
                         xticklabels=False, yticklabels=False,
                         cbar_kws={'ticks':[-1, 1]})

    fig.savefig(os.path.join("plots",
                f"{plot_filename[0]}_conditional_correlation.pdf"))

    return 0


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    main(args.filename, args.seed, args.kfolds)
