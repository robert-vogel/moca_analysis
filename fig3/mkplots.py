"""Generate plots for Figure 4.


By: Robert Vogel


Apply Umoca, wisdom-of-crowds, best individual, and
gaussian mixture model (GMM) to data specified that
conforms to the data table specification


DATA TABLE SPECIFICATION

"""

import sys
import os
import argparse
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from moca import classifiers as cls
from moca import cross_validate as cv
from moca import stats

from sklearn.mixture import GaussianMixture



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
    parser.add_argument("--stratified_cv",
                        action="store_true",
                        help=("Sample from positive and negative class"
                            " according to sample class prevalence"))
    return parser.parse_args()


class Gmm(cls.MocaABC):
    def __init__(self):
        super().__init__()

        self._gmm = GaussianMixture(n_components=2,
                                    covariance_type="diag")

    def fit(self, data):
        self.M = data.shape[0]

        self._gmm.fit(data.T)

        # means_[components, features]
        # assume that component 1 is the positive class, positive class
        # samples have higher rank
        self.prevalence = None


        self._pos_idx = 1
        self._negative_idx = 0

        self.prevalence = self._gmm.weights_[self._pos_idx]

        delta = (self._gmm.means_[self._negative_idx, :]
                 - self._gmm.means_[self._pos_idx, :])

        delta = np.where(delta > 0, 1, 0)

        # if more than half the number of classifiers
        # have a positive delta, then switch the indexes
        if self.M/2 < np.sum(delta):
            self._pos_idx = 0
            self._negative_idx = 1
            self.prevalence = self._gmm.weights_[self._pos_idx]

    def get_scores(self, data):
        if not stats.is_rank(data):
            raise ValueError

        s = np.zeros(data.shape[1])
        # loop over samples
        for i in range(data.shape[1]):

            # loop over features, o.k., because covariance
            # is diagonal

            for j in range(data.shape[0]):

                # quadratic term
                s[i] += ((1/self._gmm.covariances_[self._pos_idx, j] 
                       - 1/self._gmm.covariances_[self._negative_idx, j]) 
                      * data[j,i]**2)

                # linear term
                s[i] += (2*(self._gmm.means_[self._negative_idx, j]
                            / self._gmm.covariances_[self._negative_idx,j]
                            - self._gmm.means_[self._pos_idx, j]
                            / self._gmm.covariances_[self._pos_idx, j])
                         * data[j, i])

        return s


def cov2corr(c):
    corr= np.zeros(shape=c.shape)
    for i in range(corr.shape[0]):
        for j in range(1, corr.shape[0]):
            corr[i, j] = c[i,j] / np.sqrt(c[i, i] * c[j, j])
    return corr + corr.T + np.eye(c.shape[0])


def scale_lims(lim, scale):
    return (lim[1]-lim[0]) * scale + lim[0]


def text_lims(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    return [scale_lims(xlim, 0.4), scale_lims(ylim, 0.05)]


def read_data(fname):
    """Load data under the assumed specification.

    Data specification:

        comma delimited text file as follows

        Line 1: header labels:
            * sample ids
                may be multiple columns and have a variety
                of sample id names
            * team_{02:number}
                the sample predictions of team specified by
                number.  Alway this format.
            * class
                sample class labels
        Line 2-end
            * arbirary sample names,
            * sample scores per team, or empty -> np.nan
            * sample class labels, 0 or 1

    Args:
        fname: (str)
            name of file, including path, to be read

    Return:
        X: ((n samples, m classifiers) np.ndarray)
            Sample scores by each base classifier
        y: ((n samples,) np.ndarray)
            sample class labels, 0 and 1, representing the
            negative and positive class labels, respectively. 
        cls_names: (list)
            classifier team names in order of data matrix
    """
    delim = ","


    with open(fname, "r") as fid:

        # Decompose header and get the number of samples
        cls_names = []
        score_col_idx = []
        label_col_idx = None

        for i, tline in enumerate(fid):

            if i == 0:

                tline = tline.strip().split(delim)

                for j, field in enumerate(tline):

                    if re.match("^class$", field):
                        label_col_idx = j
                        continue

                    if field in cls_names:
                        raise ValueError("Duplicate team names")

                    if (re.match("^t?T?eam_[0-9]{2}$", field)
                        is not None):
                        cls_names.append(field)
                        score_col_idx.append(j)

                if ((m_cls := len(score_col_idx)) == 0
                        or label_col_idx is None):
                    raise ValueError("Could not decompose header")

        n_samples = i

        X = np.zeros(shape=(m_cls, n_samples))
        y = np.zeros(n_samples)

        # return file object to the beginning of file
        fid.seek(0, os.SEEK_SET)


        sample_idx = 0
        for i, tline in enumerate(fid):
            
            if i == 0:
                continue

            tline = tline.strip().split(delim)

            if (re.match("^[0-1]*\.?[0-1]*$",tline[label_col_idx])
                is None):
                raise ValueError(("Incorrect sample label"
                                 f" {tline[label_col_idx]}"))

            y[sample_idx] = np.float16(tline[label_col_idx])


            for j, cls_idx in enumerate(score_col_idx):
                val = tline[cls_idx]

                if val == "":
                    X[j, sample_idx] = np.nan
                elif (re.match("^-?[0-9]*\.?[0-9]*e?[+-]?[0-9]*$",
                               val) is not None):
                    X[j, sample_idx] = np.float64(val)
                else:
                    raise ValueError(("Error while parsing"
                                      f" file at line {i}"
                                      f"\r {','.join(tline)}"))

            sample_idx += 1

    return X, y, cls_names


def weights_by_cv(data, labels, kfolds, stratified_cv, seed):
    """Compute weight statistics by cross-validation
    

    Args:
        data : ((m classifiers, n samples) np.ndarray)
            sample rank predictions per base classifier
        labels : ((n samples,) np.ndarray)
            sample class labels, 0 and 1, representing the
            negative and positive class labels, respectively. 
        kfolds: (int)
            number of folds for cross validations, > 0
        stratified_cv: (bool)
            if true, perform cross-validation such that the
            sample class prevalence is, approximately, preserved.
        seed: (np.random.default_rng compatible seed)
            default None

    Returns:
        inferred_weights : ((kfolds, m classifiers) np.ndarray)
        empirical_weights : ((kfolds, m classifiers) np.ndarray)
    """
    rng = np.random.default_rng(seed=seed)

    if stratified_cv:
        cv_generator = cv.stratified_kfold(data, labels, kfolds, seed=rng)
    else:
        cv_generator = cv.kfold(data, labels, kfolds, seed=rng)

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


def auc_by_cv(data, labels, kfolds, stratified_cv, seed):
    """Compute AUC by cross-validation
    

    Args:
        X : ((m classifiers, n samples) np.ndarray)
            sample rank predictions per base classifier
        y : ((n samples,) np.ndarray)
            sample class labels, 0 and 1, representing the
            negative and positive class labels, respectively. 
        kfolds: (int)
            number of folds for cross validations, > 0
        stratified_cv: (bool)
            if true, perform cross-validation such that the
            sample class prevalence is, approximately, preserved.
        seed: (np.random.default_rng compatible seed)
            default None

    Returns:
        auc:((kfolds,) np.ndarray)
    """
    rng = np.random.default_rng(seed=seed)


    cl_labels = []

    auc = np.zeros(shape=(kfolds, 4))

    for i, cl in enumerate([cls.Umoca(), cls.Woc(), Gmm()]):

        cl_labels.append(cl.name)

        if stratified_cv:
            cv_generator = cv.stratified_kfold(data, labels, kfolds, seed=rng)
        else:
            cv_generator = cv.kfold(data, labels, kfolds, seed=rng)

        k = 0
        for train, test in cv_generator:
            cl.fit(train["data"])

            _, _, auc[k, i] = stats.roc(cl.get_scores(test["data"]),
                                        test["labels"])
            k += 1

    if stratified_cv:
        cv_generator = cv.stratified_kfold(data, labels, kfolds, seed=rng)
    else:
        cv_generator = cv.kfold(data, labels, kfolds, seed=rng)
 

    cl_labels.append("Best Ind")
    i += 1
    k = 0
    for train, test in cv_generator:
        cl = cls.BestBC()
        cl.fit(train["data"], train["labels"])
 
        _, _, auc[k, i] = stats.roc(cl.get_scores(test["data"]),
                                    test["labels"])
        k += 1

    return auc, cl_labels
    

def main(filename, seed, kfolds, stratified_cv):
    # make plots directory exists, otherwise create it
    if not os.path.exists("plots"):
        os.mkdir("plots")

    # instantiate random number generator
    rng = np.random.default_rng(seed=seed)

    data, labels, team_names = read_data(filename)

    # weights analysis
    inf_weights, emp_weights = weights_by_cv(data, labels,
                                             kfolds,
                                             stratified_cv,
                                             rng)

    # weights plots
    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)

    mean_emp_weights = np.mean(emp_weights, 0)
    mean_inf_weights = np.mean(inf_weights, 0)
    ax.errorbar(mean_emp_weights,
                mean_inf_weights,
                xerr=np.std(emp_weights, 0) / np.sqrt(kfolds),
                yerr=np.std(inf_weights, 0) / np.sqrt(kfolds),
                fmt='o', capsize=CAPSIZE, ms=MS,
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
    
    fig.savefig(f"plots/{plot_filename[0]}_weights.pdf")
    
    # Perform AUC analysis

    auc, cl_labels = auc_by_cv(data, labels,
                               kfolds,
                               stratified_cv,
                               rng)

    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)
    ax.barh(np.arange(len(cl_labels)),
            np.mean(auc,0),
            xerr=np.std(auc,0) / np.sqrt(kfolds),
            tick_label=cl_labels,
            capsize=CAPSIZE)

    ax.set_xlim(0.9*np.min(auc), 1.01*np.max([1., np.max(auc)]))

    ax.set_xlabel("AUC", fontsize=FONTSIZE)

    ax.set_position(POSITION)

    fig.savefig(f"plots/{plot_filename[0]}_auc.pdf")

    
    # Plot conditional correlation plot

    C = stats.moca_cov(stats.rank_transform(data), labels)

    fig = sns.clustermap(cov2corr(C),
                         cmap='bwr',
                         figsize=FIGSIZE, vmin=-1, vmax=1,
                         xticklabels=False, yticklabels=False,
                         cbar_kws={'ticks':[-1, 1]})
    fig.savefig(f'plots/{plot_filename[0]}_conditional_correlation.pdf')

    return 0


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])

    main(args.filename, args.seed, args.kfolds, args.stratified_cv)
