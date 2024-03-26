
import re
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.path.pardir))
import argparse

import numpy as np
import matplotlib.pyplot as plt
import utils

FIGSIZE = (5,5.5)
FONTSIZE = 15

AX_POSIT = (0.2, 0.15, 0.7, 0.8)
LEGEND_LOC = 0

BAR_FRAC = 0.9
XLIMS=(0.3, 0.8)


def _parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("-o",
                        default="plots",
                        type=str,
                        help="Directory to print plots")
    parser.add_argument("stat_files",
                        type=str,
                        nargs='+',
                        help="Files containing cross-validation data")

    return parser.parse_args()


def sem(data, ax=0):
    return (np.std(data, axis=ax)
            / np.sqrt(data.shape[ax]))

def fname2performance_name(fname):
    if re.search("_auc\.", fname) is not None:
        return "AUC"
    if re.search("_f1score\.", fname) is not None:
        return "F1 score"
    if re.search("_ba\.", fname) is not None:
        return "B.A."

    raise ValueError


def main(plot_dir, *stat_files):

    n_performance_sets = len(stat_files)

    performance_measure = []
    performance_labels = []
    max_num_bars_per_performance_set = 0

    for fname in stat_files:
        cvdata, cl_labels= utils.read_cross_validation_file(fname,
                                                  row_idx_regex="k_folds")

        performance_measure.append({"mean":np.mean(cvdata, 0),
                                   "sem":sem(cvdata, ax=0),
                                   "labels":cl_labels})
        performance_labels.append(fname2performance_name(fname))

        if (cl_len := len(cl_labels)) > max_num_bars_per_performance_set:
            max_num_bars_per_performance_set = cl_len
        

    # data sets are located one unit apart
    # while methods per data set occupy a total of 0.9
    # units.  Each bar is then 0.9 / number of
    # methods per data set
    data_set_midpoints = np.arange(len(performance_measure))
    data_height = 0.9
    bar_height = data_height / max_num_bars_per_performance_set


    coluer = plt.cm.tab10.colors

    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)

    i_performance_set = 0
    for data_name, data_values in zip(performance_labels,
                                      performance_measure):

        data_midpoint = data_set_midpoints[i_performance_set]

        for i, cls_name in enumerate(data_values["labels"]):
            label = None
            if i_performance_set == 0:
                label=cls_name

            ax.barh(data_midpoint - data_height/2 + bar_height*(i + 0.5) ,
                    data_values["mean"][i],
                    bar_height,
                    capsize=3,
                    color=coluer[i],
                    xerr=data_values["sem"][i],
                    label=label)

        i_performance_set += 1


    ax.legend(loc=LEGEND_LOC)
    ax.set_position(AX_POSIT)

    ax.set_yticks(data_set_midpoints, labels=performance_labels,
                  fontsize=FONTSIZE)
    ax.set_xlabel("Performance", fontsize=FONTSIZE)

    ax.set_xlim(XLIMS)

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    fig.savefig(os.path.join(plot_dir, f"performance.pdf"))

    return 0


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    main(args.o, *args.stat_files)
