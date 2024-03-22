
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.path.pardir))
import argparse

import numpy as np
import matplotlib.pyplot as plt
import utils

FIGSIZE = (8,5.5)
FONTSIZE = 15

AX_POSIT = (0.45, 0.15, 0.5, 0.8)
LEGEND_LOC = 0

BAR_FRAC = 0.9
XLIMS=(0.4, 1)


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
    parser.add_argument("--statistic",
                        type=str,
                        help="Statistic used to assess method performance.")

    return parser.parse_args()


def sem(data, ax=0):
    return (np.std(data, axis=ax)
            / np.sqrt(data.shape[ax]))

def fname2data_name(fname):
    return fname

def main(out_dir, statistic,*stat_files):

    n_data_sets = len(stat_files)

    data_set = {}
    max_num_bars_per_data_set = 0

    for fname in stat_files:
        cvdata, cl_labels= utils.read_cross_validation_file(fname,
                                                  row_idx_regex="k_folds")

        data_set[fname2data_name(fname)] = {"mean":np.mean(cvdata, 0),
                                            "sem":sem(cvdata, ax=0),
                                            "labels":cl_labels}
        if (cl_len := len(cl_labels)) > max_num_bars_per_data_set:
            max_num_bars_per_data_set = cl_len
        

    # data sets are located one unit apart
    # while methods per data set occupy a total of 0.9
    # units.  Each bar is then 0.9 / number of
    # methods per data set
    data_set_midpoints = np.arange(len(data_set))
    data_height = 0.9
    bar_height = data_height / max_num_bars_per_data_set


    coluer = plt.cm.tab10.colors

    fig, ax = plt.subplots(1,1,figsize=FIGSIZE)

    i_data_set = 0
    data_names = []
    for data_name, data_values in data_set.items():

        data_midpoint = data_set_midpoints[i_data_set]
        data_names.append(data_name)

        for i, cls_name in enumerate(data_values["labels"]):
            label = None
            if i_data_set == 0:
                label=cls_name

            ax.barh(data_midpoint - data_height/2 + bar_height*(i + 0.5) ,
                    data_values["mean"][i],
                    bar_height,
                    capsize=3,
                    color=coluer[i],
                    xerr=data_values["sem"][i],
                    label=label)

        i_data_set += 1


    ax.axvline(0.5, linestyle=":", color="k", alpha=0.5, label="Random")

    ax.legend(loc=LEGEND_LOC)
    ax.set_position(AX_POSIT)

    ax.set_yticks(data_set_midpoints, labels=data_names,
                  fontsize=FONTSIZE)
    ax.set_xlabel(statistic, fontsize=FONTSIZE)

    ax.set_xlim(XLIMS)
    fig.savefig(os.path.join("plots", f"{statistic}.pdf"))

    return 0


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    main(args.o, args.statistic, *args.stat_files)
