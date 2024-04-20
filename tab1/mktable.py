
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),
                             os.path.pardir))
import argparse

import numpy as np
import utils



def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        type=str,
                        help="Directory to write results to file.")
    parser.add_argument("--challenge_name_map",
                        type=str,
                        help="Path to stat file that maps challenge names")
    parser.add_argument("stat_files",
                        type=str,
                        nargs='+',
                        help="Files containing cross-validation data")
    parser.add_argument("--statistic",
                        type=str,
                        help="Statistic used to assess method performance.")

    return parser.parse_args()


def sem(data, ax=None):
    if ax is None:
        return np.std(data) / np.sqrt(data.size)

    return (np.std(data, axis=ax)
            / np.sqrt(data.shape[ax]))


def get_challenge_names(fname):
    output = {}
    with open(fname, "r") as fid:
        for tline in fid:
            tline = tline.strip().split('\t')
            output[tline[0]] = tline[1]
    return output


def main(out_dir, statistic, chal_file_names, *stat_files):

    chal_names = get_challenge_names(chal_file_names)

    n_data_sets = len(stat_files)

    data_set = {}

    for i, fname in enumerate(stat_files):
        cvdata, cl_labels= utils.read_cross_validation_file(fname,
                                                  row_idx_regex="k_folds")

        if i == 0:
            classifier_order = cl_labels

        chal_name = chal_names[fname]

        data_set[chal_name] = {"mean":np.zeros(cvdata.shape[1]),
                                "sem":np.zeros(cvdata.shape[1])}

        for j, cl_label in enumerate(classifier_order):
            idx = cl_labels.index(cl_label)

            data_set[chal_name]["mean"][j] = np.mean(cvdata[:, idx])
            data_set[chal_name]["sem"][j] = sem(cvdata[:, idx])


    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    with open(os.path.join(out_dir, f"{statistic}.csv"), "w") as fid:

        header = ["challenge_name"]
        header.extend(classifier_order)

        fid.write(','.join(header))

        for data_name, data_values in data_set.items():

            record = [data_name]

            for i in range(data_values["mean"].size):
                record.append(f"{data_values['mean'][i]:0.3f}"
                              "\u00B1"
                              f"{data_values['sem'][i]:0.3f}")

            record = f"\n{','.join(record)}"
            fid.write(record)

    return 0


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    main(args.o, args.statistic, 
         args.challenge_name_map, *args.stat_files)
