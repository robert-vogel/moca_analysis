"""Organize images by class

By: Robert Vogel

The `tf.keras.utils.image_dataset_from_directory` functions can infer
image class labels from directory structure.  This script makes the
class directories, and moves 
the data_set file, 
"""
import sys
import os
import shutil
import argparse



def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_list",
                        type=str,
                        required=True,
                        help=("Path to tsv file containing list of images"
                              " and class labels"))
    parser.add_argument("--img_dir",
                        type=str,
                        required=True,
                        help="Path to directory with jpeg images.")
    parser.add_argument("--img_exclude_list",
                        type=str,
                        required=True,
                        help=("List of images that are excluded from"
                              " analysis.  These images will be deleted"))

    return parser.parse_args()



def main(args):
    args = _parse_args(args)

    # mapping of class label to path
    label_to_path = {
            "benign":os.path.join(args.img_dir, "benign"),
            "malignant": os.path.join(args.img_dir, "malignant")
    }


    # make directories if necessary
    for class_label, dir_path  in label_to_path.items():

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    with open(args.img_list, "r") as fid:

        for tline in fid:
            tline = tline.strip()
            tline = tline.split("\t")

            img_filename = os.path.join(args.img_dir,
                                        f"{tline[0]}.jpeg")

            if os.path.exists(img_filename):
                shutil.move(img_filename, label_to_path[tline[1]])

    with open(args.img_exclude_list, "r") as fid:

        for tline in fid:
            tline = tline.strip()
            tline = tline.split("\t")

            img_filename = os.path.join(args.img_dir,
                                        f"{tline[0]}.jpeg")

            if os.path.exists(img_filename):
                os.remove(img_filename)



if __name__ == "__main__":
    main(sys.argv[1:])
