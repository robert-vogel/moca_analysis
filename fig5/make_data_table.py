"""Check data

Check that data:
    * License CC-0
    * valid benign_malignant field

By: Robert Vogel
"""


import os
import sys
import json
import argparse

REQUIRED_LICENSE = "CC-0"
CLINICAL_FEATURE = "benign_malignant"
BENIGN_MALIG = set(["benign", "malignant"])

def _parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        type=str,
                        default="out",
                        help="output file name.")
    parser.add_argument("--img_json",
                        required=True,
                        type=str,
                        help="Collection json file name")
    return parser.parse_args()


def main(args):

    args = _parse_args(args)

    if not os.path.exists(args.img_json):
        raise ValueError(f"{args.img_json} :file does not exist")

    with open(args.img_json, "r") as fid:
        metadata = json.load(fid)

    images_to_use = []

    for w in metadata["results"]:

        if (lic := w["copyright_license"]) != REQUIRED_LICENSE:
            print(w['isic_id'], f"incorrect_license:{lic}", sep='\t')
            continue

        if CLINICAL_FEATURE not in w["metadata"]["clinical"]:
            print(w['isic_id'],"clinical_feature_not_found", sep='\t')
            continue

        if (d := w["metadata"]["clinical"][CLINICAL_FEATURE]) not in BENIGN_MALIG:
            print(w["isic_id"], f"belign_malignant_error:{d}", sep='\t')
            continue

        img_fname = os.path.join(os.path.dirname(args.img_json),
                                 f"{w['isic_id']}.jpeg")

        if not os.path.exists(img_fname):
            print(w["isic_id"], "img_file_not_exist", sep='\t')
            continue

        images_to_use.append(f"{w['isic_id']}\t{d}")
    

    images_to_use = '\n'.join(images_to_use)


    write_mode = "w"
    if os.path.exists(args.o):
        write_mode = "a"
        images_to_use = f"\n{images_to_use}"

    with open(args.o, write_mode) as fid:
        fid.write(images_to_use)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
