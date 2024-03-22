"""Download collections meta data.

By: Robert Vogel


EXAMPLE
    python download_collections.py --dir train_imgs 289
"""


import os
import sys
import re
import time
import argparse
import urllib
import requests
import json

# don't want to send too many requests too quickly
SLEEP_TIME = 0.2

URL_METADATA = {
        "scheme":"https",
        "netloc":"api.isic-archive.com",
        "path":"/api/v2/images/search/",
        "params":"",
        "fragment":""
        }
URL_IMG = {
        "scheme":"https",
        "netloc":"content.isic-archive.com",
        "params":"",
        "fragment":""
        }

SEARCH_ISIC_URL = ("https://api.isic-archive.com/api/v2/"
                    "images/search/?collections")
COLLECTION_ID_REGEX = "^[0-9]{3}$"


def _parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir",
                        type=str,
                        help="Directory to print data")

    parser.add_argument("collection_id_num",
                        type=str,
                        help="ISIC archive collection id number")

    return parser.parse_args()


def get_metadata_download(fname, url):
    """
    Args:
        fname: str
            file name and path to print meta data json
        url: str
            url to download the meta data from
    """

    # verify url pattern
    tmp = urllib.parse.urlparse(url)

    for key, val in URL_METADATA.items():
        if getattr(tmp, key) != val:
            raise ValueError(f"Invalid url: {url}")

    r = requests.get(url)

    if r.status_code != 200: 
        raise ValueError(f"Non successful query {r.status_code}")

    metadata = r.json()

    with open(fname, "w") as fid:
        json.dump(metadata, fid)

    return metadata


def download_image(out_dir, img_id, img_url):
    """
    Args:
        out_dir: str
            directory to write image
        img_id: str
            isic image id
        img_url: str
            url to download image 

    Return:
        0 if image already exists
        1 if image downloaded
    """
    img_fname = os.path.join(out_dir, f"{img_id}.jpeg")

    if os.path.exists(img_fname):
        print(f"Already downloaded {img_id}")
        return 0

    print(img_url)

    # verify url pattern
    tmp = urllib.parse.urlparse(img_url)

    for key, val in URL_IMG.items():
        if getattr(tmp, key) != val:
            raise ValueError(f"Invalid url: {img_url}")

    r = requests.get(img_url)

    if r.status_code != 200: 
        raise ValueError(f"Non successful query {r.status_code}")

    img_fname = os.path.join(out_dir, f"{img_id}.jpeg")

    with open(img_fname, "wb") as fid:
        fid.write(r.content)

    return 1


def download_image_set(image_records, out_dir):
    """
    Args:
        image_records: list
            each item of list is meta data for a record
            in the collection file
        out_dir: str
            directory to print results

    Returns:
        num_img_download: int
            Number of images downloaded
    """
    num_img_download = 0
    for record in image_records:
        out_code = download_image(out_dir,
                                  record["isic_id"],
                                  record["files"]["thumbnail_256"]["url"])
        num_img_download += 1
        if out_code == 1:
            time.sleep(SLEEP_TIME)

    return num_img_download


def main(collection_id_num, out_dir):

    if re.match(COLLECTION_ID_REGEX, collection_id_num) is None:
        print("Invalid collection id")
        return 1

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_fname = os.path.join(out_dir,
                            f"collection_{collection_id_num}_01.json")
    print(out_fname)
    metadata = get_metadata_download(out_fname,
                            f"{SEARCH_ISIC_URL}={collection_id_num}")
    print(f"N images: {metadata['count']}")

    img_total_count = metadata["count"]

    num_img = download_image_set(metadata["results"], out_dir)
    i = 2
    while (metadata["next"] is not None
           and num_img <= img_total_count):

        out_fname = os.path.join(out_dir,
                        f"collection_{collection_id_num}_{i:02d}.json")

        metadata = get_metadata_download(out_fname,
                                        metadata["next"])
        i += 1
        num_img += download_image_set(metadata["results"], out_dir)

    return 0


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    main(args.collection_id_num, args.dir)
