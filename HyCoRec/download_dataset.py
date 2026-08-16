"""Trigger crslab's dataset auto-download without running the full training pipeline.

crslab.data.dataset.base.BaseDataset downloads its resource zip as the very first
step of __init__, before any preprocessing. Running run_crslab.py works too, but it
also builds the model afterwards and dies with FileNotFoundError if data/edger/<dataset>
hasn't been built yet (see run_edger.py). This script does only the download step.
"""

import argparse
import importlib
import os

from crslab.config import DATASET_PATH
from crslab.download import build

# dataset name -> (resources module, tokenize matching config/crs/hycorec/<dataset>.yaml)
DATASET_TOKENIZE = {
    "redial": ("crslab.data.dataset.redial.resources", "nltk"),
    "tgredial": ("crslab.data.dataset.tgredial.resources", "pkuseg"),
    "opendialkg": ("crslab.data.dataset.opendialkg.resources", "nltk"),
    "durecdial": ("crslab.data.dataset.durecdial.resources", "jieba"),
    "inspired": ("crslab.data.dataset.inspired.resources", "nltk"),
    "hredial": ("crslab.data.dataset.hredial.resources", "nltk"),
    "htgredial": ("crslab.data.dataset.htgredial.resources", "pkuseg"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        choices=sorted(DATASET_TOKENIZE),
        help="Dataset name",
    )
    parser.add_argument(
        "-t",
        "--tokenize",
        type=str,
        default=None,
        help="Override the tokenize variant to download",
    )
    args = parser.parse_args()

    module_path, default_tokenize = DATASET_TOKENIZE[args.dataset]
    tokenize = args.tokenize or default_tokenize
    resources = importlib.import_module(module_path).resources
    resource = resources[tokenize]

    dpath = os.path.join(DATASET_PATH, args.dataset, tokenize)
    print(f"Downloading {args.dataset} ({tokenize}) into {dpath}")
    build(dpath, resource["file"], version=resource["version"])
    print("Done.")
