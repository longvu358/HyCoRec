"""Post-hoc diagnostics for a trained HyCoRec / CPC-Hypergraph v2 checkpoint.

Reads only what's needed to reconstruct the model (vocab + side_data +
edger/collective files already on disk) -- it does NOT reprocess
train/valid/test_data.json, so it's fast and safe to run on CPU alongside a
GPU training job.

Prints:
  * the learned per-field scope-fusion weights softmax(alpha_C, alpha_P, alpha_G)
    -- the single cheapest signal for whether a scope is actually contributing
  * the head/tail item split saved by build_collective.py

Requires the training run to have used ``-ss`` (save_system), so
``save/<model_name>.pth`` exists.

Usage::

    python inspect_model.py -c config/crs/hycorec/redial.yaml
    python inspect_model.py -c config/crs/hycorec/ablation/A9b1.yaml -m save/HyCoRec.pth
"""

import argparse
import json
import os
import pickle

import torch

from crslab.config import DATA_PATH, DATASET_PATH, SAVE_PATH, Config
from crslab.model.crs.hycorec.hycorec import HyCoRecModel


def load_light_vocab_and_side_data(opt):
    """The pieces of HReDialDataset/_load_data the model actually needs --
    skips _data_preprocess (the expensive per-conversation pass)."""
    dpath = os.path.join(DATASET_PATH, opt["dataset"].lower(), opt["tokenize"])
    with open(os.path.join(dpath, "token2id.json"), encoding="utf-8") as f:
        tok2ind = json.load(f)
    with open(os.path.join(dpath, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(dpath, "side_data.pkl"), "rb") as f:
        side_data = pickle.load(f)
    vocab = {
        "tok2ind": tok2ind,
        "ind2tok": {v: k for k, v in tok2ind.items()},
        "entity2id": entity2id,
        "id2entity": {v: k for k, v in entity2id.items()},
        "vocab_size": len(tok2ind),
        "n_entity": max(entity2id.values()) + 1,
    }
    return vocab, side_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument(
        "-m",
        "--model_file",
        default=None,
        help="checkpoint path; default save/<model_name>.pth from the config",
    )
    args = ap.parse_args()

    opt = Config(args.config, gpu="-1").opt  # read-only inspection: always CPU
    vocab, side_data = load_light_vocab_and_side_data(opt)

    device = torch.device("cpu")
    model = HyCoRecModel(opt, device, vocab, side_data)

    model_file = args.model_file or os.path.join(SAVE_PATH, f"{opt['model_name']}.pth")
    if not os.path.isfile(model_file):
        raise SystemExit(
            f"No checkpoint at {model_file}. Re-run training with -ss (save_system)."
        )
    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[checkpoint] {model_file}")
    print(
        f"[config] scopes={model.scopes}  ei_mode={model.ei_mode}  "
        f"hconv_layers={model.hconv_layers}  context_window_w={model.context_window_w}  "
        f"k_hist={model.k_hist}  k_hop={model.k_hop}"
    )

    print("\n[fusion weights]  softmax(alpha) per field, order = (C, P, G)")
    for field, w in model.fusion.weight_table().items():
        print(f"  {field:6s}: C={w[0]:.3f}  P={w[1]:.3f}  G={w[2]:.3f}")

    collective_dir = os.path.join(DATA_PATH, "collective", opt["dataset"].lower())
    stats_path = os.path.join(collective_dir, "stats.json")
    if os.path.isfile(stats_path):
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
        if "tail" in stats:
            t = stats["tail"]
            print(
                f"\n[tail split] {t['n_tail']}/{t['n_items']} items are tail "
                f"({t['n_tail_unseen_in_train']} never mentioned in train)"
            )


if __name__ == "__main__":
    main()
