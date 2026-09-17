"""Build the Collective-scope (scope G) static hypergraph for HyCoRec / CPC-v2.

Reads the CRSLab-processed ``train_data.json`` (per-user-grouped) for a dataset,
builds the corpus-level dialogue hyperedges ``H^{G,dlg}_f = {V_f(d) : d in D_train,
|V_f(d)| >= 2}`` for f in {item, entity, word}, then precomputes and stores the
sparse propagation matrix ``A_hat^G_f = D^-1 N E^-1 N^T`` (Eq. 7).

Leakage guard (spec 2.4.1): only ``train_data.json`` is read, and each
conversation is counted once (groups repeat conversations).

Usage::

    python build_collective.py -d hredial            # or htgredial
    python build_collective.py -d hredial --tokenize nltk

Output: ``data/collective/<dataset>/A_hat_{item,entity,word}.npz`` (+ ``stats.json``).

Review / aspect-centric hyperedges (spec Eq. 4-5) are Branch 2 and are added by
re-running this with ``--review-index <path>`` once that index exists.
"""

import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import scipy.sparse as sp

from crslab.config import DATA_PATH, DATASET_PATH
from crslab.data.dataset.hycorec_common import merge_conv_turns, session_node_sets
from crslab.model.crs.hycorec.cpc import a_hat_from_incidence

DEFAULT_TOKENIZE = {"hredial": "nltk", "htgredial": "pkuseg"}
HEAD_FRACTION = 0.2  # top 20% of items by train frequency == "head"; rest == "tail"


def _dialogue_hyperedges(groups, tok2ind, entity2id, unk_idx):
    """One hyperedge per distinct training conversation, per field."""
    seen = set()
    edges = {"item": [], "entity": [], "word": []}
    for group in groups:
        for conv in group:
            cid = int(conv["conv_id"])
            if cid in seen:
                continue
            seen.add(cid)
            turns = merge_conv_turns(conv["dialog"], tok2ind, entity2id, unk_idx)
            items, entities, words = session_node_sets(turns)
            if len(items) >= 2:
                edges["item"].append(items)
            if len(entities) >= 2:
                edges["entity"].append(entities)
            if len(words) >= 2:
                edges["word"].append(words)
    return edges, len(seen)


def _tail_items(item_hyperedges, item_universe):
    """Long-tail item set by train frequency (spec 6.1 Tail-Recall@10):
    rank all items in V_I by how many training conversations mention them,
    the top HEAD_FRACTION by item COUNT (not by mass) is the "head", the rest
    -- including items never mentioned in train at all -- is the "tail".
    """
    freq = Counter()
    for members in item_hyperedges:
        freq.update(members)
    ranked = sorted(item_universe, key=lambda i: freq.get(i, 0), reverse=True)
    n_head = max(1, round(HEAD_FRACTION * len(ranked)))
    tail = sorted(ranked[n_head:])
    return tail, freq


def _incidence(hyperedges, n_nodes):
    rows, cols = [], []
    for j, members in enumerate(hyperedges):
        for v in members:
            rows.append(v)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    return sp.csc_matrix(
        (data, (rows, cols)), shape=(n_nodes, len(hyperedges))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, choices=list(DEFAULT_TOKENIZE))
    ap.add_argument("--tokenize", default=None)
    args = ap.parse_args()

    tokenize = args.tokenize or DEFAULT_TOKENIZE[args.dataset]
    dpath = os.path.join(DATASET_PATH, args.dataset, tokenize)

    with open(os.path.join(dpath, "token2id.json"), encoding="utf-8") as f:
        tok2ind = json.load(f)
    with open(os.path.join(dpath, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(dpath, "train_data.json"), encoding="utf-8") as f:
        groups = json.load(f)
    with open(os.path.join(dpath, "side_data.pkl"), "rb") as f:
        side_data = pickle.load(f)

    n_entity = max(entity2id.values()) + 1
    n_word = max(tok2ind.values()) + 1
    n_nodes = {"item": n_entity, "entity": n_entity, "word": n_word}
    unk_idx = tok2ind.get("__unk__", 3)

    print(f"[{args.dataset}] {len(groups)} groups -> extracting dialogue hyperedges ...")
    edges, n_conv = _dialogue_hyperedges(groups, tok2ind, entity2id, unk_idx)

    out_dir = os.path.join(DATA_PATH, "collective", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    stats = {"n_train_conversations": n_conv, "tokenize": tokenize, "fields": {}}
    for field in ("item", "entity", "word"):
        he = edges[field]
        incidence = _incidence(he, n_nodes[field])
        a_hat = a_hat_from_incidence(incidence).tocsr()
        sp.save_npz(os.path.join(out_dir, f"A_hat_{field}.npz"), a_hat)
        sizes = Counter(len(h) for h in he)
        stats["fields"][field] = {
            "n_hyperedges": len(he),
            "n_nodes": n_nodes[field],
            "a_hat_nnz": int(a_hat.nnz),
            "edge_size_min": min(sizes) if sizes else 0,
            "edge_size_max": max(sizes) if sizes else 0,
            "edge_size_mean": round(
                sum(k * v for k, v in sizes.items()) / max(1, len(he)), 2
            ),
        }
        print(
            f"  {field:6s}: {len(he):6d} hyperedges | Â nnz={a_hat.nnz:>9d} "
            f"| |h| mean={stats['fields'][field]['edge_size_mean']}"
        )

    item_universe = side_data.get("item_entity_ids") or sorted(
        {v for h in edges["item"] for v in h}
    )
    tail_items, freq = _tail_items(edges["item"], item_universe)
    with open(os.path.join(out_dir, "tail_items.json"), "w", encoding="utf-8") as f:
        json.dump(tail_items, f)
    n_unseen = sum(1 for i in item_universe if freq.get(i, 0) == 0)
    stats["tail"] = {
        "n_items": len(item_universe),
        "n_head": len(item_universe) - len(tail_items),
        "n_tail": len(tail_items),
        "n_tail_unseen_in_train": n_unseen,
    }
    print(
        f"  tail  : {len(tail_items)}/{len(item_universe)} items "
        f"({n_unseen} never mentioned in train) -> tail_items.json"
    )

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
