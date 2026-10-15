#!/usr/bin/env python3
"""S10: dataset statistics and per-sample node-set overlap between the C / P / G scopes (no training).

    python scripts/extra/cpg_s10.py -d htgredial            # -> results/cpg/s10_stats.csv, s10_overlap.csv, s10_summary.md
    python scripts/extra/cpg_s10.py -d hredial --split test

Per test recommendation sample (same samples as rec_process_fn), over nodes (field, id):
  C = nodes of the current-dialogue turns            (scope C reads these)
  P = history-session nodes + k_hop KG expansion (+ review entities of history items), as the model builds them
  G = nodes one hyperedge away in H^G from C u P     (non-zero columns of the rows of A_hat^G_f)
G always contains its seeds (so J(C,G), J(P,G) are inflated by construction); `novel_G` = |G \\ (C u P)| / |G| says how much of G is new evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from crslab.config import DATASET_PATH, DATA_PATH  # noqa: E402
from crslab.data.dataset.hycorec_common import process_grouped_raw_data  # noqa: E402
from crslab.model.crs.hycorec.cpc import khop_closure  # noqa: E402

FIELDS = ("item", "entity", "word")


def load_adj(edger_path, key2id):
    raw = pickle.load(open(edger_path, "rb"))
    adj = {}
    for a, lst in raw.items():
        if a in key2id:
            adj[key2id[a]] = [key2id[x] for x in lst if x in key2id]
    return adj


def jaccard(a, b):
    """a, b: sorted unique int arrays."""
    if a.size == 0 and b.size == 0:
        return float("nan")
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, choices=["hredial", "htgredial"])
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config/crs/cpg" / args.dataset / "full.yaml").read_text())
    tok = cfg["tokenize"]
    dpath = Path(DATASET_PATH) / args.dataset / tok
    tok2ind = json.load(open(dpath / "token2id.json", encoding="utf-8"))
    entity2id = json.load(open(dpath / "entity2id.json", encoding="utf-8"))
    side = pickle.load(open(dpath / "side_data.pkl", "rb"))
    unk = tok2ind.get("__unk__", 3)
    k_hist, k_hop, cap = cfg.get("k_hist", 40), cfg.get("k_hop", 1), cfg.get("khop_cap")
    coll_dir = ROOT / cfg["collective_path"]
    out_dir = ROOT / "results" / "cpg"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- tab:stats -------------------------------------------------
    allsplit = {s: json.load(open(dpath / f"{s}_data.json", encoding="utf-8")) for s in ("train", "valid", "test")}
    stats = {}
    users = {c["user_id"] for g in allsplit["train"] + allsplit["valid"] + allsplit["test"] for c in g}
    stats["n_users"] = len(users)
    stats["n_conversations"] = len({c["conv_id"] for s in allsplit.values() for g in s for c in g})
    items = side.get("item_entity_ids") or []
    stats["n_items"] = len(items)
    test_groups = allsplit["test"]  # each group = one target session + its D_u(d)
    stats["pct_test_sessions_with_history"] = round(100 * sum(len(g) > 1 for g in test_groups) / max(1, len(test_groups)), 2)
    rev_path = Path(DATA_PATH) / "reviews" / args.dataset / "review_index.json"
    if rev_path.exists():
        review_index = {int(k): [e for e, _ in v] for k, v in json.load(open(rev_path)).items()}
        stats["pct_items_with_review"] = round(100 * sum(i in review_index for i in items) / max(1, len(items)), 2)
    else:
        review_index = {}
    cs = json.load(open(coll_dir / "stats.json"))
    for f, v in cs["fields"].items():
        stats[f"G_{f}_hyperedges_dlg"] = v["n_dialogue_hyperedges"]
        stats[f"G_{f}_hyperedges_rev_or_asp"] = v["n_review_or_aspect_hyperedges"]
        stats[f"G_{f}_a_hat_nnz"] = v["a_hat_nnz"]
    stats["items_only_reachable_via_review"] = cs.get("tail", {}).get("n_tail_saved_by_review")
    stats["items_unseen_in_train_dialogues"] = cs.get("tail", {}).get("n_tail_unseen_in_train")
    with open(out_dir / f"s10_stats_{args.dataset}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "stat", "value"])
        for k, v in stats.items():
            w.writerow([args.dataset, k, v])

    # ---------------- per-sample overlap ----------------------------------------
    ent_adj = load_adj(DATA_PATH + f"/edger/{args.dataset}/entity_edger.pkl", entity2id)
    word_adj = load_adj(DATA_PATH + f"/edger/{args.dataset}/word_edger.pkl", tok2ind)
    a_hat = {}
    for f in FIELDS:
        p = coll_dir / f"A_hat_{f}.npz"
        if p.exists():
            a_hat[f] = sp.load_npz(p).tocsr()
    use_rev = bool(cfg.get("use_review_hypergraph")) and review_index
    n_e = max(entity2id.values()) + 1
    OFF = {"item": 0, "entity": n_e, "word": 2 * n_e}  # disjoint int ranges per field

    samples = process_grouped_raw_data(allsplit[args.split], tok2ind, entity2id, unk, k_hist)
    rows = []
    for s in samples:
        if s["role"] != "Recommender" or not s["items"]:
            continue
        if args.max_samples and len(rows) >= args.max_samples:
            break
        C, P = set(), set()
        for f in FIELDS:
            C |= {OFF[f] + v for turn in s[f] for v in turn}
        h_item = s.get("history_session_items") or []
        h_ent = s.get("history_session_entities") or []
        h_word = s.get("history_session_words") or []
        P |= {OFF["item"] + v for sess in h_item for v in sess}
        for f, sessions, adj in (("entity", h_ent, ent_adj), ("word", h_word, word_adj)):
            for v in {v for sess in sessions for v in sess}:
                P |= {OFF[f] + u for u in khop_closure([v], adj, k_hop, max_neighbors=cap)}
        if use_rev:
            for i in {v for sess in h_item for v in sess}:
                if review_index.get(i):
                    P |= {OFF["entity"] + e for e in [i] + review_index[i]}
        seeds = np.array(sorted(C | P), dtype=np.int64)
        parts = []
        for f, m in a_hat.items():
            lo, hi = OFF[f], OFF[f] + m.shape[0]
            for v in seeds[(seeds >= lo) & (seeds < hi)] - lo:
                parts.append(m.indices[m.indptr[v]:m.indptr[v + 1]].astype(np.int64) + lo)
        G = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        Ca, Pa = np.array(sorted(C), dtype=np.int64), np.array(sorted(P), dtype=np.int64)
        rows.append({
            "conv_id": s["conv_id"], "n_hist": len(h_item), "n_C": Ca.size, "n_P": Pa.size, "n_G": G.size,
            "J_PG": jaccard(Pa, G), "J_CP": jaccard(Ca, Pa), "J_CG": jaccard(Ca, G),
            "novel_G": np.setdiff1d(G, seeds, assume_unique=True).size / max(1, G.size),
        })
    with open(out_dir / f"s10_overlap_{args.dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"## S10 {args.dataset} ({args.split}, {len(rows)} samples)", "", "| stat | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in stats.items()]
    lines += ["", "| pair | median | p25 | p75 | n(valid) |", "|---|---|---|---|---|"]
    for k in ("J_PG", "J_CP", "J_CG", "novel_G"):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        lines.append(f"| {k} | {np.median(v):.3f} | {np.percentile(v, 25):.3f} | {np.percentile(v, 75):.3f} | {len(v)} |"
                     if len(v) else f"| {k} | - | - | - | 0 |")
    text = "\n".join(lines)
    (out_dir / f"s10_summary_{args.dataset}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
