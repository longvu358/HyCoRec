"""KG hub-degree + built review-info stats for hredial / htgredial (side by side).

Usage: python analysis/graph_review_stats.py
Output: prints a summary, writes docs/statistic_results/graph_review_stats.json
"""

import json
import os
import pickle

import numpy as np

from crslab.config import DATASET_PATH, ROOT_PATH

CFG = {"hredial": "nltk", "htgredial": "pkuseg"}
EDGER_DIR = os.path.join(ROOT_PATH, "data", "edger")


def degree_stats(ds):
    out = {}
    for t in ("item", "entity", "word"):
        with open(os.path.join(EDGER_DIR, ds, f"{t}_edger.pkl"), "rb") as f:
            d = pickle.load(f)
        deg = np.sort(np.array([len(v) for v in d.values()]))[::-1]
        top = sorted(d.items(), key=lambda kv: -len(kv[1]))[:5]
        out[t] = {
            "n_nodes": len(d),
            "mean": round(float(deg.mean()), 1),
            "median": int(np.median(deg)),
            "p90": int(np.percentile(deg, 90)),
            "p99": int(np.percentile(deg, 99)),
            "max": int(deg[0]),
            "n_deg_gt_100": int((deg > 100).sum()),
            "n_deg_gt_1000": int((deg > 1000).sum()),
            "top1pct_edge_share": round(100 * float(deg[: max(1, len(deg) // 100)].sum() / deg.sum()), 1),
            "top5": [(str(k), len(v)) for k, v in top],
        }
    return out


def review_stats(ds):
    base = os.path.join(DATASET_PATH, ds, CFG[ds])
    with open(os.path.join(base, "entity2id.json"), encoding="utf-8") as f:
        e2i = json.load(f)
    with open(os.path.join(base, "side_data.pkl"), "rb") as f:
        item_ids = set(pickle.load(f)["item_entity_ids"])
    out = {}
    for sp in ("train", "valid", "test"):
        with open(os.path.join(base, f"{sp}_conv_idx_to_review_info.pkl"), "rb") as f:
            d = pickle.load(f)
        ne = [len(v["selected_entityIds"]) for v in d.values() if v["selected_entityIds"]]
        ids = [int(x) for v in d.values() for x in v["selected_entityIds"]]
        nw = [len(s) for v in d.values() for s in v["selected_infoListListInt"]]
        out[sp] = {
            "n_conv": len(d),
            "pct_nonempty": round(100 * len(ne) / len(d), 1),
            "ents_per_nonempty_mean": round(float(np.mean(ne)), 2) if ne else None,
            "pct_ent_in_vocab": round(100 * np.mean([i < len(e2i) for i in ids]), 1) if ids else None,
            "pct_ent_is_item": round(100 * np.mean([i in item_ids for i in ids]), 1) if ids else None,
            "n_distinct_ents": len(set(ids)),
            "tokens_per_review_mean": round(float(np.mean(nw)), 1) if nw else None,
        }
    return out


def main():
    res = {ds: {"degree": degree_stats(ds), "review_info": review_stats(ds)} for ds in CFG}
    for ds, r in res.items():
        print(f"== {ds}")
        for t, s in r["degree"].items():
            print(f"  {t}: " + ", ".join(f"{k}={v}" for k, v in s.items() if k != "top5"), "| top5", s["top5"][:3])
        for sp, s in r["review_info"].items():
            print(f"  review[{sp}]", s)
    out = os.path.join(ROOT_PATH, "docs", "statistic_results", "graph_review_stats.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
