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

Output: ``data/collective/<dataset>/A_hat_{item,entity,word}.npz`` (+ ``stats.json``,
``tail_items.json``) -- "G0" in the ablation ladder (dialogue hyperedges only).

With ``--with_review`` / ``--with_aspect`` (needs ``build_review_index.py``
run first), also adds the review-hypergraph ``H^{G,rev}_E`` (Eq. 4/6, one
hyperedge per reviewed item = {item} u its top review entities, field E) and/or
the aspect-centric ``H^{G,asp}_I`` (Eq. 5/7, one hyperedge per salient entity
= the items whose reviews rank it highest, field I). ``--use_review`` is
shorthand for both together ("G" full in the ablation ladder; either flag
alone is A6/A7's "G0 + one of the two"). Written to a separate
``<dataset>_{review,aspect,full}`` directory so the G0-only artifacts used by
A0/A3 configs are untouched; point a config's ``collective_path`` at
whichever one that ablation needs.
"""

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

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


def _review_hyperedges(review_index):
    """H^{G,rev}_E (Eq. 4/6): {i} u E_top^+(R_i), one hyperedge per item that
    has review data. ``review_index``: {item_id(str): [[entity_id, tf], ...]}."""
    edges = []
    for item_id_str, pairs in review_index.items():
        members = sorted({int(item_id_str)} | {eid for eid, _ in pairs})
        if len(members) >= 2:
            edges.append(members)
    return edges


def _aspect_hyperedges(review_index, m_min, k_asp):
    """H^{G,asp}_I (Eq. 5/7): for each aspect entity c with df(c) >= m_min,
    one hyperedge = Top-k_asp items by tf^+(c,i). Returns (edges, df)."""
    item_tf_by_entity = defaultdict(dict)
    for item_id_str, pairs in review_index.items():
        item_id = int(item_id_str)
        for eid, tf in pairs:
            item_tf_by_entity[eid][item_id] = tf

    df = {c: len(items) for c, items in item_tf_by_entity.items()}
    edges = []
    for c, items in item_tf_by_entity.items():
        if df[c] < m_min:
            continue
        ranked = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:k_asp]
        members = [i for i, _ in ranked]
        if len(members) >= 2:
            edges.append(members)
    return edges, df


def _unified_dialogue_hyperedges(groups, tok2ind, entity2id, unk_idx, n_entity, k_w):
    """v2.2 Eq. 3: one hyperedge per distinct training conversation over
    V^G = V_E u V_W, h_d = V_I(d) u V_E(d) u (V_W(d) + n_E). Items are a subset
    of V_E so share the entity index space. ``k_w`` (or None) caps the words
    kept per h_d, preferring the rarest ones (highest idf)."""
    seen, convs = set(), []
    for group in groups:
        for conv in group:
            cid = int(conv["conv_id"])
            if cid in seen:
                continue
            seen.add(cid)
            turns = merge_conv_turns(conv["dialog"], tok2ind, entity2id, unk_idx)
            items, entities, words = session_node_sets(turns)
            convs.append((set(items) | set(entities), set(words)))
    w_df = Counter(w for _, ws in convs for w in ws)
    edges, n_word_total, n_word_kept = [], 0, 0
    for ents, words in convs:
        n_word_total += len(words)
        if k_w is not None and len(words) > k_w:
            words = sorted(words, key=lambda w: (w_df[w], w))[:k_w]
        n_word_kept += len(words)
        members = sorted(ents | {w + n_entity for w in words})
        if len(members) >= 2:
            edges.append(members)
    return edges, len(seen), n_word_total, n_word_kept


def _size_hist(edges):
    sizes = np.array([len(h) for h in edges]) if edges else np.zeros(1)
    return {
        "n": len(edges),
        "min": int(sizes.min()),
        "p50": float(np.percentile(sizes, 50)),
        "p90": float(np.percentile(sizes, 90)),
        "max": int(sizes.max()),
        "mean": round(float(sizes.mean()), 2),
    }


def _build_unified(args, groups, tok2ind, entity2id, side_data, n_entity, n_word, unk_idx):
    """v2.2 scope G: ONE hypergraph on V^G (Eq. 6-7) -> A_hat_G.npz."""
    n_g = n_entity + n_word
    dlg, n_conv, w_total, w_kept = _unified_dialogue_hyperedges(
        groups, tok2ind, entity2id, unk_idx, n_entity, args.k_w_dlg
    )
    parts = {"dlg": dlg, "rev": [], "asp": []}
    review_index = None
    if args.with_review or args.with_aspect or args.use_review:
        path = args.review_index or os.path.join(
            DATA_PATH, "reviews", args.dataset, "review_index.json"
        )
        with open(path, encoding="utf-8") as f:
            review_index = json.load(f)
        if args.with_review or args.use_review:
            parts["rev"] = _review_hyperedges(review_index)
        if args.with_aspect or args.use_review:
            parts["asp"], _ = _aspect_hyperedges(review_index, args.m_min, args.k_asp)
    all_edges = parts["dlg"] + parts["rev"] + parts["asp"]
    incidence = _incidence(all_edges, n_g)
    a_hat = a_hat_from_incidence(incidence).tocsr()

    suffix = "_unified" + ("_full" if review_index is not None and args.use_review else "")
    out_dir = args.out_dir or os.path.join(DATA_PATH, "collective", args.dataset + suffix)
    os.makedirs(out_dir, exist_ok=True)
    sp.save_npz(os.path.join(out_dir, "A_hat_G.npz"), a_hat)

    item_universe = side_data.get("item_entity_ids") or []
    deg = np.asarray(incidence.tocsr().sum(axis=1)).ravel()
    n_dlg_item = Counter(v for h in parts["dlg"] for v in h if v < n_entity)
    n_rev_item = Counter(v for h in parts["rev"] + parts["asp"] for v in h if v < n_entity)
    n_tail = sum(1 for i in item_universe if n_dlg_item.get(i, 0) == 0 and n_rev_item.get(i, 0) > 0)
    stats = {
        "mode": "unified",
        "n_train_conversations": n_conv,
        "n_entity": n_entity,
        "n_word": n_word,
        "n_G": n_g,
        "word_offset": n_entity,
        "k_w_dlg": args.k_w_dlg,
        "word_share_in_h_d": round(
            sum(1 for h in parts["dlg"] for v in h if v >= n_entity)
            / max(1, sum(len(h) for h in parts["dlg"])), 3,
        ),
        "words_per_conv_before_cap": round(w_total / max(1, n_conv), 1),
        "words_per_conv_after_cap": round(w_kept / max(1, n_conv), 1),
        "hyperedges": {k: _size_hist(v) for k, v in parts.items()},
        "a_hat_nnz": int(a_hat.nnz),
        "a_hat_density": float(a_hat.nnz) / (n_g * n_g),
        "n_isolated_nodes": int((deg == 0).sum()),
        "n_tail_saved_by_review": n_tail,
        "n_items": len(item_universe),
    }
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"[done] unified Â^G ({n_g}x{n_g}, nnz={a_hat.nnz}) -> {out_dir}")


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
    ap.add_argument(
        "--use_review",
        action="store_true",
        help="shorthand for --with_review --with_aspect (ablation A8/A9: 'G' full).",
    )
    ap.add_argument(
        "--with_review",
        action="store_true",
        help="add H^{G,rev}_E (Eq. 4/6, field E) from build_review_index.py's output "
        "(ablation A6: G0 + h^rev,G).",
    )
    ap.add_argument(
        "--with_aspect",
        action="store_true",
        help="add H^{G,asp}_I (Eq. 5/7, field I) from build_review_index.py's output "
        "(ablation A7: G0 + h^asp).",
    )
    ap.add_argument(
        "--review_index",
        default=None,
        help="path to review_index.json; default data/reviews/<dataset>/review_index.json",
    )
    ap.add_argument("--m_min", type=int, default=3, help="aspect-centric Eq. 5: df(c) >= m_min")
    ap.add_argument("--k_asp", type=int, default=50, help="aspect-centric Eq. 5: Top-k_asp items per aspect")
    ap.add_argument(
        "--unified",
        action="store_true",
        help="v2.2: build ONE Â^G over V^G=V_E u V_W (word ids offset by n_E), "
        "written as A_hat_G.npz, instead of three per-field matrices.",
    )
    ap.add_argument(
        "--k_w_dlg",
        type=int,
        default=None,
        help="--unified only: max words per dialogue hyperedge h_d (rarest kept); "
        "default uncapped. Inspect stats.json word_share_in_h_d first.",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="output dir; default data/collective/<dataset> (G0) or "
        "data/collective/<dataset>_full (--use_review)",
    )
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

    if args.unified:
        _build_unified(args, groups, tok2ind, entity2id, side_data, n_entity, n_word, unk_idx)
        return

    print(f"[{args.dataset}] {len(groups)} groups -> extracting dialogue hyperedges ...")
    edges, n_conv = _dialogue_hyperedges(groups, tok2ind, entity2id, unk_idx)
    n_dlg = {f: len(edges[f]) for f in edges}

    with_review = args.with_review or args.use_review
    with_aspect = args.with_aspect or args.use_review

    review_index = None
    if with_review or with_aspect:
        review_index_path = args.review_index or os.path.join(
            DATA_PATH, "reviews", args.dataset, "review_index.json"
        )
        with open(review_index_path, encoding="utf-8") as f:
            review_index = json.load(f)
        if with_review:
            review_edges = _review_hyperedges(review_index)
            edges["entity"] = edges["entity"] + review_edges
            print(f"[{args.dataset}] +{len(review_edges)} review hyperedges (field E, Eq. 4/6) from {review_index_path}")
        if with_aspect:
            aspect_edges, df = _aspect_hyperedges(review_index, args.m_min, args.k_asp)
            edges["item"] = edges["item"] + aspect_edges
            print(
                f"[{args.dataset}] +{len(aspect_edges)} aspect hyperedges (field I, Eq. 5/7, "
                f"m_min={args.m_min}, k_asp={args.k_asp}, {sum(1 for v in df.values() if v >= args.m_min)}"
                f"/{len(df)} candidate aspects pass df>=m_min)"
            )

    suffix = "_full" if (with_review and with_aspect) else "_review" if with_review else "_aspect" if with_aspect else ""
    out_dir = args.out_dir or os.path.join(DATA_PATH, "collective", args.dataset + suffix)
    os.makedirs(out_dir, exist_ok=True)

    stats = {
        "n_train_conversations": n_conv,
        "tokenize": tokenize,
        "with_review": with_review,
        "with_aspect": with_aspect,
        "fields": {},
    }
    for field in ("item", "entity", "word"):
        he = edges[field]
        incidence = _incidence(he, n_nodes[field])
        a_hat = a_hat_from_incidence(incidence).tocsr()
        sp.save_npz(os.path.join(out_dir, f"A_hat_{field}.npz"), a_hat)
        sizes = Counter(len(h) for h in he)
        stats["fields"][field] = {
            "n_hyperedges": len(he),
            "n_dialogue_hyperedges": n_dlg[field],
            "n_review_or_aspect_hyperedges": len(he) - n_dlg[field],
            "n_nodes": n_nodes[field],
            "a_hat_nnz": int(a_hat.nnz),
            "edge_size_min": min(sizes) if sizes else 0,
            "edge_size_max": max(sizes) if sizes else 0,
            "edge_size_mean": round(
                sum(k * v for k, v in sizes.items()) / max(1, len(he)), 2
            ),
        }
        print(
            f"  {field:6s}: {len(he):6d} hyperedges ({n_dlg[field]} dlg + "
            f"{len(he) - n_dlg[field]} rev/asp) | Â nnz={a_hat.nnz:>9d} "
            f"| |h| mean={stats['fields'][field]['edge_size_mean']}"
        )

    item_universe = side_data.get("item_entity_ids") or sorted(
        {v for h in edges["item"][: n_dlg["item"]] for v in h}
    )
    # deg_dlg is always computed from dialogue hyperedges only (checklist §4.1
    # step 7's Matthew-effect stat is specifically about the review signal
    # reaching items dialogue alone never touches).
    tail_items, freq = _tail_items(edges["item"][: n_dlg["item"]], item_universe)
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

    if review_index is not None:
        n_saved = sum(
            1 for i in item_universe if freq.get(i, 0) == 0 and str(i) in review_index
        )
        stats["tail"]["n_tail_saved_by_review"] = n_saved
        print(
            f"  matthew: {n_saved}/{n_unseen} items with zero dialogue mentions "
            f"gain a review hyperedge anyway (deg_dlg=0, deg_rev>0)"
        )

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
