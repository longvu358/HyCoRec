"""Pre-analysis for evidence-driven retention: is there within-sample signal beyond age?

Unit = scope-C window e (w=3, dedupe, entity field = entities + items, as the model) of a rec sample.
Label u_e = target in e, or target 1-hop connected to a node of e in A_hat^G (item + entity fields).
Evidence features use only turns <= T (the context), never the target.

Usage: python evidence_probe.py <hredial|htgredial>   (fit on valid, evaluate on test)
"""
import json
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from crslab.data.dataset.hycorec_common import merge_conv_turns  # noqa: E402
from crslab.model.crs.hycorec.cpc import sliding_window_hyperedges  # noqa: E402

DS = sys.argv[1]
TOK = {"hredial": "nltk", "htgredial": "pkuseg"}[DS]
W = 3
root = f"data/dataset/{DS}/{TOK}"
tok2ind = json.load(open(f"{root}/token2id.json"))
entity2id = json.load(open(f"{root}/entity2id.json"))
A = sum(sp.load_npz(f"data/collective/{DS}_full/A_hat_{f}.npz").tocsr() for f in ("item", "entity"))
A.data[:] = 1
NB = {}


def nb(v):
    if v not in NB:
        NB[v] = set(A.indices[A.indptr[v]:A.indptr[v + 1]].tolist()) if v < A.shape[0] else set()
    return NB[v]


FEATS = ["age", "log_age", "n_remention", "ov_max", "cont_later", "rel_now", "seeker_share", "size", "has_item"]


def windows_of(split):
    """Yield per-sample (feature matrix, labels) for every rec sample of the split."""
    raw = json.load(open(f"{root}/{split}_data.json"))
    for group in raw:
        turns = merge_conv_turns(group[-1]["dialog"], tok2ind, entity2id, tok2ind.get("__unk__", 3))
        for k in range(1, len(turns)):
            if turns[k]["role"] != "Recommender" or not turns[k]["item"]:
                continue
            ctx = turns[:k]
            node_turns = [list(dict.fromkeys(t["entity"] + t["item"])) for t in ctx]
            items = {i for t in ctx for i in t["item"]}
            T = len(ctx) - 1
            edges, ends = sliding_window_hyperedges(node_turns, W, return_end=True, dedupe=True)
            if not edges:
                continue
            now = set(node_turns[T]) or set(next((n for n in reversed(node_turns) if n), []))
            rows = []
            for e, end in zip(edges, ends):
                es = set(e)
                nbe = set().union(*(nb(v) for v in es))
                later = [set(node_turns[t]) for t in range(end + 1, T + 1)]
                rem = [len(es & s) / len(es) for s in later]
                cont = [len(s & (es | nbe)) / len(s) for s in later if s]
                first = {}
                for t in range(max(0, end - W + 1), end + 1):
                    for v in node_turns[t]:
                        first.setdefault(v, ctx[t]["role"])
                age = T - end
                rows.append([age, np.log1p(age), sum(r > 0 for r in rem), max(rem, default=0.0),
                             max(cont, default=0.0), len(now & (es | nbe)) / max(1, len(now)),
                             np.mean([first[v] == "Seeker" for v in es]), len(es), float(bool(es & items))])
            for tgt in turns[k]["item"]:
                lab = [float(tgt in set(e) or any(tgt in nb(v) for v in e)) for e in edges]
                yield np.array(rows, dtype=float), np.array(lab)


def collect(split):
    X, y, sid = [], [], []
    for i, (x, l) in enumerate(windows_of(split)):
        X.append(x)
        y.append(l)
        sid += [i] * len(l)
    return np.concatenate(X), np.concatenate(y), np.array(sid)


def within_auc(score, y, sid):
    """Pairwise AUC inside samples (what a softmax over windows can exploit), pooled over pairs."""
    num = den = 0.0
    n_mixed = 0
    for s in np.unique(sid):
        m = sid == s
        p, n = score[m][y[m] == 1], score[m][y[m] == 0]
        if len(p) and len(n):
            n_mixed += 1
            d = p[:, None] - n[None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum()
            den += d.size
    return num / den, n_mixed


Xv, yv, sv = collect("valid")
Xt, yt, st = collect("test")
n_samples = len(np.unique(st))
print(f"{DS}: test {n_samples} samples, {len(yt)} windows, P(u)={yt.mean():.3f}; valid {len(yv)} windows")

age_t = Xt[:, 0]
print("\n[1] P(u | age, re-mentioned later?)   (test)")
print("  age   no-remention (n)        remention (n)")
for a in range(0, 8):
    m = np.minimum(age_t, 7) == a
    r = Xt[:, 2] > 0
    s = lambda mm: f"{yt[mm].mean():.3f} ({mm.sum():5d})" if mm.sum() else "   -          "
    print(f"  {a if a < 7 else '>=7':>3}   {s(m & ~r)}      {s(m & r)}")

print("\n[2] P(u | age, relevance to current turn rel_now)")
for a in (1, 2, 3, 5):
    m = (age_t >= a) if a == 5 else (age_t == a)
    lo, hi = m & (Xt[:, 5] == 0), m & (Xt[:, 5] > 0)
    print(f"  age {'>=5' if a == 5 else a}: rel_now=0 {yt[lo].mean():.3f} (n={lo.sum()})   rel_now>0 {yt[hi].mean():.3f} (n={hi.sum()})")

print("\n[3] within-sample pairwise AUC on test (0.5 = no selection signal)")
sc = StandardScaler().fit(Xv)
res = {}
for name, score in [("recency (-age)", -Xt[:, 0]),
                    ("re-mentions", Xt[:, 2] - 1e-3 * Xt[:, 0]),
                    ("rel_now", Xt[:, 5] - 1e-3 * Xt[:, 0]),
                    ("cont_later", Xt[:, 4] - 1e-3 * Xt[:, 0])]:
    res[name] = within_auc(score, yt, st)
for name, cols in [("LR age-only", [0, 1]), ("LR age+size+item", [0, 1, 7, 8]),
                   ("LR evidence (all)", list(range(len(FEATS)))),
                   ("LR evidence w/o rel_now", [i for i in range(len(FEATS)) if i != 5])]:
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xv)[:, cols], yv)
    res[name] = within_auc(lr.decision_function(sc.transform(Xt)[:, cols]), yt, st)
    if name == "LR evidence (all)":
        coef = dict(zip([FEATS[c] for c in cols], np.round(lr.coef_[0], 3)))
for k, (auc, nm) in res.items():
    print(f"  {k:26s} {auc:.4f}   (samples with mixed labels: {nm}/{n_samples})")
print("  LR(all) standardized coefs:", coef)

print("\n[4] same, only samples with T >= 5 context turns (where history is long)")
long_s = {s for s in np.unique(st) if Xt[st == s, 0].max() >= 4}
m = np.isin(st, list(long_s))
lr_a = LogisticRegression(max_iter=2000).fit(sc.transform(Xv)[:, [0, 1, 7, 8]], yv)
lr_e = LogisticRegression(max_iter=2000).fit(sc.transform(Xv), yv)
for name, score in [("recency", -Xt[:, 0]), ("LR age+size+item", lr_a.decision_function(sc.transform(Xt)[:, [0, 1, 7, 8]])),
                    ("LR evidence (all)", lr_e.decision_function(sc.transform(Xt)))]:
    auc, nm = within_auc(score[m], yt[m], st[m])
    print(f"  {name:26s} {auc:.4f}   (mixed: {nm}/{len(long_s)})")
