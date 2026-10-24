"""Valid-set metrics from cpg training logs, max over epochs per metric (HyCoRec protocol: it reports valid, not test).
Rec: recall/mrr/ndcg/tail_recall. Conv (runs with rec_only=false): bleu/dist/f1.
Usage: python analysis/valid_table.py log/20261023-221027 log/20261023-221159 [--per-seed]"""
import glob, json, re, statistics as st, sys

REC = ["recall@1", "recall@10", "recall@50", "mrr@10", "ndcg@10", "tail_recall@50"]
# papers' "Dist-n" (MHIM/HyCoRec) is our dist_cnt@n
CONV = ["bleu@1", "bleu@2", "dist_cnt@2", "dist_cnt@3", "dist_cnt@4", "f1"]


def report(L, i):
    # pandas-style header/value line pairs after the evaluator line, or a json dict on one line
    if L[i + 2].lstrip().startswith("{"):
        return json.loads(L[i + 2])
    b = []
    for l in L[i + 2:]:
        if not l.startswith(" "):
            break
        b.append(l)
    d = {}
    for h, v in zip(b[0::2], b[1::2]):
        d.update(zip(h.replace("\\", "").split(), v.replace("\\", "").split()))
    return {k: float(x) for k, x in d.items()}


def parse(f):
    L = open(f).read().split("\n")
    V = {"train_recommender": [], "train_conversation": []}
    for i, l in enumerate(L):
        for stage in V:
            if stage in l and "[Valid]" in l:
                V[stage].append(report(L, i))
    rec, conv = V["train_recommender"], V["train_conversation"]
    # rec stage must have finished (killed runs have valid epochs but no rec test)
    if not rec or not any("train_recommender" in l and "[Test]" in l for l in L):
        return None
    r = {"n_ep": len(rec), "rec": {k: max(v[k] for v in rec) for k in REC},
         "argmax": [max(range(len(rec)), key=lambda e: rec[e][k]) for k in REC]}
    if conv:
        r["conv"] = {k: max(v[k] for v in conv) for k in CONV}
    return r


def main(dirs, per_seed):
    cells = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/cpg_*.log")):
            r = parse(f)
            if r:
                ds, cell, seed = re.match(r"cpg_(h?\w+?redial)_(.+)_seed(\d+)\.log", f.split("/")[-1]).groups()
                cell = f"{ds}/{cell}"
                cells.setdefault(cell, []).append((seed, r))
    for part, K in (("rec", REC), ("conv", CONV)):
        rows = {c: [(s, r) for s, r in rs if part in r] for c, rs in cells.items()}
        if not any(rows.values()):
            continue
        print(f"\n[{part}] cell | seeds | " + " | ".join(K))
        for cell, rs in rows.items():
            if not rs:
                continue
            if per_seed:
                for s, r in rs:
                    extra = f"  ep={r['n_ep']} argmax={r['argmax']}" if part == "rec" else ""
                    print(f"{cell} | {s} | " + " | ".join(f"{r[part][k]:.4f}" for k in K) + extra)
            print(f"{cell} | {','.join(s for s, _ in rs)} (mean) | "
                  + " | ".join(f"{st.mean(r[part][k] for _, r in rs):.4f}" for k in K))


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")], "--per-seed" in sys.argv)
