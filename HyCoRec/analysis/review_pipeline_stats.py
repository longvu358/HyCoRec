"""Pipeline parameters / counts for the review branch (build_review_index.py + build_collective.py).
Usage: python analysis/review_pipeline_stats.py  -> docs/statistic_results/review_pipeline_stats.json"""
import glob, json, os, pickle, statistics as st, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_review_index as B
from crslab.config import DATA_PATH, DATASET_PATH


def q(xs):
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))]
    return {"n": len(xs), "min": xs[0], "p10": p(.1), "median": p(.5), "mean": round(sum(xs) / len(xs), 2), "p90": p(.9), "max": xs[-1]}


def index_stats(ds, k_asp=50, m_min=3):
    idx = json.load(open(f"{DATA_PATH}/reviews/{ds}/review_index.json"))
    asp = defaultdict(dict)
    for i, pairs in idx.items():
        for e, tf in pairs:
            asp[e][i] = tf
    df = {e: len(v) for e, v in asp.items()}
    kept = {e: v for e, v in asp.items() if df[e] >= m_min}
    big = [e for e in kept if df[e] > k_asp]
    ties = 0
    for e in big:
        r = sorted(kept[e].values(), reverse=True)
        ties += r[k_asp - 1] == r[k_asp]
    return {
        "E_top_size": q([len(v) for v in idx.values()]),
        "tf_of_kept_entities": q([tf for v in idx.values() for _, tf in v]),
        "n_aspects_candidate": len(df), "n_aspects_df_ge_m_min": len(kept),
        "df_of_candidates": q(list(df.values())),
        "aspect_hyperedge_size_before_cap": q([df[e] for e in kept]),
        "n_aspects_truncated_by_k_asp": len(big), "n_truncations_with_tie_at_cutoff": ties,
    }


def en_text_stats():
    from nltk import sent_tokenize
    mid2uri = pickle.load(open(f"{B.REVIEW_RAW_DIR}/id2entity.pkl", "rb"))
    rev = pickle.load(open(f"{B.REVIEW_RAW_DIR}/movieid2review_dict.pkl", "rb"))
    d = f"{DATASET_PATH}/hredial/nltk"
    e2i = json.load(open(f"{d}/entity2id.json")); items = set(pickle.load(open(f"{d}/side_data.pkl", "rb"))["item_entity_ids"])
    rel = [m for m, u in mid2uri.items() if u in e2i and e2i[u] in items]
    per_item, rlen, slen, dup = [], [], [], 0
    for m in rel:
        rs = [r for r in rev.get(str(m), []) if isinstance(r, str) and len(r.split()) >= B.MIN_REVIEW_WORDS]
        dup += len(rs) - len(set(rs)); per_item.append(len(rs))
        for r in rs:
            rlen.append(len(r.split())); slen += [len(B._WORD_RE.findall(s)) for s in sent_tokenize(r)]
    return {"reviews_per_item_used": q(per_item), "n_items_with_>=1_review": sum(1 for x in per_item if x),
            "review_len_words": q(rlen), "sentence_len_tokens": q(slen), "exact_duplicate_reviews_within_item": dup}


def zh_text_stats():
    import pyarrow.compute as pc, pyarrow.parquet as pq
    d = f"{DATASET_PATH}/htgredial/pkuseg"
    e2i = json.load(open(f"{d}/entity2id.json")); items = set(pickle.load(open(f"{d}/side_data.pkl", "rb"))["item_entity_ids"])
    files = sorted(glob.glob(f"{B.DOUBAN_RAW_DIR}/*.parquet"))
    names = set()
    for f in files:
        names.update(pq.read_table(f, columns=["Movie_Name"]).column(0).unique().to_pylist())
    n2i, _ = B._douban_item_map(names, e2i, items)
    cnt, lens = Counter(), []
    for f in files:
        t = pq.read_table(f, columns=["Movie_Name", "Comment"])
        t = t.filter(pc.is_in(t["Movie_Name"], value_set=__import__("pyarrow").array(list(n2i))))
        for n, c in zip(t["Movie_Name"].to_pylist(), t["Comment"].to_pylist()):
            if isinstance(c, str) and len(c) >= B.ZH_MIN_COMMENT_CHARS:
                cnt[n] += 1; lens.append(len(c))
    return {"comments_per_item_used": q(list(cnt.values())), "comment_len_chars": q(lens)}


out = {"constants": {k: getattr(B, k) for k in ("POS_THRESHOLD", "MIN_REVIEW_WORDS", "MAX_PHRASE_WORDS", "MIN_SURFACE_CHARS",
        "POS_STAR", "ZH_MIN_COMMENT_CHARS", "ZH_MIN_SURFACE_CHARS", "ZH_MAX_PHRASE_CHARS")}}
for ds in ("hredial", "htgredial"):
    out[ds] = {"index": index_stats(ds), "raw_index_stats": json.load(open(f"{DATA_PATH}/reviews/{ds}/review_index_stats.json")),
               "text": en_text_stats() if ds == "hredial" else zh_text_stats(),
               "collective_full": json.load(open(f"{DATA_PATH}/collective/{ds}_full/stats.json"))}
os.makedirs("docs/statistic_results", exist_ok=True)
json.dump(out, open("docs/statistic_results/review_pipeline_stats.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({ds: {k: v for k, v in out[ds].items() if k != "collective_full"} for ds in ("hredial", "htgredial")}, indent=1))
