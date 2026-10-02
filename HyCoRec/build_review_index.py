"""Build E_top^+(R_i) -- the review-hypergraph index for CPC-Hypergraph v2 Branch 2
(spec 1.3, 4.1 steps 2-3; see docs/contexual_personal_collective/cpc_hypergraph_v2.1_method_spec.md).

For every item i in V_I with review text (data/reviews/raw/revcore/, see
docs/review_data_qc.md for provenance/QC), extracts the sentence-level
positive review entities:

    E_top^+(R_i) = Top-k_rev { c : c in linked-entities-of-positive-sentences } by tf^+(c,i)

Pipeline per review sentence:
  1. VADER sentiment (compound >= 0.05 -> positive, spec convention).
  2. Entity link positive sentences against a surface-form phrase index built
     from this dataset's own entity2id.json (excluding V_I -- spec: movie
     titles must not become "aspect" entities). This is a lightweight
     substitute for a hosted entity linker (e.g. DBpedia Spotlight, which
     this environment can't call): case-sensitive, greedy-longest-match
     phrase matching, restricted to entities already in the task KG. Real
     names in a KG built by CRSLab's own linker, so precision is reasonable;
     documented here as a deliberate simplification, not the paper's own
     linker.

Output: data/reviews/<dataset>/review_index.json =
    {item_id(str): [[entity_id, tf_pos], ...]}   # sorted by tf_pos desc, len <= k_rev
Consumed by build_collective.py to add H^{G,rev}_E (Eq. 4/6) and the
aspect-centric H^{G,asp}_I (Eq. 5/7, needs the tf counts to rank both
directions) and by the model for H^{P,rev}_E (Eq. 1r/1s).

htgredial (Chinese) uses the Douban short-comment corpus instead (docs/tgredial_review_corpus.md):
no VADER -- a comment is positive iff its star rating >= --pos_star (default 4); no
sentence split -- each short comment is one unit; entity linking is greedy longest-match
over the Chinese surface forms of this dataset's entity2id.json (parenthetical
disambiguation stripped), V_I excluded. Raw corpus (not committed, see PROVENANCE.md
and the licence note in the docs): data/reviews/raw/douban1000w/*.parquet.

Usage::

    python build_review_index.py -d hredial --k_rev 10
    python build_review_index.py -d htgredial --k_rev 10

Requires: hredial -- nltk vader_lexicon + punkt (nltk.download('vader_lexicon');
nltk.download('punkt')); htgredial -- pyarrow.
"""

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict

from tqdm import tqdm

from crslab.config import DATA_PATH, DATASET_PATH, ROOT_PATH

REVIEW_RAW_DIR = os.path.join(ROOT_PATH, "..", "data", "reviews", "raw", "revcore")
DOUBAN_RAW_DIR = os.path.join(DATA_PATH, "reviews", "raw", "douban1000w")
DEFAULT_TOKENIZE = {"hredial": "nltk", "htgredial": "pkuseg"}

POS_THRESHOLD = 0.05  # spec 1.3: compound >= 0.05 -> sent(q) = +1
MIN_REVIEW_WORDS = 5  # drop junk entries like "SPOILERS" (docs/review_data_qc.md)
MAX_PHRASE_WORDS = 6  # longest entity surface form we bother trying to match
MIN_SURFACE_CHARS = 4  # drop noise like "It", "Up" from the linking dictionary

_DISAMBIG_SUFFIX = re.compile(r"_\([^)]*\)$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'.]*|[0-9]+")

# zh (Douban short comments)
POS_STAR = 4  # Star >= 4 -> positive (replaces VADER, which is English-only)
ZH_MIN_COMMENT_CHARS = 5
ZH_MIN_SURFACE_CHARS = 2
ZH_MAX_PHRASE_CHARS = 12
_ZH_TRAILING_PAREN = re.compile(r"[（(][^）)]*[）)]\s*$")
_ZH_TAG = re.compile(r"<[^>]*>")
# The Baike-style KG has an entity for every common word ('喜欢（胡杨林歌曲）',
# '结局（冷漠演唱歌曲）'). A song/album/book/brand/... disambiguation => the surface
# form is almost always just the plain word in a comment, not that entity.
_ZH_NON_ASPECT_PAREN = re.compile(
    r"歌曲|演唱|专辑|单曲|歌手|乐队|小说|图书|出版|品牌|公司|化妆品|保险|游戏|漫画|软件|网站|缩写|姓氏|地名|行政"
)
_ZH_CJK = re.compile(r"[\u4e00-\u9fff]")


def build_surface_index(entity2id, item_ids):
    """{surface phrase (Title Case, as it would appear in text): entity_id},
    entities in V_I excluded (spec: no movie-title-as-aspect), ambiguous
    surface forms (two different entities rendering to the same name) dropped
    rather than guessed at.
    """
    index = {}
    ambiguous = set()
    for uri, eid in entity2id.items():
        if eid in item_ids:
            continue
        name = uri.rsplit("/", 1)[-1].rstrip(">")
        name = _DISAMBIG_SUFFIX.sub("", name).replace("_", " ").strip()
        if len(name) < MIN_SURFACE_CHARS or not name[:1].isupper():
            continue
        if name in index and index[name] != eid:
            ambiguous.add(name)
            continue
        index[name] = eid
    for name in ambiguous:
        index.pop(name, None)
    return index


def link_entities(sentence, surface_index, max_words=MAX_PHRASE_WORDS):
    """Greedy longest-match phrase lookup against `surface_index`."""
    tokens = _WORD_RE.findall(sentence)
    n = len(tokens)
    found = []
    i = 0
    while i < n:
        for w in range(min(max_words, n - i), 0, -1):
            eid = surface_index.get(" ".join(tokens[i : i + w]))
            if eid is not None:
                found.append(eid)
                i += w
                break
        else:
            i += 1
    return found


def zh_norm(name):
    """Normalise an entity / movie name for matching: drop <a> tags, a trailing
    parenthetical disambiguation ('（1962年…电影）') and whitespace."""
    name = _ZH_TAG.sub("", name)
    name = _ZH_TRAILING_PAREN.sub("", name)
    return re.sub(r"\s+", "", name).strip()


def build_zh_surface_index(entity2id, item_ids):
    """{surface string: entity_id}; V_I excluded, ambiguous names dropped
    (same policy as build_surface_index)."""
    index, ambiguous = {}, set()
    for name_raw, eid in entity2id.items():
        if eid in item_ids:
            continue
        name = zh_norm(name_raw)
        if len(name) < ZH_MIN_SURFACE_CHARS or len(name) > ZH_MAX_PHRASE_CHARS:
            continue
        paren = _ZH_TRAILING_PAREN.search(_ZH_TAG.sub("", name_raw))
        if not _ZH_CJK.search(name) or (paren and _ZH_NON_ASPECT_PAREN.search(paren.group(0))):
            continue
        if name in index and index[name] != eid:
            ambiguous.add(name)
            continue
        index[name] = eid
    for name in ambiguous:
        index.pop(name, None)
    return index


def link_entities_zh(text, surface_index, prefixes, max_chars=ZH_MAX_PHRASE_CHARS):
    """Greedy longest-match over characters; `prefixes` = set of the first
    ZH_MIN_SURFACE_CHARS chars of every surface form (cheap reject)."""
    found = []
    n = len(text)
    i = 0
    while i < n:
        if text[i : i + ZH_MIN_SURFACE_CHARS] in prefixes:
            for w in range(min(max_chars, n - i), ZH_MIN_SURFACE_CHARS - 1, -1):
                eid = surface_index.get(text[i : i + w])
                if eid is not None:
                    found.append(eid)
                    i += w
                    break
            else:
                i += 1
        else:
            i += 1
    return found


def _douban_item_map(movie_names, entity2id, item_ids):
    """{douban Movie_Name: item entity_id}. Douban names look like
    '记忆碎片 Memento' (Chinese title, then original title); KG names look like
    '记忆碎片（2000年克里斯托弗·诺兰执导电影）'. Match on the normalised Chinese
    title; skip a name when either side is ambiguous (several KG items or
    several Douban movies sharing it) rather than guess."""
    kg = {}
    for name_raw, eid in entity2id.items():
        if eid in item_ids:
            kg.setdefault(zh_norm(name_raw), set()).add(eid)
    dm = {}
    for name in movie_names:
        dm.setdefault(zh_norm(name.split(" ")[0]), []).append(name)
    out, n_amb = {}, 0
    for key, names in dm.items():
        eids = kg.get(key)
        if not eids:
            continue
        if len(eids) > 1 or len(names) > 1:
            n_amb += 1
            continue
        out[names[0]] = next(iter(eids))
    return out, n_amb


def collect_tf_zh(entity2id, item_ids, pos_star):
    import glob

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(DOUBAN_RAW_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards in {DOUBAN_RAW_DIR} (see docs/tgredial_review_corpus.md)")
    names = set()
    for f in files:
        names.update(pq.read_table(f, columns=["Movie_Name"]).column(0).unique().to_pylist())
    name2item, n_amb = _douban_item_map(names, entity2id, item_ids)
    print(f"  {len(name2item)} items in V_I resolve to a Douban movie ({n_amb} ambiguous names skipped)")

    surface_index = build_zh_surface_index(entity2id, item_ids)
    prefixes = {s[:ZH_MIN_SURFACE_CHARS] for s in surface_index}
    print(f"  {len(surface_index)} linkable surface forms (V_I excluded)")

    import pyarrow as pa

    keep = pa.array(list(name2item))
    tf_pos = defaultdict(Counter)
    n_used = n_pos = n_linked = 0
    for f in tqdm(files, desc="shards"):
        t = pq.read_table(f, columns=["Movie_Name", "Star", "Comment"])
        t = t.filter(pc.is_in(t["Movie_Name"], value_set=keep))
        for name, star, text in zip(*(t[c].to_pylist() for c in ("Movie_Name", "Star", "Comment"))):
            if not isinstance(text, str) or len(text) < ZH_MIN_COMMENT_CHARS:
                continue
            n_used += 1
            if star is None or star < pos_star:
                continue
            n_pos += 1
            linked = link_entities_zh(text, surface_index, prefixes)
            if linked:
                n_linked += 1
                tf_pos[name2item[name]].update(linked)
    return tf_pos, {
        "n_items_with_review_text": len(name2item),
        "n_reviews_used": n_used,
        "n_sentences_total": n_used,
        "n_sentences_positive": n_pos,
        "n_positive_sentences_with_a_linked_entity": n_linked,
        "pos_star": pos_star,
    }


def collect_tf_en(entity2id, item_ids):
    from nltk import sent_tokenize
    from nltk.sentiment import SentimentIntensityAnalyzer

    with open(os.path.join(REVIEW_RAW_DIR, "id2entity.pkl"), "rb") as f:
        id2entity_movie = pickle.load(f)  # revcore movie id -> uri
    with open(os.path.join(REVIEW_RAW_DIR, "movieid2review_dict.pkl"), "rb") as f:
        movie2rev = pickle.load(f)
    uri2entity_id = entity2id

    print("  building surface-form entity index ...")
    surface_index = build_surface_index(entity2id, item_ids)
    print(f"  {len(surface_index)} linkable surface forms (V_I excluded)")

    sia = SentimentIntensityAnalyzer()
    tf_pos = defaultdict(Counter)  # item_id -> Counter[entity_id] = tf+(c,i)
    n_reviews_used = n_sent_total = n_sent_pos = n_linked_pos = 0

    relevant = [
        (mid, uri2entity_id[uri])
        for mid, uri in id2entity_movie.items()
        if uri in uri2entity_id and uri2entity_id[uri] in item_ids
    ]
    print(f"  {len(relevant)} items in V_I resolve to a RevCore movie id")

    for mid, item_id in tqdm(relevant, desc="items"):
        revs = movie2rev.get(str(mid))
        if not revs:
            continue
        for review_text in revs:
            if not isinstance(review_text, str) or len(review_text.split()) < MIN_REVIEW_WORDS:
                continue
            n_reviews_used += 1
            for sent in sent_tokenize(review_text):
                n_sent_total += 1
                compound = sia.polarity_scores(sent)["compound"]
                if compound < POS_THRESHOLD:
                    continue
                n_sent_pos += 1
                linked = link_entities(sent, surface_index)
                if linked:
                    n_linked_pos += 1
                    tf_pos[item_id].update(linked)
    return tf_pos, {
        "n_items_with_review_text": len(relevant),
        "n_reviews_used": n_reviews_used,
        "n_sentences_total": n_sent_total,
        "n_sentences_positive": n_sent_pos,
        "n_positive_sentences_with_a_linked_entity": n_linked_pos,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, choices=list(DEFAULT_TOKENIZE))
    ap.add_argument("--tokenize", default=None)
    ap.add_argument(
        "--k_rev",
        type=int,
        default=10,
        help="Top-k entities kept per item (spec default 'k_rev = k' i.e. the "
        "P/G KG-expansion k-hop, currently k_hop=1 in the model config -- too "
        "restrictive for a review hyperedge, so this defaults to 10, the "
        "middle of the spec's own {5,10,20} sweep, instead of literally 1.",
    )
    ap.add_argument("--pos_star", type=int, default=POS_STAR, help="htgredial: Star >= this is positive")
    ap.add_argument(
        "--idf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="rank by tf+ * log(N/df) instead of raw tf+ (the stored value stays raw tf+). "
        "Default: on for htgredial (short comments are dominated by generic words), off for hredial.",
    )
    args = ap.parse_args()

    tokenize = args.tokenize or DEFAULT_TOKENIZE[args.dataset]
    dpath = os.path.join(DATASET_PATH, args.dataset, tokenize)

    with open(os.path.join(dpath, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(dpath, "side_data.pkl"), "rb") as f:
        side_data = pickle.load(f)
    item_ids = set(side_data["item_entity_ids"])

    print(f"[{args.dataset}] collecting tf+ ...")
    if args.dataset == "htgredial":
        tf_pos, counts = collect_tf_zh(entity2id, item_ids, args.pos_star)
    else:
        tf_pos, counts = collect_tf_en(entity2id, item_ids)

    use_idf = args.dataset == "htgredial" if args.idf is None else args.idf
    df = Counter()
    for counter in tf_pos.values():
        df.update(counter.keys())
    n_docs = max(1, len(tf_pos))

    e_top = {}
    for item_id, counter in tf_pos.items():
        if use_idf:
            top = sorted(counter.items(), key=lambda kv: -kv[1] * math.log(n_docs / df[kv[0]]))[: args.k_rev]
        else:
            top = counter.most_common(args.k_rev)
        top = [(eid, tf) for eid, tf in top if not use_idf or df[eid] < n_docs]
        if top:
            e_top[str(item_id)] = [[eid, tf] for eid, tf in top]

    n_rel = counts["n_items_with_review_text"]
    sizes = [len(v) for v in e_top.values()]
    stats = {
        "dataset": args.dataset,
        "k_rev": args.k_rev,
        "rank_by": "tf*idf" if use_idf else "tf",
        **counts,
        "n_items_with_E_top_nonempty": len(e_top),
        "pct_items_with_E_top_nonempty": round(100 * len(e_top) / max(1, n_rel), 1),
        "E_top_size": {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": round(sum(sizes) / len(sizes), 2) if sizes else 0,
        },
    }
    print(json.dumps(stats, indent=2))

    out_dir = os.path.join(DATA_PATH, "reviews", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "review_index.json"), "w", encoding="utf-8") as f:
        json.dump(e_top, f)
    with open(os.path.join(out_dir, "review_index_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[done] -> {out_dir}/review_index.json")


if __name__ == "__main__":
    main()
