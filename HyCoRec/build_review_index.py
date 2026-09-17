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

Usage::

    python build_review_index.py -d hredial --k_rev 10

Requires: nltk vader_lexicon + punkt (nltk.download('vader_lexicon'); nltk.download('punkt')).
"""

import argparse
import json
import os
import pickle
import re
from collections import Counter, defaultdict

from nltk import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from crslab.config import DATA_PATH, DATASET_PATH, ROOT_PATH

REVIEW_RAW_DIR = os.path.join(ROOT_PATH, "..", "data", "reviews", "raw", "revcore")
DEFAULT_TOKENIZE = {"hredial": "nltk", "htgredial": "pkuseg"}

POS_THRESHOLD = 0.05  # spec 1.3: compound >= 0.05 -> sent(q) = +1
MIN_REVIEW_WORDS = 5  # drop junk entries like "SPOILERS" (docs/review_data_qc.md)
MAX_PHRASE_WORDS = 6  # longest entity surface form we bother trying to match
MIN_SURFACE_CHARS = 4  # drop noise like "It", "Up" from the linking dictionary

_DISAMBIG_SUFFIX = re.compile(r"_\([^)]*\)$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'.]*|[0-9]+")


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
    args = ap.parse_args()

    tokenize = args.tokenize or DEFAULT_TOKENIZE[args.dataset]
    dpath = os.path.join(DATASET_PATH, args.dataset, tokenize)

    with open(os.path.join(dpath, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(dpath, "side_data.pkl"), "rb") as f:
        side_data = pickle.load(f)
    item_ids = set(side_data["item_entity_ids"])
    uri2entity_id = entity2id

    with open(os.path.join(REVIEW_RAW_DIR, "id2entity.pkl"), "rb") as f:
        id2entity_movie = pickle.load(f)  # revcore movie id -> uri
    with open(os.path.join(REVIEW_RAW_DIR, "movieid2review_dict.pkl"), "rb") as f:
        movie2rev = pickle.load(f)

    print(f"[{args.dataset}] building surface-form entity index ...")
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

    e_top = {}
    for item_id, counter in tf_pos.items():
        top = counter.most_common(args.k_rev)
        if top:
            e_top[str(item_id)] = [[eid, tf] for eid, tf in top]

    sizes = [len(v) for v in e_top.values()]
    stats = {
        "dataset": args.dataset,
        "k_rev": args.k_rev,
        "n_items_with_review_text": len(relevant),
        "n_reviews_used": n_reviews_used,
        "n_sentences_total": n_sent_total,
        "n_sentences_positive": n_sent_pos,
        "n_positive_sentences_with_a_linked_entity": n_linked_pos,
        "n_items_with_E_top_nonempty": len(e_top),
        "pct_items_with_E_top_nonempty": round(100 * len(e_top) / max(1, len(relevant)), 1),
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
