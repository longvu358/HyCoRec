"""QC + cross-check for the RevCore raw review corpus (CPC-Hypergraph v2 Branch 2).

Two things this checks, before any of it is trusted as input to
build_review_index.py:

1. Corpus quality on its own terms: coverage/length distribution, obvious
   junk (near-empty reviews, encoding issues), duplicate reviews per movie.
2. Cross-check against data HyCoRec's own authors already shipped
   (`*_conv_idx_to_review_info.pkl`) -- if RevCore's reviews and HyCoRec's
   own pre-selected review entities agree on which real-world movie/person
   a given id refers to, that's strong evidence the two are compatible
   (likely the same underlying review source), not just coincidentally
   joinable DBpedia URIs.

Usage::

    python analysis/review_data_check.py --dataset hredial

Output: prints a summary and writes docs/statistic_results/review_data_check.json
"""

import argparse
import json
import os
import pickle
import random
import statistics

from crslab.config import DATASET_PATH, ROOT_PATH

REVIEW_DIR = os.path.join(ROOT_PATH, "..", "data", "reviews", "raw", "revcore")


def _load_revcore():
    with open(os.path.join(REVIEW_DIR, "movieid2review_dict.pkl"), "rb") as f:
        movie2rev = pickle.load(f)
    with open(os.path.join(REVIEW_DIR, "id2entity.pkl"), "rb") as f:
        id2entity_movie = pickle.load(f)  # RevCore movie-id space -> uri
    with open(os.path.join(REVIEW_DIR, "entity2entityId.pkl"), "rb") as f:
        entity2id_rc = pickle.load(f)  # uri -> RevCore full-entity-id space (0..64361)
    return movie2rev, id2entity_movie, entity2id_rc


def corpus_quality(movie2rev):
    """Sanity-check the review text itself, independent of any HyCoRec data."""
    n_empty = n_internal_dup = 0
    words_per_review = []
    reviews_per_movie = []
    non_ascii_flagged = 0
    junk_examples = []

    for mid, revs in movie2rev.items():
        revs = [r for r in revs if isinstance(r, str) and r.strip()]
        reviews_per_movie.append(len(revs))
        if not revs:
            n_empty += 1
            continue
        if len(set(revs)) < len(revs):
            n_internal_dup += 1
        for r in revs:
            nw = len(r.split())
            words_per_review.append(nw)
            if nw < 5 and len(junk_examples) < 5:
                junk_examples.append((mid, r))
            non_ascii = sum(1 for c in r if ord(c) > 127)
            if len(r) and non_ascii / len(r) > 0.05:
                non_ascii_flagged += 1

    wpr_sorted = sorted(words_per_review)
    return {
        "n_movies": len(movie2rev),
        "n_movies_empty": n_empty,
        "n_movies_with_internal_dup_reviews": n_internal_dup,
        "reviews_per_movie": {
            "min": min(reviews_per_movie),
            "max": max(reviews_per_movie),
            "mean": round(statistics.mean(reviews_per_movie), 2),
            "median": statistics.median(reviews_per_movie),
        },
        "words_per_review": {
            "min": min(words_per_review),
            "max": max(words_per_review),
            "mean": round(statistics.mean(words_per_review), 1),
            "median": statistics.median(words_per_review),
            "p10": wpr_sorted[len(wpr_sorted) // 10],
        },
        "non_ascii_flagged": non_ascii_flagged,
        "n_reviews_total": len(words_per_review),
        "junk_examples_lt5_words": junk_examples,
    }


def coverage_vs_hycorec_items(dataset_path, id2entity_movie, movie2rev):
    with open(os.path.join(dataset_path, "entity2id.json"), encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(os.path.join(dataset_path, "side_data.pkl"), "rb") as f:
        side_data = pickle.load(f)
    id2entity_hyco = {v: k for k, v in entity2id.items()}
    item_uris = {id2entity_hyco[i] for i in side_data["item_entity_ids"] if i in id2entity_hyco}

    catalog_uris = set(id2entity_movie.values())  # every RevCore movie id, review or not
    reviewed_uris = {
        id2entity_movie[int(mid)]
        for mid, revs in movie2rev.items()
        if revs and int(mid) in id2entity_movie
    }
    return {
        "n_items_V_I": len(item_uris),
        "n_items_in_revcore_catalog": len(item_uris & catalog_uris),
        "catalog_coverage_pct": round(100 * len(item_uris & catalog_uris) / len(item_uris), 1),
        "n_items_with_actual_reviews": len(item_uris & reviewed_uris),
        "review_coverage_pct": round(100 * len(item_uris & reviewed_uris) / len(item_uris), 1),
    }


def cross_check_against_hycorec_review_pkl(dataset_path, movie2rev, id2entity_movie, entity2id_rc, split="test"):
    """Decode HyCoRec's own selected_entityIds (legacy RevCore/MHIM 64k-entity
    vocabulary, NOT this dataset's own entity2id.json -- see finding below) and
    check how often the referenced entity is itself a reviewed movie."""
    path = os.path.join(dataset_path, f"{split}_conv_idx_to_review_info.pkl")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        hyco_rev = pickle.load(f)

    id2ent_rc_full = {v: k for k, v in entity2id_rc.items()}
    uri2movieid = {v: k for k, v in id2entity_movie.items()}

    n_refs = n_resolved = n_is_movie = n_has_reviews = 0
    examples = []
    for conv_idx, info in hyco_rev.items():
        for ent_id_str in info["selected_entityIds"]:
            n_refs += 1
            uri = id2ent_rc_full.get(int(ent_id_str))
            if uri is None:
                continue
            n_resolved += 1
            mid = uri2movieid.get(uri)
            if mid is None:
                continue
            n_is_movie += 1
            revs = movie2rev.get(str(mid))
            if revs:
                n_has_reviews += 1
                if len(examples) < 3:
                    examples.append({"conv_idx": conv_idx, "entity_uri": uri, "review_excerpt": revs[0][:200]})

    return {
        "split": split,
        "n_entity_refs": n_refs,
        "resolved_via_revcore_64k_vocab": n_resolved,
        "resolved_pct": round(100 * n_resolved / n_refs, 1),
        "of_which_is_a_movie": n_is_movie,
        "of_which_movie_has_reviews": n_has_reviews,
        "expected_movie_has_reviews_pct_if_random": round(100 * len(movie2rev) / 6924, 1),
        "observed_movie_has_reviews_pct": round(100 * n_has_reviews / n_is_movie, 1) if n_is_movie else None,
        "examples": examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="hredial")
    ap.add_argument("--tokenize", default="nltk")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    random.seed(0)

    dataset_path = os.path.join(DATASET_PATH, args.dataset, args.tokenize)
    movie2rev, id2entity_movie, entity2id_rc = _load_revcore()

    quality = corpus_quality(movie2rev)
    coverage = coverage_vs_hycorec_items(dataset_path, id2entity_movie, movie2rev)
    cross = cross_check_against_hycorec_review_pkl(
        dataset_path, movie2rev, id2entity_movie, entity2id_rc, split=args.split
    )

    print(f"[quality] {quality['n_movies']} movies, {quality['n_movies_empty']} empty, "
          f"{quality['n_movies_with_internal_dup_reviews']} with internal dup reviews")
    print(f"[quality] reviews/movie: {quality['reviews_per_movie']}")
    print(f"[quality] words/review: {quality['words_per_review']}")
    print(f"[quality] non-ascii flagged: {quality['non_ascii_flagged']}/{quality['n_reviews_total']}")
    print(f"[coverage] {coverage['n_items_in_revcore_catalog']}/{coverage['n_items_V_I']} "
          f"({coverage['catalog_coverage_pct']}%) of {args.dataset} V_I is in RevCore's movie catalog")
    print(f"[coverage] {coverage['n_items_with_actual_reviews']}/{coverage['n_items_V_I']} "
          f"({coverage['review_coverage_pct']}%) of {args.dataset} V_I has actual review text")
    if cross:
        print(f"[cross-check vs HyCoRec's own {args.split}_conv_idx_to_review_info.pkl]")
        print(f"  entity refs: {cross['n_entity_refs']}, resolved via RevCore's 64k-entity vocab: "
              f"{cross['resolved_via_revcore_64k_vocab']} ({cross['resolved_pct']}%)")
        print(f"  of resolved, entity is itself a movie: {cross['of_which_is_a_movie']}")
        print(f"  of those movies, has reviews: {cross['of_which_movie_has_reviews']} "
              f"({cross['observed_movie_has_reviews_pct']}%, expected ~{cross['expected_movie_has_reviews_pct_if_random']}% at random)")
        for ex in cross["examples"]:
            print(f"  e.g. conv_idx={ex['conv_idx']} entity={ex['entity_uri']}")
            print(f"       RevCore review: {ex['review_excerpt']}")

    out_dir = os.path.join(ROOT_PATH, "docs", "statistic_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "review_data_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"quality": quality, "coverage": coverage, "cross_check": cross}, f, indent=2)
    print(f"\n[done] -> {out_path}")


if __name__ == "__main__":
    main()
