"""Dataset statistics for the sliding-window hyperedge design (ReDial / TG-ReDial).

For every session we build role-alternating *turns* (consecutive same-role
utterances merged, matching ``HReDialDataset._convert_to_id``) and, per turn,
the sets of:

* ``item``   -- movies recommended / mentioned via ``@id`` markers  (raw ``movies`` field)
* ``entity`` -- linked attribute entities: genre / actor / director / ...  (raw ``entity`` field)
* ``word``   -- content tokens of the utterance text (stop-words / punctuation removed)

It then produces the five statistics described in ``docs/statistic.md`` plus
plots, a JSON dump and a Markdown report.

Usage:
    python analysis/window_stats.py                 # both datasets, all splits
    python analysis/window_stats.py --dataset hredial
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise

import ijson
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # .../HyCoRec/HyCoRec
# dataset dirs live under either the inner package or the outer repo root
DATA_ROOTS = [REPO, os.path.dirname(REPO)]
OUT_DIR = os.path.join(REPO, "docs", "statistic_results")


def resolve(rel: str) -> str:
    for root in DATA_ROOTS:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(rel)


# candidate window sizes evaluated everywhere
W_CANDIDATES = [1, 2, 3, 5, 10]
GAP_MAX = 10  # histogram cap: bucket ">=10"
DECAY_GMAX = 15
EARLY_THRESH = (
    3  # a target mention is "recent" if within this many turns of the rec turn
)

DATASETS = {
    "hredial": {
        "lang": "EN (ReDial)",
        "paths": [
            "data/dataset/hredial/nltk/train_data.json",
            "data/dataset/hredial/nltk/valid_data.json",
            "data/dataset/hredial/nltk/test_data.json",
        ],
        "stopwords": "data/dataset/hredial/nltk/stopwords.txt",
    },
    "htgredial": {
        "lang": "ZH (TG-ReDial)",
        "paths": [
            "data/dataset/htgredial/pkuseg/train_data.json",
            "data/dataset/htgredial/pkuseg/valid_data.json",
            "data/dataset/htgredial/pkuseg/test_data.json",
        ],
        "stopwords": None,
    },
}

_CJK = re.compile(r"[一-鿿]")
_ALNUM = re.compile(r"[a-zA-Z0-9]")


# --------------------------------------------------------------------------- #
# Loading & turn construction
# --------------------------------------------------------------------------- #
@dataclass
class Turn:
    role: str
    items: set = field(default_factory=set)
    entities: set = field(default_factory=set)
    words: set = field(default_factory=set)
    n_utt: int = 0


def load_stopwords(path: str | None) -> set:
    if not path:
        return set()
    with open(resolve(path), "r", encoding="utf-8") as f:
        return {ln.strip().lower() for ln in f if ln.strip()}


def keep_word(tok: str, stop: set) -> bool:
    t = tok.strip().lower()
    if not t or t in stop:
        return False
    if _CJK.search(t):  # any Chinese char -> keep
        return len(t) >= 1
    if not _ALNUM.search(t):  # pure punctuation
        return False
    return len(t) >= 2


def iter_sessions(paths: list[str]):
    """Yield the canonical dialog for every unique conv_id (first occurrence)."""
    seen: set[str] = set()
    for rel in paths:
        with open(resolve(rel), "rb") as f:
            for conv in ijson.items(f, "item.item"):
                cid = conv["conv_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                yield cid, conv["dialog"]


def build_turns(dialog: list[dict], stop: set) -> list[Turn]:
    turns: list[Turn] = []
    last_role = None
    for utt in dialog:
        role = utt["role"]
        if role != last_role:
            turns.append(Turn(role=role))
        cur = turns[-1]
        cur.n_utt += 1
        cur.items.update(m for m in utt.get("movies", []))
        cur.entities.update(e for e in utt.get("entity", []))
        cur.words.update(w for w in (utt.get("text") or []) if keep_word(w, stop))
        last_role = role
    # an entity that also shows up as an item in the same turn: treat as item only
    for t in turns:
        t.entities -= t.items
    return turns


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def describe(xs: list[float]) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def gap_bucket(dt: int) -> int:
    return min(GAP_MAX, dt)


def length_bucket(n_turns: int) -> str:
    if n_turns < 5:
        return "short(<5)"
    if n_turns <= 10:
        return "medium(5-10)"
    return "long(>10)"


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def analyse(name: str, cfg: dict) -> dict:
    stop = load_stopwords(cfg["stopwords"])
    sessions: list[list[Turn]] = []
    n_raw_utts: list[int] = []

    for _cid, dialog in iter_sessions(cfg["paths"]):
        turns = build_turns(dialog, stop)
        if not turns:
            continue
        sessions.append(turns)
        n_raw_utts.append(sum(t.n_utt for t in turns))

    R = {"dataset": name, "lang": cfg["lang"], "n_sessions": len(sessions)}

    # ---- Stat 1: turns / session ----------------------------------------- #
    turn_counts = [len(s) for s in sessions]
    R["stat1"] = {
        "turns_per_session": describe(turn_counts),
        "utterances_per_session": describe(n_raw_utts),
        "pct_ge": {
            k: round(100 * np.mean([c >= k for c in turn_counts]), 2)
            for k in (5, 10, 15, 20)
        },
        "buckets": {
            b: int(sum(length_bucket(c) == b for c in turn_counts))
            for b in ("short(<5)", "medium(5-10)", "long(>10)")
        },
        "hist": Counter(min(c, 30) for c in turn_counts),
    }

    # ---- Stat 2: turn-gap of repeated item / entity / word ------------- #
    gap = {k: Counter() for k in ("item", "entity", "word")}
    gap_raw = {k: [] for k in ("item", "entity", "word")}
    for s in sessions:
        for kind, attr in (
            ("item", "items"),
            ("entity", "entities"),
            ("word", "words"),
        ):
            occ: dict = defaultdict(list)
            for ti, t in enumerate(s):
                for x in getattr(t, attr):
                    occ[x].append(ti)
            for ts in occ.values():
                if len(ts) < 2:
                    continue
                for a, b in pairwise(ts):
                    dt = b - a
                    gap[kind][gap_bucket(dt)] += 1
                    gap_raw[kind].append(dt)

    R["stat2"] = {}
    for kind in ("item", "entity", "word"):
        raw = gap_raw[kind]
        tot = len(raw)
        R["stat2"][kind] = {
            "n_consecutive_pairs": tot,
            "median_dt": float(np.median(raw)) if raw else None,
            "mean_dt": float(np.mean(raw)) if raw else None,
            "hist_dt": dict(sorted(gap[kind].items())),
            "pct_within_w": {
                w: (round(100 * np.mean([d <= w for d in raw]), 2) if raw else None)
                for w in W_CANDIDATES
            },
        }

    # ---- Stat 3: ground-truth item position ---------------------------- #
    cats = Counter()  # per target-item
    sess_cats = Counter()  # per session (by its primary / first target)
    dist_first = []  # turn_rec - turn_first_mention  (non-cold targets)
    dist_last = []  # turn_rec - turn_last_mention_before
    n_targets = 0
    n_sess_with_target = 0

    def classify(turns: list[Turn], turn_rec: int, e) -> str:
        occ_before = [i for i in range(turn_rec) if e in turns[i].items]
        if not occ_before:
            return "cold (never before rec turn)"
        last = occ_before[-1]
        dist_first.append(turn_rec - occ_before[0])
        dist_last.append(turn_rec - last)
        if turn_rec - last <= EARLY_THRESH:
            return f"recent (last mention <={EARLY_THRESH} before)"
        if len(occ_before) == 1:
            return "early-only (single early mention, not repeated)"
        return "mid (repeated but last mention far)"

    turn_rec_from_end = []
    n_cold = 0
    for s in sessions:
        rec_turns = [i for i, t in enumerate(s) if t.role == "Recommender" and t.items]
        if rec_turns:
            turn_rec = rec_turns[-1]
        else:
            item_turns = [i for i, t in enumerate(s) if t.items]
            if not item_turns:
                continue
            turn_rec = item_turns[-1]
        targets = list(s[turn_rec].items)
        if not targets:
            continue
        n_sess_with_target += 1
        turn_rec_from_end.append(len(s) - 1 - turn_rec)
        first_cat = None
        for e in targets:
            n_targets += 1
            c = classify(s, turn_rec, e)
            cats[c] += 1
            if c.startswith("cold"):
                n_cold += 1
            if first_cat is None:
                first_cat = c
        sess_cats[first_cat] += 1

    n_appear = n_targets - n_cold  # targets mentioned somewhere before the rec turn
    R["stat3"] = {
        "n_sessions_with_target": n_sess_with_target,
        "n_target_items": n_targets,
        "n_cold_targets": n_cold,
        "n_targets_appearing_before_rec": n_appear,
        "target_item_categories_pct": {
            k: round(100 * v / max(n_targets, 1), 2) for k, v in cats.most_common()
        },
        "categories_among_appearing_targets_pct": {
            k: round(100 * v / max(n_appear, 1), 2)
            for k, v in cats.most_common()
            if not k.startswith("cold")
        },
        "session_categories_pct": {
            k: round(100 * v / max(n_sess_with_target, 1), 2)
            for k, v in sess_cats.most_common()
        },
        "turn_rec_distance_from_last_turn": describe(turn_rec_from_end),
        "_dist_first_raw": dist_first,
        "dist_recturn_minus_firstmention": describe(dist_first),
        "dist_recturn_minus_lastmention": describe(dist_last),
        "_note": (
            "cold = target introduced by the recommender, absent from prior session "
            "context (neither local window nor session-global history contains it -> "
            "needs KG / collaborative channel). Among targets that DO appear before "
            "the rec turn, 'early-only' is the population that motivates H_global; "
            "'recent' is where a local window already suffices."
        ),
    }

    # ---- Stat 4: decay curve (co-occurrence lift vs turn gap) --------- #
    # lift(g) = mean_{turn pairs at gap g} |A_kind(t) & A_kind(t+g)|  /  sum_x p_x^2
    # p_x = fraction of turns containing x  ->  denominator = expected intersection
    #        size under independence.  lift > 1 means positive association.
    R["stat4"] = {}
    for kind, attr in (("item", "items"), ("entity", "entities"), ("word", "words")):
        n_turns_total = 0
        xcount = Counter()
        for s in sessions:
            for t in s:
                n_turns_total += 1
                for x in getattr(t, attr):
                    xcount[x] += 1
        if n_turns_total == 0:
            continue
        denom = sum((c / n_turns_total) ** 2 for c in xcount.values())
        inter_sum = defaultdict(float)
        pair_cnt = defaultdict(int)
        for s in sessions:
            L = len(s)
            for i in range(L):
                Ai = getattr(s[i], attr)
                for g in range(1, min(DECAY_GMAX, L - 1 - i) + 1):
                    Aj = getattr(s[i + g], attr)
                    pair_cnt[g] += 1
                    if Ai and Aj:
                        inter_sum[g] += len(Ai & Aj)
        curve = {
            g: (inter_sum[g] / pair_cnt[g] / denom if pair_cnt[g] and denom else None)
            for g in range(1, DECAY_GMAX + 1)
        }
        R["stat4"][kind] = {
            "denom_expected_intersection": denom,
            "lift_by_gap": curve,
            "raw_mean_shared_by_gap": {
                g: (inter_sum[g] / pair_cnt[g] if pair_cnt[g] else None)
                for g in range(1, DECAY_GMAX + 1)
            },
        }

    # ---- Stat 5: hyperedge collapse in short sessions ---------------- #
    by_bucket = defaultdict(list)
    for c in turn_counts:
        by_bucket[length_bucket(c)].append(c)
    coll = {}
    for w in W_CANDIDATES:
        overall = 100 * np.mean([w >= c for c in turn_counts])
        per_bucket = {
            b: (round(100 * np.mean([w >= c for c in v]), 2) if v else None)
            for b, v in by_bucket.items()
        }
        coll[w] = {"overall_pct": round(float(overall), 2), "by_bucket_pct": per_bucket}
    R["stat5"] = {
        "collapse_pct_by_w": coll,
        "_note": "collapse = w >= |session|  ->  local hyperedge == global hyperedge",
    }

    return R


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_dataset(R: dict):
    name = R["dataset"]

    # Stat 1
    h = R["stat1"]["hist"]
    xs = sorted(h)
    ys = [h[x] for x in xs]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(xs, ys, color="#4C72B0")
    ax[0].set_title(f"{name}: turns / session")
    ax[0].set_xlabel("turns (capped at 30)")
    ax[0].set_ylabel("sessions")
    tc = []
    for x, c in h.items():
        tc += [x] * c
    tc = np.sort(tc)
    ax[1].plot(tc, np.arange(1, len(tc) + 1) / len(tc), color="#C44E52")
    for w in (5, 10, 15, 20):
        ax[1].axvline(w, ls=":", c="grey", lw=0.8)
    ax[1].set_title(f"{name}: CDF turns / session")
    ax[1].set_xlabel("turns")
    ax[1].set_ylabel("cumulative fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_stat1_turns.png"), dpi=120)
    plt.close(fig)

    # Stat 2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, color in (
        ("item", "#4C72B0"),
        ("entity", "#55A868"),
        ("word", "#CCB974"),
    ):
        hist = R["stat2"][kind]["hist_dt"]
        if not hist:
            continue
        keys = sorted(hist)
        total = sum(hist.values())
        ax.plot(
            keys, [hist[k] / total for k in keys], marker="o", label=kind, color=color
        )
    for w in W_CANDIDATES:
        ax.axvline(w, ls=":", c="grey", lw=0.8)
    ax.set_title(f"{name}: turn-gap between consecutive re-mentions")
    ax.set_xlabel("Δt turns (10 = ≥10)")
    ax.set_ylabel("fraction of consecutive re-mention pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_stat2_turngap.png"), dpi=120)
    plt.close(fig)

    # Stat 3
    s3 = R["stat3"]
    cat = s3["target_item_categories_pct"]
    if cat:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        labels = list(cat)
        ax[0].bar(
            range(len(labels)),
            [cat[k] for k in labels],
            color=["#C44E52", "#4C72B0", "#55A868", "#CCB974", "#8172B2"][
                : len(labels)
            ],
        )
        ax[0].set_xticks(range(len(labels)))
        ax[0].set_xticklabels(
            [k.split(" (")[0] for k in labels], rotation=20, ha="right"
        )
        ax[0].set_ylabel("% of target items")
        ax[0].set_title(f"{name}: target-item location vs rec turn")
        raw = s3.get("_dist_first_raw", [])
        if raw:
            ax[1].hist(raw, bins=range(max(raw) + 2), color="#4C72B0", align="left")
            ax[1].set_xlabel("rec turn − first mention (turns)")
            ax[1].set_ylabel("target items")
            ax[1].set_title(f"{name}: how early appearing targets first surface")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{name}_stat3_target.png"), dpi=120)
        plt.close(fig)

    # Stat 4
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for kind, color in (
        ("item", "#4C72B0"),
        ("entity", "#55A868"),
        ("word", "#CCB974"),
    ):
        if kind not in R["stat4"]:
            continue
        c = R["stat4"][kind]["lift_by_gap"]
        keys = [k for k in sorted(c) if c[k] is not None]
        ax.plot(keys, [c[k] for k in keys], marker="o", label=kind, color=color)
    ax.axhline(1.0, ls="--", c="grey", lw=0.8)
    ax.set_title(f"{name}: co-occurrence lift vs turn gap (decay curve)")
    ax.set_xlabel("turn gap g")
    ax.set_ylabel("observed / expected shared elements")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}_stat4_decay.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def write_report(results: list[dict]):
    L = ["# Sliding-window hyperedge — dataset statistics\n"]
    L.append(
        "Generated by `analysis/window_stats.py`. Sessions = unique conversations "
        "(train+valid+test pooled, deduped by `conv_id`). A *turn* = consecutive "
        "same-role utterances merged (as in `HReDialDataset._convert_to_id`).\n"
    )

    for R in results:
        n = R["dataset"]
        L.append(f"\n## {n} — {R['lang']}  ({R['n_sessions']:,} sessions)\n")

        s1 = R["stat1"]
        L.append("### Stat 1 — turns / session\n")
        L.append(
            md_table(
                ["metric", "turns/session", "utterances/session"],
                [
                    [
                        k,
                        round(s1["turns_per_session"][k], 2),
                        round(s1["utterances_per_session"][k], 2),
                    ]
                    for k in (
                        "mean",
                        "median",
                        "std",
                        "p25",
                        "p75",
                        "p90",
                        "p95",
                        "min",
                        "max",
                    )
                ],
            )
        )
        L.append(
            "\n"
            + md_table(
                ["≥5 turns", "≥10", "≥15", "≥20"],
                [[f"{s1['pct_ge'][k]} %" for k in (5, 10, 15, 20)]],
            )
        )
        tot = sum(s1["buckets"].values())
        L.append(
            "\n"
            + md_table(
                ["bucket", "sessions", "share"],
                [[b, c, f"{100 * c / tot:.1f} %"] for b, c in s1["buckets"].items()],
            )
        )
        L.append(f"\n![turns]({n}_stat1_turns.png)\n")

        L.append("\n### Stat 2 — turn-gap between consecutive re-mentions (drives w)\n")
        rows = []
        for kind in ("item", "entity", "word"):
            k = R["stat2"][kind]
            rows.append(
                [kind, k["n_consecutive_pairs"], k["median_dt"]]
                + [f"{k['pct_within_w'][w]} %" for w in W_CANDIDATES]
            )
        L.append(
            md_table(
                ["kind", "#pairs", "median Δt"] + [f"Δt≤{w}" for w in W_CANDIDATES],
                rows,
            )
        )
        L.append(f"\n![turngap]({n}_stat2_turngap.png)\n")

        s3 = R["stat3"]
        L.append("\n### Stat 3 — ground-truth item position vs recommend turn\n")
        L.append(
            f"{s3['n_sessions_with_target']:,} sessions with an identifiable recommend "
            f"turn; {s3['n_target_items']:,} target items "
            f"({s3['n_cold_targets']:,} cold / "
            f"{s3['n_targets_appearing_before_rec']:,} appear before the rec turn).\n"
        )
        tre = s3["turn_rec_distance_from_last_turn"]
        if tre:
            L.append(
                f"Rec turn sits {tre['median']:.0f} turns from the session end "
                f"(mean {tre['mean']:.1f}, p90 {tre['p90']:.0f}).\n"
            )
        L.append(
            md_table(
                [
                    "target-item category",
                    "share of ALL targets",
                    "share of APPEARING targets",
                ],
                [
                    [
                        k,
                        f"{v} %",
                        (
                            f"{s3['categories_among_appearing_targets_pct'].get(k)} %"
                            if not k.startswith("cold")
                            else "—"
                        ),
                    ]
                    for k, v in s3["target_item_categories_pct"].items()
                ],
            )
        )
        df = s3["dist_recturn_minus_firstmention"]
        dl = s3["dist_recturn_minus_lastmention"]
        if df:
            L.append("\nDistance in turns (targets that appear before the rec turn):\n")
            L.append(
                md_table(
                    ["metric", "rec − first mention", "rec − last mention"],
                    [
                        [k, round(df[k], 2), round(dl[k], 2)]
                        for k in ("mean", "median", "p90", "max")
                    ],
                )
            )
        L.append(f"\n> {s3['_note']}\n")
        L.append(f"\n![target]({n}_stat3_target.png)\n")

        L.append("\n### Stat 4 — empirical decay curve (co-occurrence lift vs gap)\n")
        rows = []
        for kind in ("item", "entity", "word"):
            if kind not in R["stat4"]:
                continue
            c = R["stat4"][kind]["lift_by_gap"]
            rows.append(
                [kind]
                + [
                    None if c[g] is None else round(c[g], 2)
                    for g in (1, 2, 3, 5, 10, 15)
                ]
            )
        L.append(md_table(["kind", "g=1", "g=2", "g=3", "g=5", "g=10", "g=15"], rows))
        L.append(f"\n![decay]({n}_stat4_decay.png)\n")

        L.append("\n### Stat 5 — hyperedge collapse risk (w ≥ |session|)\n")
        rows = []
        for w in W_CANDIDATES:
            b = R["stat5"]["collapse_pct_by_w"][w]
            rows.append(
                [w, f"{b['overall_pct']} %"]
                + [
                    (
                        "n/a"
                        if b["by_bucket_pct"].get(k) is None
                        else f"{b['by_bucket_pct'][k]} %"
                    )
                    for k in ("short(<5)", "medium(5-10)", "long(>10)")
                ]
            )
        L.append(
            md_table(
                ["w", "overall collapse", "short(<5)", "medium(5-10)", "long(>10)"],
                rows,
            )
        )
        L.append(f"\n> {R['stat5']['_note']}\n")

    with open(os.path.join(OUT_DIR, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    for name in names:
        print(f"[+] analysing {name} ...")
        R = analyse(name, DATASETS[name])
        results.append(R)
        plot_dataset(R)
        print(f"    {R['n_sessions']} sessions done")

    def _json(o):
        if isinstance(o, Counter):
            return dict(o)
        raise TypeError(o)

    slim = json.loads(json.dumps(results, default=_json))
    for R in slim:
        R.get("stat3", {}).pop("_dist_first_raw", None)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    write_report(results)
    print(f"[done] -> {OUT_DIR}")


if __name__ == "__main__":
    main()
