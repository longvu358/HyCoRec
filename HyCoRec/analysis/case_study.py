"""Case-study data for the paper figures: real ReDial test conversations + each model's top-k.

Joins the per-sample test dumps of `scripts/run_cpg.py analyze` (results/cpg/dump/<ds>_<cell>_seed<s>.jsonl)
with the raw test dialogs, so every row carries
  * the user's earlier sessions (scope P, movies mentioned + text),
  * the current session up to the recommendation turn (merged turns, @ids replaced by titles),
  * the gold response turn and its target movie,
  * the scope-C relation windows with the main model's retention weight,
  * rank of the target + top-k titles for every model.

    # motivation figure: ReSCOPE vs earlier methods
    python analysis/case_study.py --main rescope_full --baselines hycorec_repro --seed 3407
    # learned-relation figure: learned retention vs latest-only / uniform / fixed decay
    python analysis/case_study.py --mode relation --main rescope_full --seed 3407

Outputs results/cpg/case_study_<mode>_<ds>_<main>_seed<s>.jsonl (all samples) and a .md with the best
candidates (main ranks the target <= `--hit`, every baseline > `--miss`). In `relation` mode a candidate must
also put its largest retention weight on an older window than the newest one, i.e. the learned weight departs
from the 0.75^age prior. ReSCOPE v2's absolute alpha collapses towards 0, so weights are relative (alpha / sum).
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DUMP_DIR = ROOT / "results" / "cpg" / "dump"
DATA = {"hredial": ROOT / "data/dataset/hredial/nltk"}
BASELINES = {"rec": ["hycorec_repro"], "relation": ["r2_latest", "r2_uniform", "r2_ip_fixed"]}


def title(uri):
    s = uri.strip("<>").rsplit("/", 1)[-1].replace("_", " ")
    return re.sub(r"\s*\((\d{4} )?film\)$", "", s)


def mention_titles(groups):
    """@movieId -> title, voted over every utterance whose @ids align 1:1 with its `movies` list."""
    votes = defaultdict(Counter)
    for split in groups:
        for group in split:
            for conv in group:
                for utt in conv["dialog"]:
                    ids = [w[1:] for w in utt["text"] if re.fullmatch(r"@\d+", w)]
                    if ids and len(ids) == len(utt["movies"]):
                        for i, m in zip(ids, utt["movies"]):
                            votes[i][title(m)] += 1
    return {i: c.most_common(1)[0][0] for i, c in votes.items()}


def render(utt, at2title):
    words = [f"[{at2title.get(w[1:], w)}]" if re.fullmatch(r"@\d+", w) else w for w in utt["text"]]
    return re.sub(r" ([.,!?;:'])", r"\1", " ".join(words))


def merged_turns(dialog, at2title):
    """Same merging as hycorec_common.merge_conv_turns (consecutive same-role utterances -> one turn)."""
    turns = []
    for utt in dialog:
        movies = [title(m) for m in utt["movies"]]
        entities = [title(e) for e in utt["entity"]]
        if turns and turns[-1]["role"] == utt["role"]:
            turns[-1]["text"] += " " + render(utt, at2title)
            turns[-1]["movies"] += movies
            turns[-1]["entities"] += entities
        else:
            turns.append({"role": utt["role"], "text": render(utt, at2title), "movies": movies, "entities": entities})
    return turns


def relations(context, retention, w, gamma):
    """Scope-C windows (turns end-w+1..end, as cpc.sliding_window_hyperedges), oldest first, with the
    relative learned weight next to the relative fixed-decay prior gamma^age."""
    if not retention or not retention.get("windows"):
        return []
    wins = sorted(retention["windows"], key=lambda x: -x[0])
    t_last = len(context) - 1
    tot = sum(a for _, a in wins) or 1.0
    ptot = sum(gamma ** age for age, _ in wins)
    out = []
    for age, a in wins:
        end = t_last - age
        span = context[max(0, end - w + 1): end + 1]
        out.append({"turns": [max(0, end - w + 1), end], "age": age,
                    "movies": list(dict.fromkeys(m for t in span for m in t["movies"])),
                    "entities": list(dict.fromkeys(e for t in span for e in t["entities"])),
                    "alpha": a, "rel_weight": a / tot, "prior_rel_weight": gamma ** age / ptot})
    return out


def load_dump(ds, cell, seed):
    lines = (DUMP_DIR / f"{ds}_{cell}_seed{seed}.jsonl").read_text().splitlines()
    return {(r["conv_id"], r["n_turn"], r["target"]): r for r in map(json.loads, lines[1:])}


def md_case(r, args):
    md = [f"## conv {r['conv_id']} · turn {r['n_turn']} · target **{r['target']}**"
          f"{' (tail)' if r['tail'] else ''} · {r['n_hist']} history session(s)", ""]
    if args.mode == "rec":
        md += [f"- *history conv {h['conv_id']}*: {', '.join(h['movies']) or '-'}" for h in r["history_sessions"]]
        md.append("")
    md += [f"> **[{i}] {t['role']}**: {t['text']}  " for i, t in enumerate(r["context"])]
    md += [f"> **{r['gold_response']['role']} (gold)**: {r['gold_response']['text']}", ""]
    if r["relations"]:
        md += ["| window (turns) | age | movies / entities | learned weight | prior 0.75^age |", "|---|---|---|---|---|"]
        md += [f"| {w['turns'][0]}-{w['turns'][1]} | {w['age']} | {'; '.join(w['movies'] + w['entities']) or '(words only)'} "
               f"| {w['rel_weight']:.3f} | {w['prior_rel_weight']:.3f} |" for w in r["relations"]]
        md.append("")
    md += ["| model | rank | top-5 |", "|---|---|---|"]
    md += [f"| {m} | {v['rank']} | {'; '.join(v[f'top{args.k}'][:5])} |" for m, v in r["models"].items()]
    return md + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(BASELINES), default="rec")
    ap.add_argument("--dataset", default="hredial")
    ap.add_argument("--main", default="rescope_full")
    ap.add_argument("--baselines", nargs="+", help="default per --mode")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--k", type=int, default=10, help="top-k titles kept per model (dump holds top-20)")
    ap.add_argument("--hit", type=int, default=5, help="candidate: main ranks the target <= hit")
    ap.add_argument("--miss", type=int, default=10, help="candidate: every baseline ranks it > miss")
    ap.add_argument("--max_turns", type=int, default=8, help="candidate: at most this many turns of context")
    ap.add_argument("--window", type=int, default=3, help="context_window_w of the main model")
    ap.add_argument("--gamma", type=float, default=0.75, help="retention prior / r2_ip_fixed decay")
    ap.add_argument("--n_md", type=int, default=30)
    args = ap.parse_args()
    baselines = args.baselines or BASELINES[args.mode]

    d = DATA[args.dataset]
    splits = [json.load(open(d / f"{s}_data.json")) for s in ("train", "valid", "test")]
    at2title = mention_titles(splits)
    id2title = {v: title(k) for k, v in json.load(open(d / "entity2id.json")).items()}
    test = {int(g[-1]["conv_id"]): g for g in splits[2]}

    models = [args.main] + baselines
    dumps = {m: load_dump(args.dataset, m, args.seed) for m in models}
    keys = [k for k in dumps[args.main] if all(k in dumps[m] for m in baselines)]
    print(f"{len(keys)} samples shared by {models}")

    rows = []
    for conv_id, n_turn, target in keys:
        group = test[conv_id]
        turns = merged_turns(group[-1]["dialog"], at2title)
        assert turns[n_turn]["role"] == "Recommender", (conv_id, n_turn)
        r0 = dumps[args.main][(conv_id, n_turn, target)]
        rows.append({
            "conv_id": conv_id, "user_id": group[-1].get("user_id"), "n_turn": n_turn,
            "n_hist": r0["n_hist"], "tail": r0["tail"],
            "target": id2title.get(target, target), "target_id": target,
            "history_sessions": [{"conv_id": c["conv_id"],
                                  "movies": sorted({title(m) for u in c["dialog"] for m in u["movies"]}),
                                  "turns": merged_turns(c["dialog"], at2title)} for c in group[:-1]],
            "context": turns[:n_turn],
            "gold_response": turns[n_turn],
            "relations": relations(turns[:n_turn], r0.get("retention"), args.window, args.gamma),
            "models": {m: {"rank": dumps[m][(conv_id, n_turn, target)]["rank"],
                           f"top{args.k}": [id2title.get(i, i) for i in dumps[m][(conv_id, n_turn, target)]["top20"][:args.k]]}
                       for m in models},
        })

    stem = f"case_study_{args.mode}_{args.dataset}_{args.main}_seed{args.seed}"
    out = ROOT / "results" / "cpg" / f"{stem}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def gap(r):  # main rank minus best baseline rank (more negative = clearer win)
        return r["models"][args.main]["rank"] - min(r["models"][m]["rank"] for m in baselines)

    cand = [r for r in rows if r["models"][args.main]["rank"] <= args.hit and r["n_turn"] <= args.max_turns
            and all(r["models"][m]["rank"] > args.miss for m in baselines)]
    if args.mode == "relation":
        # learned retention revives an older relation: top-weighted window is not the newest one
        def lift(r):
            top = max(r["relations"], key=lambda w: w["rel_weight"])
            return top["rel_weight"] / top["prior_rel_weight"] if top["age"] > r["relations"][-1]["age"] else 0.0
        cand = [r for r in cand if len(r["relations"]) >= 2 and lift(r) > 1.0]
        cand.sort(key=lambda r: (gap(r), -lift(r)))
    else:  # prefer samples that use history (P scope)
        cand.sort(key=lambda r: (-(r["n_hist"] > 0), gap(r)))
    md = [f"# Case-study candidates ({args.mode}): {args.dataset}, seed {args.seed}",
          f"{len(cand)} / {len(rows)} samples: `{args.main}` rank <= {args.hit}, every baseline "
          f"({', '.join(baselines)}) rank > {args.miss}, context <= {args.max_turns} turns"
          + (", top retention weight on an older window than the newest." if args.mode == "relation" else "."), ""]
    for r in cand[:args.n_md]:
        md += md_case(r, args)
    out.with_suffix(".md").write_text("\n".join(md))
    print(f"{len(rows)} rows -> {out}\n{len(cand)} candidates -> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
