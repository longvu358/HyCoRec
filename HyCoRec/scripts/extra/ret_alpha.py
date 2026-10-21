"""Summarise the learned retention alpha from a `run_cpg.py analyze` dump.

Usage: python scripts/extra/ret_alpha.py results/cpg/dump/hredial_r2_rescope_seed3407.jsonl

Reads the `retention` field of every test row ({"q_age", "windows": [[age, alpha], ...]}) and prints
  1. global alpha distribution (is it still ~sigmoid(3)=0.9526? collapsed to 0/1?),
  2. alpha by age (does it fall with age, or is it non-monotone?),
  3. within-sample spread (does alpha tell windows of one sample apart, or is it one global scale?),
  4. how often the latest valid window is not the last turn (q_age > 0).
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main(path):
    lines = open(path).read().splitlines()
    head = json.loads(lines[0])
    rows = [json.loads(l) for l in lines[1:]]
    rets = [r["retention"] for r in rows if r.get("retention")]
    print(f"{len(rows)} rows, {len(rets)} with retention; header alpha_lt005={head.get('alpha_lt005')}")
    if not rets:
        return
    a_all = np.array([w[1] for r in rets for w in r["windows"]])
    print(f"\n[1] alpha over {a_all.size} windows: mean {a_all.mean():.4f} std {a_all.std():.4f}")
    print("    quantiles 1/5/25/50/75/95/99%:", np.round(np.percentile(a_all, [1, 5, 25, 50, 75, 95, 99]), 4))
    print(f"    <0.05: {(a_all < .05).mean():.2%}  <0.5: {(a_all < .5).mean():.2%}  >0.9: {(a_all > .9).mean():.2%}"
          f"  |alpha-0.9526|<0.01: {(abs(a_all - 0.9526) < .01).mean():.2%}")

    by_age = defaultdict(list)
    for r in rets:
        for age, a in r["windows"]:
            by_age[age].append(a)
    print("\n[2] by age:   age      n   mean    std")
    for age in sorted(by_age)[:16]:
        v = np.array(by_age[age])
        print(f"           {age:5d} {v.size:6d} {v.mean():.4f} {v.std():.4f}")
    ages = np.array([w[0] for r in rets for w in r["windows"]], dtype=float)
    if a_all.std() > 0 and ages.std() > 0:
        print(f"    corr(age, alpha) = {np.corrcoef(ages, a_all)[0, 1]:+.3f}")

    within = [np.std([w[1] for w in r["windows"]]) for r in rets if len(r["windows"]) > 1]
    between = [np.mean([w[1] for w in r["windows"]]) for r in rets]
    print(f"\n[3] mean within-sample std {np.mean(within):.4f} vs std of sample means {np.std(between):.4f}")
    print("    (within << between => alpha acts as a per-sample scale, not as a window selector)")

    q = np.array([r["q_age"] for r in rets])
    print(f"\n[4] q_age>0 (latest valid window is stale): {(q > 0).mean():.2%}, mean q_age {q.mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
