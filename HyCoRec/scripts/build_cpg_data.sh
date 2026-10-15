#!/usr/bin/env bash
# Extra collective / review indices needed by the CPG experiments (extra-S2 w/o dialogue hyperedges, RQ5 k_asp).
# Prereq (already done for the base configs): build_review_index.py -d <ds>; build_collective.py -d <ds> [--use_review ...]
# data/ is gitignored -> run this on every training machine.   Usage: scripts/build_cpg_data.sh [hredial htgredial]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
for ds in "${@:-hredial htgredial}"; do
  for d in $ds; do
    # extra-S2: G without dialogue hyperedges (review + aspect only)  -> data/collective/<ds>_full_nodlg
    $PY build_collective.py -d "$d" --use_review --no_dialogue
    # RQ5: k_asp sweep (default 50)                              -> data/collective/<ds>_full_kasp<k>
    for k in 20 100 200; do
      $PY build_collective.py -d "$d" --use_review --k_asp "$k" --out_dir "data/collective/${d}_full_kasp${k}"
    done
  done
done
