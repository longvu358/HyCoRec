#!/usr/bin/env bash
# Rebuild everything under data/ (gitignored) on a new machine, in dependency order. Idempotent: every step
# is skipped when its output already exists. Usage:
#   scripts/setup_env.sh [hredial htgredial]        # default: both datasets
#   MHIM_DIR=~/projects/MHIM scripts/setup_env.sh   # where to copy the contrastive pretrain checkpoints from
# Not scriptable (copy by hand, see the warnings this script prints):
#   data/reviews/raw/revcore/*       RevCore review data used for hredial (conv2items.json, entity2id.json, ... )
#   data/reviews/raw/douban1000w/*.parquet   Douban comments used for htgredial (HF dirtycomputer/douban_movie_review)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASETS=(${@:-hredial htgredial})
PY=${PY:-"uv run python"}
MHIM_DIR=${MHIM_DIR:-$HOME/projects/MHIM}

step() { echo; echo "=== $*"; }
need() { [ -e "$1" ] || { echo "!! missing $1 -- $2"; return 1; }; }

step "0. python env"
uv sync
$PY -c "import pyarrow" 2>/dev/null || uv pip install pyarrow   # build_review_index.py (htgredial/Douban parquet)

step "1. datasets (crslab auto-download; redial is also needed by edger/build_en_side)"
for d in redial hredial htgredial; do
  case " ${DATASETS[*]} redial " in *" $d "*) ;; *) continue ;; esac
  $PY download_dataset.py -d "$d"
done

step "2. side word lists (ConceptNet / HowNet)"
[ -e data/conceptnet/en_side.txt ] || $PY build_en_side.py
[ -e data/conceptnet/zh_side.txt ] || $PY build_zh_side.py

step "3. edger (item / entity / word adjacency)"
for d in "${DATASETS[@]}"; do
  [ -e "data/edger/$d/word_edger.pkl" ] || $PY run_edger.py -d "$d"
done

step "4. review index (needs the raw review data)"
for d in "${DATASETS[@]}"; do
  case $d in
    hredial)   need data/reviews/raw/revcore/conv2items.json "copy the RevCore files here" || continue ;;
    htgredial) need data/reviews/raw/douban1000w/0.parquet "download HF dirtycomputer/douban_movie_review parquet shards" || continue ;;
  esac
  [ -e "data/reviews/$d/review_index.json" ] || $PY build_review_index.py -d "$d"
done

step "5. collective hypergraph G (G0 / +review / +aspect / full)"
for d in "${DATASETS[@]}"; do
  [ -e "data/collective/$d/A_hat_item.npz" ]        || $PY build_collective.py -d "$d"
  [ -e "data/reviews/$d/review_index.json" ] || { echo "!! no review index for $d, skipping review/aspect/full collectives"; continue; }
  [ -e "data/collective/${d}_review/A_hat_item.npz" ] || $PY build_collective.py -d "$d" --with_review
  [ -e "data/collective/${d}_aspect/A_hat_item.npz" ] || $PY build_collective.py -d "$d" --with_aspect
  [ -e "data/collective/${d}_full/A_hat_item.npz" ]   || $PY build_collective.py -d "$d" --use_review
done

step "6. extra collectives for the CPG experiments (RQ5 k_asp, extra-S2 no-dialogue)"
for d in "${DATASETS[@]}"; do
  [ -e "data/collective/${d}_full/A_hat_item.npz" ] && PY="$PY" scripts/build_cpg_data.sh "$d"
done

step "7. contrastive pretrain checkpoints (only extra-S3 s3_D_pre needs them; HTGReDial)"
for pair in HReDial:hredial HTGReDial:htgredial; do
  src=${pair%%:*}; key=${pair##*:}
  case " ${DATASETS[*]} " in *" $key "*) ;; *) continue ;; esac
  if [ -e "pretrain/$src/10-epoch.pth" ]; then continue; fi
  if [ -e "$MHIM_DIR/MHIM/pretrain/$src/10-epoch.pth" ]; then
    mkdir -p "pretrain/$src"; cp "$MHIM_DIR/MHIM/pretrain/$src/10-epoch.pth" "pretrain/$src/"
  else
    echo "!! no pretrain for $src (copy from MHIM/MHIM/pretrain/$src or run MHIM Contrast/run.py)"
  fi
done

step "8. generate experiment configs and show the plan"
$PY scripts/run_cpg.py gen
$PY scripts/run_cpg.py plan --exp RQ1 RQ4 RQ5 | tail -3
echo; echo "done. Smoke-check one cell:  $PY scripts/run_cpg.py run --cells full --dataset htgredial --smoke --gpu 0"
