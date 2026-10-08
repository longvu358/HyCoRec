# HTGReDial (TG-ReDial) ablation ladder

Mirrors the ReDial A-ladder (spec 6.2) with `T*` cell names. `rtx6000/` = same configs, **same batch size as base** (unlike ReDial, no x2; see "Memory").

```bash
# data prerequisites (data/ is gitignored -> build on the training machine)
python build_review_index.py -d htgredial
python build_collective.py -d htgredial                                              # G0       -> htgredial
python build_collective.py -d htgredial --use_review                                 # G full   -> htgredial_full
python build_collective.py -d htgredial --with_review --out_dir data/collective/htgredial_review   # T6
python build_collective.py -d htgredial --with_aspect --out_dir data/collective/htgredial_aspect   # T7

uv run scripts/run_ablation.py run --dataset tgredial --gpu-profile rtx6000 --gpu 0,1,2,3 --ddp --seeds 123
uv run scripts/run_ablation.py summarize --dataset tgredial      # -> results/ablation_tgredial/summary*.md
```

| cell | scopes | toggled component | ReDial analogue |
|---|---|---|---|
| T0 | P | floor | A0 |
| T1 | C+P, w=3 | sliding-window C | A1 |
| T2 | C+P, w=1 | C degenerate (per-turn) | A2 |
| T3 / T3b | P+G0, ei base / xg | dialogue-only collective | A3 / A3b |
| T5 | P + review hypergraph | R in P only | A5 |
| T6 | P+G0+h^rev | review-only G | A6 |
| T7 | P+G0+h^asp | aspect-only G | A7 |
| T8 | P+R+G full | no C | A8 |
| **T9** | C+P+R+G | **full model** | A9 |
| T9_norev | C+P+G0 | review off | A9b1 |
| T9_w2 / T9_w5 | T9 with w=2 / 5 | window size sweep | (new) |

ReDial-only v2.2 cells (`A9_v22*`) have no TG counterpart (no `*_unified_full` collective built).

## Hyper-parameters (from `docs/statistic_findings.md`, `docs/tgredial_graph_review_findings.md`)

Identical across all T* cells, so only the toggled component differs. (ReDial A-cells are now synced the same way: AdamW, rec 10 ep / conv 5 ep, see `../A*.yaml`.)

### Why TG values differ from ReDial (same method, different data)

| param | ReDial (`A*`) | HTGReDial (`T*`) | reason |
|---|---|---|---|
| `tokenize` | nltk | pkuseg | English vs Chinese corpus |
| `rec` batch size | 64 (rtx6000: 128) | 128 (rtx6000: 128) | TG needs ~5 GiB before the first batch (word RGCN over the KG + A_hat), so rtx6000 is not x2 here; see Memory |
| `khop_cap` | none | 300 | `entity_adj` hubs: TG max degree 20 647 (27 nodes > 1000) vs ReDial max 812 (15 nodes); TG p99 32 vs 4 |
| `context_window_w` | 3 (sweep 1 in A2) | 3 (sweep 2/3/5: T9_w2, T9_w5) | ReDial item recurrence within 3 turns 88.6%, entity 75.2%; TG entity 83.9%, word 80.2%; TG item window is inert (3 re-mentions in 10k sessions) |
| window effect | item + entity + word | entity + word only | TG targets are 100% cold, so C acts through entity/word |
| review branch | RevCore reviews (decoded entity space) | Douban short comments, only ~26% of V_I (59% of mentions) | no native TG review corpus; the rest relies on scope G |
| `fusion_mode`, `alpha`, lr, optimizer | rows7, AdamW 5e-4 / 1e-3 | same | kept identical so only the data differs |


| param | value | evidence |
|---|---|---|
| `tokenize` | pkuseg | Chinese corpus |
| `context_window_w` | 3 (sweep 2/5) | entity recurrence within 3 turns = 83.9% (5: 94.7%), word 80.2% (5: 92.5%); zero collapse for w<=5 (sessions 10-16 turns); decay cliff at g~9-10 so w>=10 is pointless |
| item window | no separate setting | only 3 item re-mentions in 10k sessions and 100% of targets are cold: the C scope acts through entity/word, so T1/T2 measure that, not item recall |
| `khop_cap` | 300 | `entity_adj` mean degree 8.4 (ReDial 2.4), max 20 647 (generic tags such as `影视作品`) |
| `rec` | 10 epochs, bs 128, AdamW lr 2e-4 wd 1e-2, early stop on recall@50 (impatience 2; lr halved after 1 non-improving epoch) | as T9 / ReDial |
| `conv` | 5 epochs, bs 32, AdamW lr 5e-4 wd 1e-2, early stop on valid gen_loss (impatience 2), best epoch restored before test | as T9 / ReDial |
| `fusion_mode` | rows7 | v2.2 Eq. 11-13 (final choice); alpha is unused, so no frozen-alpha (A10/T10) cells |
| review | Douban index; ~26% of items, ~59% of item mentions | `docs/tgredial_review_corpus.md`; the rest relies on G |

Not done: hub filtering (dropping nodes with degree > 1000 in the k-hop expansion) is suggested by the graph analysis but has no code path; `khop_cap` is the only mitigation.

## Memory (why rtx6000 is not x2 here)

Evidence: `log/20261004-011141` (seed 3535) and `log/20261004-013754` (seed 123) run concurrently on the same GPUs 0-3 (DDP, 4 ranks each).
- ReDial cells are light, so both fit. HTGReDial T9 at rec bs256 (64/GPU) took 9.9 GiB/GPU (seed 3535, ran fine alone); ~5 GiB of that is already allocated before the first batch (word RGCN over the KG + A_hat), the rest is batch-dependent.
- seed 123 reached T9 while seed 3535 was in T9 -> `CUDA out of memory` in `_word_table` on all 4 GPUs (5.8 GiB ours + 9.9 GiB theirs > 15.77 GiB), then T9_norev hit the same. Not a config bug.
- Two concurrent jobs need <= ~7.5 GiB each, so TG uses bs128/32 (32 and 8 per GPU). If it still OOMs, halve rec to 64. Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (OOM log shows ~300 MiB reserved-but-unallocated).
- Alternative: run the two seeds sequentially on TG with the x2 sizes; wall time is similar.

## What the seed-3535 TG runs say (T9, T9_norev)

- Valid rec_loss rises from epoch 0 (10.1 -> 16.5) while train loss falls (9.9 -> 4.4): the model memorises 8k sessions. Valid recall@50 still improves until epoch 6 (0.0606), then plateaus, so early stop on recall@50 is correct and `epoch: 10, impatience: 3` is kept (stops after epoch 9). No lr/dropout change is made: nothing in the logs isolates them.
- Test recall@50: T9 0.0615 vs T9_norev 0.0591. Valid n~3000 gives SE ~0.004 on recall@50, so this gap is noise. Differences below ~0.005 between T cells need both seeds (123, 3535) averaged before being read as an effect.
- Cost per cell (solo): rec ~7 min/epoch x 10 + conv ~18 min/epoch x 5 = ~3.3 h; the full 13-cell ladder is ~43 h per seed. Cheapest cells to drop if time is short: T3b, T9_w2, T9_w5.
