# Pretrain design for HTGReDial (MHIM-style: pretrain KG + hypergraph, then tune rec / conv)

Status: design only (nothing in section 2 is implemented). Section 1 is runnable today with MHIM's checkpoint.

## Why

HTGReDial rec supervision is tiny vs. the label space (measured on `data/dataset/htgredial/pkuseg`):

| quantity | value |
|---|---|
| train rec samples | 24 003 (3 per session, 8 001 sessions) |
| distinct train targets | 10 819 of 33 531 candidate items (about 2.2 samples/item) |
| test targets never seen as a train target | 22.2 % |
| popularity baseline R@1/10/50 (test) | 0.0003 / 0.0037 / 0.0190 |
| entity-mention targets in train dialogs (all turns) | 149 241 over 29 756 entities (about 6x the rec labels) |

MHIM (reference: `~/projects/MHIM`, same data) reaches R@10 .031 / R@50 .083 with a contrastively pretrained R-GCN, Adam 1e-3, bs 64, early stop on valid rec_loss (min, impatience 2). HyCoRec ablation cells without pretrain reach R@10 .007-.022 / R@50 .025-.058.

Note: valid rec_loss rising while recall rises is normal here (MHIM shows the same); do not use it as an overfitting signal, and do not add weight decay / lower lr on that basis (T0 got worse with it).

## 1. KG pretrain (R-GCN)

What MHIM does: `Contrast/run.py -d tgredial` trains a GCC/MoCo contrastive objective over random-walk subgraphs of the entity KG and saves **only** the `RGCNConv` weights (`weight 8x128x128`, `comp 56x8`, `root`, `bias`); node embeddings are not saved and are learned from scratch in rec.

Ready-made checkpoints: `~/projects/MHIM/MHIM/pretrain/{HReDial,HTGReDial}/10-epoch.pth` (key `encoder`; shapes match HyCoRec's `RGCNConv(128,128,56,num_bases=8)`).

Gaps in HyCoRec (before this change):
- `-p -e N` existed but loaded only `item_encoder`; `entity_encoder` and `word_encoder` stayed random. MHIM has one `kg_encoder` shared by all fields. Now `pretrain_load: all` (default) loads all three, `item` keeps the old behaviour.
- no ablation config ever passed `-p`, and no `pretrain/` dir exists in HyCoRec.

Run (server):

```bash
mkdir -p pretrain/HTGReDial && cp ~/projects/MHIM/MHIM/pretrain/HTGReDial/10-epoch.pth pretrain/HTGReDial/
uv run run_crslab.py -c config/crs/hycorec/ablation/tgredial/T3b.yaml -g 0 -s 123 -p -e 10
```

(`scripts/run_ablation.py` needs a `--pretrain` pass-through to do this for the whole ladder.)

Open questions to settle by experiment, in this order:
1. T3b / T9 with `-p -e 10`, `pretrain_load: all` vs `item` (does loading all three matter?).
2. Same recipe as MHIM for rec (Adam 1e-3, bs 64, early stop on valid rec_loss) vs the current AdamW 2e-4/wd 1e-2 config.
3. Only if the MHIM checkpoint helps: port `Contrast/` into this repo so the pretrain can be regenerated (needs `data.bin` / `embedding_meta_data.pkl` from MHIM's graph build) and try freezing the encoder for the first rec epochs.

## 2. Hypergraph pretrain (C / P / G stacks)

Goal: give the HConv stacks, fusion and the entity / word tables a useful starting point from the 6x larger entity-level signal, before the 24k item labels are used.

Trainable parts to pretrain: `hconv`, `hconv_p`, `hconv_g` (3 fields each), `fusion`, `item_attn`, `entity_embedding`, `word_embedding` (+ the RGCNs from section 1, optionally frozen).

Objective (self-supervised, train split only): for every turn t >= 2 of a train target session, build the user vector exactly as `encode_user` does from turns < t (scope C windows, scope P history, scope G) and score it against the entity table; loss = multi-positive InfoNCE / sampled softmax with positives = entities mentioned in turn t, negatives = in-batch + degree-weighted samples.
- Supervision: 149k entity targets (vs 24k item targets). Item targets of the train rec turns can be added as positives.
- Same code path as fine-tuning (`recommend()` minus the `rec_bias`), so no train/test mismatch in how the user vector is formed.
- Candidate table: `entity_embedding` after the entity RGCN, same as `_candidate_table`.

Checkpoint: save the state dict of the parts listed above to `pretrain/HTGReDial/hyper-<epoch>.pth`; loader mirrors section 1 (`-p`), fine-tune with encoder lr x0.1 or frozen for the first epoch.

Constraints:
- Pretrain on the train split only (valid / test are different users, never use their dialogs).
- The per-sample hyperedge construction in `_cpc_field_preferences` is the cost driver (about 80 min per rec epoch at 24k samples); pretraining over about 95k turns would be about 4x that per epoch unless the turn-level user vector reuses the session's incremental hyperedges. Budget this before implementing; a first version can subsample turns.
- `exclude=target` logic must also hide the predicted turn's entities from the hyperedges.

Decision gate: implement section 2 only if section 1 closes most of the gap to MHIM (target R@10 about .03, R@50 about .08); if section 1 alone gets there, the extra pretrain is a paper-level ablation, not a necessity.

## 3. Seen-item masking (diagnostic, already implemented)

2/3 of test samples have a previously recommended item in the same session and the target never repeats one (0 / 2994). `rec_evaluate` now also reports `mrecall@k / mndcg@k / mmrr@k` with in-session items masked; standard metrics are unchanged. To score an existing checkpoint without training:

```bash
uv run run_crslab.py -c config/crs/hycorec/ablation/tgredial/T3b.yaml -g 0 --rec_eval_from save/<ckpt>.pth
```
