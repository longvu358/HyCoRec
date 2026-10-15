# CPG-Hypergraph paper experiments (`docs/CPG-Hypergraph_Experiment.pdf`)

Configs are GENERATED (`python scripts/run_cpg.py gen`) from `scripts/cpg_manifest.py` = dataset base yaml
(`../hycorec/redial.yaml`, `../hycorec/tgredial.yaml`, i.e. the full model) + per-cell overrides. Edit the manifest, not these files.
`<dataset>/extra/` = analyses outside the RQ set. Isolated from the old ladder: results -> `results/cpg/`,
logs -> `log/<stamp>/cpg_*.log`, checkpoints -> `save/cpg_*`. Only RQ1 cells run the conv stage.

| RQ | what | cells / how | runs |
|---|---|---|---|
| RQ1, RQ2 | rec table + Dist-n | `full`, `hycorec_repro` (HyCoRec reproduced = P-only) | 2 cells x 2 ds x 3 seeds |
| RQ3 | C@5/10/15/20, LT@10 | no new run: `analyze` the ckpts of RQ1, `report --exp RQ3` | - |
| RQ4 | "w/o X" | `wo_C wo_P wo_G wo_G_rev wo_G_asp wo_cand G_to_mhim` (+ `full`) | 7 x 2 x 3 |
| RQ5 | sensitivity, R@10 | `sens_d*` `sens_L*` `sens_w*` `sens_kasp*` (default point = `full`) | 13 x 2 x 1 seed |
| RQ6 | case table | no new run: `report --exp RQ6` (tail target, HyCoRec rank > 5, CPG rank <= 5) | - |
| extra-S3 | Eq. 17 candidate diagnostic (run BEFORE fixing `wo_cand`) | `s3_D_pre` (htgredial only) `s3_D_free` `s3_D_perm` `s3_D_det` vs `full`, `wo_cand` | 4 x 2 x 3 |
| extra-S4 | cost vs MHIM | `lad_A3` (P+G_dlg) `lad_A4` (P+MHIM ext) + `full`; `analyze`, `report --exp extra-S4` | 2 x 2 x 3 |
| extra-S6/S7 | attention by turn / history length | `analyze` full + hycorec_repro, `report --exp extra-S6 extra-S7` | - |
| extra-S8 | fusion design | `s8_F1..F4 s8_A10 s8_A12 s8_A13` | 7 x 2 x 3 |
| extra-S10 | stats + scope overlap | `python scripts/extra/cpg_s10.py -d <ds>` | no training |

    scripts/build_cpg_data.sh            # RQ5 k_asp + extra-S2 no-dialogue collectives (data/ is gitignored)
    python scripts/run_cpg.py plan --exp RQ1 RQ4
    python scripts/run_cpg.py run  --exp RQ1 --gpu 0
    python scripts/run_cpg.py run  --exp RQ4 RQ5 --gpu 0
    python scripts/run_cpg.py analyze --cells full hycorec_repro --gpu 0
    python scripts/run_cpg.py report --exp RQ3 RQ6
    python scripts/run_cpg.py summarize --exp RQ1 RQ4 RQ5

Baselines other than `hycorec_repro` are copied from the original papers (no std), as in the doc.

## Running on a GPU box
    python scripts/run_cpg.py run --exp RQ1 --gpu 0,1,2,3 --parallel          # 4 independent single-GPU runs at a time
    python scripts/run_cpg.py run --exp RQ1 --gpu 0,1 --ddp                   # one run split over 2 GPUs (batch_size is global)
    python scripts/run_cpg.py run --exp RQ4 --gpu 0 --rec-bs 64 --conv-bs 16  # smaller batch if OOM (changes the result slightly: keep it fixed within an RQ)
Resumable: finished (dataset, cell, seed) are skipped; `--force` reruns. Nohup it: `nohup python scripts/run_cpg.py run ... &`.
