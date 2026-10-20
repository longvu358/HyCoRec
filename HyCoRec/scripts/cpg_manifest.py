"""Experiment manifest for docs/CPG-Hypergraph_Experiment.pdf (RQ1-RQ6 + extra analyses).

Single source of truth: every cell is ``base config of the dataset`` + ``overrides``.
``scripts/run_cpg.py gen`` writes them to config/crs/cpg/<dataset>/<cell>.yaml
(``extra-*`` cells to config/crs/cpg/<dataset>/extra/<cell>.yaml).

Cell fields
  exp        RQ1 | RQ4 | RQ5 | extra-S2 | extra-S3 | extra-S4 | extra-S8   (RQ2/RQ3/RQ6 reuse the RQ1 runs)
  over       dict of yaml overrides on top of the dataset base (= full model), or callable(collective_prefix)
  rec_only   True -> skip the conv stage; only RQ1 cells need Dist-n
  ckpt       True -> run with -ss and keep the checkpoint (input of `run_cpg.py analyze`)
  seeds      per-cell default seeds (else SEEDS); RQ5 is a one-seed curve
  status     ready | needs_data (collective index; scripts/build_cpg_data.sh; flips to ready once paths exist)
  cli        extra run_crslab.py flags (e.g. ["-p", "-e", "10"] = load the contrastive pretrain)
"""
from __future__ import annotations

DATASETS = {
    # dataset key -> (base yaml, collective dir prefix)
    "hredial": ("config/crs/hycorec/redial.yaml", "data/collective/hredial"),
    "htgredial": ("config/crs/hycorec/tgredial.yaml", "data/collective/htgredial"),
}
SEEDS = [3407, 42, 123]

# overrides that turn the full model into the P-only floor == "HyCoRec (reproduced)"
P_ONLY = {"scopes": ["P"], "collective_path": None, "use_review_hypergraph": False, "ei_mode": "base"}


def _cell(exp, over=None, *, rec_only=True, ckpt=False, status="ready", prio=1, need="", datasets=None, note="",
          cli=None, seeds=None):
    return dict(exp=exp, over=over or {}, rec_only=rec_only, ckpt=ckpt, status=status, prio=prio, need=need,
                datasets=datasets, note=note, cli=cli or [], seeds=seeds)


CELLS: dict[str, dict] = {
    # ---------------- RQ1 / RQ2 (+ RQ3, RQ6 from the dump of these checkpoints) --------
    "full": _cell("RQ1", {}, rec_only=False, ckpt=True, note="CPG-Hypergraph (C+P+G, review on)"),
    "hycorec_repro": _cell("RQ1", P_ONLY, rec_only=False, ckpt=True,
                           note="HyCoRec reproduced on this pipeline = P-only floor (A0/T0); must match the paper's HyCoRec numbers"),

    # ---------------- RQ4 : "w/o X" ablation (rows = full + these 7), rec only ----------
    "wo_C": _cell("RQ4", {"scopes": ["P", "G"]}),
    "wo_P": _cell("RQ4", {"scopes": ["C", "G"], "use_review_hypergraph": False},
                  note="no history -> also drops the review hyperedges of P"),
    "wo_G": _cell("RQ4", {"scopes": ["C", "P"], "collective_path": None, "ei_mode": "base"}),
    "wo_G_rev": _cell("RQ4", lambda p: {"collective_path": p + "_aspect"}, note="G = dialogue + aspect"),
    "wo_G_asp": _cell("RQ4", lambda p: {"collective_path": p + "_review"}, note="G = dialogue + review"),
    "wo_cand": _cell("RQ4", {"ei_mode": "base"}, note="Eq. 17 off: E = X0_I. Run extra-S3 first, then decide whether to keep this row"),
    "G_to_mhim": _cell("RQ4", {"scopes": ["C", "P"], "collective_path": None, "ei_mode": "base",
                               "mhim_extension": "Adaptive"},
                       note="G replaced by MHIM's retrieval extension (similar-conversation item lists join P's item sessions)"),
}

# ---------------- RQ5 : sensitivity, R@10 only, one seed per point --------------------
# The default point of each panel is the full model and is skipped (== full).
# panels: d (kg_emb_dim), L (hconv_layers), w (context_window_w), k_asp (build_collective --k_asp, default 50)
for _tag, (_key, _vals, _dflt) in {
    "d": ("kg_emb_dim", [64, 128, 256, 512], 128),
    "L": ("hconv_layers", [1, 2, 3, 4], 2),
    "w": ("context_window_w", [1, 2, 3, 4, 5], 3),
}.items():
    for _v in _vals:
        if _v != _dflt:
            CELLS[f"sens_{_tag}{_v}"] = _cell("RQ5", {_key: _v}, prio=2, seeds=[SEEDS[0]])
for _v in [20, 100, 200]:
    CELLS[f"sens_kasp{_v}"] = _cell(
        "RQ5", (lambda v: lambda p: {"collective_path": f"{p}_full_kasp{v}"})(_v), prio=2, seeds=[SEEDS[0]],
        status="needs_data", need=f"build_collective.py --use_review --k_asp {_v} --out_dir <prefix>_full_kasp{_v}")

# ======================= extra/ : analyses outside the RQ set ==============================
CELLS.update({
    # S3 : where does the candidate-side gain (Eq. 17) come from?  RUN FIRST, before fixing the RQ4 row
    # "w/o candidate enrichment". D_cur == full, D_none == wo_cand.
    "s3_D_pre": _cell("extra-S3", {"ei_mode": "base", "pretrain_load": "all"}, cli=["-p", "-e", "10"],
                      datasets=["htgredial"],
                      note="H1: no G on the candidate side, but contrastively pretrained R-GCNs (pretrain/HTGReDial/10-epoch.pth only)"),
    "s3_D_free": _cell("extra-S3", {"ei_mode": "free"}, note="H3: free item table instead of X^G"),
    "s3_D_perm": _cell("extra-S3", {"ei_mode": "xg_perm"}, note="H4: X^G rows permuted (fixed seed)"),
    "s3_D_det": _cell("extra-S3", {"ei_mode": "xg_detach"}, note="H4: X^G detached on the candidate path"),
    # (H2 D_a "item emb + X^G_item" is dropped: with the per-field G of hredial_full / htgredial_full, Eq. 17
    #  already adds A^G_I X^(0)_I, so D_a == D_cur.)

    # S4 : cost vs MHIM -- `analyze` these ckpts, `report --exp extra-S4`
    "lad_A3": _cell("extra-S4", lambda p: {"scopes": ["P", "G"], "collective_path": p, "use_review_hypergraph": False,
                                           "ei_mode": "base"}, prio=3, ckpt=True, note="P + G_dlg"),
    "lad_A4": _cell("extra-S4", {**P_ONLY, "mhim_extension": "Adaptive"}, prio=3, ckpt=True,
                    note="P + MHIM retrieval extension"),

    # S2 : ablation rows that are not in the 7 of RQ4
    "wo_P_rev": _cell("extra-S2", {"use_review_hypergraph": False}, prio=3),
    "wo_G_dlg": _cell("extra-S2", lambda p: {"collective_path": p + "_full_nodlg"}, prio=3, status="needs_data",
                      need="build_collective.py --use_review --no_dialogue"),

    # S8 : fusion / encoder design (F1 = mean of rows)
    "s8_F1": _cell("extra-S8", {"fusion_mode": "mean7"}, prio=3, note="no query attention: mean of the non-empty rows"),
    "s8_F2": _cell("extra-S8", {"fusion_mode": "global_alpha"}, prio=3, note="learned 7-vector softmax weights, same for every sample"),
    "s8_F3": _cell("extra-S8", {"fusion_keys": "node"}, prio=3, note="every node of every row is a key (no per-row pooling)"),
    "s8_F4": _cell("extra-S8", {"pooling": "Attn"}, prio=3, note="attention pooling for the final reduce (SelfAttentionBatch)"),
    "s8_A10": _cell("extra-S8", {"scope_embed": True}, prio=3, note="learned scope embedding e_k added to each row"),
    "s8_A12": _cell("extra-S8", {"scope_weights": "shared"}, prio=3, note="one HConv stack shared by C/P/G"),
    "s8_A13": _cell("extra-S8", {"g_readout": "balanced"}, prio=3, note="r^G = mean of per-node-type means"),

    # RQ2 (ReSCOPE): relation formation (window w) x relation persistence (retention). All cells dedupe
    # identical windows, so only `full` (no dedupe, no retention) is the legacy sliding-window floor.
    "r2_latest": _cell("extra-S12", {"window_dedupe": True, "retention_use_history": False}, prio=2,
                       note="Latest relation only: just the latest valid window (sliding-window baseline)"),
    "r2_turn": _cell("extra-S12", {"window_dedupe": True, "context_window_w": 1}, prio=2,
                     note="Turn-level full history: w=1, no cross-turn formation, no retention"),
    "r2_uniform": _cell("extra-S12", {"window_dedupe": True}, prio=2, ckpt=True,
                        note="Uniform relational history: all windows, alpha = 1"),
    "r2_readout": _cell("extra-S12", {"window_dedupe": True, "retention": "learned", "retention_apply": "readout"},
                        prio=2, note="Learned readout only"),
    "r2_noage": _cell("extra-S12", {"window_dedupe": True, "retention": "learned", "retention_use_age": False},
                      prio=2, note="ReSCOPE w/o age (content only)"),
    "r2_shuffle": _cell("extra-S12", {"window_dedupe": True, "retention": "learned", "retention_shuffle": True},
                        prio=2, ckpt=True, note="control: learned alpha permuted among the windows of a sample"),
    "r2_rescope": _cell("extra-S12", {"window_dedupe": True, "retention": "learned"}, prio=2, ckpt=True,
                        rec_only=True, note="ReSCOPE: learned alpha on propagation + readout"),
})
# Fixed temporal decay alpha = gamma^age; gamma tuned on valid with the same budget as the other variants.
# gamma = 1 must reproduce r2_uniform (sanity); small gamma approaches Latest-only.
for _g in (1.0, 0.95, 0.85, 0.7, 0.5):
    CELLS[f"r2_fixed_g{str(_g).replace('.', '')}"] = _cell(
        "extra-S12", {"window_dedupe": True, "retention": "fixed", "retention_gamma": _g}, prio=2,
        note=f"Fixed temporal decay gamma={_g}")

# Analyses on the saved checkpoints (no extra training):
#   RQ3  C@5/10/15/20 + LT@10  : `analyze --cells full hycorec_repro`, `report --exp RQ3`
#   RQ6  top-5 case table      : same dumps, `report --exp RQ6`
#   extra-S4 cost, S6 attention by turn, S7 by history length : `analyze` + `report --exp extra-S4 extra-S6 extra-S7`
#   extra-S10 overlap/stats    : scripts/extra/cpg_s10.py (no model)
