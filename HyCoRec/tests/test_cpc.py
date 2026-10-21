"""Unit tests for the CPC-Hypergraph v2 building blocks.

Covers the anti-leakage guarantees of spec 2.4 and the numeric identities of
``crslab.model.crs.hycorec.cpc``. Does not need a GPU or the full dataset.

Run: ``pytest HyCoRec/tests/test_cpc.py``
"""

import json
import os

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from crslab.data.dataset.hycorec_common import (
    build_history_sessions,
    merge_conv_turns,
    process_grouped_raw_data,
)
from crslab.model.crs.hycorec.cpc import (
    ScopeFusion,
    a_hat_from_incidence,
    build_incidence,
    khop_closure,
    sliding_window_hyperedges,
)

DP = os.path.join(
    os.path.dirname(__file__), "..", "data", "dataset", "hredial", "nltk"
)
_HAVE_DATA = os.path.isfile(os.path.join(DP, "valid_data.json"))


def _load(name):
    with open(os.path.join(DP, name), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# cpc.py primitives
# --------------------------------------------------------------------------
def test_khop_closure_levels():
    adj = {1: [2, 3], 2: [4], 3: [], 4: [5]}
    assert khop_closure([1], adj, 0) == {1}
    assert khop_closure([1], adj, 1) == {1, 2, 3}
    assert khop_closure([1], adj, 2) == {1, 2, 3, 4}
    assert khop_closure([1], adj, 99) == {1, 2, 3, 4, 5}


def test_build_incidence_dedup_and_empty():
    uniq, coo, n_edges = build_incidence([[1, 1, 2], [], [2, 3]])
    assert n_edges == 2  # empty hyperedge dropped
    assert set(uniq) == {1, 2, 3}
    assert len(coo[0]) == len(coo[1])


def test_build_incidence_excludes_target():
    """spec 2.4.5: the sample's target item must not appear in any C/P hyperedge."""
    y = 7
    uniq, coo, n_edges = build_incidence([[1, 7, 2], [7], [7, 3]], exclude=y)
    assert y not in uniq
    assert y not in coo[0]
    assert n_edges == 2  # {7} collapses to empty and is dropped


def test_sliding_window_w1_degenerates_to_per_turn():
    """spec v2.1 2.2: w=1 == v2's per-turn co-mention hyperedges."""
    turns = [[1], [2, 3], [], [4]]
    edges = sliding_window_hyperedges(turns, w=1)
    # each turn is its own hyperedge; size<2 turns ([1], [], [4]) are dropped
    assert edges == [[2, 3]]


def test_sliding_window_positions_and_overlap():
    turns = [[1], [2], [3], [4], [5]]
    edges = sliding_window_hyperedges(turns, w=3)
    # t'=0: {1} size 1 -> dropped
    # t'=1: {1,2} ; t'=2: {1,2,3} ; t'=3: {2,3,4} ; t'=4: {3,4,5}
    assert edges == [[1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    # consecutive windows overlap by w-1 turns -> node 3 appears in 3 hyperedges
    flat = [v for e in edges for v in e]
    assert flat.count(3) == 3


def test_sliding_window_drops_undersized_windows():
    # a lone non-empty turn surrounded by empty turns never reaches size 2
    edges = sliding_window_hyperedges([[], [7], []], w=1)
    assert edges == []


def test_a_hat_matches_manual():
    # one hyperedge over nodes {0,1,2}: D^-1 N E^-1 N^T -> all entries 1/3
    inc = sp.csc_matrix(np.ones((3, 1)))
    a = a_hat_from_incidence(inc).toarray()
    assert np.allclose(a, np.full((3, 3), 1 / 3))


def test_a_hat_isolated_node_zero_row():
    # node 3 in no hyperedge -> row/col all zero, no inf/nan
    inc = sp.csc_matrix(np.array([[1.0], [1.0], [0.0]]))
    a = a_hat_from_incidence(inc).toarray()
    assert np.isfinite(a).all()
    assert np.allclose(a[2], 0)


def test_scope_fusion_init_close_to_hycorec():
    fus = ScopeFusion()
    w = torch.softmax(fus.logits["item"], 0)
    assert w.argmax().item() == ScopeFusion.P  # P-dominated at init
    assert abs(w[ScopeFusion.P].item() - 0.576) < 0.02


def test_scope_fusion_cold_start_masks_p_gradient():
    """spec 3.4: a history-less sample must not update alpha[P]."""
    fus = ScopeFusion()
    pc = torch.ones(4, requires_grad=True)
    out = fus.fuse("item", pc, pc, torch.zeros(4), has_history=False).sum()
    out.backward()
    g = fus.logits["item"].grad
    assert g is None or g[ScopeFusion.P].item() == 0.0
    # C / G still learn from the cold sample
    assert g is None or g[ScopeFusion.C].item() != 0.0


def test_scope_fusion_with_history_updates_all():
    fus = ScopeFusion()
    pc, pp, pg = (torch.ones(4) * k for k in (1.0, 2.0, 3.0))
    pp = pp.clone().requires_grad_(True)
    fus.fuse("item", pc, pp, pg, has_history=True).sum().backward()
    assert fus.logits["item"].grad is not None
    assert fus.logits["item"].grad.abs().sum() > 0


# --------------------------------------------------------------------------
# dataset processing (needs the hredial data on disk)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_DATA, reason="hredial nltk data not present")
def test_turn_local_lists_are_distinct():
    tok2ind = _load("token2id.json")
    entity2id = _load("entity2id.json")
    raw = _load("valid_data.json")
    samples = process_grouped_raw_data(raw[:200], tok2ind, entity2id, 3, 40)
    # find a sample with >=3 non-empty turns and check they are not all identical
    multi = [
        s for s in samples if sum(1 for t in s["item"] if t) >= 2
    ]
    assert multi, "expected some samples with multiple item-bearing turns"
    s = multi[0]
    assert len({tuple(t) for t in s["item"]}) > 1


@pytest.mark.skipif(not _HAVE_DATA, reason="hredial nltk data not present")
def test_no_target_leak_into_context_or_history_hyperedges():
    """After the per-sample exclude=target guard, y must be absent from every
    C hyperedge and every P hyperedge (spec 2.4.5)."""
    tok2ind = _load("token2id.json")
    entity2id = _load("entity2id.json")
    raw = _load("valid_data.json")
    samples = process_grouped_raw_data(raw[:300], tok2ind, entity2id, 3, 40)
    checked = 0
    for s in samples:
        if s["role"] != "Recommender":
            continue
        for y in s["items"]:
            # scope C hyperedges = sliding-window hyperedges (spec v2.1 Eq. 2),
            # guarded by exclude=y
            c_edges = sliding_window_hyperedges(s["item"], w=3)
            _, c_coo, _ = build_incidence(c_edges, exclude=y)
            assert y not in c_coo[0]
            # scope P item hyperedges = per-session item sets, guarded by exclude=y
            _, p_coo, _ = build_incidence(s["history_session_items"], exclude=y)
            assert y not in p_coo[0]
            checked += 1
    assert checked > 0


@pytest.mark.skipif(not _HAVE_DATA, reason="hredial nltk data not present")
def test_history_sessions_capped_and_item_bearing():
    entity2id = _load("entity2id.json")
    movie = next(iter(entity2id))  # any linkable entity string

    def turns_with_movie():
        dialog = [
            {"role": "Seeker", "text": ["hi"], "movies": [], "entity": []},
            {"role": "Recommender", "text": ["try"], "movies": [movie], "entity": []},
        ]
        return merge_conv_turns(dialog, {"hi": 0, "try": 1}, entity2id, 3)

    empty = merge_conv_turns(
        [{"role": "Seeker", "text": ["x"], "movies": [], "entity": []}],
        {"x": 0},
        entity2id,
        3,
    )
    history = [turns_with_movie() for _ in range(50)] + [empty]
    sessions = build_history_sessions(history, k_hist=40)
    assert 0 < len(sessions) <= 40  # capped, and the movie-less session dropped
    assert all(len(items) >= 1 for items, _e, _w in sessions)


# --------------------------------------------------------------------------
# v2.2 fusion (rows7): Eq. 12-13 and the 3.4 cold-start table
# --------------------------------------------------------------------------
def _rows7_stub(d=8):
    from types import SimpleNamespace

    from crslab.model.crs.hycorec.attention import MHItemAttention
    from crslab.model.crs.hycorec.hycorec import HyCoRecModel

    m = SimpleNamespace(
        kg_emb_dim=d,
        device=torch.device("cpu"),
        pooling="Mean",
        item_attn=MHItemAttention(d, 4),
    )
    m._attention_and_gating = lambda r, c: HyCoRecModel._attention_and_gating(m, r, c)
    return m, lambda r, c: HyCoRecModel._user_from_rows(m, r, c)


def test_rows7_cold_start_table():
    d = 8
    _, user = _rows7_stub(d)
    rows, ctx = torch.randn(5, d), torch.randn(3, d)
    assert torch.allclose(user(rows, None), rows.mean(0), atol=1e-6)  # n_c = 0
    empty = torch.zeros(0, d)
    assert torch.allclose(user(empty, ctx), ctx.mean(0))  # no rows
    assert torch.equal(user(empty, None), torch.zeros(d))  # both empty


def test_rows7_matches_eq13_mean_pool():
    d = 8
    m, user = _rows7_stub(d)
    rows, ctx = torch.randn(4, d), torch.randn(3, d)
    n_tilde = m.item_attn(rows, ctx)
    expected = (n_tilde.mean(0) + ctx.sum(0)) / (ctx.size(0) + 1)
    assert torch.allclose(user(rows, ctx), expected, atol=1e-5)


def test_scope_weights_separate_vs_shared():
    from types import SimpleNamespace

    from torch import nn

    from crslab.model.crs.hycorec.hycorec import HyCoRecModel

    mk = lambda: nn.ModuleDict({f: nn.ModuleList([nn.Linear(2, 2)]) for f in ("item", "entity", "word", "G")})
    m = SimpleNamespace(hconv=mk(), hconv_p=mk(), hconv_g=mk(), scope_weights="separate")
    st = lambda sc, f: HyCoRecModel._stack_for(m, sc, f)
    assert st("C", "entity") is m.hconv["entity"]
    assert st("P", "entity") is m.hconv_p["entity"]
    assert st("G", "G") is m.hconv_g["G"]
    ids = {id(st(sc, "entity")) for sc in "CPG"}
    assert len(ids) == 3
    m.scope_weights = "shared"
    assert all(st(sc, "entity") is m.hconv["entity"] for sc in "CPG")


# --------------------------------------------------------------------------
# scope-C retention (ReSCOPE)
# --------------------------------------------------------------------------
def test_sliding_window_return_end_tracks_dropped_edges():
    turns = [[1], [2], [3, 4], [5]]  # w=1: turn 0 and 1 and 3 have <2 members
    edges, ends = sliding_window_hyperedges(turns, 1, return_end=True)
    assert edges == [[3, 4]] and ends == [2]
    edges, ends = sliding_window_hyperedges(turns, 2, return_end=True)
    assert ends == [1, 2, 3]


def test_build_incidence_kept_alignment():
    uniq, coo, n, kept = build_incidence([[7], [], [1, 2], [7, 3]], exclude=7, return_kept=True)
    assert kept == [2, 3] and n == 2


def _conv():
    from crslab.model.crs.hycorec.hycorec import CustomHypergraphConv

    torch.manual_seed(0)
    return CustomHypergraphConv(4, 4)


def test_edge_gate_one_is_identity_and_zero_removes_edge():
    conv = _conv()
    x = torch.randn(4, 4)
    ei = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 1, 1]])
    base = conv(x, ei, num_edges=2)
    assert torch.allclose(conv(x, ei, num_edges=2, edge_gate=torch.ones(2)), base)
    gated = conv(x, ei, num_edges=2, edge_gate=torch.tensor([1.0, 0.0]))
    # node 3 sits only in the muted edge: it keeps just the bias; D stays unweighted
    assert torch.allclose(gated[3], conv.bias)
    # node 0 sits only in edge 0 -> unchanged vs. ungated
    assert torch.allclose(gated[0], base[0])


def test_retention_gate_init_and_age_bucket():
    from crslab.model.crs.hycorec.cpc import RetentionGate, age_bucket

    assert [age_bucket(a) for a in (0, 7, 8, 11, 16, 32, 10**6)] == [0, 7, 8, 8, 10, 12, 15]
    gate = RetentionGate(4, n_fields=3, init_bias=3.0)
    h = torch.randn(2, 12)
    r = gate(h, h[-1], [0, 5])
    assert torch.allclose(r, torch.sigmoid(torch.tensor(3.0)).expand(2))
    r.sum().backward()
    assert gate.mlp[2].bias.grad is not None
    noage = RetentionGate(4, n_fields=3, use_age=False)
    assert not hasattr(noage, "age_emb") and noage(h, h[-1], [0, 5]).shape == (2,)


def test_window_dedupe_keeps_smallest_age():
    turns = [[1, 2], [], [], [3, 4]]  # w=2: windows end 0,1 identical ([1,2]); 2 -> [], 3 -> [3,4]
    edges, ends = sliding_window_hyperedges(turns, 2, return_end=True)
    assert edges == [[1, 2], [1, 2], [3, 4]] and ends == [0, 1, 3]
    edges, ends = sliding_window_hyperedges(turns, 2, return_end=True, dedupe=True)
    assert edges == [[1, 2], [3, 4]] and ends == [1, 3]


def test_edge_gate_interp_identity_and_self_fallback():
    conv = _conv()
    x = torch.randn(4, 4)
    ei = torch.tensor([[0, 1, 1, 2, 3], [0, 0, 1, 1, 1]])
    base = conv(x, ei, num_edges=2)
    assert torch.allclose(
        conv(x, ei, num_edges=2, edge_gate=torch.ones(2), gate_interp=True), base, atol=1e-6
    )
    g = conv(x, ei, num_edges=2, edge_gate=torch.tensor([1.0, 0.0]), gate_interp=True)
    # node 3 only lies in the muted edge: it keeps its own (projected) embedding + bias, not just the bias
    assert torch.allclose(g[3], conv.lin(x)[3] + conv.bias, atol=1e-6)
    assert torch.allclose(g[0], base[0], atol=1e-6)


def test_retention_gate_decay_prior_starts_at_gamma_pow_age():
    from crslab.model.crs.hycorec.cpc import RetentionGate

    gate = RetentionGate(4, n_fields=3, prior="decay", gamma=0.75)
    h = torch.randn(3, 12)
    r = gate(h, h[-1], [0, 2, 4])
    assert torch.allclose(r, torch.tensor([0.98, 0.75**2, 0.75**4]).clamp(0.02, 0.98), atol=1e-5)
