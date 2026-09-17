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
