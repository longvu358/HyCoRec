"""CPC-Hypergraph v2 building blocks for :class:`HyCoRecModel`.

Implements the reusable pieces of
``docs/contexual_personal_collective/cpc_hypergraph_v2_method_spec.md``:

* :func:`khop_closure`         -- k-hop neighbourhood over an adjacency dict (H^P_{E,W}).
* :func:`build_incidence`      -- sparse COO incidence for a list of hyperedges,
  with the target-item leakage guard (spec 2.4.5).
* :func:`load_collective` / :class:`CollectivePropagation` -- precomputed sparse
  propagation matrices Â^G_f (Eq. 7) and the L-layer shared-weight conv over them.
* :class:`ScopeFusion`         -- per-field learned softmax fusion of the C/P/G
  pooled preference vectors (Eq. 11-12) with cold-start gradient masking (3.4).

The per-scope hypergraph convolution itself reuses ``CustomHypergraphConv`` from
``hycorec.py`` (same weights across scopes -- spec 3.1).
"""

import os

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

FIELDS = ("item", "entity", "word")


def khop_closure(seeds, adj, k):
    """BFS k-hop closure over ``adj`` (``dict[int, list[int]]``).

    Returns the set of node ids reachable within ``k`` hops of ``seeds``
    (seeds included). ``k <= 0`` returns just the seed set.
    """
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(max(0, k)):
        nxt = set()
        for u in frontier:
            nxt.update(adj.get(u, ()))
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def sliding_window_hyperedges(turn_lists, w):
    """Scope C hyperedges (spec v2.1 Eq. 2 / Eq. 4 in the simple write-up).

    One hyperedge per turn position ``t' <= t``, each the union of the ``w``
    turns ending at ``t'`` -- not one hyperedge per turn (v2's co-mention) and
    not a single hyperedge over the last ``w`` turns (v2's optional "window"
    variant). ``w=1`` degenerates exactly to v2's per-turn co-mention
    hyperedges (used as the ablation floor for scope C, A2).

    ``turn_lists``: ordered list of per-turn node-id lists (turn-local, as
    shipped by the dataloader's ``context_turn_<field>``). Hyperedges with
    fewer than 2 members after merging are dropped (spec: no degree-0 edges).
    """
    edges = []
    for t_prime in range(len(turn_lists)):
        window = turn_lists[max(0, t_prime - w + 1) : t_prime + 1]
        merged = set()
        for turn in window:
            merged.update(turn)
        if len(merged) >= 2:
            edges.append(sorted(merged))
    return edges


def build_incidence(hyperedges, exclude=None):
    """Build a binary hypergraph incidence in COO form.

    Args:
        hyperedges: iterable of node-id iterables, one per hyperedge.
        exclude: a node id to drop from every hyperedge (target-item leakage
            guard, spec 2.4.5), or ``None``.

    Returns:
        ``(unique_nodes, [node_ids, edge_ids], n_edges)`` -- plain python lists,
        kept on CPU (the caller remaps to a local sub-index before uploading a
        single tensor, as in ``HyCoRecModel._before_hyperconv``).
    """
    node_ids, edge_ids = [], []
    n_edges = 0
    for members in hyperedges:
        uniq = [v for v in dict.fromkeys(members) if v != exclude]
        if not uniq:
            continue
        node_ids.extend(uniq)
        edge_ids.extend([n_edges] * len(uniq))
        n_edges += 1
    return list(dict.fromkeys(node_ids)), [node_ids, edge_ids], n_edges


def a_hat_from_incidence(incidence_csc):
    """Â = D^-1 N E^-1 N^T  (Eq. 7), all sparse. ``incidence_csc`` is n_f x m."""
    n = incidence_csc.tocsr().astype(np.float64)
    node_deg = np.asarray(n.sum(axis=1)).ravel()
    edge_deg = np.asarray(n.sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        inv_node = np.where(node_deg > 0, 1.0 / node_deg, 0.0)
        inv_edge = np.where(edge_deg > 0, 1.0 / edge_deg, 0.0)
    d_inv = sp.diags(inv_node)
    e_inv = sp.diags(inv_edge)
    return (d_inv @ n @ e_inv @ n.T).tocoo()


class CollectivePropagation:
    """Precomputed sparse Â^G_f held on a device; applies the L-layer
    shared-weight conv  X <- Â (X W_l) + b_l  (Eq. 9 with s = G)."""

    def __init__(self, a_hat):
        self.a_hat = a_hat  # {field: torch.sparse_coo_tensor (n_f x n_f)}

    def to(self, device):
        self.a_hat = {f: m.to(device) for f, m in self.a_hat.items()}
        return self

    def run(self, field, x0, hconv_layers):
        """``x0``: (n_f, d) initial embeddings. ``hconv_layers``: iterable of
        ``CustomHypergraphConv`` (their ``.lin`` weight + ``.bias`` are reused)."""
        x = x0
        a = self.a_hat[field]
        for layer in hconv_layers:
            x = torch.sparse.mm(a, layer.lin(x))
            if layer.bias is not None:
                x = x + layer.bias
        return x


def load_collective(path, fields=FIELDS):
    """Load ``A_hat_<field>.npz`` (scipy CSR) written by ``scripts/build_collective.py``."""
    a_hat = {}
    for f in fields:
        fp = os.path.join(path, f"A_hat_{f}.npz")
        m = sp.load_npz(fp).tocoo()
        idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
        val = torch.tensor(m.data, dtype=torch.float)
        a_hat[f] = torch.sparse_coo_tensor(idx, val, tuple(m.shape)).coalesce()
    return CollectivePropagation(a_hat)


class ScopeFusion(nn.Module):
    """Per-field learned fusion  P_f = sum_s softmax(a_f)[s] * p^s_f  (Eq. 11-12).

    Scope order is ``(C, P, G)``. ``a_f`` init ``(-1, 0, -1)`` -> weights
    ~ (0.21, 0.58, 0.21), i.e. starts close to HyCoRec (P-dominated).
    """

    C, P, G = 0, 1, 2

    def __init__(self, fields=FIELDS, init=(-1.0, 0.0, -1.0), freeze=False):
        super().__init__()
        self.freeze = freeze
        self.logits = nn.ParameterDict(
            {
                f: nn.Parameter(
                    torch.tensor(list(init), dtype=torch.float),
                    requires_grad=not freeze,
                )
                for f in fields
            }
        )

    def fuse(self, field, p_c, p_p, p_g, has_history=True):
        """``p_*``: (d,) tensors. When ``has_history`` is False the P branch does
        not update ``a_f[P]`` (spec 3.4) -- the caller is expected to have set
        ``p_p = p_c`` already."""
        a = self.logits[field]
        if not has_history:
            a = torch.stack([a[self.C], a[self.P].detach(), a[self.G]])
        w = torch.softmax(a, dim=0)
        return w[self.C] * p_c + w[self.P] * p_p + w[self.G] * p_g

    def weight_table(self):
        return {
            f: torch.softmax(self.logits[f], dim=0).detach().cpu().tolist()
            for f in self.logits
        }
