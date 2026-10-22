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

import math
import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn

FIELDS = ("item", "entity", "word")


def khop_closure(seeds, adj, k, max_neighbors=None):
    """BFS k-hop closure over ``adj`` (``dict[int, list[int]]``).

    Returns the set of node ids reachable within ``k`` hops of ``seeds``
    (seeds included). ``k <= 0`` returns just the seed set.

    ``max_neighbors``, if set, caps how many neighbors a single hop
    contributes per frontier node (deterministic subsample seeded by the
    node id). Some KG adjacencies here have extreme long-tail degree (e.g.
    HTGReDial's entity_adj: max degree ~20.6k, mean 8.4 vs HReDial's mean
    2.4) -- one mentioned hub entity can otherwise blow a single sample's
    P-scope hyperedge up to tens of thousands of nodes, which is what made
    HTGReDial's full run ~16x slower than HReDial's and OOM'd at a larger
    batch size.
    """
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(max(0, k)):
        nxt = set()
        for u in frontier:
            neighbors = adj.get(u, ())
            if max_neighbors is not None and len(neighbors) > max_neighbors:
                neighbors = random.Random(u).sample(list(neighbors), max_neighbors)
            nxt.update(neighbors)
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return seen


def sliding_window_hyperedges(turn_lists, w, return_end=False, dedupe=False):
    """Scope C hyperedges (spec v2.1 Eq. 2 / Eq. 4 in the simple write-up).

    One hyperedge per turn position ``t' <= t``, each the union of the ``w``
    turns ending at ``t'`` -- not one hyperedge per turn (v2's co-mention) and
    not a single hyperedge over the last ``w`` turns (v2's optional "window"
    variant). ``w=1`` degenerates exactly to v2's per-turn co-mention
    hyperedges (used as the ablation floor for scope C, A2).

    ``turn_lists``: ordered list of per-turn node-id lists (turn-local, as
    shipped by the dataloader's ``context_turn_<field>``). Hyperedges with
    fewer than 2 members after merging are dropped (spec: no degree-0 edges).

    ``return_end=True`` returns ``(edges, ends)`` where ``ends[i]`` is the turn
    position ``t'`` the i-th kept hyperedge ends at (needed for the retention
    age ``T - t'``; dropped edges make the list index differ from ``t'``).

    ``dedupe=True`` merges windows with an identical node set (turns that
    mention nothing make consecutive windows coincide) and keeps the one with
    the smallest age, i.e. the largest ``t'``. Off by default: it changes the
    binary incidence of the original sliding-window baseline.
    """
    edges, ends = [], []
    for t_prime in range(len(turn_lists)):
        window = turn_lists[max(0, t_prime - w + 1) : t_prime + 1]
        merged = set()
        for turn in window:
            merged.update(turn)
        if len(merged) >= 2:
            members = sorted(merged)
            if dedupe and edges and edges[-1] == members:
                ends[-1] = t_prime  # same node set as the previous window: newer end wins
                continue
            edges.append(members)
            ends.append(t_prime)
    return (edges, ends) if return_end else edges


def build_incidence(hyperedges, exclude=None, return_kept=False):
    """Build a binary hypergraph incidence in COO form.

    Args:
        hyperedges: iterable of node-id iterables, one per hyperedge.
        exclude: a node id to drop from every hyperedge (target-item leakage
            guard, spec 2.4.5), or ``None``.

    Returns:
        ``(unique_nodes, [node_ids, edge_ids], n_edges)`` -- plain python lists,
        kept on CPU (the caller remaps to a local sub-index before uploading a
        single tensor, as in ``HyCoRecModel._before_hyperconv``).
        With ``return_kept=True`` a 4th element lists, for every kept
        hyperedge, its index in the input ``hyperedges`` (to align per-edge
        quantities such as retention gates after empty edges are dropped).
    """
    node_ids, edge_ids, kept = [], [], []
    n_edges = 0
    for src, members in enumerate(hyperedges):
        uniq = [v for v in dict.fromkeys(members) if v != exclude]
        if not uniq:
            continue
        kept.append(src)
        node_ids.extend(uniq)
        edge_ids.extend([n_edges] * len(uniq))
        n_edges += 1
    out = list(dict.fromkeys(node_ids)), [node_ids, edge_ids], n_edges
    return out + (kept,) if return_kept else out


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
        self._replicas = {}  # (field, device) -> copy for DataParallel replicas

    def to(self, device):
        self.a_hat = {f: m.to(device) for f, m in self.a_hat.items()}
        self._replicas = {}
        return self

    def _on(self, field, device):
        a = self.a_hat[field]
        if a.device == device:
            return a
        key = (field, device)
        if key not in self._replicas:
            self._replicas[key] = a.to(device)
        return self._replicas[key]

    def run(self, field, x0, hconv_layers, dropout=0.0, training=False):
        """``x0``: (n_f, d) initial embeddings. ``hconv_layers``: iterable of
        ``CustomHypergraphConv`` (their ``.lin`` weight + ``.bias`` are reused)."""
        x = x0
        a = self._on(field, x0.device)
        for layer in hconv_layers:
            x = F.dropout(x, p=dropout, training=training)
            x = torch.sparse.mm(a, layer.lin(x))
            if layer.bias is not None:
                x = x + layer.bias
        return x


def load_collective(path, fields=FIELDS):
    """Load ``A_hat_<field>.npz`` (scipy CSR) written by ``scripts/build_collective.py``.

    Stored as torch sparse CSR (not COO): ``torch.sparse.mm`` on CUDA is
    ~6-7x faster in CSR for these matrices (measured on the HTGReDial word
    field, n=30001, nnz=21.3M: 43ms/mm COO vs 6ms/mm CSR), with numerically
    identical output -- this is the dominant per-step cost of the G scope.
    """
    a_hat = {}
    for f in fields:
        fp = os.path.join(path, f"A_hat_{f}.npz")
        m = sp.load_npz(fp).tocsr()
        crow = torch.tensor(m.indptr, dtype=torch.int64)
        col = torch.tensor(m.indices, dtype=torch.int64)
        val = torch.tensor(m.data, dtype=torch.float)
        a_hat[f] = torch.sparse_csr_tensor(crow, col, val, tuple(m.shape))
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


def age_bucket(age, n_buckets=16):
    """Age 0..7 keep their own bucket, older ages are binned on a log2 scale
    (two buckets per octave) and clipped at ``n_buckets - 1``."""
    age = max(0, int(age))
    if age < 8:
        return min(age, n_buckets - 1)
    return min(8 + int(2 * math.log2(age / 8)), n_buckets - 1)


class RetentionGate(nn.Module):
    """Scope-C retention, one alpha per window ``e`` shared by the three fields.

    ``alpha_e = sigmoid(MLP([h_e; h_e * q; phi(age_e)]) + b)`` with ``h_e`` the
    concatenation of the per-field mean node embeddings of the window (before
    propagation, so alpha does not depend on its own propagated state), ``q`` the
    same vector for the latest valid window, and ``phi`` a learned embedding of
    the bucketed age (``use_age=False`` drops it: content-only). The last MLP
    layer starts at 0 and a fixed offset ``init_bias`` so alpha starts at
    ``sigmoid(init_bias) ~ 0.95`` (close to uniform history), or at ``gamma ** age`` with
    ``prior='decay'`` (the empirical decay of the dialogue statistics).
    """

    def __init__(
        self, dim, n_fields=3, hidden=64, use_age=True, n_buckets=16, init_bias=3.0,
        prior="uniform", gamma=0.75,
    ):
        super().__init__()
        assert prior in ("uniform", "decay")
        self.use_age, self.n_buckets = use_age, n_buckets
        self.prior, self.gamma = prior, float(gamma)
        d_in = 2 * n_fields * dim
        if use_age:
            self.age_emb = nn.Embedding(n_buckets, 16)
            nn.init.normal_(self.age_emb.weight, std=0.1)
            d_in += 16
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        # fixed offset instead of a learned bias: AdamW's weight decay then pulls the
        # gate towards the neutral alpha = sigmoid(init_bias) (uniform history), not towards 0.5
        self.register_buffer("base_logit", torch.tensor(float(init_bias)))

    def forward(self, h, q, ages):
        """``h``: (m, F*d), ``q``: (F*d,), ``ages``: list[int] of length m -> (m,)."""
        feats = [h, h * q.unsqueeze(0)]
        if self.use_age:
            idx = torch.tensor(
                [age_bucket(a, self.n_buckets) for a in ages], dtype=torch.long, device=h.device
            )
            feats.append(self.age_emb(idx))
        base = self.base_logit
        if self.prior == "decay":  # start from alpha(age) = gamma ** age, weight decay pulls back to it
            p = torch.tensor(
                [self.gamma ** max(0, int(a)) for a in ages], dtype=h.dtype, device=h.device
            ).clamp(0.02, 0.98)
            base = torch.log(p) - torch.log1p(-p)
        return torch.sigmoid(self.mlp(torch.cat(feats, dim=-1)).squeeze(-1) + base)


class EvidenceRetention(nn.Module):
    """Scope-C retention as a per-relation forget gate driven by what happened after the relation.

    A relation is a window ``e`` (end turn ``t_e``). Each later context turn ``t in (t_e, T]`` can
    weaken it: ``f_{e,t} = sigmoid(MLP([h_e; u_t; h_e * u_t; ov(e, t)]) + b0)``, so
    ``log rho_e = sum_t log(1 - f_{e,t})`` is non-increasing in time and the latest window has
    ``rho = 1`` (empty sum) -- the anchor the older relations are measured against. ``ov`` is the
    share of e's nodes re-mentioned at ``t`` (explicit reinforcement). The readout score is
    ``(W_q u_T) . (W_k h_e) / sqrt(d) + log rho_e`` (softmax over windows outside this module):
    relevance to the current turn selects, retention discounts. ``h_e`` / ``u_t`` are pre-propagation
    field-concatenated mean embeddings of the window / turn, so the score never sees the target.

    forget: evidence (above) | age (one learned per-turn forget rate) | const (``1 - gamma``, i.e.
    ``rho = gamma ** age``) | none (``rho = 1``). ``relevance=False`` drops the attention term.
    Both learned terms start neutral: W_k = 0 (no relevance), forget ~ sigmoid(b0) ~ 0.018 per turn.
    """

    def __init__(self, dim, n_fields=3, hidden=64, forget="evidence", relevance=True, overlap=True,
                 gamma=0.75, init_forget=-4.0, d_att=64):
        super().__init__()
        assert forget in ("evidence", "age", "const", "none")
        self.forget, self.relevance, self.overlap = forget, relevance, overlap
        self.gamma, self.d_att = float(gamma), d_att
        d = n_fields * dim
        if relevance:
            self.w_q = nn.Linear(d, d_att, bias=False)
            self.w_k = nn.Linear(d, d_att, bias=False)
            nn.init.zeros_(self.w_k.weight)
        if forget == "evidence":
            self.mlp = nn.Sequential(nn.Linear(3 * d + int(overlap), hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.mlp[2].weight)
            nn.init.zeros_(self.mlp[2].bias)
            # fixed offset (not a learned bias): weight decay pulls towards "almost never forget"
            self.register_buffer("base_logit", torch.tensor(float(init_forget)))
        elif forget == "age":
            self.rate_logit = nn.Parameter(torch.tensor(float(init_forget)))

    def log_retention(self, h, ends, u, ov):
        """``h``: (m, D) windows, ``ends``: (m,) long, ``u``: (n_turns, D), ``ov``: (m, n_turns) -> (m,)."""
        n_turns = u.size(0)
        t = torch.arange(n_turns, device=h.device)
        later = (t.unsqueeze(0) > ends.unsqueeze(1)).to(h.dtype)  # (m, n_turns): turns after the window
        if self.forget == "none":
            return h.new_zeros(h.size(0))
        if self.forget == "const":
            return later.sum(1) * math.log(self.gamma)
        if self.forget == "age":
            return later.sum(1) * F.logsigmoid(-self.rate_logit)
        m = h.size(0)
        he, ut = h.unsqueeze(1).expand(m, n_turns, -1), u.unsqueeze(0).expand(m, n_turns, -1)
        feats = [he, ut, he * ut] + ([ov.unsqueeze(-1)] if self.overlap else [])
        logit = self.mlp(torch.cat(feats, -1)).squeeze(-1) + self.base_logit
        return (later * F.logsigmoid(-logit)).sum(1)  # log(1 - sigmoid(z)) = logsigmoid(-z)

    def forward(self, h, ends, u, ov, q=None):
        """``q``: (D,) current-turn query (default ``u[-1]``). Returns (score (m,), log rho (m,))."""
        log_rho = self.log_retention(h, ends, u, ov)
        score = log_rho
        if self.relevance:
            q = self.w_q(u[-1] if q is None else q)
            score = score + (self.w_k(h) @ q) / math.sqrt(self.d_att)
        return score, log_rho
