# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

r"""
PCR
====
References:
    Chen, Qibin, et al. `"Towards Knowledge-Based Recommender Dialog System."`_ in EMNLP 2019.

.. _`"Towards Knowledge-Based Recommender Dialog System."`:
   https://www.aclweb.org/anthology/D19-1189/

"""

import json
import os.path
import random
import pickle
from typing import List
from time import perf_counter

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from tqdm import tqdm
from torch_geometric.nn import RGCNConv

from crslab.config import DATA_PATH, DATASET_PATH
from crslab.model.base import BaseModel
from crslab.model.crs.hycorec.attention import MHItemAttention
from crslab.model.crs.hycorec.cpc import (
    ScopeFusion,
    build_incidence,
    khop_closure,
    load_collective,
    sliding_window_hyperedges,
)
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionBatch, SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from crslab.model.crs.hycorec.decoder import TransformerDecoderKG


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


class CustomHypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = "node",
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(flow="source_to_target", node_dim=0, **kwargs)

        assert attention_mode in ["node", "edge"]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(
                in_channels, out_channels, bias=False, weight_initializer="glorot"
            )

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=["num_edges"])
    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        hyperedge_weight: Optional[Tensor] = None,
        hyperedge_attr: Optional[Tensor] = None,
        num_edges: Optional[int] = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (int, optional) : The number of edges :math:`M`.
                (default: :obj:`None`)
        """
        num_nodes = x.size(0)

        if num_edges is None:
            # Falling back here forces a blocking GPU->CPU sync
            # (int(tensor.max())) on every single call. Callers in this
            # model already know the edge count for free on the CPU side
            # (from building the hyperedge index in cpc.build_incidence),
            # so they pass it explicitly.
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            if hyperedge_weight is None:
                hyperedge_weight = x.new_ones(num_edges)
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == "node":
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        node_idx, edge_idx = hyperedge_index[0], hyperedge_index[1]

        if hyperedge_weight is None:
            D = scatter(
                x.new_ones(node_idx.numel()),
                node_idx,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        else:
            D = scatter(
                hyperedge_weight[edge_idx],
                node_idx,
                dim=0,
                dim_size=num_nodes,
                reduce="sum",
            )
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(
            x.new_ones(edge_idx.numel()),
            edge_idx,
            dim=0,
            dim_size=num_edges,
            reduce="sum",
        )
        B = 1.0 / B
        B[B == float("inf")] = 0

        if self.use_attention:
            out = self.propagate(
                hyperedge_index, x=x, norm=B, alpha=alpha, size=(num_nodes, num_edges)
            )
            out = self.propagate(
                hyperedge_index.flip([0]),
                x=out,
                norm=D,
                alpha=alpha,
                size=(num_edges, num_nodes),
            )
        else:
            # Fast path: with no attention, message() collapses to a plain
            # norm_i * x_j scale (heads=1), so the two propagate() calls
            # below just do a scatter-add from nodes to hyperedges and back.
            # Doing that directly instead of going through
            # MessagePassing.propagate()'s generic dispatch (arg
            # collection/validation for a general message()/aggregate()/
            # update() pipeline) skips a lot of fixed per-call Python
            # overhead. That overhead dominates here because these
            # hypergraphs are usually tiny (median ~6 nodes in this
            # project's real data) - actual compute is negligible, so
            # cutting the fixed cost per call is what matters. Verified
            # bit-identical to the stock propagate()-based implementation
            # via torch.allclose across hundreds of random graphs.
            msg1 = B.index_select(0, edge_idx).unsqueeze(-1) * x.index_select(
                0, node_idx
            )
            out = x.new_zeros(num_edges, x.size(-1))
            out.index_add_(0, edge_idx, msg1)

            msg2 = D.index_select(0, node_idx).unsqueeze(-1) * out.index_select(
                0, edge_idx
            )
            out = x.new_zeros(num_nodes, x.size(-1))
            out.index_add_(0, node_idx, msg2)

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


class HyCoRecModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_entity: A integer indicating the number of entities.
        n_relation: A integer indicating the number of relation in KG.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        user_emb_dim: A integer indicating the dimension of user embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        longest_label: A integer indicating the longest length for response generation.
        user_proj_dim: A integer indicating dim to project for user embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        self.device = device
        self.gpu = opt.get("gpu", -1)
        self.dataset = opt.get("dataset", None)
        self.llm = opt.get("llm", "chatgpt-4o")
        assert self.dataset in [
            "HReDial",
            "HTGReDial",
            "DuRecDial",
            "OpenDialKG",
            "ReDial",
            "TGReDial",
        ]
        # vocab
        self.pad_token_idx = vocab["tok2ind"]["__pad__"]
        self.start_token_idx = vocab["tok2ind"]["__start__"]
        self.end_token_idx = vocab["tok2ind"]["__end__"]
        self.vocab_size = vocab["vocab_size"]
        self.token_emb_dim = opt.get("token_emb_dim", 300)
        self.pretrain_embedding = side_data.get("embedding", None)
        self.token2id = json.load(
            open(
                os.path.join(
                    DATASET_PATH, self.dataset.lower(), opt["tokenize"], "token2id.json"
                ),
                "r",
                encoding="utf-8",
            )
        )
        self.entity2id = json.load(
            open(
                os.path.join(
                    DATASET_PATH,
                    self.dataset.lower(),
                    opt["tokenize"],
                    "entity2id.json",
                ),
                "r",
                encoding="utf-8",
            )
        )
        # kg
        self.n_entity = vocab["n_entity"]
        self.entity_kg = side_data["entity_kg"]
        self.n_relation = self.entity_kg["n_relation"]
        self.edge_idx, self.edge_type = edge_to_pyg_format(
            self.entity_kg["edge"], "RGCN"
        )
        self.edge_idx = self.edge_idx.to(device)
        self.edge_type = self.edge_type.to(device)
        self.num_bases = opt.get("num_bases", 8)
        self.kg_emb_dim = opt.get("kg_emb_dim", 300)
        self.user_emb_dim = self.kg_emb_dim
        # transformer
        self.n_heads = opt.get("n_heads", 2)
        self.n_layers = opt.get("n_layers", 2)
        self.ffn_size = opt.get("ffn_size", 300)
        self.dropout = opt.get("dropout", 0.1)
        self.attention_dropout = opt.get("attention_dropout", 0.0)
        self.relu_dropout = opt.get("relu_dropout", 0.1)
        self.embeddings_scale = opt.get("embedding_scale", True)
        self.learn_positional_embeddings = opt.get("learn_positional_embeddings", False)
        self.reduction = opt.get("reduction", False)
        self.n_positions = opt.get("n_positions", 1024)
        self.longest_label = opt.get("longest_label", 30)
        self.user_proj_dim = opt.get("user_proj_dim", 512)
        # pooling
        self.pooling = opt.get("pooling", None)
        assert self.pooling == "Attn" or self.pooling == "Mean"
        # MHA
        self.mha_n_heads = opt.get("mha_n_heads", 4)
        self.extension_strategy = opt.get("extension_strategy", None)
        # ---- CPC-Hypergraph v2 (docs/contexual_personal_collective) ----------
        # active preference scopes: C=Contextual (current dialogue),
        # P=Personal (D_u(d) historical sessions), G=Collective (static corpus)
        self.scopes = [s.upper() for s in opt.get("scopes", ["C", "P", "G"])]
        self.k_hist = opt.get("k_hist", 40)
        self.k_hop = opt.get("k_hop", 1)  # k-hop for H^P_{E,W}; 1 == HyCoRec baseline
        # cap on neighbors/hop for the H^P_{E,W} k-hop expansion (None = uncapped,
        # matches original behaviour). Guards against hub nodes in entity_adj/
        # word_adj blowing a single sample's P-scope hyperedge up to thousands
        # of nodes -- see khop_closure's docstring in cpc.py.
        self.khop_cap = opt.get("khop_cap", None)
        # sliding-window size w for scope C (spec v2.1 Eq. 2): one hyperedge per
        # turn position t'<=t, merging the w turns ending at t'. w=1 degenerates
        # to per-turn co-mention hyperedges (the v2.1 ablation floor for C, A2).
        self.context_window_w = (
            opt.get("context_window_w", opt.get("hyperedge_window_k", None)) or 3
        )
        self.hconv_layers = opt.get("hconv_layers", 2)
        # scope_weights: "shared" = one HConv stack per field reused by C, P and G
        # (spec 3.1, checkpoint-compatible); "separate" = each scope owns its own
        # stack (C: self.hconv, P: self.hconv_p, G: self.hconv_g).
        self.scope_weights = opt.get("scope_weights", "shared")
        assert self.scope_weights in ("shared", "separate")
        self.collective_path = opt.get("collective_path", None)
        self.alpha_init = tuple(opt.get("alpha_init", (-1.0, 0.0, -1.0)))
        self.freeze_alpha = opt.get("freeze_alpha", False)
        # fusion_mode: "field" = v2.1 per-field softmax alpha over (C,P,G) then
        # attention on 3 rows; "rows7" = v2.2 Eq. 11-13 (MHA(P_c, R, R) over the
        # up-to-7 pooled scope rows, empty rows dropped from the key set).
        self.fusion_mode = opt.get("fusion_mode", "field")
        self.word_rgcn = opt.get("word_rgcn", True)
        assert self.fusion_mode in ("field", "rows7")
        self.ei_mode = opt.get("ei_mode", "xg")  # base | xg  (Eq. 14a / 14b)
        self.item_entity_ids = list(side_data.get("item_entity_ids", []))
        self.collective = None
        self.g_unified = False
        # Branch 2: review-hypergraph in scope P, field E (Eq. 1r/1s). Scope G's
        # review + aspect hyperedges (Eq. 4/6, 5/7) are baked into Â^G_{E,I}
        # offline by build_collective.py --use_review; nothing to load here for G.
        self.use_review_hypergraph = opt.get("use_review_hypergraph", False)
        self.review_index_path = opt.get("review_index_path", None)
        self.review_index = None
        self.pretrain = opt.get("pretrain", False)
        self.pretrain_data = None
        self.pretrain_epoch = opt.get("pretrain_epoch", 9999)

        super(HyCoRecModel, self).__init__(opt, device)
        return

    # 构建模型
    def build_model(self, *args, **kwargs):
        if self.pretrain:
            pretrain_file = os.path.join(
                "pretrain", self.dataset, str(self.pretrain_epoch) + "-epoch.pth"
            )
            self.pretrain_data = torch.load(
                pretrain_file, map_location=torch.device("cuda:" + str(self.gpu[0]))
            )
            logger.info(f"[Load Pretrain Weights from {pretrain_file}]")
        if self.dataset == "HReDial":
            self._build_hredial_copy_mask()
        self._build_adjacent_matrix()
        # self._build_hllm_data()
        self._build_embedding()
        self._build_kg_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()
        self._load_collective()
        self._load_review_index()

    def _load_review_index(self):
        """E_top^+(R_i) per item, precomputed by build_review_index.py
        (spec 1.3/4.1). Only used by scope P, field E (Eq. 1r/1s); scope G's
        review/aspect hyperedges are already baked into Â^G_{E,I}."""
        if not self.use_review_hypergraph:
            return
        path = self.review_index_path or os.path.join(
            DATA_PATH, "reviews", self.dataset.lower(), "review_index.json"
        )
        if not os.path.isfile(path):
            logger.warning(
                f"[CPC] use_review_hypergraph=true but no review_index.json at {path}; "
                "run build_review_index.py. Disabling review-hypergraph in scope P."
            )
            self.use_review_hypergraph = False
            return
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # {item_id(int): [entity_id, ...]} -- tf counts (Eq. 1r/1s only needs
        # the top-k set itself, not the ranking used to build it)
        self.review_index = {
            int(item_id): [eid for eid, _tf in pairs] for item_id, pairs in raw.items()
        }
        logger.info(f"[CPC] loaded review index ({len(self.review_index)} items) from {path}")

    def _load_collective(self):
        self.n_word = max(self.token2id.values()) + 1
        if "G" not in self.scopes:
            return
        path = self.collective_path or os.path.join(
            DATA_PATH, "collective", self.dataset.lower()
        )
        if not os.path.isdir(path):
            logger.warning(
                f"[CPC] scope G requested but no collective dir at {path}; "
                "run build_collective.py. Disabling scope G."
            )
            self.scopes = [s for s in self.scopes if s != "G"]
            return
        # v2.2: a single unified Â^G over V^G = V_E u V_W (A_hat_G.npz)
        self.g_unified = os.path.isfile(os.path.join(path, "A_hat_G.npz"))
        if self.g_unified:
            assert (
                self.fusion_mode == "rows7"
            ), "unified Â^G (A_hat_G.npz) needs fusion_mode: rows7"
            self.collective = load_collective(path, fields=("G",)).to(self.device)
        else:
            self.collective = load_collective(path).to(self.device)
        if self.scope_weights == "separate":
            keys = ("G",) if self.g_unified else ("item", "entity", "word")
            for k in keys:
                self.hconv_g[k] = nn.ModuleList(
                    CustomHypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
                    for _ in range(self.hconv_layers)
                )
        logger.info(
            f"[CPC] loaded collective propagation matrices from {path} "
            f"(unified={self.g_unified})"
        )

    # 构建 mask
    def _build_hredial_copy_mask(self):
        token_filename = os.path.join(DATASET_PATH, "hredial", "nltk", "token2id.json")
        token_file = open(token_filename, "r", encoding="utf-8")
        token2id = json.load(token_file)
        id2token = {token2id[token]: token for token in token2id}
        self.hredial_copy_mask = list()
        for i in range(len(id2token)):
            token = id2token[i]
            if token[0] == "@":
                self.hredial_copy_mask.append(True)
            else:
                self.hredial_copy_mask.append(False)
        self.hredial_copy_mask = torch.as_tensor(self.hredial_copy_mask).to(self.device)
        return

    def _build_hllm_data(self):
        self.hllm_data_table = {
            "train": pickle.load(
                open(
                    os.path.join(
                        DATA_PATH,
                        "hllm",
                        self.dataset.lower(),
                        self.llm,
                        "hllm_train_data.pkl",
                    ),
                    "rb",
                )
            ),
            "valid": pickle.load(
                open(
                    os.path.join(
                        DATA_PATH,
                        "hllm",
                        self.dataset.lower(),
                        self.llm,
                        "hllm_valid_data.pkl",
                    ),
                    "rb",
                )
            ),
            "test": pickle.load(
                open(
                    os.path.join(
                        DATA_PATH,
                        "hllm",
                        self.dataset.lower(),
                        self.llm,
                        "hllm_test_data.pkl",
                    ),
                    "rb",
                )
            ),
        }
        return

    # 构建关联矩阵
    def _build_adjacent_matrix(self):
        entity2id = self.entity2id
        token2id = self.token2id
        item_edger = pickle.load(
            open(
                os.path.join(
                    DATA_PATH, "edger", self.dataset.lower(), "item_edger.pkl"
                ),
                "rb",
            )
        )
        entity_edger = pickle.load(
            open(
                os.path.join(
                    DATA_PATH, "edger", self.dataset.lower(), "entity_edger.pkl"
                ),
                "rb",
            )
        )
        word_edger = pickle.load(
            open(
                os.path.join(
                    DATA_PATH, "edger", self.dataset.lower(), "word_edger.pkl"
                ),
                "rb",
            )
        )

        item_adj = {}
        for item_a in item_edger:
            item_list = item_edger[item_a]
            if item_a not in entity2id:
                continue
            item_a = entity2id[item_a]
            mapped_item_list = []
            for item in item_list:
                if item not in entity2id:
                    continue
                mapped_item_list.append(entity2id[item])
            item_adj[item_a] = mapped_item_list
        self.item_adj = item_adj

        entity_adj = {}
        for entity_a in entity_edger:
            entity_list = entity_edger[entity_a]
            if entity_a not in entity2id:
                continue
            entity_a = entity2id[entity_a]
            mapped_entity_list = []
            for entity in entity_list:
                if entity not in entity2id:
                    continue
                mapped_entity_list.append(entity2id[entity])
            entity_adj[entity_a] = mapped_entity_list
        self.entity_adj = entity_adj

        word_adj = {}
        for word_a in word_edger:
            word_list = word_edger[word_a]
            if word_a not in token2id:
                continue
            word_a = token2id[word_a]
            mapped_word_list = []
            for word in word_list:
                if word not in token2id:
                    continue
                mapped_word_list.append(token2id[word])
            word_adj[word_a] = mapped_word_list
        self.word_adj = word_adj

        logger.info(f"[Adjacent Matrix built.]")
        return

    # 构建编码层
    def _build_embedding(self):
        if self.pretrain_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrain_embedding, dtype=torch.float),
                freeze=False,
                padding_idx=self.pad_token_idx,
            )
        else:
            self.token_embedding = nn.Embedding(
                self.vocab_size, self.token_emb_dim, self.pad_token_idx
            )
            nn.init.normal_(
                self.token_embedding.weight, mean=0, std=self.kg_emb_dim**-0.5
            )
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        self.entity_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.entity_embedding.weight, mean=0, std=self.kg_emb_dim**-0.5)
        nn.init.constant_(self.entity_embedding.weight[0], 0)
        self.word_embedding = nn.Embedding(self.n_entity, self.kg_emb_dim, 0)
        nn.init.normal_(self.word_embedding.weight, mean=0, std=self.kg_emb_dim**-0.5)
        nn.init.constant_(self.word_embedding.weight[0], 0)
        logger.debug("[Build embedding]")
        return

    # 构建超图编码层
    def _build_kg_layer(self):
        # graph encoder
        self.item_encoder = RGCNConv(
            self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases
        )
        self.entity_encoder = RGCNConv(
            self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases
        )
        self.word_encoder = RGCNConv(
            self.kg_emb_dim, self.kg_emb_dim, self.n_relation, num_bases=self.num_bases
        )
        if self.pretrain:
            self.item_encoder.load_state_dict(self.pretrain_data["encoder"])
        # hypergraph convolution. scope_weights="shared": ONE stack of L layers
        # per field reused by C/P/G (spec 3.1). "separate": every hypergraph owns
        # its stack -- C and P one per field (3 + 3), G one (unified) -- so
        # 7 stacks in the v2.2 layout, one per row of R.
        def _stack():
            return nn.ModuleList(
                CustomHypergraphConv(self.kg_emb_dim, self.kg_emb_dim)
                for _ in range(self.hconv_layers)
            )

        self.hconv = nn.ModuleDict({f: _stack() for f in ("item", "entity", "word")})
        if self.scope_weights == "separate":
            self.hconv_p = nn.ModuleDict(
                {f: _stack() for f in ("item", "entity", "word")}
            )
            # G's stacks are created in _load_collective, once it is known whether
            # Â^G is unified (one stack "G") or per-field (one stack per field).
            self.hconv_g = nn.ModuleDict()

        # learned per-field softmax fusion of the C/P/G pooled vectors (Eq. 11-12)
        self.fusion = ScopeFusion(init=self.alpha_init, freeze=self.freeze_alpha)
        # attention type
        self.item_attn = MHItemAttention(self.kg_emb_dim, self.mha_n_heads)
        # pooling
        if self.pooling == "Attn":
            self.kg_attn = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
            self.kg_attn_his = SelfAttentionBatch(self.kg_emb_dim, self.kg_emb_dim)
        logger.debug("[Build kg layer]")
        return

    # 构建推荐模块
    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()
        logger.debug("[Build recommendation layer]")
        return

    # 构建对话模块
    def _build_conversation_layer(self):
        self.register_buffer(
            "START", torch.tensor([self.start_token_idx], dtype=torch.long)
        )
        self.entity_to_token = nn.Linear(self.kg_emb_dim, self.token_emb_dim, bias=True)
        self.related_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions,
        )
        self.context_encoder = TransformerEncoder(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.pad_token_idx,
            self.learn_positional_embeddings,
            self.embeddings_scale,
            self.reduction,
            self.n_positions,
        )
        self.decoder = TransformerDecoderKG(
            self.n_heads,
            self.n_layers,
            self.token_emb_dim,
            self.ffn_size,
            self.vocab_size,
            self.token_embedding,
            self.dropout,
            self.attention_dropout,
            self.relu_dropout,
            self.embeddings_scale,
            self.learn_positional_embeddings,
            self.pad_token_idx,
            self.n_positions,
        )
        self.user_proj_1 = nn.Linear(self.user_emb_dim, self.user_proj_dim)
        self.user_proj_2 = nn.Linear(self.user_proj_dim, self.vocab_size)
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)

        self.copy_proj_1 = nn.Linear(2 * self.token_emb_dim, self.token_emb_dim)
        self.copy_proj_2 = nn.Linear(self.token_emb_dim, self.vocab_size)
        logger.debug("[Build conversation layer]")
        return

    @staticmethod
    def flatten(inputs):
        outputs = set()
        for li in inputs:
            for i in li:
                outputs.add(i)
        return list(outputs)

    # 注意力融合特征向量
    def _attention_and_gating(self, related_embedding, context_embedding):
        """Fold the fused per-field preference vectors (n_fields, d) into a
        single user vector, cross-attending to the mentioned entities of the
        current dialogue (the P_c analogue). Same structure as the original
        HyCoRec gating, only the input is [P_I; P_E; P_W] instead of a bag of
        hyperedge node embeddings."""
        if context_embedding is None:
            if self.pooling == "Attn":
                return self.kg_attn_his(related_embedding)
            assert self.pooling == "Mean"
            return torch.mean(related_embedding, dim=0)
        attentive = self.item_attn(related_embedding, context_embedding)
        if self.pooling == "Attn":
            user_repr = self.kg_attn_his(attentive).unsqueeze(0)
            user_repr = torch.cat((context_embedding, user_repr), dim=0)
            return self.kg_attn(user_repr)
        assert self.pooling == "Mean"
        user_repr = torch.mean(attentive, dim=0).unsqueeze(0)
        user_repr = torch.cat((context_embedding, user_repr), dim=0)
        return torch.mean(user_repr, dim=0)

    # ---- CPC-Hypergraph v2 scope machinery -------------------------------
    _FIELDS = ("item", "entity", "word")

    def _field_adj(self, field):
        return {
            "item": self.item_adj,
            "entity": self.entity_adj,
            "word": self.word_adj,
        }[field]

    def _stack_for(self, scope, field):
        if self.scope_weights == "shared" or scope == "C":
            return self.hconv[field]
        return (self.hconv_p if scope == "P" else self.hconv_g)[field]

    def _scope_hconv(self, scope, hyperedges, field, tot_embedding, exclude=None):
        """Shared L-layer conv over a locally built incidence.
        Returns (x_sub: (n_sub, d), tot2sub: dict) or (None, {})."""
        uniq, coo, n_edges = build_incidence(hyperedges, exclude=exclude)
        if n_edges == 0 or not uniq:
            return None, {}
        tot2sub = {t: s for s, t in enumerate(uniq)}
        x = tot_embedding[uniq]
        ei = torch.tensor(
            [[tot2sub[v] for v in coo[0]], coo[1]],
            dtype=torch.long,
            device=self.device,
        )
        for layer in self._stack_for(scope, field):
            x = layer(x, ei, num_edges=n_edges)
        return x, tot2sub

    def _empty_row(self, none_if_empty):
        return None if none_if_empty else torch.zeros(self.kg_emb_dim, device=self.device)

    def _pool_local(self, x, tot2sub, query_nodes, none_if_empty=False):
        if x is None:
            return self._empty_row(none_if_empty)
        rows = [tot2sub[v] for v in dict.fromkeys(query_nodes) if v in tot2sub]
        if not rows:
            return self._empty_row(none_if_empty)
        return x[rows].mean(dim=0)

    def _gather_dense(self, xg, query_nodes):
        rows = sorted({v for v in query_nodes if 0 <= v < xg.size(0)})
        return xg[rows] if rows else None

    def _pool_dense(self, xg, query_nodes, none_if_empty=False):
        g = self._gather_dense(xg, query_nodes)
        return self._empty_row(none_if_empty) if g is None else g.mean(dim=0)

    def _personal_hyperedges(self, field, hist_field_sessions, item_sessions=None):
        """H^P_f node sets + the readout node set Q^P_f.

        * field ``item``  -> one hyperedge per historical session (its item set),
          no expansion (spec Eq. 1, ``H^P_I``).
        * field ``entity`` / ``word`` -> one star hyperedge ``{v} u N^(k)(v)`` per
          node ``v`` mentioned across the historical sessions, expanded k-hop over
          the corresponding auxiliary KG (``entity_adj`` / ``word_adj``). For
          field ``entity`` with the review-hypergraph enabled, also appends
          ``{i} u E_top^+(R_i)`` for every historical item ``i`` (Eq. 1r),
          nested into the same incidence as the KG-expansion (Eq. 1s: N^P_E
          = [N^{P,KG}_E | N^{P,rev}_E], one shared HConv call for both).
        """
        if field == "item":
            edges = [sorted(s) for s in hist_field_sessions if s]
            q = sorted({v for s in hist_field_sessions for v in s})
            return edges, q
        seeds = sorted({v for s in hist_field_sessions for v in s})
        adj = self._field_adj(field)
        edges = [
            sorted(khop_closure([v], adj, self.k_hop, max_neighbors=self.khop_cap))
            for v in seeds
        ]
        if field == "entity" and self.use_review_hypergraph and self.review_index and item_sessions:
            i_p = sorted({v for s in item_sessions for v in s})
            edges += [
                sorted({i} | set(self.review_index[i]))
                for i in i_p
                if self.review_index.get(i)
            ]
        q = sorted({v for e in edges for v in e})
        return edges, q

    def _cpc_field_preferences(self, batch, field_x0, xg_by_field):
        """Per sample -> (fused rows, context embedding).

        ``fusion_mode == "field"``: rows is (3, d) = fused (P_I, P_E, P_W).
        ``fusion_mode == "rows7"``: rows is (n_rows<=7, d) = non-empty pooled
        rows [r^C_I,r^C_E,r^C_W, r^P_I,r^P_E,r^P_W, r^G] (spec v2.2 Eq. 11).
        """
        rows7 = self.fusion_mode == "rows7"
        bsz = len(batch["conv_id"])
        fused_all, ctx_all = [], []
        for b in range(bsz):
            target = int(batch["item"][b]) if "item" in batch else None
            has_history = bool(batch["has_history"][b])
            per_field = []
            c_rows, p_rows, g_parts, g_ids = [], [], [], []
            for field in self._FIELDS:
                x0 = field_x0[field]
                readout = batch[f"readout_{field}"][b]

                # scope C: sliding-window hyperedges (spec v2.1 Eq. 2)
                if "C" in self.scopes:
                    ctx_edges = sliding_window_hyperedges(
                        batch[f"context_turn_{field}"][b], self.context_window_w
                    )
                    xc, mc = self._scope_hconv("C", ctx_edges, field, x0, exclude=target)
                    p_c = self._pool_local(xc, mc, readout, none_if_empty=rows7)
                else:
                    p_c = self._empty_row(rows7)

                # scope P
                p_used_history = has_history and "P" in self.scopes
                if p_used_history:
                    p_edges, q_p = self._personal_hyperedges(
                        field,
                        batch[f"history_session_{field}"][b],
                        item_sessions=batch["history_session_item"][b],
                    )
                    xp, mp = self._scope_hconv("P", p_edges, field, x0, exclude=target)
                    p_p = self._pool_local(xp, mp, q_p, none_if_empty=rows7)
                else:
                    p_p, q_p = (None if rows7 else p_c), []

                # scope G
                if "G" in self.scopes and xg_by_field is not None and self.g_unified:
                    off = field_x0["entity"].size(0) if field == "word" else 0
                    g_ids.extend(v + off for v in list(readout) + list(q_p))
                    p_g = None
                elif "G" in self.scopes and xg_by_field is not None:
                    q_g = list(readout) + list(q_p)
                    if rows7:
                        g = self._gather_dense(xg_by_field[field], q_g)
                        if g is not None:
                            g_parts.append(g)
                        p_g = None
                    else:
                        p_g = self._pool_dense(xg_by_field[field], q_g)
                else:
                    p_g = self._empty_row(rows7)

                if rows7:
                    c_rows.append(p_c)
                    p_rows.append(p_p)
                else:
                    per_field.append(
                        self.fusion.fuse(field, p_c, p_p, p_g, has_history=p_used_history)
                    )
            if rows7:
                # r^G: unified Â^G (Eq. 9-10) if loaded, else the interim mean
                # over Q^G gathered across the three per-field Â^G_f.
                if self.g_unified and xg_by_field is not None:
                    # Eq. 9-10: Q^G over the unified index space, one mean
                    g = self._gather_dense(xg_by_field["G"], g_ids)
                    r_g = None if g is None else g.mean(0)
                else:
                    r_g = torch.cat(g_parts, 0).mean(0) if g_parts else None
                rows = [r for r in c_rows + p_rows + [r_g] if r is not None]
                fused_all.append(
                    torch.stack(rows, 0)
                    if rows
                    else torch.zeros(0, self.kg_emb_dim, device=self.device)
                )
            else:
                fused_all.append(torch.stack(per_field, dim=0))
            ctx_ids = [v for v in dict.fromkeys(batch["readout_entity"][b])]
            ctx_all.append(field_x0["entity"][ctx_ids] if ctx_ids else None)
        return fused_all, ctx_all

    def _field_x0(self, item_embedding, entity_embedding, token_embedding):
        return {
            "item": item_embedding,
            "entity": entity_embedding,
            "word": token_embedding[: self.n_word],
        }

    def _collective_by_field(self, field_x0):
        if "G" not in self.scopes or self.collective is None:
            return None
        if self.g_unified:
            # Eq. 8G: one conv over V^G. shared: W_G := W_E (tied to the
            # entity-field HConv layers); separate: G's own stack. Word ids sit at offset n_entity.
            x0 = torch.cat([field_x0["entity"], field_x0["word"]], dim=0)
            stack = self._stack_for("G", "G" if self.scope_weights == "separate" else "entity")
            return {"G": self.collective.run("G", x0, stack)}
        return {
            f: self.collective.run(f, field_x0[f], self._stack_for("G", f))
            for f in self._FIELDS
        }

    def _get_hllm_embedding(self, tot_embedding, hllm_hyper_graph, adj, conv):
        hllm_hyper_edge_A = []
        hllm_hyper_edge_B = []
        for idx, hyper_edge in enumerate(hllm_hyper_graph):
            hllm_hyper_edge_A += [item for item in hyper_edge]
            hllm_hyper_edge_B += [idx] * len(hyper_edge)

        hllm_items = list(set(hllm_hyper_edge_A))
        sub_item2id = {item: idx for idx, item in enumerate(hllm_items)}
        sub_embedding = tot_embedding[hllm_items]

        hllm_hyper_edge = [
            [sub_item2id[item] for item in hllm_hyper_edge_A],
            hllm_hyper_edge_B,
        ]
        hllm_hyper_edge = torch.LongTensor(hllm_hyper_edge).to(self.device)

        embedding = conv(sub_embedding, hllm_hyper_edge)

        return embedding

    def process_hllm(self, hllm_data, id_dict):
        res_data = []
        for raw_hyper_grapth in hllm_data:
            if not isinstance(raw_hyper_grapth, list):
                continue
            temp_hyper_grapth = []
            for meta_data in raw_hyper_grapth:
                if not isinstance(meta_data, int):
                    continue
                if meta_data not in id_dict:
                    continue
                temp_hyper_grapth.append(id_dict[meta_data])
            res_data.append(temp_hyper_grapth)
        return res_data

    def _user_from_rows(self, rows, ctx):
        """Spec v2.2 3.4 cold-start handling around Eq. 12-13. In ``field`` mode
        ``rows`` is never empty so this reduces to ``_attention_and_gating``."""
        if rows.size(0) == 0:
            if ctx is None:
                return torch.zeros(self.kg_emb_dim, device=self.device)
            return ctx.mean(dim=0)
        return self._attention_and_gating(rows, ctx)

    # 获取用户编码
    def encode_user(self, batch, item_embedding, entity_embedding, token_embedding):
        """CPC user encoder: per field fuse the C/P/G pooled preference vectors
        (Eq. 12) into P_I/P_E/P_W, then gate with the current-dialogue entities
        (Eq. 13, minus the review Transformer P_r which this codebase lacks)."""
        field_x0 = self._field_x0(item_embedding, entity_embedding, token_embedding)
        xg_by_field = self._collective_by_field(field_x0)
        fused_all, ctx_all = self._cpc_field_preferences(batch, field_x0, xg_by_field)

        user_repr_list = [
            self._user_from_rows(fused_all[i], ctx_all[i])
            for i in range(len(fused_all))
        ]
        return torch.stack(user_repr_list, dim=0), xg_by_field

    def _candidate_table(self, entity_embedding, xg_by_field):
        """E_I for the recommendation head (Eq. 14a / 14b)."""
        if self.ei_mode != "xg" or xg_by_field is None or not self.item_entity_ids:
            return entity_embedding
        if self.g_unified:
            return entity_embedding + xg_by_field["G"][: entity_embedding.size(0)]
        xg_i = xg_by_field["item"]  # (n_entity, d) -- item field lives in entity space
        return entity_embedding + xg_i

    def _word_table(self):
        """Word X^(0). ``word_rgcn: false`` skips the RGCN, which runs over the
        *entity* KG and so couples word id k to entity id k (control for the
        word-index issue, spec checklist 0)."""
        if not self.word_rgcn:
            return self.word_embedding.weight
        return self.word_encoder(
            self.word_embedding.weight, self.edge_idx, self.edge_type
        )

    # 推荐模块
    def recommend(self, batch, mode):
        item = batch["item"]
        item_embedding = self.item_encoder(
            self.entity_embedding.weight, self.edge_idx, self.edge_type
        )
        entity_embedding = self.entity_encoder(
            self.entity_embedding.weight, self.edge_idx, self.edge_type
        )
        token_embedding = self._word_table()

        user_embedding, xg_by_field = self.encode_user(
            batch, item_embedding, entity_embedding, token_embedding
        )

        e_i = self._candidate_table(entity_embedding, xg_by_field)
        scores = F.linear(user_embedding, e_i, self.rec_bias.bias)
        loss = self.rec_loss(scores, item)
        return loss, scores

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def freeze_parameters(self):
        freeze_models = [
            self.entity_embedding,
            self.token_embedding,
            self.item_encoder,
            self.entity_encoder,
            self.word_encoder,
            self.hconv,
            *([self.hconv_p, self.hconv_g] if self.scope_weights == "separate" else []),
            self.fusion,
            self.item_attn,
            self.rec_bias,
        ]
        if self.pooling == "Attn":
            freeze_models.append(self.kg_attn)
            freeze_models.append(self.kg_attn_his)
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    # 获取超图后数据
    def encode_session(self, batch, item_embedding, entity_embedding, token_embedding):
        """Build the decoder cross-attention memory from the CPC scope vectors.

        Return: session_repr (batch_size, seq_len, token_emb_dim),
                mask (batch_size, seq_len)  (True = real token)
        """
        field_x0 = self._field_x0(item_embedding, entity_embedding, token_embedding)
        xg_by_field = self._collective_by_field(field_x0)
        fused_all, ctx_all = self._cpc_field_preferences(batch, field_x0, xg_by_field)

        session_repr_list = []
        for i in range(len(fused_all)):
            parts = [fused_all[i]]  # (3, d) -- P_I, P_E, P_W
            if ctx_all[i] is not None:
                parts.append(ctx_all[i])
            session_repr_list.append(torch.cat(parts, dim=0))

        batch_seq_len = max(
            [
                session_repr.size(0)
                for session_repr in session_repr_list
                if session_repr is not None
            ]
        )
        mask_list = []
        for i in range(len(session_repr_list)):
            if session_repr_list[i] is None:
                mask_list.append([False] * batch_seq_len)
                zero_repr = torch.zeros(
                    (batch_seq_len, self.kg_emb_dim),
                    device=self.device,
                    dtype=torch.float,
                )
                session_repr_list[i] = zero_repr
            else:
                mask_list.append(
                    [False] * (batch_seq_len - session_repr_list[i].size(0))
                    + [True] * session_repr_list[i].size(0)
                )
                zero_repr = torch.zeros(
                    (batch_seq_len - session_repr_list[i].size(0), self.kg_emb_dim),
                    device=self.device,
                    dtype=torch.float,
                )
                session_repr_list[i] = torch.cat(
                    (zero_repr, session_repr_list[i]), dim=0
                )

        session_repr_embedding = torch.stack(session_repr_list, dim=0)
        session_repr_embedding = self.entity_to_token(session_repr_embedding)
        # print("session_repr_embedding.shape", session_repr_embedding.shape) # [6, 7, 300]
        return session_repr_embedding, torch.tensor(
            mask_list, device=self.device, dtype=torch.bool
        )

    # 生成对话
    def decode_forced(
        self,
        related_encoder_state,
        context_encoder_state,
        session_state,
        user_embedding,
        resp,
    ):
        bsz = resp.size(0)
        seqlen = resp.size(1)
        inputs = resp.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(
            inputs, related_encoder_state, context_encoder_state, session_state
        )
        token_logits = F.linear(latent, self.token_embedding.weight)
        user_logits = self.user_proj_2(
            torch.relu(self.user_proj_1(user_embedding))
        ).unsqueeze(1)

        user_latent = self.entity_to_token(user_embedding)
        user_latent = user_latent.unsqueeze(1).expand(-1, seqlen, -1)
        copy_latent = torch.cat((user_latent, latent), dim=-1)
        copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
        if self.dataset == "HReDial":
            copy_logits = copy_logits * self.hredial_copy_mask.unsqueeze(0).unsqueeze(
                0
            )  # not for tg-redial
        sum_logits = token_logits + user_logits + copy_logits
        _, preds = sum_logits.max(dim=-1)
        return sum_logits, preds

    # 生成对话 - test
    def decode_greedy(
        self,
        related_encoder_state,
        context_encoder_state,
        session_state,
        user_embedding,
    ):
        bsz = context_encoder_state[0].shape[0]
        xs = self._starts(bsz)
        # KV-cached decoding: feed only the newest token each step and let
        # incr_state carry cached self/cross-attention K,V forward, instead
        # of reprocessing the whole growing sequence from scratch every step
        # (see TransformerDecoderKG.forward_incremental - ~25x less
        # attention work over a typical greedy decode, verified numerically
        # identical to the old full-reprocess path via torch.allclose).
        new_token = xs
        incr_state = None
        logits = []
        for i in range(self.longest_label):
            scores, incr_state = self.decoder.forward_incremental(
                new_token,
                related_encoder_state,
                context_encoder_state,
                session_state,
                incr_state,
            )
            token_logits = F.linear(scores, self.token_embedding.weight)
            user_logits = self.user_proj_2(
                torch.relu(self.user_proj_1(user_embedding))
            ).unsqueeze(1)

            user_latent = self.entity_to_token(user_embedding)
            user_latent = user_latent.unsqueeze(1).expand(-1, 1, -1)
            copy_latent = torch.cat((user_latent, scores), dim=-1)
            copy_logits = self.copy_proj_2(torch.relu(self.copy_proj_1(copy_latent)))
            if self.dataset == "HReDial":
                copy_logits = copy_logits * self.hredial_copy_mask.unsqueeze(
                    0
                ).unsqueeze(0)  # not for tg-redial
            sum_logits = token_logits + user_logits + copy_logits
            probs, preds = sum_logits.max(dim=-1)
            logits.append(scores)
            xs = torch.cat([xs, preds], dim=1)
            new_token = preds
            # check if everyone has generated an end token
            all_finished = (
                (xs == self.end_token_idx).sum(dim=1) > 0
            ).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    # 对话模块训练
    def converse(self, batch, mode):
        response = batch["response"]
        related_tokens = batch["related_tokens"]
        context_tokens = batch["context_tokens"]

        item_embedding = self.item_encoder(
            self.entity_embedding.weight, self.edge_idx, self.edge_type
        )
        entity_embedding = self.entity_encoder(
            self.entity_embedding.weight, self.edge_idx, self.edge_type
        )
        token_embedding = self._word_table()

        # 获取对话编码
        session_state = self.encode_session(
            batch, item_embedding, entity_embedding, token_embedding
        )

        # 获取用户编码
        user_embedding, _ = self.encode_user(
            batch, item_embedding, entity_embedding, token_embedding
        )  # (batch_size, emb_dim)

        # 获取 X_c、X_h
        related_encoder_state = self.related_encoder(related_tokens)
        context_encoder_state = self.context_encoder(context_tokens)

        # 对话生成
        if mode != "test":
            self.longest_label = max(self.longest_label, response.shape[1])
            logits, preds = self.decode_forced(
                related_encoder_state,
                context_encoder_state,
                session_state,
                user_embedding,
                response,
            )
            logits = logits.view(-1, logits.shape[-1])
            labels = response.view(-1)
            return self.conv_loss(logits, labels), preds
        else:
            _, preds = self.decode_greedy(
                related_encoder_state,
                context_encoder_state,
                session_state,
                user_embedding,
            )
            return preds

    # 推荐模块和对话模块分开训练
    def forward(self, batch, mode, stage):
        if len(self.gpu) >= 2:
            self.edge_idx = self.edge_idx.cuda(torch.cuda.current_device())
            self.edge_type = self.edge_type.cuda(torch.cuda.current_device())
        if stage == "conv":
            return self.converse(batch, mode)
        if stage == "rec":
            # start = perf_counter()
            res = self.recommend(batch, mode)
            # print(f"{perf_counter() - start:.2f}")
            return res
