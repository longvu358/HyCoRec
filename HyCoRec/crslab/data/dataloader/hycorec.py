# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import (
    add_start_end_token_idx,
    merge_utt,
    padded_tensor,
    truncate,
)


class HyCoRecDataLoader(BaseDataLoader):
    """Dataloader for HyCoRec / CPC-Hypergraph v2.

    Ships ragged id lists for the three preference scopes; the model builds the
    incidence matrices (scope C / P) or applies the precomputed propagation
    matrix (scope G):

    * ``context_turn_{item,entity,word}``  -- list-of-turns, turn-local (scope C)
    * ``history_session_{item,entity,word}`` -- list-of-sessions (scope P, D_u(d))
    * ``readout_{item,entity,word}``       -- flat V_f(d<=t) (pooling supports Q^C, Q^G)
    * ``has_history``                      -- whether D_u(d) is non-empty (cold-start)
    """

    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab["tok2ind"]["__pad__"]
        self.start_token_idx = vocab["tok2ind"]["__start__"]
        self.end_token_idx = vocab["tok2ind"]["__end__"]
        self.split_token_idx = vocab["tok2ind"].get("_split_", None)
        self.related_truncate = opt.get("related_truncate", None)
        self.context_truncate = opt.get("context_truncate", None)
        self.response_truncate = opt.get("response_truncate", None)
        self.entity_truncate = opt.get("entity_truncate", None)

    @staticmethod
    def _flatten(turns, dedup=True):
        flat = [x for turn in turns for x in turn]
        return list(dict.fromkeys(flat)) if dedup else flat

    def _scope_fields(self, conv_dict):
        """Common per-sample scope payload (shared by rec and conv)."""
        history = conv_dict.get("history_session_items") or []
        return {
            "context_turn_item": conv_dict["item"],
            "context_turn_entity": conv_dict["entity"],
            "context_turn_word": conv_dict["word"],
            "history_session_item": conv_dict.get("history_session_items") or [],
            "history_session_entity": conv_dict.get("history_session_entities") or [],
            "history_session_word": conv_dict.get("history_session_words") or [],
            "readout_item": conv_dict.get("item_global")
            or self._flatten(conv_dict["item"], dedup=False),
            "readout_entity": conv_dict.get("entity_global")
            or self._flatten(conv_dict["entity"]),
            "readout_word": conv_dict.get("word_global")
            or self._flatten(conv_dict["word"]),
            "has_history": len(history) > 0,
            "conv_id": conv_dict["conv_id"],
        }

    # ---- recommendation ----------------------------------------------------
    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict["role"] != "Recommender":
                continue
            scope = self._scope_fields(conv_dict)
            for item in conv_dict["items"]:
                augment_dataset.append({**scope, "item": item})
        return augment_dataset

    @staticmethod
    def _collate_scope(batch):
        keys = (
            "context_turn_item",
            "context_turn_entity",
            "context_turn_word",
            "history_session_item",
            "history_session_entity",
            "history_session_word",
            "readout_item",
            "readout_entity",
            "readout_word",
        )
        out = {k: [d[k] for d in batch] for k in keys}
        out["has_history"] = [d["has_history"] for d in batch]
        out["conv_id"] = [d["conv_id"] for d in batch]
        return out

    def rec_batchify(self, batch):
        res = self._collate_scope(batch)
        res["item"] = torch.tensor([d["item"] for d in batch], dtype=torch.long)
        return res

    # ---- conversation ----------------------------------------------------
    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        res = self._collate_scope([self._scope_fields(d) for d in batch])
        batch_related_tokens, batch_context_tokens, batch_response = [], [], []
        for conv_dict in batch:
            batch_related_tokens.append(
                truncate(
                    conv_dict["tokens"][-1], self.related_truncate, truncate_tail=False
                )
            )
            batch_context_tokens.append(
                truncate(
                    merge_utt(
                        conv_dict["tokens"],
                        start_token_idx=self.start_token_idx,
                        split_token_idx=self.split_token_idx,
                        final_token_idx=self.end_token_idx,
                    ),
                    self.context_truncate,
                    truncate_tail=False,
                )
            )
            batch_response.append(
                add_start_end_token_idx(
                    truncate(conv_dict["response"], self.response_truncate - 2),
                    start_token_idx=self.start_token_idx,
                    end_token_idx=self.end_token_idx,
                )
            )
        res["related_tokens"] = padded_tensor(
            batch_related_tokens, self.pad_token_idx, pad_tail=False
        )
        res["context_tokens"] = padded_tensor(
            batch_context_tokens, self.pad_token_idx, pad_tail=False
        )
        res["response"] = padded_tensor(batch_response, self.pad_token_idx)
        return res

    def policy_batchify(self, *args, **kwargs):
        pass
