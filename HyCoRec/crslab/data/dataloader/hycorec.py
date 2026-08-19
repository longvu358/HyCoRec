# -*- encoding: utf-8 -*-
# @Time    :   2021/5/26
# @Author  :   Chenzhan Shang
# @email   :   czshang@outlook.com

import pickle
import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, truncate, merge_utt


class HyCoRecDataLoader(BaseDataLoader):
    """Dataloader for model KBRD.

    Notes:
        You can set the following parameters in config:

        - ``"context_truncate"``: the maximum length of context.
        - ``"response_truncate"``: the maximum length of response.
        - ``"entity_truncate"``: the maximum length of mentioned entities in context.

        The following values must be specified in ``vocab``:

        - ``"pad"``
        - ``"start"``
        - ``"end"``
        - ``"pad_entity"``

        the above values specify the id of needed special token.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.pad_token_idx = vocab["tok2ind"]["__pad__"]
        self.start_token_idx = vocab["tok2ind"]["__start__"]
        self.end_token_idx = vocab["tok2ind"]["__end__"]
        self.split_token_idx = vocab["tok2ind"].get("_split_", None)
        self.related_truncate = opt.get("related_truncate", None)
        self.context_truncate = opt.get("context_truncate", None)
        self.response_truncate = opt.get("response_truncate", None)
        self.entity_truncate = opt.get("entity_truncate", None)
        self.hyperedge_window_k = opt.get("hyperedge_window_k", None)
        self.review_entity2id = vocab["entity2id"]
        return

    @staticmethod
    def _flatten_turns(turns, k=None, dedup=True):
        """Flatten a per-turn list-of-lists into a single list.

        k=None flattens the full history (global branch, matches the
        pre-windowing behavior). k=N keeps only the last N turns (local
        branch). dedup=True keeps first-occurrence order, matching the old
        entity/word accumulation; dedup=False preserves duplicates, matching
        the old item accumulation.
        """
        if k is None:
            window = turns
        elif k <= 0:
            window = []
        else:
            window = turns[-k:]
        flat = [x for turn in window for x in turn]
        return list(dict.fromkeys(flat)) if dedup else flat

    @staticmethod
    def _global_related(conv_dict, global_key, turns_key, dedup):
        """Full-history related list: use the dataset's precomputed
        incremental accumulator (see hredial.py's _augment_and_add) when
        available, since it avoids re-flattening the whole per-conversation
        history at every turn. Datasets that don't provide it fall back to
        flattening conv_dict[turns_key] here.
        """
        precomputed = conv_dict.get(global_key)
        if precomputed is not None:
            return precomputed
        return HyCoRecDataLoader._flatten_turns(conv_dict[turns_key], dedup=dedup)

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict["role"] == "Recommender":
                related_item = self._global_related(conv_dict, "item_global", "item", dedup=False)
                related_entity = self._global_related(conv_dict, "entity_global", "entity", dedup=True)
                related_word = self._global_related(conv_dict, "word_global", "word", dedup=True)
                related_item_local = self._flatten_turns(conv_dict["item"], k=self.hyperedge_window_k, dedup=False)
                related_entity_local = self._flatten_turns(conv_dict["entity"], k=self.hyperedge_window_k)
                related_word_local = self._flatten_turns(conv_dict["word"], k=self.hyperedge_window_k)
                for item in conv_dict["items"]:
                    augment_conv_dict = {
                        "conv_id": conv_dict["conv_id"],
                        "related_item": related_item,
                        "related_entity": related_entity,
                        "related_word": related_word,
                        "related_item_local": related_item_local,
                        "related_entity_local": related_entity_local,
                        "related_word_local": related_word_local,
                        "item": item,
                    }
                    augment_dataset.append(augment_conv_dict)

        return augment_dataset

    def rec_batchify(self, batch):
        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []
        batch_related_item_local = []
        batch_related_entity_local = []
        batch_related_word_local = []
        batch_movies = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_item.append(conv_dict["related_item"])
            batch_related_entity.append(conv_dict["related_entity"])
            batch_related_word.append(conv_dict["related_word"])
            batch_related_item_local.append(conv_dict["related_item_local"])
            batch_related_entity_local.append(conv_dict["related_entity_local"])
            batch_related_word_local.append(conv_dict["related_word_local"])
            batch_movies.append(conv_dict["item"])
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "conv_id": batch_conv_id,
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "related_item_local": batch_related_item_local,
            "related_entity_local": batch_related_entity_local,
            "related_word_local": batch_related_word_local,
            "item": torch.tensor(batch_movies, dtype=torch.long),
        }

        return res

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_related_tokens = []
        batch_context_tokens = []

        batch_related_item = []
        batch_related_entity = []
        batch_related_word = []
        batch_related_item_local = []
        batch_related_entity_local = []
        batch_related_word_local = []

        batch_response = []
        batch_conv_id = []
        for conv_dict in batch:
            batch_related_tokens.append(
                truncate(conv_dict["tokens"][-1], self.related_truncate, truncate_tail=False)
            )
            batch_context_tokens.append(
                truncate(merge_utt(
                    conv_dict["tokens"],
                    start_token_idx=self.start_token_idx,
                    split_token_idx=self.split_token_idx,
                    final_token_idx=self.end_token_idx
                ), self.context_truncate, truncate_tail=False)
            )

            batch_related_item.append(self._global_related(conv_dict, "item_global", "item", dedup=False))
            batch_related_entity.append(self._global_related(conv_dict, "entity_global", "entity", dedup=True))
            batch_related_word.append(self._global_related(conv_dict, "word_global", "word", dedup=True))
            batch_related_item_local.append(self._flatten_turns(conv_dict["item"], k=self.hyperedge_window_k, dedup=False))
            batch_related_entity_local.append(self._flatten_turns(conv_dict["entity"], k=self.hyperedge_window_k))
            batch_related_word_local.append(self._flatten_turns(conv_dict["word"], k=self.hyperedge_window_k))

            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict["response"], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            batch_conv_id.append(conv_dict["conv_id"])

        res = {
            "related_tokens": padded_tensor(batch_related_tokens, self.pad_token_idx, pad_tail=False),
            "context_tokens": padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
            "related_item": batch_related_item,
            "related_entity": batch_related_entity,
            "related_word": batch_related_word,
            "related_item_local": batch_related_item_local,
            "related_entity_local": batch_related_entity_local,
            "related_word_local": batch_related_word_local,
            "response": padded_tensor(batch_response, self.pad_token_idx),
            "conv_id": batch_conv_id,
        }

        return res

    def policy_batchify(self, *args, **kwargs):
        pass
