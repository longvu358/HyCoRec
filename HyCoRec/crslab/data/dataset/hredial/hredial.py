# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2021/1/3, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com, sdzyh002@gmail

r"""
ReDial
======
References:
    Li, Raymond, et al. `"Towards deep conversational recommendations."`_ in NeurIPS 2018.

.. _`"Towards deep conversational recommendations."`:
   https://papers.nips.cc/paper/2018/hash/800de15c79c8d840f4e78d3af937d4d4-Abstract.html

"""

import json
import os
import pickle as pkl

from loguru import logger

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from crslab.data.dataset.hredial.resources import resources
from crslab.data.dataset.hycorec_common import process_grouped_raw_data


class HReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        dpath = os.path.join(DATASET_PATH, "hredial", tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            "tok2ind": self.tok2ind,
            "ind2tok": self.ind2tok,
            "entity2id": self.entity2id,
            "id2entity": self.id2entity,
            "vocab_size": len(self.tok2ind),
            "n_entity": self.n_entity,
        }

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(
            os.path.join(self.dpath, "train_data.json"), "r", encoding="utf-8"
        ) as f:
            train_data = json.load(f)
            logger.debug(
                f"[Load train data from {os.path.join(self.dpath, 'train_data.json')}]"
            )
        with open(
            os.path.join(self.dpath, "valid_data.json"), "r", encoding="utf-8"
        ) as f:
            valid_data = json.load(f)
            logger.debug(
                f"[Load valid data from {os.path.join(self.dpath, 'valid_data.json')}]"
            )
        with open(
            os.path.join(self.dpath, "test_data.json"), "r", encoding="utf-8"
        ) as f:
            test_data = json.load(f)
            logger.debug(
                f"[Load test data from {os.path.join(self.dpath, 'test_data.json')}]"
            )

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(
            open(os.path.join(self.dpath, "token2id.json"), "r", encoding="utf-8")
        )
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

    def _load_other_data(self):
        # edge extension data
        self.conv2items = json.load(
            open(os.path.join(self.dpath, "conv2items.json"), "r", encoding="utf-8")
        )
        # dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, "entity2id.json"), "r", encoding="utf-8")
        )  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        self.side_data = pkl.load(open(os.path.join(self.dpath, "side_data.pkl"), "rb"))
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'dbpedia_subkg.json')}]"
        )

    def _data_preprocess(self, train_data, valid_data, test_data):
        processed_train_data = self._raw_data_process(train_data)
        logger.debug("[Finish train data process]")
        processed_valid_data = self._raw_data_process(valid_data)
        logger.debug("[Finish valid data process]")
        processed_test_data = self._raw_data_process(test_data)
        logger.debug("[Finish test data process]")
        processed_side_data = self.side_data
        logger.debug("[Finish side data process]")
        return (
            processed_train_data,
            processed_valid_data,
            processed_test_data,
            processed_side_data,
        )

    def _raw_data_process(self, raw_data):
        """Process the per-user-grouped raw data into per-turn samples.

        ``raw_data`` is ``list[list[conv]]`` grouped by user; the last conv of a
        group is the target session and the earlier ones are ``D_u(d)``. See
        ``crslab.data.dataset.hycorec_common``.
        """
        unk_idx = self.tok2ind.get("__unk__", 3)
        k_hist = self.opt.get("k_hist", 40)
        return process_grouped_raw_data(
            raw_data, self.tok2ind, self.entity2id, unk_idx, k_hist
        )
