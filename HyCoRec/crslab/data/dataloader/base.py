# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
from abc import ABC
from math import ceil

import torch
from loguru import logger
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm


class _SequenceDataset(TorchDataset):
    """Thin ``torch.utils.data.Dataset`` wrapper around an in-memory list,
    so that batching/shuffling/(optionally) parallel prefetching can be
    delegated to ``torch.utils.data.DataLoader`` instead of a hand-rolled
    python loop."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BaseDataLoader(ABC):
    """Abstract class of dataloader

    Notes:
        ``'scale'`` can be set in config to limit the size of dataset.

        ``'num_workers'`` and ``'pin_memory'`` can be set in config to tune
        ``torch.utils.data.DataLoader`` prefetching; they default to ``0``
        (single-process, matching the historical behavior) and to whether
        CUDA is available, respectively.

    """

    def __init__(self, opt, dataset):
        """
        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: dataset

        """
        self.opt = opt
        self.dataset = dataset
        self.scale = opt.get("scale", 1)
        assert 0 < self.scale <= 1
        self.num_workers = opt.get("num_workers", os.cpu_count())
        self.pin_memory = opt.get("pin_memory", torch.cuda.is_available())
        # process_fn (e.g. rec_process_fn/conv_process_fn) is a deterministic
        # function of self.dataset, but get_data is invoked once per epoch.
        # Cache its (scaled) output per underlying function so the whole
        # dataset isn't rescanned/rebuilt every epoch.
        self._processed_cache = {}

    def _get_processed_dataset(self, process_fn):
        dataset = self.dataset if process_fn is None else None
        key = getattr(process_fn, "__func__", process_fn)
        if key not in self._processed_cache:
            if process_fn is not None:
                dataset = process_fn()
            self._processed_cache[key] = dataset[: ceil(len(dataset) * self.scale)]
        return self._processed_cache[key]

    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        dataset = self._get_processed_dataset(process_fn)
        logger.debug(f"[Dataset size: {len(dataset)}]")

        loader = TorchDataLoader(
            _SequenceDataset(dataset),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=batch_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        for batch in tqdm(loader, total=len(loader)):
            if batch is False:
                continue
            else:
                yield (batch)

    def get_conv_data(self, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(
            self.conv_batchify, batch_size, shuffle, self.conv_process_fn
        )

    def get_rec_data(self, batch_size, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(
            self.rec_batchify, batch_size, shuffle, self.rec_process_fn
        )

    def get_policy_data(self, batch_size, shuffle=True):
        """get_data wrapper for policy.

        You can implement your own process_fn in ``self.policy_process_fn``, batch_fn in ``policy_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for policy.

        """
        return self.get_data(
            self.policy_batchify, batch_size, shuffle, self.policy_process_fn
        )

    def conv_process_fn(self):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError("dataloader must implement conv_batchify() method")

    def rec_process_fn(self):
        """Process whole data for recommendation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def rec_batchify(self, batch):
        """batchify data for recommendation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train recommendation part.
        """
        raise NotImplementedError("dataloader must implement rec_batchify() method")

    def policy_process_fn(self):
        """Process whole data for policy before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def policy_batchify(self, batch):
        """batchify data for policy after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train policy part.
        """
        raise NotImplementedError("dataloader must implement policy_batchify() method")

    def retain_recommender_target(self):
        """keep data whose role is recommender.

        Returns:
            Recommender part of ``self.dataset``.

        """
        dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict["role"] == "Recommender":
                dataset.append(conv_dict)
        return dataset

    def rec_interact(self, data):
        """process user input data for system to recommend.

        Args:
            data: user input data.

        Returns:
            data for system to recommend.
        """

    def conv_interact(self, data):
        """Process user input data for system to converse.

        Args:
            data: user input data.

        Returns:
            data for system in converse.
        """
