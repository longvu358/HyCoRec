"""Thin helpers for single-node DistributedDataParallel (launched by ``torchrun``).

Everything degrades to a no-op when ``WORLD_SIZE`` is unset/1, so the
single-process and CPU paths are unchanged.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank():
    return int(os.environ.get("RANK", "0"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_ddp():
    return world_size() > 1


def is_main():
    return rank() == 0


def setup():
    """Bind this process to its GPU and join the process group (idempotent)."""
    if not is_ddp() or dist.is_initialized():
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        raise RuntimeError("DDP needs GPUs: pass -g 0,1,... (not -1)")
    torch.cuda.set_device(local_rank())
    # rank 0 evaluates alone while the others wait, so keep the timeout generous
    dist.init_process_group("nccl", timeout=timedelta(hours=2))


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if is_ddp():
        dist.barrier()


def broadcast(obj):
    """Return rank 0's ``obj`` on every rank."""
    if not is_ddp():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=0)
    return box[0]
