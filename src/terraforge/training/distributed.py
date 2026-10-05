"""PyTorch DistributedDataParallel (DDP) utilities.

How DDP works: every process holds a full model replica and
sees a different shard of each batch (DistributedSampler). After backward(), gradients
are all-reduced (averaged) across processes - overlapped with the backward pass in
buckets - so every replica applies the identical update and stays in sync. Effective
batch size = per-process batch x world size, so LR should be scaled accordingly.

This module is backend-agnostic: `gloo` runs on CPU (used by the tests here), `nccl`
on GPUs. The same entry point is launched by torchrun or SLURM on a cluster
(see docs/hpc_training.md).
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


def setup(rank: int, world_size: int, backend: str = "gloo", port: str = "29511") -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def make_loader(dataset, batch_size: int, rank: int, world_size: int, epoch: int = 0):
    """Sharded loader. set_epoch changes the shuffle each epoch (otherwise every epoch
    reuses the same order - a classic DDP bug)."""
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, drop_last=True)
    sampler.set_epoch(epoch)
    return DataLoader(dataset, batch_size, sampler=sampler)


def _worker(rank: int, world_size: int, out_dir: str) -> None:
    """Smoke test: train a tiny model with DDP and check replicas stay identical."""
    setup(rank, world_size)
    torch.manual_seed(0)  # same init everywhere; DDP also broadcasts rank 0's weights
    model = DDP(nn.Linear(8, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    g = torch.Generator().manual_seed(rank)  # different data per rank
    for _ in range(5):
        x = torch.randn(16, 8, generator=g)
        y = torch.randint(0, 2, (16,), generator=g)
        opt.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()
        opt.step()
    flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    gathered = [torch.zeros_like(flat) for _ in range(world_size)]
    dist.all_gather(gathered, flat)
    if rank == 0:
        diff = max(float((gathered[0] - t).abs().max()) for t in gathered)
        with open(os.path.join(out_dir, "ddp_diff.txt"), "w") as f:
            f.write(str(diff))
    cleanup()


def run_ddp_smoke(world_size: int, out_dir: str) -> float:
    """Spawn `world_size` CPU processes; return max parameter divergence (expect ~0)."""
    mp.spawn(_worker, args=(world_size, out_dir), nprocs=world_size, join=True)
    with open(os.path.join(out_dir, "ddp_diff.txt")) as f:
        return float(f.read())
