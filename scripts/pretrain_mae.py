"""Self-supervised masked-autoencoder pretraining of the ViT encoder (no labels used).

Uses the TRAIN split only. The saved state_dict is the encoder (a full ViT, classifier head
included but untouched) and loads straight into `train.py --model vit --pretrained`.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from terraforge.data.torch_dataset import CachedEuroSATDataset
from terraforge.models.mae import MaskedAutoencoder
from terraforge.models.vit import ViT
from terraforge.training.schedule import warmup_cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--subset", type=int, default=None, help="cap samples (smoke runs)")
    ap.add_argument("--out", default="runs/mae_encoder.pt")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Path(a.processed)
    ds = CachedEuroSATDataset(proc / "cache", proc / "band_stats.json", "train", augment=True)
    if a.subset:
        ds.idx = ds.idx[:: max(1, len(ds.idx) // a.subset)][: a.subset]
    dl = DataLoader(ds, a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers,
                    pin_memory=dev == "cuda")
    mae = MaskedAutoencoder(ViT()).to(dev)
    opt = torch.optim.AdamW(mae.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, total, total // 20))
    for ep in range(a.epochs):
        run, n = 0.0, 0
        for x, _ in dl:  # labels are deliberately ignored
            x = x.to(dev)
            opt.zero_grad(set_to_none=True)
            loss, _, _ = mae(x, a.mask_ratio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item() * len(x); n += len(x)
        print(f"[mae] epoch {ep+1}/{a.epochs} recon_loss {run/n:.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(mae.encoder.state_dict(), a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
