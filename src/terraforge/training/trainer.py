"""Model-agnostic trainer: AdamW, warmup+cosine LR, EMA weights, label smoothing, AMP."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from terraforge.training.evaluation import summarize
from terraforge.training.schedule import EMA, warmup_cosine


class ModelTrainer:
    def __init__(self, model: nn.Module, n_classes: int, lr: float = 1e-3,
                 weight_decay: float = 1e-2, label_smoothing: float = 0.0,
                 total_steps: int | None = None, warmup_steps: int = 0,
                 ema_decay: float | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_classes = n_classes
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.sched = None
        if total_steps:
            self.sched = torch.optim.lr_scheduler.LambdaLR(
                self.opt, lambda s: warmup_cosine(s, total_steps, warmup_steps))
        self.ema = EMA(model, ema_decay) if ema_decay else None
        # AMP only on CUDA: CPU autocast gives little and complicates determinism.
        self.amp = self.device == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

    @property
    def eval_model(self) -> nn.Module:
        """EMA weights if enabled (usually generalise better), else the raw model."""
        return self.ema.shadow if self.ema else self.model

    def fit_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            with torch.autocast(self.device, enabled=self.amp):
                loss = self.loss_fn(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.sched:
                self.sched.step()
            if self.ema:
                self.ema.update(self.model)
            total += loss.item() * len(y)
            n += len(y)
        return total / n

    @torch.no_grad()
    def logits(self, loader: DataLoader):
        """(logits, labels) over a loader using the evaluation model."""
        m = self.eval_model.eval()
        outs, ys = [], []
        for x, y in loader:
            outs.append(m(x.to(self.device)).float().cpu())
            ys.append(y)
        return torch.cat(outs), torch.cat(ys)

    def evaluate(self, loader: DataLoader) -> dict:
        t0 = time.perf_counter()
        logits, y = self.logits(loader)
        out = summarize(y.numpy(), logits.argmax(1).numpy(), self.n_classes)
        out["ms_per_sample"] = 1000 * (time.perf_counter() - t0) / len(y)
        return out
