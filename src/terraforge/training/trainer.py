"""Model-agnostic trainer."""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from terraforge.training.evaluation import summarize


class ModelTrainer:
    def __init__(self, model: nn.Module, n_classes: int, lr: float = 1e-3,
                 weight_decay: float = 1e-2, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.n_classes = n_classes
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()

    def fit_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad(set_to_none=True)
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item() * len(y)
            n += len(y)
        return total / n

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        ys, ps = [], []
        t0 = time.perf_counter()
        for x, y in loader:
            ps.append(self.model(x.to(self.device)).argmax(1).cpu().numpy())
            ys.append(y.numpy())
        y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
        out = summarize(y_true, y_pred, self.n_classes)
        out["ms_per_sample"] = 1000 * (time.perf_counter() - t0) / len(y_true)
        return out
