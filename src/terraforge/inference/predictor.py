"""Inference engine: batching, calibration, embeddings, optional Hub download.

The model is loaded once and shared. Batches are chunked so a large request cannot
exhaust memory, and chunks run in a thread pool: PyTorch releases the GIL inside
kernels, so threads give real parallelism for CPU inference without the cost of
copying the model into worker processes.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from terraforge.data.eurosat import CLASSES
from terraforge.models.cnn import SmallCNN
from terraforge.models.vit import ViT

ARCHS = {"cnn": SmallCNN, "vit": ViT}


class InferenceEngine:
    def __init__(self, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray,
                 temperature: float = 1.0, version: str = "untrained-demo",
                 chunk: int = 32, workers: int = 2, conformal: dict | None = None):
        self.model = model.eval()
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1, 1) + 1e-6
        self.temperature = temperature
        self.version = version
        self.chunk = chunk
        # {"method": "lac"|"aps", "q": float | list (Mondrian), "alpha": float} fitted on the
        # VALIDATION split by scripts/analyze.py; None disables prediction sets.
        self.conformal = conformal
        self.pool = ThreadPoolExecutor(max_workers=workers)

    @classmethod
    def load(cls, arch: str, weights: str | Path | None, stats: dict,
             temperature: float = 1.0, **kw) -> "InferenceEngine":
        model = ARCHS[arch]()
        version = f"{arch}-untrained-demo"
        if weights:
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            version = f"{arch}-{Path(weights).stem}"
        return cls(model, np.array(stats["mean"]), np.array(stats["std"]),
                   temperature, version, **kw)

    @classmethod
    def from_hub(cls, repo_id: str, arch: str, stats: dict, filename: str = "model.pt", **kw):
        """Fetch weights from the Hugging Face Hub (cached locally by huggingface_hub)."""
        from huggingface_hub import hf_hub_download
        return cls.load(arch, hf_hub_download(repo_id, filename), stats, **kw)

    def _prep(self, x: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.ndim != 4 or t.shape[1] != self.mean.shape[1]:
            raise ValueError(f"expected (N,{self.mean.shape[1]},H,W), got {tuple(t.shape)}")
        return (t - self.mean) / self.std

    @torch.no_grad()
    def _chunk_predict(self, t: torch.Tensor):
        logits = self.model(t) / self.temperature
        emb = self.model.features(t) if isinstance(self.model, ViT) else self.model.embed(t)
        return logits.softmax(-1).numpy(), emb.numpy()

    def predict(self, x: np.ndarray) -> dict:
        """x: raw reflectance-scale patches (N,C,H,W). Returns probs, labels, embeddings."""
        t0 = time.perf_counter()
        t = self._prep(x)
        parts = list(self.pool.map(self._chunk_predict, t.split(self.chunk)))
        probs = np.concatenate([p for p, _ in parts])
        emb = np.concatenate([e for _, e in parts])
        idx = probs.argmax(1)
        sets = None
        if self.conformal:
            from terraforge.training import conformal as cf
            mask = cf.prediction_sets(probs, np.asarray(self.conformal["q"]), self.conformal["method"])
            sets = [cf.named_sets(m, CLASSES, p) for m, p in zip(mask, probs)]
        return {
            "probs": probs, "embeddings": emb, "index": idx,
            "labels": [CLASSES[i] for i in idx], "confidence": probs.max(1),
            "latency_ms": 1000 * (time.perf_counter() - t0), "model_version": self.version,
            "sets": sets,
        }
