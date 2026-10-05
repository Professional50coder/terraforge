"""Vector index over latent embeddings: "find patches that look like this one".

Why it earns its place: a classifier answers "which of 10 classes"; an embedding index
answers "what known scenes is this most similar to", which supports retrieval-based
explanation (RAG), near-duplicate/leakage detection, active-learning (find unlabeled
patches near a confusing one) and label-noise auditing.

Exact cosine search with numpy: for ~27k x 128-d vectors a brute-force matrix product
takes milliseconds, so an approximate index (FAISS/HNSW) would add complexity with no
benefit at this scale. The interface is deliberately small so one can be swapped in when
the corpus grows to millions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.meta: list[dict] = []

    def __len__(self):
        return len(self.meta)

    def add(self, embeddings: np.ndarray, meta: list[dict]) -> None:
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {embeddings.shape[1]}")
        if len(embeddings) != len(meta):
            raise ValueError("embeddings and meta length differ")
        self.vectors = np.vstack([self.vectors, _unit(embeddings.astype("float32"))])
        self.meta += meta

    def search(self, query: np.ndarray, k: int = 5) -> list[list[dict]]:
        """Top-k by cosine similarity for each query row."""
        if not len(self):
            return [[] for _ in range(len(query))]
        sims = _unit(query.astype("float32")) @ self.vectors.T
        k = min(k, len(self))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out = []
        for row, idx in enumerate(top):
            idx = idx[np.argsort(-sims[row, idx])]
            out.append([{**self.meta[i], "score": float(sims[row, i])} for i in idx])
        return out

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "meta.json").write_text(json.dumps(self.meta))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        d = Path(directory)
        vecs = np.load(d / "vectors.npy")
        idx = cls(vecs.shape[1])
        idx.vectors = vecs
        idx.meta = json.loads((d / "meta.json").read_text())
        return idx
