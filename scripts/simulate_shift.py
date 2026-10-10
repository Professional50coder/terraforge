"""Reproduce the controlled-simulation tables in docs/algorithm.md (CPU, ~1 min, no data needed).

    python scripts/simulate_shift.py --out runs/simulation.json

A simulator with KNOWN severity s: higher s weakens the class signal and inflates embedding
noise. It checks the mechanism and the code, not satellite data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from terraforge.training import selective
from terraforge.training import shift_conformal as sc

K, D, ALPHA = 6, 12, 0.1


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class World:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.centers = self.rng.normal(size=(K, D)) * 2.0
        self.train_emb = self.sample(3000, 0.0)[1]

    def sample(self, n, s):
        y = self.rng.integers(0, K, n)
        emb = self.centers[y] + self.rng.normal(size=(n, D)) * (0.7 + 2.0 * s)
        logits = self.rng.normal(size=(n, K))
        logits[np.arange(n), y] += 3.0 * (1 - 0.85 * s)
        return y, emb, softmax(logits)

    def batch(self, n, s):
        y, emb, p = self.sample(n, s)
        return sc.Batch(p, emb, selective.knn_novelty(self.train_emb, emb, k=5), y, np.full(n, s))

    def augmented(self, n_per, levels=(0.15, 0.3, 0.45, 0.6, 0.75)):
        return sc.Batch.concat([self.batch(n_per, s) for s in levels])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/simulation.json")
    a = ap.parse_args()
    validity_levels, eff_levels = (0.0, 0.2, 0.4, 0.6, 0.8), (0.0, 0.3, 0.6)
    validity: dict = {}
    efficiency: dict = {}
    for seed in range(a.seeds):
        w = World(seed)
        clean, aug = w.batch(3000, 0.0), w.augmented(800)
        conds = {str(s): w.batch(3000, s) for s in sorted(set(validity_levels + eff_levels))}
        for s, per_method in sc.compare(clean, aug, {str(s): conds[str(s)] for s in validity_levels}, ALPHA).items():
            for m, r in per_method.items():
                validity.setdefault(m, {}).setdefault(s, []).append((r["coverage"], r["mean_size"]))
        if seed < 4:
            for s in eff_levels:
                for m, size in sc.efficiency_at_coverage(clean, aug, conds[str(s)], 1 - ALPHA).items():
                    efficiency.setdefault(m, {}).setdefault(str(s), []).append(size)
    out = {"validity_coverage_size": {m: {s: [float(np.mean([c for c, _ in v])), float(np.mean([z for _, z in v]))]
                                         for s, v in d.items()} for m, d in validity.items()},
           "size_at_90pct_coverage": {m: {s: float(np.mean(v)) for s, v in d.items()} for m, d in efficiency.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"{'method':14s}" + "".join(f"s={s:<12}" for s in map(str, validity_levels)))
    for m, d in out["validity_coverage_size"].items():
        print(f"{m:14s}" + "".join(f"{d[str(s)][0]:.3f}/{d[str(s)][1]:.2f}  " for s in validity_levels))
    print("\nsize needed for 90% coverage (inf = never)")
    for m, d in out["size_at_90pct_coverage"].items():
        print(f"{m:14s}" + "".join(f"{d[str(s)]:.2f}  " for s in eff_levels))


if __name__ == "__main__":
    main()
