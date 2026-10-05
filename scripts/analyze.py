"""Trust analysis of a trained model: conformal coverage, selective prediction, leakage audit.

Answers, with numbers, the questions a raw accuracy score cannot:
  Q1  Do conformal prediction sets reach their coverage target on clean test data - and how fast
      does that guarantee degrade under haze, band failure, cloud and noise?
  Q2  Does confidence (or embedding novelty) rank the model's own errors last, so abstaining
      on the least-confident fraction removes most mistakes?
  Q3  How much of the test accuracy could come from near-duplicate neighbours in train
      (spatial leakage between patches cut from the same scenes)?

Calibration data for temperature and conformal quantiles is the VALIDATION split only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from terraforge.data.eurosat import CLASSES
from terraforge.data.torch_dataset import CachedEuroSATDataset
from terraforge.models.cnn import SmallCNN
from terraforge.models.vit import ViT
from terraforge.training import conformal as cf
from terraforge.training import selective as sel
from terraforge.training.calibration import fit_temperature
from terraforge.training.robustness import CORRUPTIONS

ARCH = {"cnn": SmallCNN, "vit": ViT}


@torch.no_grad()
def collect(model, loader, corrupt=None, seed=0):
    """logits, labels, embeddings. `corrupt=(fn, severity)` applies a corruption per batch."""
    gen = torch.Generator().manual_seed(seed)
    logits, ys, embs = [], [], []
    for x, y in loader:
        if corrupt:
            x = corrupt[0](x, corrupt[1], gen)
        logits.append(model(x))
        embs.append(model.features(x) if isinstance(model, ViT) else model.embed(x))
        ys.append(y)
    return torch.cat(logits), torch.cat(ys).numpy(), torch.cat(embs).numpy()


def spread(idx, n):
    return idx[:: max(1, len(idx) // n)][:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=ARCH, required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=None, help="cap eval samples (smoke runs)")
    ap.add_argument("--leak-threshold", type=float, default=0.97)
    ap.add_argument("--out", default="runs/analysis.json")
    a = ap.parse_args()
    torch.manual_seed(0)
    proc = Path(a.processed)
    model = ARCH[a.arch]()
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    model.eval()
    ds = {s: CachedEuroSATDataset(proc / "cache", proc / "band_stats.json", s) for s in ("train", "val", "test")}
    ds["train"].idx = spread(ds["train"].idx, a.limit or 6000)   # reference embeddings only
    if a.limit:
        for s in ("val", "test"):
            ds[s].idx = spread(ds[s].idx, a.limit)
    dl = {s: DataLoader(d, 256) for s, d in ds.items()}

    vlog, vy, _ = collect(model, dl["val"])
    T = fit_temperature(vlog, torch.as_tensor(vy))
    vprob = (vlog / T).softmax(1).numpy()
    tlog, ty, temb = collect(model, dl["test"])
    tprob = (tlog / T).softmax(1).numpy()
    _, _, train_emb = collect(model, dl["train"])
    n_cls = len(CLASSES)
    report = {"arch": a.arch, "weights": a.weights, "alpha": a.alpha, "temperature": T,
              "n_val": len(vy), "n_test": len(ty)}

    # ---- Q1: conformal coverage, clean and under corruption ----------------------------------
    quant = {}
    for method in ("lac", "aps"):
        sc = cf.true_class_scores(cf.SCORERS[method](vprob), vy)
        quant[method] = cf.conformal_quantile(sc, a.alpha)
        quant[method + "_mondrian"] = cf.class_conditional_quantiles(sc, vy, a.alpha, n_cls)
    report["conformal_clean"] = {}
    for name, q in quant.items():
        method = name.split("_")[0]
        report["conformal_clean"][name] = cf.evaluate_sets(cf.prediction_sets(tprob, q, method), ty, n_cls)
    report["conformal_under_shift"] = {}
    for cname, fn in CORRUPTIONS.items():
        report["conformal_under_shift"][cname] = {}
        for sev in (0.1, 0.25, 0.5):
            lg, y2, _ = collect(model, dl["test"], corrupt=(fn, sev))
            p2 = (lg / T).softmax(1).numpy()
            r = cf.evaluate_sets(cf.prediction_sets(p2, quant["lac"], "lac"), y2, n_cls)
            report["conformal_under_shift"][cname][str(sev)] = {
                "coverage": r["coverage"], "mean_size": r["mean_size"],
                "accuracy": float((p2.argmax(1) == y2).mean())}

    # ---- Q1b: conformal calibration methods under HELD-OUT corruption families --------------
    # Severity is a nominal 0-1 index, not physical units, and its scale differs per family;
    # this is a known weakness of the setup and is reported, not hidden.
    from terraforge.training import shift_conformal as sc

    def make_batch(loader, corrupt=None, severity=0.0):
        lg, y, emb = collect(model, loader, corrupt=corrupt)
        return sc.Batch((lg / T).softmax(1).numpy(), emb, sel.knn_novelty(train_emb, emb, k=5),
                        y, np.full(len(y), severity))

    fams, levels, test_levels = list(CORRUPTIONS), (0.15, 0.3, 0.45, 0.6), (0.25, 0.5)
    clean_cal = make_batch(dl["val"])
    cal_by_fam = {f: sc.Batch.concat([make_batch(dl["val"], (CORRUPTIONS[f], s), s) for s in levels])
                  for f in fams}
    per_method: dict[str, list] = {}
    held_out = {}
    for f in fams:
        aug = sc.Batch.concat([cal_by_fam[g] for g in fams if g != f])   # f is NEVER seen in calibration
        conds = {f"{s}": make_batch(dl["test"], (CORRUPTIONS[f], s), s) for s in test_levels}
        res = sc.compare(clean_cal, aug, conds, a.alpha)
        # Matched-coverage efficiency and paired bootstrap CIs (SACP vs the strongest baselines).
        for s_name, tb in conds.items():
            eff = sc.efficiency_at_coverage(clean_cal, aug, tb, target=1 - a.alpha)
            masks = sc.predict_sets_all(clean_cal, aug, tb, a.alpha)
            for m, r in res[s_name].items():
                r["size_at_target_coverage"] = eff[m]
            for base in ("cond_novelty", "cond_entropy", "raps", "hybrid_union"):
                res[s_name]["sacp"].setdefault("vs", {})[base] = sc.paired_bootstrap(
                    masks["sacp"], masks[base], tb.y)
        held_out[f] = res
        for cond in res.values():
            for m, r in cond.items():
                per_method.setdefault(m, []).append(r)
    report["shift_conformal"] = {
        "protocol": "calibrate on clean + all-but-one corruption family; test on the held-out family",
        "held_out": held_out,
        "summary": {m: {"mean_coverage": float(np.mean([r["coverage"] for r in rs])),
                        "worst_undercoverage": float(np.max([r["undercoverage"] for r in rs])),
                        "mean_size": float(np.mean([r["mean_size"] for r in rs])),
                        "mean_size_at_target_coverage": float(np.mean(
                            [r["size_at_target_coverage"] for r in rs
                             if np.isfinite(r["size_at_target_coverage"])] or [np.inf])),
                        "unreachable_conditions": int(sum(
                            not np.isfinite(r["size_at_target_coverage"]) for r in rs)),
                        "mean_outside_envelope": float(np.mean([r["outside_envelope_fraction"] for r in rs]))}
                    for m, rs in per_method.items()}}

    # ---- Q2: selective prediction -------------------------------------------------------------
    conf, correct = tprob.max(1), tprob.argmax(1) == ty
    novelty = sel.knn_novelty(train_emb, temb, k=5)
    report["selective"] = {
        "accuracy": float(correct.mean()),
        "aurc_confidence": sel.aurc(conf, correct),
        "aurc_novelty": sel.aurc(-novelty, correct),
        "coverage_at_95pct_accuracy_confidence": sel.coverage_at_accuracy(conf, correct, 0.95),
        "coverage_at_99pct_accuracy_confidence": sel.coverage_at_accuracy(conf, correct, 0.99),
        "error_auroc_confidence": sel.error_auroc(1 - conf, correct),
        "error_auroc_novelty": sel.error_auroc(novelty, correct),
        "n_errors": int((~correct).sum()),
    }

    # ---- Q3: spatial-leakage audit -----------------------------------------------------------
    sim = sel.nearest_train_similarity(train_emb, temb)
    near = sim >= a.leak_threshold
    report["leakage_audit"] = {
        "threshold_cosine": a.leak_threshold, "n_train_reference": len(train_emb),
        "fraction_test_with_near_duplicate": float(near.mean()),
        "accuracy_near_duplicate": float(correct[near].mean()) if near.any() else None,
        "accuracy_others": float(correct[~near].mean()) if (~near).any() else None,
        "similarity_quantiles": {str(q): float(np.quantile(sim, q)) for q in (0.5, 0.9, 0.99)},
        "note": "reference set is a subsample of train, so the near-duplicate fraction is a "
                "lower bound; accuracy gap indicates how much neighbours may flatter the score",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    # Deployable calibration artefact: temperature + LAC quantile, fitted on validation only.
    (Path(a.out).parent / (Path(a.out).stem + "_conformal.json")).write_text(json.dumps({
        "arch": a.arch, "weights": a.weights, "temperature": T,
        "conformal": {"method": "lac", "q": quant["lac"], "alpha": a.alpha}}, indent=2))
    c = report["conformal_clean"]["lac"]
    print(f"[{a.arch}] acc {report['selective']['accuracy']:.4f} | LAC coverage {c['coverage']:.3f} "
          f"(target {1 - a.alpha:.2f}) size {c['mean_size']:.2f} | AURC {report['selective']['aurc_confidence']:.4f} "
          f"| near-dup {report['leakage_audit']['fraction_test_with_near_duplicate']:.1%}", flush=True)


if __name__ == "__main__":
    main()
