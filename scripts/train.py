"""Train and evaluate a model on EuroSAT-MS from the uint16 cache.

Protocol (fixed across models): AdamW, warmup + cosine LR, label smoothing, EMA weights,
best epoch chosen on VALIDATION macro-F1, test evaluated once, temperature fitted on
validation logits, robustness measured on the test set.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from terraforge.data.torch_dataset import CachedEuroSATDataset
from terraforge.models.cnn import SmallCNN
from terraforge.models.vit import ViT
from terraforge.training.calibration import expected_calibration_error, fit_temperature
from terraforge.training.robustness import robustness_report
from terraforge.training.trainer import ModelTrainer

MODELS = {"cnn": SmallCNN, "vit": ViT}


def seed_everything(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="cnn")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.99)
    ap.add_argument("--pretrained", default=None, help="MAE-pretrained ViT state_dict")
    ap.add_argument("--subset", type=int, default=None, help="cap train samples (smoke runs)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    seed_everything(a.seed)
    proc = Path(a.processed)
    ds = {s: CachedEuroSATDataset(proc / "cache", proc / "band_stats.json", s, augment=(s == "train"))
          for s in ("train", "val", "test")}
    if a.subset:  # smoke runs only; never used for reported results
        # The manifest is class-ordered, so take evenly spaced samples, not the first N.
        def spread(idx, n):
            return idx[:: max(1, len(idx) // n)][:n]
        ds["train"].idx = spread(ds["train"].idx, a.subset)
        for s in ("val", "test"):
            ds[s].idx = spread(ds[s].idx, max(64, a.subset // 2))
    gen = torch.Generator().manual_seed(a.seed)
    dl = {s: DataLoader(d, a.batch_size, shuffle=(s == "train"), generator=gen if s == "train" else None,
                        num_workers=a.workers, pin_memory=torch.cuda.is_available())
          for s, d in ds.items()}
    steps = a.epochs * len(dl["train"])
    model = MODELS[a.model]()
    if a.pretrained:
        model.load_state_dict(torch.load(a.pretrained, map_location="cpu"))
    tr = ModelTrainer(model, 10, lr=a.lr, label_smoothing=a.label_smoothing, total_steps=steps,
                      warmup_steps=max(1, steps // 20), ema_decay=a.ema)
    tag = a.tag or f"{a.model}_s{a.seed}"
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    best, best_state, history = -1.0, None, []
    for ep in range(a.epochs):
        loss = tr.fit_epoch(dl["train"])
        val = tr.evaluate(dl["val"])
        history.append({"epoch": ep + 1, "train_loss": loss, "val_macro_f1": val["macro_f1"],
                        "val_accuracy": val["accuracy"]})
        print(f"[{tag}] epoch {ep+1}/{a.epochs} loss {loss:.4f} val_acc {val['accuracy']:.4f} "
              f"val_f1 {val['macro_f1']:.4f}", flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            best_state = {k: v.clone() for k, v in tr.eval_model.state_dict().items()}
    tr.eval_model.load_state_dict(best_state)
    torch.save(best_state, out / f"{tag}.pt")

    vlog, vy = tr.logits(dl["val"])
    temp = fit_temperature(vlog, vy)
    tlog, ty = tr.logits(dl["test"])
    test = tr.evaluate(dl["test"])
    probs_raw = tlog.softmax(1).numpy()
    probs_cal = (tlog / temp).softmax(1).numpy()
    test["temperature"] = temp
    test["ece_raw"] = expected_calibration_error(probs_raw, ty.numpy())
    test["ece_calibrated"] = expected_calibration_error(probs_cal, ty.numpy())
    test["robustness"] = robustness_report(tr.eval_model, dl["test"])
    test["history"] = history
    test["args"] = vars(a)
    n_params = sum(p.numel() for p in model.parameters())
    test["n_params"] = n_params
    (out / f"{tag}.json").write_text(json.dumps(test, indent=2))
    print(f"[{tag}] TEST acc {test['accuracy']:.4f} macro_f1 {test['macro_f1']:.4f} "
          f"ECE {test['ece_raw']:.4f}->{test['ece_calibrated']:.4f} T={temp:.2f} params={n_params:,}",
          flush=True)


if __name__ == "__main__":
    main()
