"""Train and evaluate a model on EuroSAT-MS; logs to MLflow if installed."""
import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from terraforge.data.torch_dataset import EuroSATDataset
from terraforge.models.cnn import SmallCNN
from terraforge.models.vit import ViT
from terraforge.training.trainer import ModelTrainer

MODELS = {"cnn": SmallCNN, "vit": ViT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="cnn")
    ap.add_argument("--root", required=True, help="EuroSAT class-folder root")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    proc = Path(a.processed)
    ds = {s: EuroSATDataset(a.root, proc / "manifest.csv", proc / "band_stats.json",
                            s, augment=(s == "train")) for s in ("train", "val", "test")}
    dl = {s: DataLoader(d, a.batch_size, shuffle=(s == "train"), num_workers=2)
          for s, d in ds.items()}
    tr = ModelTrainer(MODELS[a.model](), n_classes=10, lr=a.lr)
    try:
        import mlflow
        mlflow.set_experiment("terraforge-eurosat")
        mlflow.start_run(run_name=a.model)
        mlflow.log_params(vars(a))
    except ImportError:
        mlflow = None
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        loss = tr.fit_epoch(dl["train"])
        val = tr.evaluate(dl["val"])
        print(f"epoch {ep+1} loss {loss:.4f} val_f1 {val['macro_f1']:.4f}")
        if mlflow:
            mlflow.log_metrics({"train_loss": loss, "val_macro_f1": val["macro_f1"]}, step=ep)
        if val["macro_f1"] > best:  # model selection on VAL, never on test
            best = val["macro_f1"]
            best_state = {k: v.clone() for k, v in tr.model.state_dict().items()}
    tr.model.load_state_dict(best_state)
    test = tr.evaluate(dl["test"])
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / f"{a.model}_test.json").write_text(json.dumps(test, indent=2))
    print(f"TEST acc {test['accuracy']:.4f} macro_f1 {test['macro_f1']:.4f}")
    if mlflow:
        mlflow.log_metrics({"test_accuracy": test["accuracy"],
                            "test_macro_f1": test["macro_f1"]})
        mlflow.end_run()


if __name__ == "__main__":
    main()
