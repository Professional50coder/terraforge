"""Run the TerraForge service.

    python scripts/serve.py                      # untrained demo model + synthetic stream
    python scripts/serve.py --weights runs/cnn_s0.pt --arch cnn --cache data/processed

Secrets: set TERRAFORGE_ENV_FILE to a .env path OUTSIDE the repo; it is loaded into the
environment at startup (existing variables win) and never printed.
"""
import argparse
import json
import os
from pathlib import Path

import uvicorn

from terraforge.api.main import build_reference_index, create_app
from terraforge.api.sources import EuroSATSource, SyntheticSource
from terraforge.inference.predictor import InferenceEngine
from terraforge.multimodal.explainer import Explainer, load_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="cnn", choices=["cnn", "vit"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--root", default=None, help="EuroSAT class-folder root for real replay")
    ap.add_argument("--explainer", default="template", choices=["template", "groq", "hf", "ollama"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if os.environ.get("TERRAFORGE_ENV_FILE"):
        load_env(os.environ["TERRAFORGE_ENV_FILE"], allow=("GROQ_API_KEY", "HF_TOKEN"))
    proc = Path(a.processed)
    stats = json.loads((proc / "band_stats.json").read_text()) if (proc / "band_stats.json").exists() \
        else {"mean": [1500.0] * 13, "std": [800.0] * 13}
    engine = InferenceEngine.load(a.arch, a.weights, stats, temperature=a.temperature)
    source = EuroSATSource(a.root, proc / "manifest.csv") if a.root and (proc / "manifest.csv").exists() \
        else SyntheticSource()
    index = build_reference_index(engine, source, 512)
    app = create_app(engine, source, index, Explainer(a.explainer))
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
