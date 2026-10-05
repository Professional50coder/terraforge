"""Export a trained model to ONNX and verify numerical parity with PyTorch.

ONNX decouples the model from the training stack: it can be served with
onnxruntime (CPU/GPU, no PyTorch), which shrinks the inference container and
removes the Python-version coupling that makes EO stacks painful to deploy.
Parity is checked, not assumed: export bugs are silent.
"""
import argparse

import numpy as np
import torch

from terraforge.models.cnn import SmallCNN
from terraforge.models.vit import ViT

MODELS = {"cnn": SmallCNN, "vit": ViT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="cnn")
    ap.add_argument("--weights", default=None, help="state_dict .pt (random init if omitted)")
    ap.add_argument("--out", default="model.onnx")
    a = ap.parse_args()
    m = MODELS[a.model]().eval()
    if a.weights:
        m.load_state_dict(torch.load(a.weights, map_location="cpu"))
    x = torch.randn(2, 13, 64, 64)
    torch.onnx.export(m, x, a.out, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    err = float(np.abs(got - m(x).detach().numpy()).max())
    print(f"max abs diff vs PyTorch: {err:.2e}")
    assert err < 1e-3, "ONNX output diverges from PyTorch"


if __name__ == "__main__":
    main()
