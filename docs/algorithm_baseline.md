# Algorithm baseline

Task: 10-class land-cover classification of 13-band Sentinel-2 patches (EuroSAT-MS).

## Models

1. **SmallCNN** (`models/cnn.py`) - 4 conv-BN-ReLU-pool stages, global average pool, linear head.
   The control. Convolution assumes locality and translation equivariance, which suits small
   patches with limited data.
2. **ViT** (`models/vit.py`) - written from scratch: 8x8 patches -> 64 tokens + CLS, learned
   positional embeddings, 6 pre-norm blocks, 4 heads, dim 128. Attention is global from layer 1
   but has no spatial prior, so it usually needs more data or pretraining to match a CNN.
3. **EO foundation model** (planned) - fine-tune a pretrained model via TerraTorch.

## Protocol

Same split, same normalisation, same optimiser (AdamW, grad-clip 1.0), same epochs. Best epoch
chosen on validation macro-F1; test evaluated once. Metrics: accuracy, macro precision/recall/F1,
per-class F1, confusion matrix, inference ms/sample. Tracked in MLflow.

## Hypothesis to test, not assume

The CNN will match or beat the from-scratch ViT at this data scale; the pretrained foundation
model will beat both. If results differ, the report must say so and why.

## Results

_Not yet run._ Filled in from `runs/*_test.json` after training.
