# Decision log

Each row: what was chosen, what it was weighed against, and the reason. Measured
results are added to this log only after a run in this repo produces them.

## Data

| Decision | Alternatives | Reason |
|---|---|---|
| EuroSAT-MS (13 bands, real labels) | RGB EuroSAT; BigEarthNet; weak NDVI labels | Real labels and multispectral signal, small enough for free hardware. BigEarthNet is far larger; weak labels would make every metric circular. |
| Mirror list for download (Hugging Face first, original host second) | Single URL | The original host timed out during setup; a single point of failure should not block a build. |
| uint16 memmap cache | Per-file GeoTIFF reads each epoch | Per-file reads dominated runtime on CPU. The source is uint16, so the cache is lossless and half the size of float32. |
| Stratified, seeded 70/15/15 split | Random split | Per-class proportions are identical in every split and reruns reproduce exactly. |
| Normalisation statistics from train only | All data | Statistics from held-out data leak information about it. |

## Models and training

| Decision | Alternatives | Reason |
|---|---|---|
| ViT written from scratch | timm / torchvision | Every component is visible and unit-tested (attention rows sum to 1, tiny-batch overfit). |
| Masked-autoencoder pretraining | Supervised only | Unlabeled imagery is abundant; MAE learns spectral and spatial structure without labels, and the encoder is the classifier's own architecture so weights transfer directly. |
| Warmup + cosine LR, EMA weights, label smoothing | Constant LR | Transformers are unstable at full LR from step 0; EMA is a cheap ensemble over the training trajectory. |
| Best epoch chosen on validation macro-F1; test used once | Choosing on test | Selecting on test makes the reported number optimistic. |
| Temperature scaling fitted on validation | None | Confidence is shown to users, so it must be calibrated; scaling cannot change predictions. |
| Python 3.11 in the project venv | Python 3.14 (system) | The geospatial stack has no 3.14 wheels; pip began compiling rasterio from source. |

## Serving

| Decision | Alternatives | Reason |
|---|---|---|
| Thread pool for CPU inference chunks | Process pool | PyTorch releases the GIL inside kernels, and threads avoid copying the model. |
| Exact cosine search in numpy | FAISS / HNSW | At ~27k vectors a matrix product takes milliseconds; an approximate index adds complexity for no gain. The interface allows a swap. |
| Agent has a step cap, validated tool arguments, contained errors, rule-based fallback | Unbounded loop | A confused model must not loop forever, crash the request, or leave the product dead when the LLM is down. |
| Model-name fallback chain for the hosted LLM | Single hard-coded model | Hosted catalogues retire models; the first live request after setup failed for exactly this reason. A 404 moves on; an auth error does not. |
| Honest `User-Agent` on API calls | Library default | The default Python agent was blocked by the provider's gateway (HTTP 403). |
| Secrets loaded from an external `.env` with an allowlist | Load everything | A shared `.env` holds unrelated credentials the service has no need to see. |
| CPU-only, non-root container with health check | CUDA image | Inference on 64x64 patches needs no GPU; the CPU image is far smaller. Not yet built or verified here (no Docker on the dev machine). |

## Honest limits

- EuroSAT labels land cover, not vegetation stress.
- EuroSAT is L1C (top-of-atmosphere); live STAC data is L2A. The domain gap is not yet measured.
- The live dashboard shows an untrained model on synthetic input until trained weights are served.
