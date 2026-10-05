# TerraForge

Earth Observation AI pipeline for satellite-based vegetation stress detection.

Sentinel-2 via STAC -> geospatial preprocessing (rasterio, xarray, GeoPandas) -> dataset builder -> CNN / Vision Transformer / EO foundation model (PyTorch, TerraTorch) -> MLflow tracking -> FastAPI inference -> Docker.

Classes: Healthy / Moderate stress / Severe stress.

## Status
Scaffold. Built in stages: EO fundamentals, STAC and dataset engineering, CNN baseline, ViT, foundation model, multimodal explanation, MLflow, API and Docker, distributed training.
