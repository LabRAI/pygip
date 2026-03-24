# PyHazards Architecture & Public API Sketch (Hazard-Centric)

PyHazards targets hazard prediction (earthquake, wildfire, flood, hurricane, etc.) with an easy, batteries-included API. The design is hazard-first, GPU/multi-GPU ready, and keeps the API familiar to users of popular ML libraries.

## Design Principles
- **Minimal onboarding**: one-liner load/train/evaluate for common hazards.
- **Backend-agnostic ML**: standard PyTorch models (tabular, image, sequence); plug-and-play custom models.
- **Hazard-aware datasets**: consistent interface across raster (remote sensing), tabular (climate/soil), time-series (buoys/stations), and vector/geospatial metadata.
- **GPU-first**: `device="auto"` everywhere; optional multi-GPU via DDP; mixed precision for speed.
- **Composable pipelines**: data → transforms → model → metrics → reporting with sensible defaults.
- **Extensible registry**: datasets, models, transforms, metrics, and pipelines are all discoverable by string names.

## Proposed Package Layout (building on current `pyhazards/`)
- `pyhazards.datasets`
  - `Dataset` base: unified API (`.load(split) -> DataBundle`, exposes `feature_spec`, `label_spec`, `splits`, `metadata`).
  - `hazards/` (new): curated loaders
    - `EarthquakeUSGS`, `WildfireMODIS`, `FloodCopernicus`, `HurricaneNOAA`, `LandslideNASA`, etc.
    - Each handles download/cache, normalization, split logic, CRS handling for geospatial data, and returns tensors ready for models.
  - `transforms/`: common preprocessing (standardize, log-scale precip, NDVI/NDWI indices, temporal windowing, patch extraction for rasters).
  - `registry.py`: `load_dataset(name, split="train", cache_dir=None, **kwargs)`.
- `pyhazards.models`
  - `backbones.py`: generic modules (MLP, CNN patch encoder, temporal encoder with GRU/Transformer-lite).
  - `heads.py`: task heads (`ClassificationHead`, `RegressionHead`, `SegmentationHead` for raster masks).
  - `builder.py`: `build_model(name="mlp"|"cnn"|"temporal", task="regression"|..., **kwargs)`.
  - `registry.py`: map model names to builders + metadata (input types supported).
- `pyhazards.engine`
  - `Trainer`: fit/eval/predict abstraction with callbacks, early stopping, checkpointing, mixed precision, gradient accumulation.
  - `distributed`: thin wrapper for PyTorch DDP; `strategy="auto"|"ddp"|"dp"`.
  - `inference.py`: batch/stream inference; sliding-window raster tiling for large scenes.
- `pyhazards.metrics`
  - Classification: Acc/F1/Precision/Recall/AUROC.
  - Regression: MAE/RMSE/R².
  - Segmentation: IoU/Dice.
  - Calibration: ECE/Brier.
- `pyhazards.utils`
  - `hardware.py`: `auto_device(prefer="cuda")`, `num_devices()`, simple device validation.
  - `seed_all`, logging helpers (stdout/JSON/CSV), timer/memory profilers.
- `pyhazards.cli`
  - `pyhazards run --dataset wildfire_modis --model cnn --task segmentation --device auto --strategy ddp --mixed-precision`.

## Core Python Flow
```python
from pyhazards import datasets, models
from pyhazards.engine import Trainer
from pyhazards.metrics import RegressionMetrics
from pyhazards.utils import auto_device

# 1) Load data
data = datasets.load("flood_copernicus", split_config="standard", cache_dir="~/.pyhazards")

# 2) Build model
model = models.build(
    name="temporal",
    task="regression",
    in_dim=data.feature_spec.input_dim,
    out_dim=data.label_spec.num_targets,
    hidden_dim=256,
)

# 3) Train & evaluate
trainer = Trainer(
    model=model,
    device=auto_device(),
    strategy="auto",          # promotes to DDP on multi-GPU
    mixed_precision=True,
    metrics=[RegressionMetrics()],
)
trainer.fit(data, max_epochs=50, train_split="train", val_split="val")
results = trainer.evaluate(data, split="test")
print(results)  # {"RMSE": ..., "MAE": ...}

# 4) Predict / export
preds = trainer.predict(data, split="test")
trainer.save_checkpoint("checkpoints/flood_temporal.pt")
```

## CLI Flow
```bash
pyhazards run \
  --dataset earthquake_usgs \
  --model mlp \
  --task classification \
  --device auto \
  --strategy auto \
  --mixed-precision
```

## Data Model
- `DataBundle`: typed container with `tensors`, `splits` (`train/val/test`), `spatial_meta` (CRS, bounding boxes), `temporal_index`, and `feature_spec/label_spec`.
- Supports tabular (climate/soil), raster patches (satellite), and time-series (sensors).
- Transform pipelines apply lazily (composition over datasets) and can be reused for inference.

## GPU & Multi-GPU
- `auto_device()` picks `cuda:0` if available else CPU; `num_devices()` detects multi-GPU.
- `Trainer(strategy="auto")`:
  - Single GPU: optional AMP for speed.
  - Multi-GPU: PyTorch DDP (`torchrun`) with synchronized metrics and gradient accumulation.
  - CPU fallback: same API, no code changes.

## Extensibility Checklist
- New dataset: subclass `Dataset`, implement `load(split, transforms=None)`, register via `datasets.registry`.
- New transform: add a callable, register via `datasets.transforms`.
- New model: implement `nn.Module`, expose via `models.registry` with metadata (`supported_inputs`, `task`).
- New metric: subclass `MetricBase`, add to `metrics` namespace.
- New pipeline/workflow: compose dataset + transforms + model + metrics in `pyhazards.workflows`.

This design keeps the surface area small (load → build → train → evaluate) while being hazard-first, GPU-ready, and easy to extend as new hazards, data modalities, or models are added.
