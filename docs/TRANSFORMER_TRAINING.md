# Transformer Training Guide

How to train masked and patched temporal transformers using preprocessed datasets, with both single-run and W&B sweep workflows.

## Data Preprocessing

### Create a preprocessed dataset

```bash
python -m src.behavex.models.make_dataset \
  --data /path/to/sessions/ \
  --output /path/to/data_transformer.npz \
  --val_ratio 0.15 --test_ratio 0.2 --window_size 128
```

- Expects session files: `m{N}_s{N}_cricket.xlsx` or `m{N}_s{N}_object.xlsx`
- Optional: `--drop_cols height` to keep only `height_scaled`
- Optional: `--stride N` for non-overlapping or sparse windows (default: 1)

### What gets saved

- `train_windows`, `val_windows`, `test_windows`: (N, T, F) float32 arrays
- `feature_names`: list of column names
- `*_event_mask`: optional (N, T) bool masks when event boundaries are used (pre-event positions downweighted in loss)

---

## Model types

| Model | Masking | Predictive head | Use case |
|-------|---------|-----------------|----------|
| **masked** | Timestep (per-frame) | No | Reconstruction only |
| **predictive** | Timestep | Yes (t+1..t+K) | Reconstruction + multi-step prediction |
| **patched** | Patch (P frames) | Optional | Lower resolution, faster; K patch-ahead |

**Patched model:** `window_size` must be divisible by `patch_length` (e.g. 128/4 = 32 patches).

---

## Single run (no wandb)

### With preprocessed data

```bash
python -m src.behavex.models.train_transformer \
  --config train_config.yaml \
  --preprocessed-data /path/to/data_transformer.npz \
  --output-dir ./runs/exp1
```

### Without config (CLI; masked or predictive)

```bash
python -m src.behavex.models.train_transformer \
  --preprocessed-data /path/to/data_transformer.npz \
  --model predictive --n_predict_steps 3 \
  --output-dir ./runs/exp1
```

### Patched (CLI)

```bash
python -m src.behavex.models.train_transformer \
  --preprocessed-data /path/to/data_transformer.npz \
  --model patched --patch-length 4 \
  --n_predict_steps 3 \
  --output-dir ./runs/patched_exp1
```

### Patched via config

```bash
python -m src.behavex.models.train_transformer \
  --config train_config_patched_gpu.yaml \
  --preprocessed-data /path/to/data_transformer.npz
```

---

## W&B sweep

1. Create a sweep from YAML:

```bash
# Timestep-level (masked/predictive)
python -m src.behavex.models.sweep_transformer \
  --config sweep_config.yaml \
  --preprocessed-data /path/to/data_transformer.npz \
  --project my-project --count 20

# Patched sweep
python -m src.behavex.models.sweep_transformer \
  --config sweep_config_patched.yaml \
  --preprocessed-data /path/to/data_transformer.npz \
  --project my-project --count 20
```

2. The sweep injects `run_id` so each trial gets a unique output dir. `run_train(cfg)` receives `wandb.config` and reuses the active wandb run (no `wandb_project` needed in config).

3. Single-run wandb: set `wandb_project` in YAML or `--wandb-project` on CLI.

---

## Config keys for preprocessed data

- `preprocessed_data`: path to .npz (overrides raw `data` when present)
- `mmap: true`: memory-map .npz (requires uncompressed file; use `--no-compress` when saving dataset)

---

## Example YAML snippets

**Masked/predictive** (`train_config.yaml`):

```yaml
preprocessed_data: /path/to/data_transformer.npz
model: predictive
n_predict_steps: 3
causal: true
```

**Patched** (`train_config_patched_gpu.yaml`):

```yaml
preprocessed_data: /path/to/data_transformer.npz
model: patched
patch_length: 4
n_predict_steps: 3
```
