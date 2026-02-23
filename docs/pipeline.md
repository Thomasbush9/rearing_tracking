# Behavioral Analysis Pipeline

Transformer + HMM pipeline for unsupervised behavioral state discovery.

---

## Overview

```
Raw session .xlsx files
        │
        ▼
  [1] Preprocess & save dataset   prepare_and_save_dataset()
        │  out_s16.npz
        ▼
  [2] Sweep / train               wandb sweep sweep_config_patched.yaml
        │  runs/<run>/best.pt
        ▼
  [3] Single-session sanity check  test_hmm.py
        │  session_latents.npy, hhm_session_1.pkl
        ▼
  [4] Multi-session extraction     scripts/extract_and_save_all_sessions.py
        │  all_sessions_latents.npz
        │  all_sessions_hmm.npz
        │  all_sessions_hmm.pkl
        ▼
  [5] Analysis notebook
```

---

## Step 1 — Build the preprocessed dataset

Normalises all sessions together (cross-session stats), windows, splits, and
saves norm statistics into the `.npz` for consistent extraction later.

```bash
python -m behavex.models.train_transformer \
    --data     /path/to/sessions/ \
    --output   /Users/thomasbush/Downloads/out_s16.npz \
    --window-size 128 \
    --save-dataset-only
```

The resulting `.npz` contains `norm_mean`, `norm_std`, `norm_cols` arrays that
are threaded into every downstream checkpoint so that single-session extraction
reuses the same statistics as training.

---

## Step 2 — Sweep hyperparameters

```bash
wandb sweep sweep_config_patched.yaml      # prints <sweep_id>
wandb agent <sweep_id>
```

Key swept parameters: `patch_length ∈ {4, 16}`, `n_predict_steps ∈ {5,10,15}`,
`d_model`, `nhead`, `num_layers`, `mask_ratio`, `batch_size`, `learning_rate`.

**Selecting the best checkpoint**

1. Open the wandb project in your browser.
2. Sort runs by `val/loss` ascending.
3. Click the best run → Files → download `best.pt`.
4. Note the run path: `<entity>/<project>/<run_id>`.

---

## Step 3 — Single-session sanity check

Before scaling to all sessions, verify that the transformer produces
interpretable states on one representative session.

```bash
python -m behavex.models.test_hmm \
    --model-path  runs/<best_run>/best.pt \
    --session-path /path/to/one_session.xlsx
```

This runs:
1. `extract_session_latents_for_hmm()` → `(N, d_model)`
2. BIC model selection over K = 4 … 16
3. Viterbi decoding
4. Saves `session_latents.npy` and `hhm_session_1.pkl`

**Pass criteria before proceeding:**
- BIC curve shows a clear elbow (not monotone)
- At least 3 states have distinct mean feature profiles
- Median dwell time per state > 5 frames

---

## Step 4 — Multi-session extraction + HMM

Single command for extraction only:

```bash
python scripts/extract_and_save_all_sessions.py \
    --checkpoint runs/<best_run>/best.pt \
    --data-dir   /path/to/sessions/ \
    --output     outputs/all_sessions_latents.npz \
    --window-size 128 \
    --batch-size  64 \
    --device      mps
```

With automatic BIC state selection and HMM fitting:

```bash
python scripts/extract_and_save_all_sessions.py \
    --checkpoint runs/<best_run>/best.pt \
    --data-dir   /path/to/sessions/ \
    --output     outputs/all_sessions_latents.npz \
    --fit-hmm \
    --hmm-output outputs/all_sessions_hmm.npz \
    --state-min 4 \
    --state-max 16
```

With a fixed number of states (skip BIC):

```bash
    ... --fit-hmm --n-states 8
```

### Output files

**`all_sessions_latents.npz`**

| Key | Shape | Description |
|-----|-------|-------------|
| `latents` | `(N_total, d_model)` | All sessions concatenated |
| `session_names` | `(n_sessions,)` | File stems |
| `session_conditions` | `(n_sessions,)` | `"cricket"` or `"object"` |
| `session_starts` | `(n_sessions,)` | First row index per session |
| `session_lengths` | `(n_sessions,)` | Rows per session |
| `checkpoint` | scalar | Path used |
| `window_size` | scalar | |
| `d_model` | scalar | |

**`all_sessions_hmm.npz`** (with `--fit-hmm`)

| Key | Shape | Description |
|-----|-------|-------------|
| `latents` | `(N_total, d_model)` | Same as above |
| `states` | `(N_total,)` | Viterbi state per timestep |
| `session_names` | `(n_sessions,)` | |
| `session_conditions` | `(n_sessions,)` | |
| `session_starts` | `(n_sessions,)` | |
| `session_lengths` | `(n_sessions,)` | |
| `transition_matrix` | `(K, K)` | HMM transition probabilities |
| `stationary` | `(K,)` | Stationary state distribution |
| `n_states` | scalar | |
| `log_probs` | `(n_sessions,)` | Viterbi log-prob per session |
| `bic_scores` | `(n_candidates, 2)` | `[[K, BIC], …]` if selection ran |

**`all_sessions_hmm.pkl`** — full `HiddenMarkovModelTrainer` for reuse.

---

## Step 5 — Analysis (quick-load pattern)

```python
import numpy as np
from behavex.models.hmm_trainer import HiddenMarkovModelTrainer

data = np.load("outputs/all_sessions_hmm.npz", allow_pickle=True)

latents    = data["latents"]          # (N_total, d_model)
states     = data["states"]           # (N_total,)
names      = list(data["session_names"])
conditions = list(data["session_conditions"])
starts     = data["session_starts"]
lengths    = data["session_lengths"]
A          = data["transition_matrix"]
K          = int(data["n_states"])

trainer = HiddenMarkovModelTrainer.load("outputs/all_sessions_hmm.pkl")

# Per-session state sequence
def session_states(i):
    s, l = starts[i], lengths[i]
    return states[s:s+l]

# Cricket vs. object occupancy
cricket_idx = [i for i, c in enumerate(conditions) if c == "cricket"]
object_idx  = [i for i, c in enumerate(conditions) if c == "object"]
```

### Recommended analyses (for thesis)

1. **Latent space structure** — UMAP of `latents[::10]` coloured by `states[::10]`
2. **State profiles** — mean ± std of raw behavioral features per state
3. **Event sensitivity** — state occupancy vs. time relative to cricket/object entry
4. **Condition comparison** — state occupancy fractions: cricket sessions vs. object sessions

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: No session files` | Wrong `--data-dir` or files don't match `m*_s*_cricket\|object.xlsx` | Check filename pattern |
| States flicker (dwell < 3 frames) | K too large or `sticky_prior` too low | Reduce `--state-max` or increase `--sticky-prior` |
| BIC monotone decreasing | Latents not informative enough | Try a better checkpoint or re-run sweep |
| `ImportError: hmmlearn` | Missing dependency | `pip install hmmlearn` |
| Norm stats missing from checkpoint | Old checkpoint trained before norm-stats fix | Regenerate dataset → retrain one run |
