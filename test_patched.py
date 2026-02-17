"""Quick comparison: original vs patched masked transformer on a single session."""
import torch
import numpy as np
import random
import sys

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from behavex.models.train_transformer import (
    prepare_masked_transformer_data,
    load_data_path,
    temporal_train_val_test_split,
    MaskedTemporalTrainingArgs,
    MaskedTemporalTrainer,
    PatchedTrainingArgs,
    PatchedMaskedTemporalTrainer,
)

DATA_PATH = "/Users/thomasbush/Downloads/shared WithTWB/m001_s001_cricket.xlsx"
DEVICE = "mps"
EPOCHS = 15
PATCH_LEN = 8

print("Loading data...")
data, valid_mask = load_data_path(DATA_PATH, trim_before_dist_head=True)
windows, feature_names, _ = prepare_masked_transformer_data(
    data, window_size=128, stride=1, drop_cols=["height"],
)
train_w, val_w, test_w = temporal_train_val_test_split(windows, 0.15, 0.15)
f_in = train_w.shape[-1]
print(f"Data: {train_w.shape[0]} train, {val_w.shape[0]} val, {test_w.shape[0]} test | F={f_in}")
print(f"Window: {train_w.shape[1]} timesteps | Patches: {train_w.shape[1] // PATCH_LEN} (P={PATCH_LEN})")

# ── Original (point-wise) model ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ORIGINAL (point-wise tokens, bidirectional)")
print("=" * 60)
orig_args = MaskedTemporalTrainingArgs(
    batch_size=32, device=DEVICE, learning_rate=1e-3,
    f_in=f_in, train_windows=train_w, val_windows=val_w, test_windows=test_w,
    d_model=64, nhead=2, num_layers=4,
    mask_ratio=0.15, epochs=EPOCHS, patience=EPOCHS + 1,
    verbose=True, save_model=False,
    eval_every_n_epochs=5, feature_names=feature_names,
)
orig_trainer = MaskedTemporalTrainer(orig_args)
n_params_orig = sum(p.numel() for p in orig_trainer.model.parameters())
print(f"Parameters: {n_params_orig:,}")
orig_best = orig_trainer.train()

# ── Patched model ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"PATCHED (P={PATCH_LEN}, {train_w.shape[1] // PATCH_LEN} tokens, bidirectional)")
print("=" * 60)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

patched_args = PatchedTrainingArgs(
    batch_size=32, device=DEVICE, learning_rate=1e-3,
    f_in=f_in, patch_length=PATCH_LEN,
    train_windows=train_w, val_windows=val_w, test_windows=test_w,
    d_model=64, nhead=2, num_layers=4,
    mask_ratio=0.15, epochs=EPOCHS, patience=EPOCHS + 1,
    verbose=True, save_model=False,
    eval_every_n_epochs=5, feature_names=feature_names,
)
patched_trainer = PatchedMaskedTemporalTrainer(patched_args)
n_params_patched = sum(p.numel() for p in patched_trainer.model.parameters())
print(f"Parameters: {n_params_patched:,}")
patched_best = patched_trainer.train()

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"Original  | params: {n_params_orig:>8,} | best val loss: {orig_best:.4f}")
print(f"Patched   | params: {n_params_patched:>8,} | best val loss: {patched_best:.4f}")
print(f"Tokens per sample: original={train_w.shape[1]}, patched={train_w.shape[1] // PATCH_LEN}")
