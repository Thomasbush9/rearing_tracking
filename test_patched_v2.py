"""Quick test: patched model with larger d_model to match compression ratio."""
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

from behavex.models.train_transformer import (
    prepare_masked_transformer_data,
    load_data_path,
    temporal_train_val_test_split,
    PatchedTrainingArgs,
    PatchedMaskedTemporalTrainer,
)

DATA_PATH = "/Users/thomasbush/Downloads/shared WithTWB/m001_s001_cricket.xlsx"

print("Loading data...")
data, _ = load_data_path(DATA_PATH, trim_before_dist_head=True)
windows, feature_names, _ = prepare_masked_transformer_data(
    data, window_size=128, stride=1, drop_cols=["height"],
)
train_w, val_w, test_w = temporal_train_val_test_split(windows, 0.15, 0.15)
f_in = train_w.shape[-1]

for d_model in [128, 192]:
    for patch_len in [4, 8, 16]:
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        n_patches = 128 // patch_len
        print(f"\n{'='*60}")
        print(f"d_model={d_model}, P={patch_len}, N={n_patches} tokens")
        print(f"Compression: {patch_len*f_in} -> {d_model} ({patch_len*f_in/d_model:.1f}x)")
        print(f"{'='*60}")
        args = PatchedTrainingArgs(
            batch_size=32, device="mps", learning_rate=1e-3,
            f_in=f_in, patch_length=patch_len,
            train_windows=train_w, val_windows=val_w, test_windows=test_w,
            d_model=d_model, nhead=4, num_layers=4,
            mask_ratio=0.15, epochs=25, patience=26,
            verbose=True, save_model=False,
            eval_every_n_epochs=5, feature_names=feature_names,
        )
        trainer = PatchedMaskedTemporalTrainer(args)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        print(f"Parameters: {n_params:,}")
        best = trainer.train()
        print(f">> d_model={d_model} P={patch_len}: best val={best:.4f}, params={n_params:,}")
