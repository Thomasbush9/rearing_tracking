"""Extract latent representations from a trained masked transformer for downstream HMM analysis."""

from pathlib import Path
import argparse
import sys

import torch
import numpy as np

from behavex.models.transformer import (
    MaskedTemporalTransformer,
    MaskedPredictiveTemporalTransformer,
    PatchedMaskedTemporalTransformer,
    PatchedMaskedPredictiveTemporalTransformer,
)
from behavex.models.data import MaskedTemporalDataset
from torch.utils.data import DataLoader

from .train_transformer import (
    load_preprocessed_dataset,
    prepare_masked_transformer_data,
    load_data_path,
    temporal_train_val_test_split,
)


def load_single_session_windows(
    session_path: str | Path,
    start_times_path: str | Path | None = None,
    window_size: int = 128,
    stride: int = 1,
    full_file: bool = False,
) -> tuple[np.ndarray, list[str] | None]:
    """Load one session and build windows.

    Args:
        session_path: Path to the session file (.xlsx).
        start_times_path: Optional path to startTimes.xlsx for timestamp-based trimming.
        window_size: Number of timesteps per window.
        stride: Step between consecutive windows. Use stride=window_size for non-overlapping windows.
        full_file: If True, load the entire recording without trimming pre-event frames.

    Returns:
        windows: (N, window_size, F) float32
        feature_names: list or None
    """
    data, _ = load_data_path(
        str(session_path),
        trim_before_dist_head=not full_file,
        start_times_path=Path(start_times_path) if start_times_path else None,
    )
    windows, feature_names, _ = prepare_masked_transformer_data(
        data, window_size=window_size, stride=stride
    )
    return windows, feature_names


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    return_layer_mean: bool = False,
):
    """Load model from checkpoint. Infers architecture from state_dict and training_params.

    Supports both standard (MaskedTemporalTransformer) and patched
    (PatchedMaskedTemporalTransformer) checkpoints. Patched checkpoints are detected
    by the presence of 'patch_embedding.proj.weight' in the state dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model_state", ckpt)
    tp = ckpt.get("training_params", {})

    d_model = tp.get("d_model", 64)
    nhead = tp.get("nhead", 2)
    num_layers = tp.get("num_layers", 4)
    mask_ratio = tp.get("mask_ratio", 0.15)
    value_mask_ratio = tp.get("value_mask_ratio", 0.0)
    n_predict_steps = tp.get("n_predict_steps")
    causal = tp.get("causal", False)

    is_patched = "patch_embedding.proj.weight" in model_state
    if is_patched:
        patch_len = tp.get("patch_length", 8)
        proj_w = model_state["patch_embedding.proj.weight"]  # (d_model, patch_len * f_in)
        f_in = proj_w.shape[1] // patch_len
        dropout = tp.get("dropout", 0.1)
        if n_predict_steps is not None:
            model = PatchedMaskedPredictiveTemporalTransformer(
                f_in=f_in,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                patch_len=patch_len,
                dropout=dropout,
                mask_ratio=mask_ratio,
                value_mask_ratio=value_mask_ratio,
                n_predict_steps=n_predict_steps,
                causal=causal,
            )
        else:
            model = PatchedMaskedTemporalTransformer(
                f_in=f_in,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                patch_len=patch_len,
                dropout=dropout,
                mask_ratio=mask_ratio,
                value_mask_ratio=value_mask_ratio,
                causal=causal,
            )
    else:
        f_in = model_state["embedding.weight"].shape[1]
        if n_predict_steps is not None:
            model = MaskedPredictiveTemporalTransformer(
                f_in=f_in,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mask_ratio=mask_ratio,
                value_mask_ratio=value_mask_ratio,
                n_predict_steps=n_predict_steps,
                return_layer_mean=return_layer_mean,
                causal=causal,
            )
        else:
            model = MaskedTemporalTransformer(
                f_in=f_in,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                mask_ratio=mask_ratio,
                value_mask_ratio=value_mask_ratio,
                return_layer_mean=return_layer_mean,
                causal=causal,
            )
    model.load_state_dict(model_state, strict=True)
    return model.to(device)


def extract_latents(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    no_mask: bool = True,
) -> np.ndarray:
    """Run model in eval mode and collect latents per batch.

    Returns:
        For standard models: (N_windows, T, d_model)
        For patched models:  (N_windows, N_patches, d_model)  where N_patches = T // patch_len
    """
    model.eval()
    loader = DataLoader(
        MaskedTemporalDataset(windows),
        batch_size=batch_size,
        shuffle=False,
    )
    latents_list: list[np.ndarray] = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            B, T, _ = X.shape
            mask = torch.zeros(B, T, dtype=torch.bool, device=device) if no_mask else None
            out = model(X, mask=mask)
            latent = out[1]  # (B, T, d_model) or (B, N_patches, d_model) for patched
            latents_list.append(latent.cpu().numpy())
    return np.concatenate(latents_list, axis=0)


def extract_session_latents_for_hmm(
    session_path: str | Path,
    checkpoint_path: str | Path,
    window_size: int = 128,
    batch_size: int = 64,
    device: torch.device | None = None,
    no_mask: bool = True,
) -> np.ndarray:
    """Extract patch-level latents from a single session for HMM fitting.

    Uses non-overlapping windows (stride=window_size) so each frame appears exactly once.
    Loads the full session without trimming pre-event frames.

    Returns:
        latents: (N_total, d_model) where N_total = N_windows * N_patches_per_window
                 for patched models, or N_windows * window_size for standard models.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = _load_model_from_checkpoint(Path(checkpoint_path), device)
    windows, _ = load_single_session_windows(
        session_path, stride=window_size, full_file=True, window_size=window_size
    )
    latents = extract_latents(model, windows, device, batch_size=batch_size, no_mask=no_mask)
    # (N_windows, T_or_N, d_model) → (N_total, d_model)
    return latents.reshape(-1, latents.shape[-1])


def main():
    parser = argparse.ArgumentParser(description="Extract transformer latents for HMM analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--data", type=str, default=None, help="Path to raw data file or dir")
    parser.add_argument("--preprocessed-data", type=str, default=None, help="Path to preprocessed .npz")
    parser.add_argument("--output", type=str, required=True, help="Output .npz path")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--no-mask", action="store_true", default=True, help="Use unmasked forward (default: True)")
    parser.add_argument("--mask", action="store_true", dest="use_mask", help="Use random masking")
    parser.add_argument("--format", type=str, default="per_timestep", choices=["per_window", "per_timestep"])
    parser.add_argument("--layer-mean", action="store_true", help="Use mean over encoder layers")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1, help="Window stride. Use --stride=<window-size> for non-overlapping windows (recommended for HMM)")
    parser.add_argument("--full-file", action="store_true", help="Load full file, no trim")
    parser.add_argument("--start-times", type=str, default=None)
    parser.add_argument("--mmap", action="store_true", help="Memory-map preprocessed .npz")
    args = parser.parse_args()

    no_mask = not args.use_mask
    device = torch.device(args.device or "mps" if torch.backends.mps.is_available() else "cpu")

    # Load checkpoint and model
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    model = _load_model_from_checkpoint(ckpt_path, device, return_layer_mean=args.layer_mean)

    # Load data
    if args.preprocessed_data:
        mmap_mode = "r" if args.mmap else None
        train_w, val_w, test_w, feature_names, _, _, _ = load_preprocessed_dataset(
            args.preprocessed_data, mmap_mode=mmap_mode
        )
        with np.load(args.preprocessed_data, allow_pickle=True) as npz:
            window_size = int(npz["window_size"]) if "window_size" in npz else args.window_size
    elif args.data:
        data, _ = load_data_path(
            args.data,
            trim_before_dist_head=not args.full_file,
            start_times_path=args.start_times,
        )
        windows, feature_names, _ = prepare_masked_transformer_data(
            data, window_size=args.window_size, stride=args.stride
        )
        train_w, val_w, test_w = temporal_train_val_test_split(
            windows, args.val_ratio, args.test_ratio
        )
        window_size = args.window_size
    else:
        print("Provide --data or --preprocessed-data", file=sys.stderr)
        sys.exit(1)

    # Select split(s)
    if args.split == "train":
        windows_list = [train_w]
        split_names = ["train"]
    elif args.split == "val":
        windows_list = [val_w]
        split_names = ["val"]
    elif args.split == "test":
        windows_list = [test_w]
        split_names = ["test"]
    else:
        windows_list = [train_w, val_w, test_w]
        split_names = ["train", "val", "test"]

    all_latents: list[np.ndarray] = []
    all_splits: list[str] = []

    for wins, name in zip(windows_list, split_names):
        if wins is None or len(wins) == 0:
            continue
        wins_arr = np.asarray(wins, dtype=np.float32)
        lat = extract_latents(
            model, wins_arr, device,
            batch_size=args.batch_size,
            no_mask=no_mask,
        )
        if args.format == "per_timestep":
            # Use last position of each window: (N, d_model)
            lat = lat[:, -1, :]
        all_latents.append(lat)
        all_splits.extend([name] * len(lat))

    latents = np.concatenate(all_latents, axis=0)
    d_model = latents.shape[-1]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fn_arr = np.array(feature_names, dtype=object) if feature_names else np.array([])
    np.savez(
        out_path,
        latents=latents,
        window_size=window_size,
        d_model=d_model,
        split=np.array(all_splits, dtype=object),
        feature_names=fn_arr,
        format=args.format,
    )
    print(f"Saved {latents.shape} latents to {out_path}")


if __name__ == "__main__":
    main()
