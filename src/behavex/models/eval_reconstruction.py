"""Per-feature reconstruction quality of the masked transformer.

Runs the trained model on a held-out split with random patch masking, then
measures per-feature MSE on the masked positions only.  This answers:
"Which behavioral features does the transformer reconstruct most accurately
from temporal context?" — high-accuracy reconstruction of a feature means
the model has learned its temporal structure and cross-feature dependencies.

Usage (on the VM)
-----------------
python -m behavex.models.eval_reconstruction \\
    --checkpoint  /workspace/runs/best/model.pt \\
    --dataset     /workspace/data/out_s16.npz \\
    --split       test \\
    --batch_size  512 \\
    --mask_ratio  0.5 \\
    --n_batches   200 \\
    --output_dir  /workspace/eval_recon

Outputs
-------
    mse_per_feature.json      — per-feature MSE dict (feature_name → mse)
    mse_barplot.png           — bar chart sorted by MSE (ascending)
    mse_vs_fstat.png          — scatter: reconstruction MSE vs K-means F-stat
                                (only if --fstat_json provided)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from behavex.models.extract_latents import _load_model_from_checkpoint
from behavex.models.train_transformer import load_preprocessed_dataset
from behavex.models.transformer import create_patch_mask


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _eval_reconstruction(
    checkpoint_path: Path,
    dataset_path: Path,
    split: str,
    batch_size: int,
    mask_ratio: Optional[float],
    n_batches: Optional[int],
    device: torch.device,
) -> tuple[np.ndarray, List[str]]:
    """Return (mse_per_feature, feature_names).

    mse_per_feature[i] = mean squared error for feature i, averaged over all
    masked patch positions across all processed batches.
    """
    # ── Load model ────────────────────────────────────────────────────────────
    model, feature_names_ckpt, _ = _load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    is_patched = hasattr(model, "patch_embedding")
    effective_mask_ratio = mask_ratio if mask_ratio is not None else model.mask_ratio
    print(f"Model type    : {'patched' if is_patched else 'standard'}")
    print(f"Mask ratio    : {effective_mask_ratio}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    train_w, val_w, test_w, feature_names_ds, _, _, _, _ = load_preprocessed_dataset(
        dataset_path
    )
    split_map = {"train": train_w, "val": val_w, "test": test_w}
    if split not in split_map:
        raise ValueError(f"--split must be one of {list(split_map)}, got '{split}'")
    windows_np = split_map[split]
    print(f"Split '{split}' : {windows_np.shape[0]:,} windows × "
          f"{windows_np.shape[1]} timesteps × {windows_np.shape[2]} features")

    feature_names = feature_names_ds or feature_names_ckpt or [
        f"f{i}" for i in range(windows_np.shape[2])
    ]

    dataset = TensorDataset(torch.from_numpy(windows_np.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    F = windows_np.shape[2]
    per_feature_sq_sum = np.zeros(F, dtype=np.float64)
    per_feature_count = np.zeros(F, dtype=np.float64)

    # ── Inference ─────────────────────────────────────────────────────────────
    batch_iter = iter(loader)
    n_done = 0
    with torch.no_grad():
        for (X,) in batch_iter:
            X = X.to(device)  # (B, T, F)
            B, T, _ = X.shape

            if is_patched:
                P = model.patch_len
                N = T // P
                patch_mask = create_patch_mask(B, N, effective_mask_ratio, device)
                out = model(X, mask=patch_mask)
            else:
                from behavex.models.transformer import create_timestep_mask
                ts_mask = create_timestep_mask(B, T, effective_mask_ratio, device)
                out = model(X, mask=ts_mask)
            recon, _, timestep_mask = out[0], out[1], out[2]

            # Only compute MSE on masked positions
            # timestep_mask: (B, T) bool, True = masked
            err2 = (recon - X) ** 2  # (B, T, F)
            mask_3d = timestep_mask.unsqueeze(-1).float()  # (B, T, 1) broadcast → (B, T, F)
            per_feature_sq_sum += (err2 * mask_3d).sum(dim=(0, 1)).cpu().numpy()
            per_feature_count += mask_3d.sum(dim=(0, 1)).cpu().numpy()

            n_done += 1
            if n_batches is not None and n_done >= n_batches:
                break

    mse = per_feature_sq_sum / np.maximum(per_feature_count, 1.0)
    print(f"Evaluated {n_done} batches ({int(per_feature_count.max()):,} masked positions per feature max)")
    return mse, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _barplot(mse: np.ndarray, feature_names: List[str], out_path: Path) -> None:
    sort_idx = np.argsort(mse)  # ascending: best reconstructed first
    names_sorted = [feature_names[i] for i in sort_idx]
    mse_sorted = mse[sort_idx]

    n = len(names_sorted)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.4), 4.5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, n))
    ax.bar(range(n), mse_sorted, color=colors, edgecolor="none")
    ax.set_xticks(range(n))
    ax.set_xticklabels(names_sorted, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Reconstruction MSE (masked positions)")
    ax.set_title("Per-feature reconstruction MSE  (left = best reconstructed)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved MSE bar plot         → {out_path}")


def _scatter_vs_fstat(
    mse: np.ndarray,
    feature_names: List[str],
    fstat_path: Path,
    out_path: Path,
) -> None:
    """Scatter of reconstruction MSE vs K-means F-statistic per feature.

    fstat_path: JSON saved by eval_reconstruction itself or produced externally.
    Expected format: {"feature_name": f_statistic, ...}
    """
    with open(fstat_path) as fh:
        fstat_dict = json.load(fh)

    xs, ys, labels = [], [], []
    for i, name in enumerate(feature_names):
        if name in fstat_dict:
            xs.append(fstat_dict[name])
            ys.append(mse[i])
            labels.append(name)

    if not xs:
        print("  No matching features in fstat_json — skipping scatter plot.")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xs, ys, s=30, alpha=0.8)
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("K-means F-statistic (state discriminability)")
    ax.set_ylabel("Reconstruction MSE (masked positions)")
    ax.set_title("Reconstruction quality vs state discriminability")
    ax.grid(alpha=0.3)
    # Correlation annotation
    if len(xs) >= 3:
        r = float(np.corrcoef(xs, ys)[0, 1])
        ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes,
                fontsize=9, va="top")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved MSE vs F-stat scatter → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Per-feature reconstruction MSE of the trained masked transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--dataset", required=True,
                   help="Path to preprocessed dataset (.npz or .h5)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Which data split to evaluate on")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Batch size for inference")
    p.add_argument("--mask_ratio", type=float, default=None,
                   help="Mask ratio for evaluation (default: use model's trained mask_ratio)")
    p.add_argument("--n_batches", type=int, default=None,
                   help="Max number of batches to process (default: all)")
    p.add_argument("--output_dir", default="eval_recon",
                   help="Directory for output files")
    p.add_argument("--fstat_json", default=None,
                   help="Optional JSON file mapping feature_name → F-statistic "
                        "(from compare_sessions or a prior eval run) for scatter plot")
    p.add_argument("--device", default=None,
                   help="Device string (e.g. 'cuda', 'cpu'). Auto-detected if not set.")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    mse, feature_names = _eval_reconstruction(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=Path(args.dataset),
        split=args.split,
        batch_size=args.batch_size,
        mask_ratio=args.mask_ratio,
        n_batches=args.n_batches,
        device=device,
    )

    # Save JSON
    mse_dict = {name: float(mse[i]) for i, name in enumerate(feature_names)}
    json_path = out / "mse_per_feature.json"
    with open(json_path, "w") as fh:
        json.dump(mse_dict, fh, indent=2)
    print(f"  Saved MSE JSON             → {json_path}")

    # Plots
    _barplot(mse, feature_names, out / "mse_barplot.png")

    if args.fstat_json:
        _scatter_vs_fstat(mse, feature_names, Path(args.fstat_json),
                          out / "mse_vs_fstat.png")

    # Print top-5 best and worst
    sort_idx = np.argsort(mse)
    print("\nBest reconstructed features (lowest MSE):")
    for i in sort_idx[:5]:
        print(f"  {feature_names[i]:35s}  MSE={mse[i]:.4f}")
    print("\nWorst reconstructed features (highest MSE):")
    for i in sort_idx[-5:][::-1]:
        print(f"  {feature_names[i]:35s}  MSE={mse[i]:.4f}")

    print(f"\nDone. Results in {out.resolve()}")


if __name__ == "__main__":
    main()
