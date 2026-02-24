"""Single-session behavioral segmentation via window-mean latents + K-means.

Fixes the "artificially long state" artifact from patch-level clustering:
  patch-level latents  →  average every group_size patches  →  one vector per window
  → K-means  →  one state label per ~2 s of behaviour

Usage
-----
uv run python -m behavex.models.analyze_session \\
    --latents   /Users/thomasbush/session_latents.npy \\
    --group_size 16 \\
    --k_values 5 8 10 15 \\
    --output_dir /Users/thomasbush/session_analysis \\
    --fps 62.4

The --group_size should equal window_size // patch_len from your checkpoint
(window_size=128, patch_len=8 → group_size=16).
If you are unsure, run with --detect_group_size to print candidate sizes.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


def pool_patches(latents: np.ndarray, group_size: int) -> np.ndarray:
    """Average every group_size consecutive patch vectors into one window vector.

    Args:
        latents:    (N_patches_total, D) — patch-level latents.
        group_size: Number of patches per window (window_size // patch_len).

    Returns:
        (N_windows, D) — one mean vector per window.
    """
    n, d = latents.shape
    n_complete = (n // group_size) * group_size
    if n_complete < n:
        latents = latents[:n_complete]
    return latents.reshape(-1, group_size, d).mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def _dwell_times(states: np.ndarray, K: int) -> np.ndarray:
    runs: list[list[int]] = [[] for _ in range(K)]
    i = 0
    while i < len(states):
        s = int(states[i])
        j = i + 1
        while j < len(states) and states[j] == s:
            j += 1
        runs[s].append(j - i)
        i = j
    return np.array([float(np.mean(r)) if r else 0.0 for r in runs])


def _occupancy(states: np.ndarray, K: int) -> np.ndarray:
    counts = np.array([(states == s).sum() for s in range(K)], dtype=float)
    return counts / max(len(states), 1)


def _entropy(occ: np.ndarray) -> float:
    return float(-np.sum(occ * np.log(occ + 1e-12)))


def _transition_matrix(states: np.ndarray, K: int) -> np.ndarray:
    T = np.zeros((K, K), dtype=float)
    for i in range(len(states) - 1):
        T[int(states[i]), int(states[i + 1])] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _ethogram(states: np.ndarray, K: int, fps: float,
              out_path: Path, n_show: int = 1500) -> None:
    """Coloured ethogram for the first n_show windows."""
    subset = states[:n_show]
    T = len(subset)
    time = np.arange(T) / fps

    cmap = plt.get_cmap("tab10", K)
    fig, ax = plt.subplots(figsize=(14, 2))
    for t in range(T - 1):
        ax.axvspan(time[t], time[t + 1],
                   facecolor=cmap(subset[t]), alpha=0.9, linewidth=0)
    ax.axvspan(time[-1], time[-1] + 1 / fps,
               facecolor=cmap(subset[-1]), alpha=0.9, linewidth=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(s)) for s in range(K)]
    ax.legend(handles, [f"S{s}" for s in range(K)],
              loc="upper right", ncol=K, fontsize=7, framealpha=0.8)
    ax.set_xlim(0, time[-1] + 1 / fps)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (windows)")
    ax.set_title(f"Ethogram — first {T} windows ({T / fps:.1f} s @ window-level fps)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _occupancy_bar(occ: np.ndarray, K: int, dwell: np.ndarray,
                   out_path: Path, window_sec: float) -> None:
    cmap = plt.get_cmap("tab10", K)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    ax = axes[0]
    ax.bar(range(K), occ, color=[cmap(i) for i in range(K)], edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_ylabel("Occupancy fraction")
    ax.set_title("State occupancy")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    dwell_sec = dwell * window_sec
    ax.bar(range(K), dwell_sec, color=[cmap(i) for i in range(K)], edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_ylabel("Mean dwell (seconds)")
    ax.set_title("Mean dwell time per state")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _transition_heatmap(T_mat: np.ndarray, K: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(4, K * 0.7), max(4, K * 0.7)))
    im = ax.imshow(T_mat, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Transition probability")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)], fontsize=8)
    ax.set_yticklabels([f"S{i}" for i in range(K)], fontsize=8)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Empirical transition matrix")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{T_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if T_mat[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _umap_plot(latents_norm: np.ndarray, labels: np.ndarray,
               K: int, out_path: Path,
               n_sub: int = 20_000) -> None:
    """UMAP scatter coloured by cluster label (subsampled for speed)."""
    try:
        from umap import UMAP
    except ImportError:
        warnings.warn("umap-learn not installed; skipping UMAP plot. "
                      "Install with: pip install umap-learn")
        return

    rng = np.random.default_rng(0)
    n = len(latents_norm)
    idx = rng.choice(n, min(n_sub, n), replace=False)
    sub = latents_norm[idx]
    sub_labels = labels[idx]

    reducer = UMAP(n_components=2, random_state=0, n_neighbors=30, min_dist=0.1)
    emb = reducer.fit_transform(sub)

    cmap = plt.get_cmap("tab10", K)
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in range(K):
        mask = sub_labels == s
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   s=2, alpha=0.4, color=cmap(s), label=f"S{s}", rasterized=True)
    ax.legend(markerscale=4, fontsize=8, loc="best")
    ax.set_title(f"UMAP of window-mean latents (K={K}, n={len(sub):,})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved UMAP              → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis loop
# ─────────────────────────────────────────────────────────────────────────────


def run_analysis(
    latents_path: str,
    group_size: int,
    k_values: List[int],
    output_dir: str,
    fps: float = 62.4,
    patch_len: int = 4,
    budget: int = 200_000,
    seed: int = 0,
    umap: bool = True,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load & pool ──────────────────────────────────────────────────────────
    print(f"Loading {latents_path} …")
    raw = np.load(latents_path, mmap_mode="r")
    print(f"  Patch-level shape : {raw.shape}  "
          f"({raw.nbytes / 1e9:.2f} GB if fully loaded)")

    print(f"  Averaging every {group_size} patches → window-level …")
    win_latents = pool_patches(np.asarray(raw), group_size)
    # Each row = one window = group_size patches × patch_len frames / fps
    window_sec = group_size * patch_len / fps
    print(f"  Window-level shape: {win_latents.shape}  "
          f"({window_sec:.3f} s per window  "
          f"[{group_size} patches × {patch_len} frames / {fps} fps])")

    # ── Normalise ────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    n = len(win_latents)
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(n, min(budget, n), replace=False)
    sub_norm = scaler.fit_transform(win_latents[sub_idx])
    win_norm = scaler.transform(win_latents)   # window-level, fits in RAM

    summary = {}

    for K in sorted(k_values):
        print(f"\n── K={K} ──")
        k_dir = out / f"K_{K:02d}"
        k_dir.mkdir(exist_ok=True)

        km = MiniBatchKMeans(
            n_clusters=K, random_state=seed,
            n_init=5, batch_size=min(10_000, len(sub_norm)),
        )
        km.fit(sub_norm)
        labels = km.predict(win_norm).astype(np.int64)

        # Silhouette on subsample
        n_sil = min(5_000, len(sub_norm))
        sil_idx = rng.choice(len(sub_norm), n_sil, replace=False)
        sil = float(silhouette_score(sub_norm[sil_idx],
                                     km.predict(sub_norm[sil_idx])))

        occ   = _occupancy(labels, K)
        dwell = _dwell_times(labels, K)                 # in windows
        T_mat = _transition_matrix(labels, K)
        ent   = _entropy(occ)

        print(f"  silhouette     : {sil:.4f}")
        print(f"  entropy        : {ent:.4f}  (max {np.log(K):.4f})")
        print(f"  mean dwell     : {dwell.mean():.1f} windows "
              f"= {dwell.mean() * window_sec:.1f} s")
        print(f"  per-state dwell: "
              + "  ".join(f"S{s}={dwell[s] * window_sec:.1f}s" for s in range(K)))

        # Save labels
        np.save(k_dir / "labels.npy", labels)

        # Plots
        _ethogram(labels, K, fps=1 / window_sec,   # windows/s → treat as fps
                  out_path=k_dir / "ethogram.png")
        _occupancy_bar(occ, K, dwell,
                       out_path=k_dir / "occupancy_dwell.png",
                       window_sec=window_sec)
        _transition_heatmap(T_mat, K, k_dir / "transition_matrix.png")
        if umap:
            _umap_plot(win_norm, labels, K, k_dir / "umap.png")

        summary[K] = {
            "silhouette": sil,
            "entropy": ent,
            "mean_dwell_windows": float(dwell.mean()),
            "mean_dwell_sec": float(dwell.mean() * window_sec),
            "per_state_dwell_sec": (dwell * window_sec).tolist(),
            "occupancy": occ.tolist(),
        }

    # ── K-sweep summary plot ─────────────────────────────────────────────────
    ks = sorted(summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (metric, label) in zip(axes, [
        ("silhouette",     "Silhouette (↑ better)"),
        ("entropy",        "Occupancy entropy (nats)"),
        ("mean_dwell_sec", "Mean dwell (seconds)"),
    ]):
        vals = [summary[k][metric] for k in ks]
        ax.plot(ks, vals, "o-", color="C0", lw=2)
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ks)
    plt.suptitle(f"Window-mean K-means sweep  "
                 f"(group_size={group_size}, {window_sec:.2f} s/window)")
    plt.tight_layout()
    fig.savefig(out / "k_sweep_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nDone. Results in {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-session K-means on window-mean latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--latents",     required=True,
                   help="Path to session_latents.npy (patch-level, shape N×D)")
    p.add_argument("--group_size",  type=int, default=16,
                   help="Patches per window (= window_size // patch_len). "
                        "Consecutive patches are averaged into one vector.")
    p.add_argument("--k_values",    type=int, nargs="+", default=[5, 8, 10, 15],
                   help="K values to try")
    p.add_argument("--output_dir",  default="session_analysis",
                   help="Output directory")
    p.add_argument("--fps",         type=float, default=62.4,
                   help="Camera frame rate (used for dwell time labels)")
    p.add_argument("--patch_len",   type=int,   default=4,
                   help="Frames per patch (from transformer config). "
                        "Used to convert patch-counts to seconds.")
    p.add_argument("--budget",      type=int, default=200_000,
                   help="Frames used for KMeans fitting (subsample)")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--no_umap",     action="store_true",
                   help="Skip UMAP (faster if umap-learn not installed)")
    p.add_argument("--detect_group_size", action="store_true",
                   help="Print likely group sizes given N and quit")
    args = p.parse_args()

    if args.detect_group_size:
        raw = np.load(args.latents, mmap_mode="r")
        n = raw.shape[0]
        print(f"N patches = {n}")
        for g in [8, 10, 12, 16, 20, 32]:
            print(f"  group_size={g:2d} → {n // g:,} windows "
                  f"({g * args.patch_len / args.fps:.3f} s/window)")
        return

    run_analysis(
        latents_path=args.latents,
        group_size=args.group_size,
        k_values=args.k_values,
        output_dir=args.output_dir,
        fps=args.fps,
        patch_len=args.patch_len,
        budget=args.budget,
        seed=args.seed,
        umap=not args.no_umap,
    )


if __name__ == "__main__":
    main()
