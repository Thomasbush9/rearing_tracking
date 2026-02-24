"""K-means + temporal smoothing behavioral segmentation (Plan B).

Three-phase pipeline as fallback / comparison to HMM segmentation:
  A – K sweep: MiniBatchKMeans at several K values + silhouette / Davies-Bouldin
  B – GMM comparison at the best K chosen from Phase A
  C – Temporal smoothing ablation (several window widths at best K)

Direct comparison story
-----------------------
  HMM        → finds temporal structure via EM (global sequence model)
  K-means    → finds spatial structure in latent space (no temporal constraint)
  K-means + smoothing → approximates temporal structure post-hoc

Example
-------
uv run python -m behavex.models.run_kmeans_strategy \\
    --latents_path workspace/hmm_strategy_results/latents.npy \\
    --output_dir   workspace/kmeans1 \\
    --k_values 5 10 15 20 30 \\
    --smooth_windows 1 5 11 25 \\
    --budget 200000 \\
    --seed 0
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from behavex.models.test_hmm import plot_ethogram, plot_state_occupancy


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to Python native types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _compute_dwell_times(states: np.ndarray, n_states: int) -> np.ndarray:
    """Return mean dwell time (frames) per state via run-length encoding."""
    runs: List[List[int]] = [[] for _ in range(n_states)]
    i = 0
    while i < len(states):
        s = int(states[i])
        j = i + 1
        while j < len(states) and states[j] == s:
            j += 1
        runs[s].append(j - i)
        i = j
    return np.array([float(np.mean(r)) if r else 0.0 for r in runs])


def _occupancy_entropy(states: np.ndarray, n_states: int) -> float:
    """Shannon entropy (nats) of state occupancy fractions."""
    counts = np.array(
        [(states == s).sum() for s in range(n_states)], dtype=float
    )
    fracs = counts / max(len(states), 1)
    return float(-np.sum(fracs * np.log(fracs + 1e-12)))


def _compute_transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    """Empirical row-normalised transition matrix from a state sequence."""
    T = np.zeros((n_states, n_states), dtype=float)
    for i in range(len(states) - 1):
        T[int(states[i]), int(states[i + 1])] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def _fit_kmeans_chunked(
    km: "MiniBatchKMeans",
    scaler: "StandardScaler",
    latents: np.ndarray,
    chunk_size: int,
    n_epochs: int = 3,
) -> "MiniBatchKMeans":
    """Fit MiniBatchKMeans incrementally over the full dataset using partial_fit.

    Scaler must already be fitted (e.g. on a subsample); latents are normalised
    on-the-fly per chunk.  Uses n_epochs passes so centroids converge.

    Args:
        km:         Unfitted MiniBatchKMeans instance.
        scaler:     Already-fitted StandardScaler.
        latents:    Full latent array (N, D).  May be memory-mapped.
        chunk_size: Rows per chunk.
        n_epochs:   Number of full passes through the data.

    Returns:
        Fitted km.
    """
    n = len(latents)
    for epoch in range(n_epochs):
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_norm = scaler.transform(np.asarray(latents[start:end]))
            km.partial_fit(chunk_norm)
    return km


def _predict_chunked(
    model,
    scaler: "StandardScaler",
    latents: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Transform + predict in chunks to avoid materialising the full normalised array.

    Args:
        model:      Fitted sklearn model with a .predict() method.
        scaler:     Fitted StandardScaler to apply before prediction.
        latents:    Raw latent array (N, D).  May be a memory-mapped array.
        chunk_size: Number of rows to process at once.

    Returns:
        Integer label array, shape (N,).
    """
    n = len(latents)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = np.asarray(latents[start:end])          # materialise mmap slice
        chunk_norm = scaler.transform(chunk)
        labels[start:end] = model.predict(chunk_norm)
    return labels


def smooth_states(
    states: np.ndarray, window: int, n_states: int
) -> np.ndarray:
    """Smooth discrete state sequence via windowed one-hot averaging.

    Args:
        states:   Integer state sequence, shape (T,).
        window:   Smoothing window width in frames.  window=1 → identity.
        n_states: Number of distinct states K.

    Returns:
        Smoothed integer state sequence, shape (T,).
    """
    if window <= 1:
        return states.copy()
    one_hot = np.eye(n_states, dtype=np.float32)[states]  # (T, K)
    smoothed = uniform_filter1d(one_hot, size=window, axis=0)
    return smoothed.argmax(axis=1).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — K sweep
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase_a(
    latents: np.ndarray,
    k_values: List[int],
    budget: int,
    seed: int,
    smooth_window: int,
    out_dir: Path,
    chunk_size: int = 500_000,
    fit_all: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Fit MiniBatchKMeans at each K and compute cluster quality metrics.

    Args:
        latents:       Full latent array (N, D).
        k_values:      List of K values to evaluate.
        budget:        Max frames used for fitting.
        seed:          Random seed.
        smooth_window: Window width for temporal smoothing used in metric
                       reporting (does not affect silhouette / DB on subsample).
        out_dir:       Directory for artefacts.

    Returns:
        Dict mapping K → metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(latents)
    n_sub = min(budget, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, n_sub, replace=False)
    sub = np.asarray(latents[idx])  # materialise subsample (mmap-safe)

    scaler = StandardScaler()
    sub_norm = scaler.fit_transform(sub)
    # full normalisation is done lazily inside _predict_chunked (no full copy)

    results: Dict[int, Dict[str, Any]] = {}
    for K in sorted(k_values):
        print(f"  [Phase A] K={K:2d} …", end=" ", flush=True)

        km = MiniBatchKMeans(
            n_clusters=K,
            random_state=seed,
            n_init=3,
            batch_size=min(10_000, n_sub),
        )
        if fit_all:
            _fit_kmeans_chunked(km, scaler, latents, chunk_size)
            sub_labels = km.predict(sub_norm)
        else:
            sub_labels = km.fit_predict(sub_norm)
        full_labels = _predict_chunked(km, scaler, latents, chunk_size)

        # Temporal smoothing for dwell / entropy metrics
        smoothed = smooth_states(full_labels, window=smooth_window, n_states=K)

        # Silhouette + Davies-Bouldin on subsample (too expensive on full data)
        n_sil = min(5_000, n_sub)
        sil_idx = rng.choice(n_sub, n_sil, replace=False)
        sil = float(silhouette_score(sub_norm[sil_idx], sub_labels[sil_idx]))
        db = float(davies_bouldin_score(sub_norm, sub_labels))

        dwell = _compute_dwell_times(smoothed, K)
        occ_ent = _occupancy_entropy(smoothed, K)

        results[K] = {
            "K": K,
            "silhouette": sil,
            "davies_bouldin": db,
            "inertia": float(km.inertia_),
            "mean_dwell": float(dwell.mean()),
            "occupancy_entropy": occ_ent,
            "state_occupancy": [
                float((smoothed == s).sum()) / max(len(smoothed), 1)
                for s in range(K)
            ],
        }

        print(
            f"sil={sil:.4f}  DB={db:.4f}  "
            f"dwell={dwell.mean():.2f}  entropy={occ_ent:.4f}"
        )

        k_dir = out_dir / f"K_{K:02d}"
        k_dir.mkdir(exist_ok=True)
        np.save(k_dir / "labels_raw.npy", full_labels)
        np.save(k_dir / "labels_smoothed.npy", smoothed)
        with open(k_dir / "metrics.json", "w") as fh:
            json.dump(_to_json_safe(results[K]), fh, indent=2)

        try:
            plot_ethogram(smoothed, K, k_dir / "ethogram.png")
            plot_state_occupancy(smoothed, K, k_dir / "state_occupancy.png")
        except Exception as exc:
            warnings.warn(f"Phase A plotting failed (K={K}): {exc}")

    # ── Best K by silhouette ──────────────────────────────────────────────────
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(
        f"\n  [Phase A] Best K by silhouette: {best_k} "
        f"(sil={results[best_k]['silhouette']:.4f})"
    )
    with open(out_dir / "best_k.json", "w") as fh:
        json.dump({"best_k": best_k, "criterion": "silhouette"}, fh)

    # ── 4-panel summary plot ──────────────────────────────────────────────────
    ks = sorted(results)
    metrics_to_plot = [
        ("silhouette", "Silhouette score (↑ better)"),
        ("davies_bouldin", "Davies-Bouldin index (↓ better)"),
        ("occupancy_entropy", "Occupancy entropy (nats)"),
        ("mean_dwell", f"Mean dwell time (frames, window={smooth_window})"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, label) in zip(axes.flat, metrics_to_plot):
        vals = [results[k][metric] for k in ks]
        ax.plot(ks, vals, "o-", color="C0", lw=2)
        ax.axvline(best_k, ls="--", color="C1", alpha=0.7, label=f"best K={best_k}")
        ax.set_xlabel("Number of clusters K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Phase A: K sweep  (budget={budget:,}, smooth_window={smooth_window})"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "summary_k_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Inertia elbow plot ────────────────────────────────────────────────────
    inertias = [results[k]["inertia"] for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "o-", color="C0", lw=2)
    ax.axvline(best_k, ls="--", color="C1", alpha=0.7, label=f"best K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means inertia (elbow method)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "inertia_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "all_metrics.json", "w") as fh:
        json.dump(_to_json_safe(results), fh, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — GMM comparison
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase_b(
    latents: np.ndarray,
    best_k: int,
    budget: int,
    seed: int,
    smooth_window: int,
    out_dir: Path,
    chunk_size: int = 500_000,
) -> Dict[str, Any]:
    """Fit GMM at best_k and compare to K-means via BIC and dwell time.

    Args:
        latents:       Full latent array (N, D).
        best_k:        Number of mixture components.
        budget:        Max frames used for fitting.
        seed:          Random seed.
        smooth_window: Window width for temporal smoothing in metric reporting.
        out_dir:       Directory for artefacts.
        chunk_size:    Rows per chunk for memory-efficient prediction.

    Returns:
        Metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(latents)
    n_sub = min(budget, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, n_sub, replace=False)
    sub = np.asarray(latents[idx])  # materialise subsample (mmap-safe)

    scaler = StandardScaler()
    sub_norm = scaler.fit_transform(sub)

    print(f"  [Phase B] Fitting GMM (K={best_k}) …", end=" ", flush=True)
    gmm = GaussianMixture(
        n_components=best_k,
        random_state=seed,
        n_init=3,
        covariance_type="diag",
    )
    gmm.fit(sub_norm)

    bic = float(gmm.bic(sub_norm))
    ll_per_frame = float(gmm.score(sub_norm))  # already per sample

    full_labels_gmm = _predict_chunked(gmm, scaler, latents, chunk_size)
    smoothed_gmm = smooth_states(full_labels_gmm, window=smooth_window, n_states=best_k)

    dwell = _compute_dwell_times(smoothed_gmm, best_k)
    occ_ent = _occupancy_entropy(smoothed_gmm, best_k)

    print(
        f"BIC={bic:.1f}  ll/frame={ll_per_frame:.4f}  "
        f"dwell={dwell.mean():.2f}  entropy={occ_ent:.4f}"
    )

    results: Dict[str, Any] = {
        "K": best_k,
        "bic": bic,
        "ll_per_frame": ll_per_frame,
        "mean_dwell": float(dwell.mean()),
        "occupancy_entropy": occ_ent,
        "state_occupancy": [
            float((smoothed_gmm == s).sum()) / max(len(smoothed_gmm), 1)
            for s in range(best_k)
        ],
    }

    np.save(out_dir / "labels_raw.npy", full_labels_gmm)
    np.save(out_dir / "labels_smoothed.npy", smoothed_gmm)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(_to_json_safe(results), fh, indent=2)

    try:
        plot_ethogram(smoothed_gmm, best_k, out_dir / "ethogram.png")
        plot_state_occupancy(smoothed_gmm, best_k, out_dir / "state_occupancy.png")
    except Exception as exc:
        warnings.warn(f"Phase B plotting failed: {exc}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — Temporal smoothing ablation
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase_c(
    latents: np.ndarray,
    best_k: int,
    budget: int,
    seed: int,
    smooth_windows: List[int],
    out_dir: Path,
    chunk_size: int = 500_000,
    fit_all: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """Ablation over smoothing window widths at best K.

    Fits KMeans once; sweeps window sizes.  Reports mean dwell time,
    occupancy entropy, and mean self-transition probability.

    Args:
        latents:        Full latent array (N, D).
        best_k:         Number of clusters.
        budget:         Max frames used for fitting.
        seed:           Random seed.
        smooth_windows: List of window widths to compare.
        out_dir:        Directory for artefacts.
        chunk_size:     Rows per chunk for memory-efficient prediction.

    Returns:
        Dict mapping window → metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(latents)
    n_sub = min(budget, n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, n_sub, replace=False)
    sub = np.asarray(latents[idx])  # materialise subsample (mmap-safe)

    scaler = StandardScaler()
    sub_norm = scaler.fit_transform(sub)

    # Fit KMeans once; vary smoothing
    km = MiniBatchKMeans(
        n_clusters=best_k,
        random_state=seed,
        n_init=3,
        batch_size=min(10_000, n_sub),
    )
    if fit_all:
        _fit_kmeans_chunked(km, scaler, latents, chunk_size)
    else:
        km.fit(sub_norm)
    full_labels = _predict_chunked(km, scaler, latents, chunk_size)

    results: Dict[int, Dict[str, Any]] = {}
    for w in sorted(smooth_windows):
        smoothed = smooth_states(full_labels, window=w, n_states=best_k)
        dwell = _compute_dwell_times(smoothed, best_k)
        occ_ent = _occupancy_entropy(smoothed, best_k)
        T_mat = _compute_transition_matrix(smoothed, best_k)
        self_trans = float(np.diag(T_mat).mean())

        results[w] = {
            "window": w,
            "mean_dwell": float(dwell.mean()),
            "occupancy_entropy": occ_ent,
            "mean_self_transition": self_trans,
        }

        print(
            f"  [Phase C] window={w:2d}: dwell={dwell.mean():.2f}  "
            f"entropy={occ_ent:.4f}  self_trans={self_trans:.4f}"
        )

        w_dir = out_dir / f"window_{w:02d}"
        w_dir.mkdir(exist_ok=True)
        np.save(w_dir / "labels.npy", smoothed)
        np.save(w_dir / "transmat.npy", T_mat)
        with open(w_dir / "metrics.json", "w") as fh:
            json.dump(_to_json_safe(results[w]), fh, indent=2)

        try:
            plot_ethogram(smoothed, best_k, w_dir / "ethogram.png")
            plot_state_occupancy(smoothed, best_k, w_dir / "state_occupancy.png")
        except Exception as exc:
            warnings.warn(f"Phase C plotting failed (window={w}): {exc}")

    # ── 3-panel ablation summary ──────────────────────────────────────────────
    ws = sorted(results)
    panels = [
        ("mean_dwell", "Mean dwell time (frames)"),
        ("occupancy_entropy", "Occupancy entropy (nats)"),
        ("mean_self_transition", "Mean self-transition P(s→s)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (metric, label) in zip(axes, panels):
        vals = [results[w][metric] for w in ws]
        ax.plot(ws, vals, "o-", color="C0", lw=2)
        ax.set_xlabel("Smoothing window (frames)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ws)
    plt.suptitle(f"Phase C: Temporal smoothing ablation (K={best_k})")
    plt.tight_layout()
    fig.savefig(out_dir / "smoothing_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "all_metrics.json", "w") as fh:
        json.dump(_to_json_safe(results), fh, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def run_strategy(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mmap_mode = "r" if args.mmap else None
    print(f"Loading latents from {args.latents_path} …"
          + (" (mmap)" if args.mmap else ""))
    latents = np.load(args.latents_path, mmap_mode=mmap_mode)
    ram_gb = latents.nbytes / 1e9
    print(f"  Latents shape: {latents.shape}  ({ram_gb:.2f} GB if fully loaded)")

    phases = {p.upper() for p in args.phases}
    best_k: Optional[int] = None

    if "A" in phases:
        print("\n=== Phase A: K sweep ===")
        a_results = _run_phase_a(
            latents=latents,
            k_values=args.k_values,
            budget=args.budget,
            seed=args.seed,
            smooth_window=args.default_smooth_window,
            out_dir=out_dir / "phase_a_k_sweep",
            chunk_size=args.chunk_size,
            fit_all=args.fit_all,
        )
        best_k = max(a_results, key=lambda k: a_results[k]["silhouette"])

    if "B" in phases:
        if best_k is None:
            if args.best_k is None:
                raise ValueError(
                    "--best_k is required when Phase A is skipped."
                )
            best_k = args.best_k
        print(f"\n=== Phase B: GMM comparison (K={best_k}) ===")
        _run_phase_b(
            latents=latents,
            best_k=best_k,
            budget=args.budget,
            seed=args.seed,
            smooth_window=args.default_smooth_window,
            out_dir=out_dir / "phase_b_gmm",
            chunk_size=args.chunk_size,
        )

    if "C" in phases:
        if best_k is None:
            if args.best_k is None:
                raise ValueError(
                    "--best_k is required when Phase A is skipped."
                )
            best_k = args.best_k
        print(f"\n=== Phase C: Smoothing ablation (K={best_k}) ===")
        _run_phase_c(
            latents=latents,
            best_k=best_k,
            budget=args.budget,
            seed=args.seed,
            smooth_windows=args.smooth_windows,
            out_dir=out_dir / "phase_c_smoothing",
            chunk_size=args.chunk_size,
            fit_all=args.fit_all,
        )

    print(f"\nAll done. Results in: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="K-means + temporal smoothing behavioral segmentation (Plan B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--latents_path",
        type=str,
        required=True,
        help="Path to latents .npy file, shape (N, D)",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="kmeans_strategy_results",
        help="Output directory for all artefacts",
    )
    p.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["A", "B", "C"],
        help="Phases to run: A (K sweep), B (GMM comparison), C (smoothing ablation)",
    )
    p.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 30],
        help="K values to evaluate in Phase A",
    )
    p.add_argument(
        "--smooth_windows",
        type=int,
        nargs="+",
        default=[1, 3, 5, 11, 25],
        help="Smoothing window sizes (frames) for Phase C ablation",
    )
    p.add_argument(
        "--default_smooth_window",
        type=int,
        default=5,
        help="Default smoothing window used in Phase A and B for metric reporting",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=200_000,
        help="Max frames used for fitting (subsampled randomly if N > budget)",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument(
        "--best_k",
        type=int,
        default=None,
        help="Override best K (required when Phase A is skipped)",
    )
    p.add_argument(
        "--chunk_size",
        type=int,
        default=500_000,
        help=(
            "Rows processed at once during full-data prediction. "
            "Reduces peak RAM usage: only chunk_size × D × 4 bytes of "
            "normalised data are held in memory at any time (default: 500 000)."
        ),
    )
    p.add_argument(
        "--mmap",
        action="store_true",
        default=False,
        help=(
            "Memory-map the latents file instead of loading it fully into RAM. "
            "Use when the file is larger than available RAM. "
            "Prediction is still chunked; fitting subsample is loaded normally."
        ),
    )
    p.add_argument(
        "--fit_all",
        action="store_true",
        default=False,
        help=(
            "Fit KMeans on the full dataset using partial_fit (streamed in chunks). "
            "Scaler stats are computed from --budget subsample; KMeans sees all N frames "
            "for n_epochs=3 passes. Ignored for GMM in Phase B (no partial_fit in sklearn)."
        ),
    )

    args = p.parse_args()
    run_strategy(args)


if __name__ == "__main__":
    main()
