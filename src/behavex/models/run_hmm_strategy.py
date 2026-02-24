"""Structured HMM experiment runner implementing a 5-phase evaluation protocol.

Phases:
  0 – Representation sanity (PCA spectrum, autocorrelation, smoothness)
  1 – Scale discovery (held-out LL vs data budget)
  2 – K sweep (model capacity selection via LL / BIC / occupancy / dwell)
  3 – Robustness to data selection (transition-matrix stability across seeds)
  4 – Null models & controls (shuffled / IID Gaussian Mixture)

Example
-------
# Phases 0-2
uv run python -m behavex.models.run_hmm_strategy \\
    --preprocessed_data /data/out_s16.npz \\
    --model_path        /runs/best.pt \\
    --output_dir        /hmm/strategy \\
    --phases 0 1 2 \\
    --n_workers 6

# Phases 3-4 (after picking K from Phase 2)
uv run python -m behavex.models.run_hmm_strategy \\
    --latents_path /hmm/strategy/latents.npy \\
    --output_dir   /hmm/strategy \\
    --phases 3 4 \\
    --phase3_k 20 --phase4_k 20
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from behavex.models.extract_latents import (
    _load_model_from_checkpoint,
    extract_latents,
)
from behavex.models.hmm_trainer import (
    HiddenMarkovModelTrainer,
    estimate_n_params,
)
from behavex.models.train_transformer import load_preprocessed_dataset
from behavex.models.test_hmm import (
    plot_ethogram,
    plot_state_occupancy,
    plot_transition_matrix,
)


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
    counts = np.array([(states == s).sum() for s in range(n_states)], dtype=float)
    fracs = counts / max(len(states), 1)
    return float(-np.sum(fracs * np.log(fracs + 1e-12)))


def _subsample_split(
    latents: np.ndarray,
    budget: int,
    seed: int,
    test_frac: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random subsample of *budget* points split (1-test_frac) / test_frac."""
    rng = np.random.default_rng(seed)
    n = len(latents)
    budget = min(budget, n)
    idx = rng.choice(n, budget, replace=False)
    n_test = max(1, int(budget * test_frac))
    return latents[idx[n_test:]], latents[idx[:n_test]]


def _align_transmat(A_ref: np.ndarray, A_query: np.ndarray) -> np.ndarray:
    """Permute rows/cols of A_query to minimise Frobenius distance to A_ref.

    Uses the Hungarian algorithm on pairwise row-distance cost matrix.
    """
    K = A_ref.shape[0]
    cost = np.array(
        [[float(np.sum((A_query[i] - A_ref[j]) ** 2)) for j in range(K)]
         for i in range(K)]
    )
    _, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = ref state matching query state i → reorder query to ref ordering
    inv_perm = np.argsort(col_ind)
    return A_query[inv_perm, :][:, inv_perm]


# ─────────────────────────────────────────────────────────────────────────────
# Top-level worker (module-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────


def _fit_single(
    latents: np.ndarray,
    n_states: int,
    budget: int,
    seed: int,
    out_dir_str: str,
    covariance_type: str,
    n_iter: int,
    compute_bic: bool = False,
) -> Dict[str, Any]:
    """Fit one HMM run. Saves artefacts to *out_dir_str*, returns metrics dict."""
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = _subsample_split(latents, budget, seed)
    n_train = len(train_data)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trainer = HiddenMarkovModelTrainer(
        n_states=n_states, covariance_type=covariance_type, device=device
    )
    trainer.fit([train_data], n_iter=n_iter, random_state=seed, sticky_prior=0.95)

    # Log-likelihoods (per frame)
    total_train_ll = float(np.sum(trainer.score_sequences([train_data])))
    total_test_ll = float(np.sum(trainer.score_sequences([test_data])))
    train_ll_per_frame = total_train_ll / n_train
    test_ll_per_frame = total_test_ll / len(test_data)

    # State decoding on train set
    states, _ = trainer.decode_states([train_data])

    # Dwell times and occupancy
    dwell = _compute_dwell_times(states, n_states)
    counts = np.array([(states == s).sum() for s in range(n_states)])
    occupancy = (counts / len(states)).tolist()

    metrics: Dict[str, Any] = {
        "n_states": n_states,
        "budget": budget,
        "seed": seed,
        "n_train": n_train,
        "train_ll": train_ll_per_frame,
        "test_ll": test_ll_per_frame,
        "mean_dwell": float(dwell.mean()),
        "dwell_per_state": dwell.tolist(),
        "state_occupancy": occupancy,
    }

    if compute_bic:
        n_params = estimate_n_params(trainer.model, "gaussian")
        bic = -2.0 * total_train_ll + n_params * np.log(n_train)
        metrics["bic"] = float(bic)
        metrics["n_params"] = int(n_params)
        metrics["occupancy_entropy"] = _occupancy_entropy(states, n_states)

    # ── Save artefacts ────────────────────────────────────────────────────────
    trainer.save(out_dir / "hmm.pkl")
    np.save(out_dir / "states.npy", states)
    # Save transition matrix separately so Phase 3 can load it without
    # triggering hmmlearn's covariance shape validation on model reload.
    np.save(out_dir / "transmat.npy", trainer.get_transition_matrix())
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(_to_json_safe(metrics), fh, indent=2)

    # ── Standard plots ────────────────────────────────────────────────────────
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    try:
        plot_transition_matrix(trainer, plots_dir / "transition_matrix.png")
        plot_ethogram(states, n_states, plots_dir / "ethogram.png")
        plot_state_occupancy(states, n_states, plots_dir / "state_occupancy.png")
    except Exception as exc:
        warnings.warn(f"Plotting failed in worker ({out_dir_str}): {exc}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Representation sanity
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase0(
    latents: np.ndarray, args: argparse.Namespace, out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(latents)

    # ── PCA ───────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    n_sub = min(args.phase0_n_samples, n)
    idx = rng.choice(n, n_sub, replace=False)
    sub = latents[idx]

    n_comp = min(64, sub.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(sub)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, n_comp + 1), cumvar, marker=".", lw=1.5, color="C0")
    ax.axhline(0.90, ls="--", color="C1", label=f"90 % (n={n90})")
    ax.axvline(n90, ls="--", color="C1", alpha=0.5)
    ax.axhline(0.95, ls="--", color="C2", label=f"95 % (n={n95})")
    ax.axvline(n95, ls="--", color="C2", alpha=0.5)
    ax.set_xlabel("PC index")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA spectrum of latent representations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "pca_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Autocorrelation (use contiguous prefix for temporal validity) ──────────
    max_lag = 50
    ac_data = latents[: min(n, 200_000)]
    T, D = ac_data.shape
    mu = ac_data.mean(axis=0)
    var = ac_data.var(axis=0) + 1e-10
    centered = ac_data - mu

    lags = np.arange(1, max_lag + 1)
    ac_matrix = np.stack(
        [(centered[:-lag] * centered[lag:]).mean(axis=0) / var for lag in lags],
        axis=0,
    )  # (max_lag, D)
    mean_ac = ac_matrix.mean(axis=1)
    std_ac = ac_matrix.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, mean_ac, lw=2, color="C0", label="Mean autocorrelation")
    ax.fill_between(lags, mean_ac - std_ac, mean_ac + std_ac, alpha=0.25, color="C0",
                    label="±1 std over features")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Per-feature lag autocorrelation (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "autocorrelation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Smoothness (frame-to-frame L2 norm) ──────────────────────────────────
    diffs = np.linalg.norm(ac_data[1:] - ac_data[:-1], axis=1)
    win = min(200, len(diffs) // 10)
    running_mean = np.convolve(diffs, np.ones(win) / win, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(diffs, bins=100, edgecolor="none", color="C0")
    axes[0].set_xlabel("Frame-to-frame L2 norm")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Smoothness — distribution")
    axes[1].plot(np.arange(len(running_mean)), running_mean, lw=1, color="C1")
    axes[1].set_xlabel("Frame index")
    axes[1].set_ylabel(f"Running mean L2 (window={win})")
    axes[1].set_title("Smoothness over time")
    plt.tight_layout()
    fig.savefig(out_dir / "smoothness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_samples_used": int(n_sub),
        "n_components_90pct": int(n90),
        "n_components_95pct": int(n95),
        "mean_lag1_autocorr": float(mean_ac[0]),
        "mean_step_norm": float(diffs.mean()),
        "std_step_norm": float(diffs.std()),
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Phase 0 summary: {summary}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Scale discovery
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase1(
    latents: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    n_workers: int,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets: List[int] = sorted(args.phase1_budgets)
    seeds: List[int] = args.phase1_seeds
    K: int = args.phase1_k

    jobs = [(b, s) for b in budgets for s in seeds]
    all_metrics: Dict[Tuple[int, int], Any] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _fit_single,
                latents, K, b, s,
                str(out_dir / f"budget_{b:07d}" / f"seed_{s}"),
                args.covariance_type, args.n_iter, False,
            ): (b, s)
            for b, s in jobs
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                all_metrics[key] = fut.result()
                print(f"  [Phase 1] budget={key[0]:,}, seed={key[1]} → "
                      f"test_ll={all_metrics[key]['test_ll']:.4f}")
            except Exception as exc:
                warnings.warn(f"Phase 1 job {key} failed: {exc}")
                all_metrics[key] = {"test_ll": float("nan")}

    # ── Summary plot: test LL vs budget ──────────────────────────────────────
    means = np.array([
        np.nanmean([all_metrics.get((b, s), {}).get("test_ll", float("nan"))
                    for s in seeds])
        for b in budgets
    ])
    stds = np.array([
        np.nanstd([all_metrics.get((b, s), {}).get("test_ll", float("nan"))
                   for s in seeds])
        for b in budgets
    ])
    bxs = np.array(budgets)
    valid = ~np.isnan(means)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(bxs[valid], means[valid], yerr=stds[valid],
                fmt="o-", capsize=4, color="C0", lw=2)
    ax.fill_between(bxs[valid], means[valid] - stds[valid],
                    means[valid] + stds[valid], alpha=0.2, color="C0")
    ax.set_xlabel("Training budget (frames)")
    ax.set_ylabel("Test LL per frame")
    ax.set_title(f"Phase 1: Scale discovery (K={K})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_ll_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — K sweep
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase2(
    latents: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    n_workers: int,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    k_values: List[int] = sorted(args.phase2_k_values)
    seeds: List[int] = args.phase2_seeds
    budget: int = args.phase2_budget

    jobs = [(k, s) for k in k_values for s in seeds]
    all_metrics: Dict[Tuple[int, int], Any] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _fit_single,
                latents, k, budget, s,
                str(out_dir / f"K_{k:02d}" / f"seed_{s}"),
                args.covariance_type, args.n_iter, True,
            ): (k, s)
            for k, s in jobs
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                all_metrics[key] = fut.result()
                m = all_metrics[key]
                print(f"  [Phase 2] K={key[0]:2d}, seed={key[1]} → "
                      f"test_ll={m.get('test_ll', float('nan')):.4f}  "
                      f"bic={m.get('bic', float('nan')):.1f}")
            except Exception as exc:
                warnings.warn(f"Phase 2 job {key} failed: {exc}")
                all_metrics[key] = {}

    # ── 4-panel summary figure ────────────────────────────────────────────────
    def _agg(metric: str):
        ms = np.array([
            np.nanmean([all_metrics.get((k, s), {}).get(metric, float("nan"))
                        for s in seeds])
            for k in k_values
        ])
        ss = np.array([
            np.nanstd([all_metrics.get((k, s), {}).get(metric, float("nan"))
                       for s in seeds])
            for k in k_values
        ])
        return ms, ss

    panels = [
        ("test_ll", "Test LL per frame"),
        ("bic", "BIC"),
        ("occupancy_entropy", "Occupancy entropy (nats)"),
        ("mean_dwell", "Mean dwell time (frames)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    kxs = np.array(k_values)
    for ax, (metric, label) in zip(axes.flat, panels):
        means, stds = _agg(metric)
        valid = ~np.isnan(means)
        ax.plot(kxs[valid], means[valid], "o-", color="C0", lw=2)
        ax.fill_between(kxs[valid], means[valid] - stds[valid],
                        means[valid] + stds[valid], alpha=0.2, color="C0")
        ax.set_xlabel("Number of states K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Phase 2: K sweep (budget={budget:,})")
    plt.tight_layout()
    fig.savefig(out_dir / "summary_k_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Robustness
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase3(
    latents: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    n_workers: int,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    K: int = args.phase3_k
    budget: int = args.phase3_budget
    seeds: List[int] = args.phase3_seeds

    all_metrics: Dict[int, Any] = {}

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _fit_single,
                latents, K, budget, s,
                str(out_dir / f"seed_{s}"),
                args.covariance_type, args.n_iter, False,
            ): s
            for s in seeds
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                all_metrics[s] = fut.result()
                print(f"  [Phase 3] seed={s} → "
                      f"test_ll={all_metrics[s]['test_ll']:.4f}")
            except Exception as exc:
                warnings.warn(f"Phase 3 seed {s} failed: {exc}")

    # ── Load transition matrices and occupancies ──────────────────────────────
    trans_matrices: Dict[int, np.ndarray] = {}
    occupancies: Dict[int, np.ndarray] = {}
    for s in seeds:
        # Prefer the lightweight transmat.npy saved by _fit_single
        tm_path = out_dir / f"seed_{s}" / "transmat.npy"
        states_path = out_dir / f"seed_{s}" / "states.npy"
        if tm_path.exists():
            trans_matrices[s] = np.load(tm_path)
        if states_path.exists():
            st = np.load(states_path)
            counts = np.array([(st == k).sum() for k in range(K)], dtype=float)
            occupancies[s] = counts / max(len(st), 1)

    if len(trans_matrices) < 2:
        print("  Phase 3: fewer than 2 successful fits — skipping stability plot.")
        return all_metrics

    # ── Align all matrices to first available seed ────────────────────────────
    ref_seed = next(s for s in seeds if s in trans_matrices)
    A_ref = trans_matrices[ref_seed]
    aligned = {ref_seed: A_ref}
    for s in seeds:
        if s != ref_seed and s in trans_matrices:
            aligned[s] = _align_transmat(A_ref, trans_matrices[s])

    s_list = [s for s in seeds if s in aligned]
    n_s = len(s_list)
    dist_matrix = np.zeros((n_s, n_s))
    for i, si in enumerate(s_list):
        for j, sj in enumerate(s_list):
            if i != j:
                dist_matrix[i, j] = float(
                    np.linalg.norm(aligned[si] - aligned[sj], "fro")
                )

    # ── Stability summary figure ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(dist_matrix, cmap="viridis")
    axes[0].set_xticks(range(n_s))
    axes[0].set_xticklabels([f"s{s}" for s in s_list])
    axes[0].set_yticks(range(n_s))
    axes[0].set_yticklabels([f"s{s}" for s in s_list])
    axes[0].set_title(
        f"Pairwise Frobenius distance of aligned\ntransition matrices (K={K})"
    )
    plt.colorbar(im0, ax=axes[0], label="Frobenius distance")
    dmax = dist_matrix.max() if dist_matrix.max() > 0 else 1.0
    for i in range(n_s):
        for j in range(n_s):
            axes[0].text(
                j, i, f"{dist_matrix[i, j]:.2f}", ha="center", va="center",
                fontsize=8,
                color="white" if dist_matrix[i, j] > 0.5 * dmax else "black",
            )

    occ_rows = np.array([occupancies[s] for s in s_list if s in occupancies])
    if len(occ_rows) > 0:
        im1 = axes[1].imshow(occ_rows, aspect="auto", cmap="Blues")
        axes[1].set_xlabel("State")
        axes[1].set_ylabel("Seed")
        axes[1].set_yticks(range(len(s_list)))
        axes[1].set_yticklabels([f"s{s}" for s in s_list])
        axes[1].set_title("State occupancy per seed")
        plt.colorbar(im1, ax=axes[1], label="Occupancy fraction")

    plt.suptitle(f"Phase 3: Robustness (K={K}, budget={budget:,})")
    plt.tight_layout()
    fig.savefig(out_dir / "stability_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.save(out_dir / "frob_distances.npy", dist_matrix)
    off_diag = dist_matrix[dist_matrix > 0]
    print(f"  Phase 3: mean pairwise Frobenius dist = {off_diag.mean():.4f} "
          f"(std={off_diag.std():.4f})")
    return all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Null models
# ─────────────────────────────────────────────────────────────────────────────


def _run_phase4(
    latents: np.ndarray, args: argparse.Namespace, out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    K: int = args.phase4_k
    budget: int = args.phase4_budget
    seed = 0

    train_data, test_data = _subsample_split(latents, budget, seed)
    rng = np.random.default_rng(seed + 1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    results: Dict[str, Any] = {}

    # ── Real HMM ──────────────────────────────────────────────────────────────
    real_dir = out_dir / "real"
    real_dir.mkdir(exist_ok=True)
    trainer_real = HiddenMarkovModelTrainer(
        n_states=K, covariance_type=args.covariance_type, device=device
    )
    trainer_real.fit([train_data], n_iter=args.n_iter, random_state=seed,
                     sticky_prior=0.95)
    real_test_ll = (
        float(np.sum(trainer_real.score_sequences([test_data]))) / len(test_data)
    )
    states_real, _ = trainer_real.decode_states([train_data])
    dwell_real = _compute_dwell_times(states_real, K)
    trainer_real.save(real_dir / "hmm.pkl")
    results["real"] = {
        "test_ll": real_test_ll,
        "mean_dwell": float(dwell_real.mean()),
    }
    print(f"  [Phase 4] real HMM   → test_ll={real_test_ll:.4f}  "
          f"mean_dwell={dwell_real.mean():.2f}")

    # ── Shuffled control ──────────────────────────────────────────────────────
    shuf_dir = out_dir / "shuffled"
    shuf_dir.mkdir(exist_ok=True)
    shuffled_train = rng.permutation(train_data)
    shuffled_test = rng.permutation(test_data)
    trainer_shuf = HiddenMarkovModelTrainer(
        n_states=K, covariance_type=args.covariance_type, device=device
    )
    trainer_shuf.fit([shuffled_train], n_iter=args.n_iter, random_state=seed,
                     sticky_prior=0.95)
    shuf_test_ll = (
        float(np.sum(trainer_shuf.score_sequences([shuffled_test]))) / len(shuffled_test)
    )
    states_shuf, _ = trainer_shuf.decode_states([shuffled_train])
    dwell_shuf = _compute_dwell_times(states_shuf, K)
    trainer_shuf.save(shuf_dir / "hmm.pkl")
    results["shuffled"] = {
        "test_ll": shuf_test_ll,
        "mean_dwell": float(dwell_shuf.mean()),
    }
    print(f"  [Phase 4] shuffled   → test_ll={shuf_test_ll:.4f}  "
          f"mean_dwell={dwell_shuf.mean():.2f}")

    # ── IID Gaussian Mixture ──────────────────────────────────────────────────
    iid_dir = out_dir / "iid_mixture"
    iid_dir.mkdir(exist_ok=True)

    # Normalise with real-HMM scaler for fair comparison
    scaler = trainer_real.scaler
    train_norm = scaler.transform(train_data) if scaler is not None else train_data
    test_norm = scaler.transform(test_data) if scaler is not None else test_data

    gmm = GaussianMixture(
        n_components=K,
        covariance_type=args.covariance_type,
        random_state=seed,
        n_init=3,
    )
    gmm.fit(train_norm)
    iid_test_ll = float(gmm.score(test_norm))  # already per-sample
    gm_labels = gmm.predict(train_norm)
    dwell_iid = _compute_dwell_times(gm_labels, K)

    with open(iid_dir / "gmm_score.json", "w") as fh:
        json.dump({"test_ll": iid_test_ll, "mean_dwell": float(dwell_iid.mean())}, fh)
    results["iid_mixture"] = {
        "test_ll": iid_test_ll,
        "mean_dwell": float(dwell_iid.mean()),
    }
    print(f"  [Phase 4] IID GMM    → test_ll={iid_test_ll:.4f}  "
          f"mean_dwell={dwell_iid.mean():.2f}")

    with open(out_dir / "null_results.json", "w") as fh:
        json.dump(_to_json_safe(results), fh, indent=2)

    # ── Summary bar chart ─────────────────────────────────────────────────────
    labels = ["Real HMM", "Shuffled", "IID Mixture"]
    ll_vals = [results["real"]["test_ll"], results["shuffled"]["test_ll"],
               results["iid_mixture"]["test_ll"]]
    dwell_vals = [results["real"]["mean_dwell"], results["shuffled"]["mean_dwell"],
                  results["iid_mixture"]["mean_dwell"]]
    colors = ["C0", "C1", "C2"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, ll_vals, color=colors, edgecolor="white")
    axes[0].set_ylabel("Test LL per frame")
    axes[0].set_title("Log-likelihood: real vs null models")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, dwell_vals, color=colors, edgecolor="white")
    axes[1].set_ylabel("Mean dwell time (frames)")
    axes[1].set_title("Mean dwell time: real vs null models")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(f"Phase 4: Null model comparison (K={K})")
    plt.tight_layout()
    fig.savefig(out_dir / "null_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Dwell-time distribution overlay ──────────────────────────────────────
    all_dwells = np.concatenate([dwell_real, dwell_shuf, dwell_iid])
    if all_dwells.max() > all_dwells.min():
        bins = np.histogram_bin_edges(all_dwells, bins=30)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(dwell_real, bins=bins, alpha=0.5, label="Real HMM",
                color="C0", density=True)
        ax.hist(dwell_shuf, bins=bins, alpha=0.5, label="Shuffled",
                color="C1", density=True)
        ax.hist(dwell_iid, bins=bins, alpha=0.5, label="IID Mixture",
                color="C2", density=True)
        ax.set_xlabel("Mean dwell time per state (frames)")
        ax.set_ylabel("Density")
        ax.set_title(f"Dwell-time distributions: real vs null models (K={K})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_dir / "dwell_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


def run_strategy(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Extract or load latents ───────────────────────────────────────────────
    cached_path = out_dir / "latents.npy"
    if args.latents_path:
        print(f"Loading latents from {args.latents_path} …")
        latents = np.load(args.latents_path)
    elif cached_path.exists():
        print(f"Loading cached latents from {cached_path} …")
        latents = np.load(cached_path)
    else:
        if not args.preprocessed_data or not args.model_path:
            raise ValueError(
                "--preprocessed_data and --model_path are required when latents "
                "are not cached (use --latents_path to skip extraction)."
            )
        print("Extracting latents …")
        device = torch.device(
            args.device if args.device
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        train_w, val_w, test_w, _, _, _, _, _ = load_preprocessed_dataset(
            args.preprocessed_data
        )
        split_map = {
            "train": [train_w],
            "val": [val_w],
            "test": [test_w],
            "all": [train_w, val_w, test_w],
        }
        selected = [w for w in split_map[args.split]
                    if w is not None and len(w) > 0]
        if not selected:
            raise ValueError(f"No windows found for split='{args.split}'")
        windows = np.concatenate(selected, axis=0).astype(np.float32)

        model, _, _ = _load_model_from_checkpoint(Path(args.model_path), device)
        raw_latents = extract_latents(model, windows, device, no_mask=True)
        latents = raw_latents.reshape(-1, raw_latents.shape[-1])

        np.save(cached_path, latents)
        print(f"  Saved latents → {cached_path}")

    print(f"  Latents shape: {latents.shape}")

    phases = set(args.phases)
    n_workers = args.n_workers

    if 0 in phases:
        print("\n=== Phase 0: Representation sanity ===")
        _run_phase0(latents, args, out_dir / "phase0_representation")

    if 1 in phases:
        print("\n=== Phase 1: Scale discovery ===")
        _run_phase1(latents, args, out_dir / "phase1_scale", n_workers)

    if 2 in phases:
        print("\n=== Phase 2: K sweep ===")
        _run_phase2(latents, args, out_dir / "phase2_k_sweep", n_workers)

    if 3 in phases:
        if args.phase3_k is None:
            raise ValueError("--phase3_k is required when phase 3 is selected.")
        print(f"\n=== Phase 3: Robustness (K={args.phase3_k}) ===")
        _run_phase3(latents, args, out_dir / "phase3_robustness", n_workers)

    if 4 in phases:
        if args.phase4_k is None:
            raise ValueError("--phase4_k is required when phase 4 is selected.")
        print(f"\n=== Phase 4: Null models (K={args.phase4_k}) ===")
        _run_phase4(latents, args, out_dir / "phase4_null_models")

    print(f"\nAll done. Results in: {out_dir.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Structured HMM evaluation protocol (phases 0–4)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument("--model_path", type=str, default=None,
                   help="Path to transformer checkpoint (best.pt)")
    p.add_argument("--preprocessed_data", type=str, default=None,
                   help="Path to preprocessed dataset (.npz)")
    p.add_argument("--output_dir", type=str, default="hmm_strategy_results")
    p.add_argument("--split", type=str, default="all",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--device", type=str, default=None,
                   help="mps | cuda | cpu (default: auto)")
    p.add_argument("--phases", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--covariance_type", type=str, default="diag",
                   choices=["full", "diag", "tied", "spherical"])
    p.add_argument("--n_iter", type=int, default=50,
                   help="Max EM iterations per HMM fit")
    p.add_argument("--latents_path", type=str, default=None,
                   help="Skip extraction; load cached latents (.npy)")

    # Phase 0
    p.add_argument("--phase0_n_samples", type=int, default=50_000)

    # Phase 1
    p.add_argument("--phase1_k", type=int, default=10)
    p.add_argument("--phase1_budgets", type=int, nargs="+",
                   default=[50_000, 150_000, 300_000, 500_000])
    p.add_argument("--phase1_seeds", type=int, nargs="+", default=[0, 1, 2])

    # Phase 2
    p.add_argument("--phase2_k_values", type=int, nargs="+",
                   default=[5, 10, 15, 20, 30, 40, 50, 60])
    p.add_argument("--phase2_budget", type=int, default=200_000)
    p.add_argument("--phase2_seeds", type=int, nargs="+", default=[0, 1, 2])

    # Phase 3
    p.add_argument("--phase3_k", type=int, default=None,
                   help="[REQUIRED for phase 3]")
    p.add_argument("--phase3_budget", type=int, default=200_000)
    p.add_argument("--phase3_seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4])

    # Phase 4
    p.add_argument("--phase4_k", type=int, default=None,
                   help="[REQUIRED for phase 4]")
    p.add_argument("--phase4_budget", type=int, default=200_000)

    args = p.parse_args()
    run_strategy(args)


if __name__ == "__main__":
    main()
