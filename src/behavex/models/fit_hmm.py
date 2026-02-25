"""Gaussian HMM on per-session window-level latents.

Fits a single GaussianHMM (diagonal covariance) on all sessions found in
--latents_dir concatenated, then Viterbi-decodes each session.  Produces
the same family of plots as compare_sessions.py so results are directly
comparable to the K-means baseline.

Prerequisite
------------
Run compare_sessions.py with --save_winlatents to generate
    {session}_winlatents.npy
files in the output directory.

Outputs (all in --output_dir)
------------------------------
  hmm_model.pkl                 — fitted hmmlearn GaussianHMM (reusable)
  {session}_hmm_labels.npy      — Viterbi state sequence per session
  hmm_occupancy_trajectory.png  — per-state occupancy vs session, cricket vs object
  hmm_transition_comparison.png — early cricket / late cricket / object + diffs
  hmm_loglik_per_session.png    — normalised log-likelihood per session
  hmm_stationary.png            — HMM stationary distribution

Usage
-----
python -m behavex.models.fit_hmm \\
    --latents_dir /path/to/compare_output/ \\
    --k 8 \\
    --group_size 32 --patch_len 4 --fps 62.4 \\
    --early_threshold 3 \\
    --output_dir  /path/to/compare_output/
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm

from behavex.models.analyze_session import _transition_matrix
from behavex.models.compare_sessions import _parse_session_name


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_winlatents(latents_dir: Path) -> Dict[str, np.ndarray]:
    """Load all *_winlatents.npy files. Returns {session_stem: array (N, D)}."""
    result: Dict[str, np.ndarray] = {}
    for p in sorted(latents_dir.glob("*_winlatents.npy")):
        key = p.stem[: -len("_winlatents")]
        result[key] = np.load(p)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HMM fit & decode
# ─────────────────────────────────────────────────────────────────────────────


def fit_hmm(
    all_winlatents: Dict[str, np.ndarray],
    K: int,
    n_iter: int = 200,
    seed: int = 0,
) -> hmm.GaussianHMM:
    """Fit a single GaussianHMM on all sessions concatenated."""
    names   = sorted(all_winlatents)
    lengths = [len(all_winlatents[n]) for n in names]
    X       = np.concatenate([all_winlatents[n] for n in names], axis=0)

    print(f"  Fitting GaussianHMM  K={K}, diag cov, n_iter={n_iter}")
    print(f"  Input: {X.shape[0]:,} windows ({len(names)} sessions), D={X.shape[1]}")

    model = hmm.GaussianHMM(
        n_components=K,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        verbose=False,
    )
    model.fit(X, lengths=lengths)

    # Check convergence
    log_prob = model.monitor_.history[-1]
    print(f"  Final log-prob (train): {log_prob:.2f}")
    if not model.monitor_.converged:
        print("  WARNING: HMM did not converge — consider increasing --n_iter")

    return model


def decode_sessions(
    model: hmm.GaussianHMM,
    all_winlatents: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Viterbi-decode each session. Returns {session_stem: label_array}."""
    hmm_labels: Dict[str, np.ndarray] = {}
    for name, X in sorted(all_winlatents.items()):
        logprob, states = model.decode(X, algorithm="viterbi")
        hmm_labels[name] = states.astype(np.int64)
        print(f"  {name}: loglik/win = {logprob / len(X):.4f}")
    return hmm_labels


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────


def _occupancy_from_labels(labels: np.ndarray, K: int) -> np.ndarray:
    counts = np.array([(labels == s).sum() for s in range(K)], dtype=float)
    return counts / max(len(labels), 1)


def plot_stationary(model: hmm.GaussianHMM, K: int, out_path: Path) -> None:
    """Bar chart of HMM stationary distribution."""
    # Compute stationary as left eigenvector of transition matrix
    A = model.transmat_
    eigvals, eigvecs = np.linalg.eig(A.T)
    # Eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigvals - 1.0))
    stat = np.real(eigvecs[:, idx])
    stat = np.abs(stat) / np.abs(stat).sum()

    cmap = plt.get_cmap("tab10", K)
    fig, ax = plt.subplots(figsize=(max(6, K * 0.9), 3.5))
    ax.bar(range(K), stat, color=[cmap(s) for s in range(K)], edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"S{s}" for s in range(K)], fontsize=9)
    ax.set_ylabel("Stationary probability")
    ax.set_title(f"HMM stationary distribution  (K={K})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stationary dist      → {out_path}")


def plot_occupancy_trajectory(
    hmm_labels: Dict[str, np.ndarray],
    K: int,
    out_path: Path,
) -> None:
    """One panel per HMM state: occupancy vs session number, cricket vs object."""
    cricket: Dict[int, np.ndarray] = {}
    object_: Dict[int, np.ndarray] = {}

    for name, labels in hmm_labels.items():
        parsed = _parse_session_name(name)
        if parsed is None:
            continue
        sess_num, cond = parsed
        occ = _occupancy_from_labels(labels, K)
        (cricket if cond == "cricket" else object_)[sess_num] = occ

    sessions = sorted(set(cricket) & set(object_))
    if len(sessions) < 2:
        print("  Trajectory plot skipped: need ≥2 matched session numbers")
        return

    ncols = min(4, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8), squeeze=False)
    axes_flat = axes.flatten()

    for s in range(K):
        ax = axes_flat[s]
        c_vals = [cricket[sess][s] for sess in sessions]
        o_vals = [object_[sess][s] for sess in sessions]
        ax.plot(sessions, c_vals, "o-", color="C1", lw=1.8, ms=5, label="cricket")
        ax.plot(sessions, o_vals, "s--", color="C0", lw=1.8, ms=5, label="object")
        ax.set_title(f"HMM-S{s}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Session", fontsize=7)
        ax.set_ylabel("Occupancy", fontsize=7)
        ax.set_xticks(sessions)
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if s == 0:
            ax.legend(fontsize=7)

    for idx in range(K, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f"HMM state occupancy over sessions  (K={K})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved occupancy trajectory → {out_path}")


def plot_transition_comparison(
    hmm_labels: Dict[str, np.ndarray],
    K: int,
    early_threshold: int,
    out_path: Path,
) -> None:
    """Three transition matrices: early cricket / late cricket / object + two diffs."""
    early_arrs: List[np.ndarray] = []
    late_arrs:  List[np.ndarray] = []
    obj_arrs:   List[np.ndarray] = []

    for name, labels in hmm_labels.items():
        parsed = _parse_session_name(name)
        if parsed is None:
            continue
        sess_num, cond = parsed
        if cond == "cricket":
            (early_arrs if sess_num <= early_threshold else late_arrs).append(labels)
        else:
            obj_arrs.append(labels)

    if not early_arrs or not late_arrs:
        print("  [transition] Not enough sessions for early/late split — skipping")
        return

    T_early = _transition_matrix(np.concatenate(early_arrs), K)
    T_late  = _transition_matrix(np.concatenate(late_arrs),  K)
    T_obj   = _transition_matrix(np.concatenate(obj_arrs),   K) if obj_arrs else None
    T_diff  = T_late - T_early

    n_panels = 4 if T_obj is not None else 3
    panel_size = max(3.0, K * 0.65)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(panel_size * n_panels + 1.0, panel_size + 1.5))

    tick_labels = [f"S{s}" for s in range(K)]

    def _draw(ax: plt.Axes, mat: np.ndarray, title: str,
              cmap: str, vmin: float, vmax: float) -> None:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(tick_labels, fontsize=6)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("To state", fontsize=7)
        ax.set_ylabel("From state", fontsize=7)
        ax.set_title(title, fontsize=9)
        for i in range(K):
            for j in range(K):
                v = mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if abs(v) > 0.4 else "black")

    n_e, n_l = len(early_arrs), len(late_arrs)
    _draw(axes[0], T_early, f"Early cricket (n={n_e})",  "Blues",  0,    1)
    _draw(axes[1], T_late,  f"Late cricket  (n={n_l})",  "Blues",  0,    1)
    _draw(axes[2], T_diff,  "Diff: late − early",        "RdBu_r", -0.3, 0.3)
    if T_obj is not None:
        _draw(axes[3], T_obj, f"Object (n={len(obj_arrs)})", "Blues", 0, 1)

    plt.suptitle(
        f"HMM transition matrices — early (≤s{early_threshold:02d}) vs "
        f"late (>s{early_threshold:02d}) cricket  (K={K})",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition comparison → {out_path}")


def plot_loglik_per_session(
    model: hmm.GaussianHMM,
    all_winlatents: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Normalised log-likelihood per session, coloured by condition."""
    cricket_pts: List[Tuple[int, float]] = []
    object_pts:  List[Tuple[int, float]] = []

    for name, X in sorted(all_winlatents.items()):
        parsed = _parse_session_name(name)
        if parsed is None:
            continue
        sess_num, cond = parsed
        ll_per_win = model.score(X) / len(X)
        (cricket_pts if cond == "cricket" else object_pts).append((sess_num, ll_per_win))
        print(f"  loglik/win  {name}: {ll_per_win:.4f}")

    fig, ax = plt.subplots(figsize=(max(6, (len(cricket_pts) + len(object_pts)) * 0.6), 4))

    if cricket_pts:
        c_sess, c_ll = zip(*sorted(cricket_pts))
        ax.plot(c_sess, c_ll, "o-", color="C1", lw=1.8, ms=6, label="cricket")
    if object_pts:
        o_sess, o_ll = zip(*sorted(object_pts))
        ax.plot(o_sess, o_ll, "s--", color="C0", lw=1.8, ms=6, label="object")

    ax.set_xlabel("Session number", fontsize=9)
    ax.set_ylabel("Log-likelihood per window (nats)", fontsize=9)
    ax.set_title("HMM model fit quality per session\n"
                 "(higher = session is more consistent with the learned model)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved log-likelihood plot  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_hmm(
    latents_dir: str,
    k: int,
    output_dir: str,
    group_size: int = 32,
    patch_len: int = 4,
    fps: float = 62.4,
    early_threshold: int = 3,
    n_iter: int = 200,
    seed: int = 0,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    window_sec = group_size * patch_len / fps
    print(f"Window duration : {window_sec:.3f} s")

    # ── Load latents ─────────────────────────────────────────────────────────
    print(f"\nLoading window latents from {latents_dir} …")
    all_winlatents = _load_winlatents(Path(latents_dir))
    if not all_winlatents:
        raise FileNotFoundError(
            "No *_winlatents.npy found. "
            "Re-run compare_sessions.py with --save_winlatents."
        )
    print(f"  Found {len(all_winlatents)} sessions: {sorted(all_winlatents)}")

    # ── Fit HMM ──────────────────────────────────────────────────────────────
    print("\nFitting HMM …")
    model = fit_hmm(all_winlatents, K=k, n_iter=n_iter, seed=seed)

    model_path = out / "hmm_model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Saved HMM model            → {model_path}")

    # ── Decode ───────────────────────────────────────────────────────────────
    print("\nViterbi decoding …")
    hmm_labels = decode_sessions(model, all_winlatents)

    for name, labels in hmm_labels.items():
        lbl_path = out / f"{name}_hmm_labels.npy"
        np.save(lbl_path, labels)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_stationary(model, k, out / "hmm_stationary.png")
    plot_occupancy_trajectory(hmm_labels, k, out / "hmm_occupancy_trajectory.png")
    plot_transition_comparison(
        hmm_labels, k, early_threshold,
        out / "hmm_transition_comparison.png",
    )
    plot_loglik_per_session(model, all_winlatents, out / "hmm_loglik_per_session.png")

    print(f"\nDone. Results in {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Gaussian HMM on per-session window-level latents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--latents_dir", required=True,
                   help="Directory containing *_winlatents.npy files "
                        "(from compare_sessions.py --save_winlatents)")
    p.add_argument("--k",           type=int, required=True,
                   help="Number of HMM states (use same K as K-means for comparability)")
    p.add_argument("--output_dir",  default="hmm_output",
                   help="Output directory for model and plots")
    p.add_argument("--group_size",  type=int, default=32)
    p.add_argument("--patch_len",   type=int, default=4)
    p.add_argument("--fps",         type=float, default=62.4)
    p.add_argument("--early_threshold", type=int, default=3,
                   help="Sessions ≤ this are 'early' for transition comparison")
    p.add_argument("--n_iter",      type=int, default=200,
                   help="Max EM iterations for HMM fitting")
    p.add_argument("--seed",        type=int, default=0)
    args = p.parse_args()

    run_hmm(
        latents_dir=args.latents_dir,
        k=args.k,
        output_dir=args.output_dir,
        group_size=args.group_size,
        patch_len=args.patch_len,
        fps=args.fps,
        early_threshold=args.early_threshold,
        n_iter=args.n_iter,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
