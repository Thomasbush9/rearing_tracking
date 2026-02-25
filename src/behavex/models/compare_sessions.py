"""Cross-session stability of K-means behavioral states.

Splits a concatenated latents.npy into per-session slices using row counts
derived from the original Excel files (same sorted order the extractor used),
then applies the saved K-means model to each slice.

No GPU or model re-loading required — only the already-downloaded latents.npy
and the Excel files (read in read-only mode for row counts only).

Usage
-----
uv run python -m behavex.models.compare_sessions \\
    --latents     /path/to/session_latents.npy \\
    --data_dir    /path/to/excel_dir \\
    --model_dir   /path/to/session_analysis \\
    --k           5 \\
    --group_size  16 \\
    --patch_len   8 \\
    --window_size 128 \\
    --output_dir  compare_output

The model_dir must contain:
  scaler.pkl        — saved by analyze_session.py
  K_05/km_K05.pkl   — saved by analyze_session.py for K=5

Alignment assumption
--------------------
The concatenated latents.npy was built by extract_session_latents_for_hmm()
which uses stride=window_size (non-overlapping windows) and loads the full
file (full_file=True, no pre-event trimming).  For each session file with T
rows, the patch count contributed is:

    n_windows = (T - window_size) // window_size + 1
    n_patches = n_windows * (window_size // patch_len)   [= n_windows * group_size]

The script validates that the sum of expected patches equals the latent file
length before slicing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_rel, wilcoxon, zscore

from behavex.models.analyze_session import (
    _dwell_times,
    _entropy,
    _occupancy,
    _transition_matrix,
    pool_patches,
)

# Same pattern as train_transformer._SESSION_PATTERN
_SESSION_PATTERN = re.compile(
    r"^m\d+_s\d+_(cricket|object)\.(xlsx|xls)$", re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# Session file discovery & row-count
# ─────────────────────────────────────────────────────────────────────────────


def _discover_sessions(data_dir: Path) -> List[Path]:
    """Return session Excel files in the same sorted() order the extractor used."""
    return sorted(
        f for f in data_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".xlsx", ".xls")
        and _SESSION_PATTERN.match(f.name)
        and f.name != "startTimes.xlsx"
    )


def _row_count(xlsx_path: Path) -> int:
    """Fast row count: reads only the first column (avoids loading all data)."""
    try:
        return len(pd.read_excel(xlsx_path, usecols=[0]))
    except Exception as exc:
        warnings.warn(f"Could not read {xlsx_path}: {exc}. Using 0 rows.", stacklevel=2)
        return 0


def _patches_for_session(
    n_frames: int, window_size: int, group_size: int, extraction_stride: int
) -> int:
    """Number of patches a session with n_frames rows contributes to latents.npy.

    extraction_stride: the stride used when building the preprocessed dataset
    (often stride=patch_len, e.g. 16 for window_size=128, patch_len=4 with s16 preset).
    group_size = window_size // patch_len (patches per transformer window).
    """
    if n_frames < window_size:
        return 0
    n_windows = (n_frames - window_size) // extraction_stride + 1
    return n_windows * group_size


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_dir: Path, K: int):
    scaler_path = model_dir / "scaler.pkl"
    km_path = model_dir / f"K_{K:02d}" / f"km_K{K:02d}.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"scaler.pkl not found in {model_dir}. "
            "Re-run analyze_session.py — it now saves scaler.pkl automatically."
        )
    if not km_path.exists():
        raise FileNotFoundError(
            f"{km_path} not found. Re-run analyze_session.py with --k_values {K}."
        )

    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)
    with open(km_path, "rb") as fh:
        km = pickle.load(fh)
    return scaler, km


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────


def _occupancy_heatmap(
    session_names: List[str],
    occupancies: np.ndarray,  # (n_sessions, K)
    K: int,
    out_path: Path,
) -> None:
    n_sess = len(session_names)
    fig, ax = plt.subplots(figsize=(max(5.0, K * 0.9 + 2.0),
                                    max(3.5, n_sess * 0.55 + 1.5)))
    im = ax.imshow(occupancies, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Occupancy fraction")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"S{s}" for s in range(K)], fontsize=9)
    ax.set_yticks(range(n_sess))
    ax.set_yticklabels(session_names, fontsize=8)
    ax.set_xlabel("State")
    ax.set_ylabel("Session")
    ax.set_title(f"State occupancy across sessions  (K={K})")
    for i in range(n_sess):
        for j in range(K):
            v = occupancies[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved occupancy heatmap    → {out_path}")


def _transition_small_multiples(
    session_names: List[str],
    transition_mats: List[np.ndarray],
    K: int,
    out_path: Path,
) -> None:
    n_sess = len(session_names)
    ncols = min(4, n_sess)
    nrows = (n_sess + ncols - 1) // ncols
    ps = max(2.5, K * 0.55)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * ps, nrows * ps))
    axes_flat = np.array(axes).flatten()

    for idx, (name, T_mat) in enumerate(zip(session_names, transition_mats)):
        ax = axes_flat[idx]
        im = ax.imshow(T_mat, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"S{s}" for s in range(K)], fontsize=6)
        ax.set_yticklabels([f"S{s}" for s in range(K)], fontsize=6)
        ax.set_title(name, fontsize=8)
        for i in range(K):
            for j in range(K):
                v = T_mat[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if v > 0.5 else "black")

    for idx in range(n_sess, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f"Transition matrices per session  (K={K})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition multiples → {out_path}")


def _entropy_bar(
    session_names: List[str],
    entropies: List[float],
    K: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(5, len(session_names) * 0.7), 3.5))
    ax.bar(range(len(session_names)), entropies, color="C0", edgecolor="white")
    ax.axhline(np.log(K), ls="--", color="red", lw=1, label=f"max entropy (ln {K})")
    ax.set_xticks(range(len(session_names)))
    ax.set_xticklabels(session_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Occupancy entropy (nats)")
    ax.set_title(f"Per-session state entropy  (K={K})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved entropy bar          → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Session trajectory & cricket-vs-object statistics
# ─────────────────────────────────────────────────────────────────────────────

_NAME_RE = re.compile(r"_s(\d+)_(cricket|object)$", re.IGNORECASE)


def _parse_session_name(name: str) -> Optional[Tuple[int, str]]:
    """'m003_s006_cricket' → (6, 'cricket'). Returns None if no match."""
    m = _NAME_RE.search(name)
    return (int(m.group(1)), m.group(2).lower()) if m else None


def _trajectory_plot(
    session_names: List[str],
    occupancies: np.ndarray,  # (n_sessions, K)
    K: int,
    out_path: Path,
) -> None:
    """One panel per state: occupancy vs session number, cricket vs object."""
    parsed = [(_parse_session_name(n), occupancies[i]) for i, n in enumerate(session_names)]
    parsed = [(p, occ) for p, occ in parsed if p is not None]

    cricket: Dict[int, np.ndarray] = {}
    object_: Dict[int, np.ndarray] = {}
    for (sess_num, cond), occ in parsed:
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
        ax.set_title(f"S{s}", fontsize=9, fontweight="bold")
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

    plt.suptitle(f"State occupancy over sessions  (K={K})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory plot      → {out_path}")


def _cricket_vs_object_stats(
    session_names: List[str],
    occupancies: np.ndarray,  # (n_sessions, K)
    K: int,
    out_path: Path,
) -> None:
    """Paired cricket-vs-object bar chart with significance markers per state.

    Uses Wilcoxon signed-rank (≥5 pairs) or paired t-test (<5 pairs).
    FDR correction (Benjamini-Hochberg) across states.
    """
    parsed = [(_parse_session_name(n), occupancies[i]) for i, n in enumerate(session_names)]
    parsed = [(p, occ) for p, occ in parsed if p is not None]

    cricket: Dict[int, np.ndarray] = {}
    object_: Dict[int, np.ndarray] = {}
    for (sess_num, cond), occ in parsed:
        (cricket if cond == "cricket" else object_)[sess_num] = occ

    sessions = sorted(set(cricket) & set(object_))
    n_pairs = len(sessions)
    if n_pairs < 3:
        print(f"  Stats plot skipped: need ≥3 matched sessions, got {n_pairs}")
        return

    c_mat = np.array([[cricket[s][k] for k in range(K)] for s in sessions])  # (n, K)
    o_mat = np.array([[object_[s][k] for k in range(K)] for s in sessions])  # (n, K)

    c_mean = c_mat.mean(axis=0)
    o_mean = o_mat.mean(axis=0)
    c_sem  = c_mat.std(axis=0, ddof=1) / np.sqrt(n_pairs)
    o_sem  = o_mat.std(axis=0, ddof=1) / np.sqrt(n_pairs)

    # Per-state paired test
    raw_p = []
    for k in range(K):
        c_vals = c_mat[:, k].tolist()
        o_vals = o_mat[:, k].tolist()
        try:
            if n_pairs >= 5:
                _, p = wilcoxon(c_vals, o_vals)
            else:
                _, p = ttest_rel(c_vals, o_vals)
        except Exception:
            p = 1.0
        raw_p.append(p)

    # FDR correction (BH) — fallback to raw if statsmodels absent
    try:
        from statsmodels.stats.multitest import multipletests
        _, p_corr, _, _ = multipletests(raw_p, method="fdr_bh")
    except ImportError:
        p_corr = np.array(raw_p)
        print("  (statsmodels not found — showing uncorrected p-values)")

    def _sig_label(p: float) -> str:
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    x = np.arange(K)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, K * 1.1), 4.5))
    ax.bar(x - w/2, c_mean, w, yerr=c_sem, label="cricket",
           color="C1", alpha=0.85, capsize=3)
    ax.bar(x + w/2, o_mean, w, yerr=o_sem, label="object",
           color="C0", alpha=0.85, capsize=3)

    # Significance markers above each pair
    y_top = np.maximum(c_mean + c_sem, o_mean + o_sem)
    y_offset = y_top.max() * 0.04
    for k, (p, yt) in enumerate(zip(p_corr, y_top)):
        ax.text(x[k], yt + y_offset, _sig_label(p),
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in range(K)], fontsize=9)
    ax.set_ylabel("Mean occupancy ± SE")
    ax.set_ylim(0, (y_top + y_offset * 3).max())
    stat_label = "Wilcoxon" if n_pairs >= 5 else "paired t"
    ax.set_title(
        f"Cricket vs Object  (K={K}, n={n_pairs} sessions, {stat_label}, FDR-BH)\n"
        "* p<0.05   ** p<0.01   *** p<0.001   ns = not significant"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved stats bar chart      → {out_path}")

    # Print table
    print(f"\n  Cricket vs Object (n={n_pairs} paired sessions, {stat_label} + FDR-BH):")
    print(f"  {'State':6}  {'Cricket':8}  {'Object':8}  {'p_corr':8}  Sig")
    for k in range(K):
        print(f"  S{k:<5}  {c_mean[k]:.3f}     {o_mean[k]:.3f}     "
              f"{p_corr[k]:.4f}    {_sig_label(p_corr[k])}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-session feature loading & correlation plots
# ─────────────────────────────────────────────────────────────────────────────


def _load_session_features(
    session_path: Path,
    window_size: int,
    extraction_stride: int,
    n_windows: int,
) -> Tuple[np.ndarray, List[str]]:
    """Load one session's raw behavioral features, windowed to match latent windows.

    Uses the same stride as latent extraction so feature windows align 1-to-1
    with the pooled latent windows produced by pool_patches.
    """
    from behavex.models.train_transformer import load_data_path, prepare_masked_transformer_data

    data, valid_mask = load_data_path(str(session_path), trim_before_dist_head=False)
    windows, feature_names, _, _ = prepare_masked_transformer_data(
        data,
        window_size=window_size,
        stride=extraction_stride,
        valid_mask=valid_mask,
    )
    # (N, window_size, F) → (N, F) per-window means
    win_features = windows.mean(axis=1)
    if len(win_features) > n_windows:
        win_features = win_features[:n_windows]
    return win_features, feature_names


def _feature_heatmap_global(
    all_features: np.ndarray,
    all_labels: np.ndarray,
    feature_names: List[str],
    K: int,
    out_path: Path,
) -> None:
    """Z-scored per-state mean feature heatmap across all pooled sessions."""
    feat_z = zscore(all_features, axis=0, nan_policy="omit")
    feat_z = np.nan_to_num(feat_z)
    state_means = np.array([
        feat_z[all_labels == s].mean(axis=0) if (all_labels == s).any()
        else np.zeros(all_features.shape[1])
        for s in range(K)
    ])
    sort_idx = np.argsort(-state_means.var(axis=0))
    sm = state_means[:, sort_idx]
    fn = [feature_names[i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(max(12, len(fn) * 0.35), max(3, K * 0.6 + 1.5)))
    im = ax.imshow(sm, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    plt.colorbar(im, ax=ax, label="Z-scored mean")
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"S{s}" for s in range(K)], fontsize=9)
    ax.set_xticks(range(len(fn)))
    ax.set_xticklabels(fn, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Feature profile per state  (K={K}, n={len(all_labels):,} windows)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature heatmap      → {out_path}")


def _feature_boxplots_global(
    all_features: np.ndarray,
    all_labels: np.ndarray,
    feature_names: List[str],
    K: int,
    out_path: Path,
    top_n: int = 8,
) -> Dict[str, float]:
    """Violin plots for top-N features by F-statistic, pooled across sessions.

    Returns a dict mapping feature_name -> F-statistic (for all features, not just top-N).
    This can be saved and passed to eval_reconstruction.py as --fstat_json.
    """
    f_stats: List[float] = []
    for fi in range(all_features.shape[1]):
        groups = [all_features[all_labels == s, fi] for s in range(K)]
        valid = [g for g in groups if len(g) > 1]
        if len(valid) >= 2:
            fval = float(f_oneway(*valid).statistic)
            f_stats.append(fval if np.isfinite(fval) else 0.0)
        else:
            f_stats.append(0.0)

    # Save F-stats alongside the plot for use with eval_reconstruction --fstat_json
    fstat_dict = {feature_names[i]: f_stats[i] for i in range(len(feature_names))}
    fstat_path = out_path.parent / "fstat_per_feature.json"
    with open(fstat_path, "w") as fh:
        json.dump(fstat_dict, fh, indent=2)
    print(f"  Saved F-stat JSON          → {fstat_path}")

    top_idx = np.argsort(f_stats)[-top_n:][::-1]
    ncols = min(4, len(top_idx))
    nrows = (len(top_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_flat = np.array(axes).flatten()
    cmap = plt.get_cmap("tab10", K)

    for panel, fi in enumerate(top_idx):
        ax = axes_flat[panel]
        data_plot = [
            all_features[all_labels == s, fi] if (all_labels == s).any()
            else np.array([0.0])
            for s in range(K)
        ]
        parts = ax.violinplot(data_plot, positions=range(K), showmedians=True)
        for pc, s in zip(parts["bodies"], range(K)):
            pc.set_facecolor(cmap(s))
            pc.set_alpha(0.7)
        ax.set_xticks(range(K))
        ax.set_xticklabels([f"S{s}" for s in range(K)], fontsize=8)
        ax.set_title(f"{feature_names[fi]}\nF={f_stats[fi]:.1f}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for panel in range(len(top_idx), len(axes_flat)):
        axes_flat[panel].set_visible(False)
    plt.suptitle(f"Top-{len(top_idx)} discriminative features  (K={K})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature boxplots     → {out_path}")
    return fstat_dict


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_compare(
    latents_path: str,
    data_dir: str,
    model_dir: str,
    k: int,
    group_size: int,
    window_size: int,
    output_dir: str,
    patch_len: int = 4,
    fps: float = 62.4,
    extraction_stride: Optional[int] = None,
    filter_pattern: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    save_labels: bool = False,
    save_winlatents: bool = False,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(model_dir)
    data_path = Path(data_dir)
    window_sec = group_size * patch_len / fps

    # Default: stride = patch_len (the s16 / overlapping-window preset)
    if extraction_stride is None:
        extraction_stride = patch_len
    print(f"Extraction stride : {extraction_stride}  "
          f"(patches/window={group_size}, window_size={window_size})")

    # ── Discover sessions ────────────────────────────────────────────────────
    session_files = _discover_sessions(data_path)
    if not session_files:
        raise FileNotFoundError(
            f"No m*_s*_cricket|object.xlsx files found in {data_path}"
        )
    print(f"Found {len(session_files)} session files in {data_path}")
    if filter_pattern:
        import re as _re
        session_files_all = session_files
        session_files = [f for f in session_files if _re.search(filter_pattern, f.stem)]
        print(f"Filter '{filter_pattern}': {len(session_files)} / {len(session_files_all)} sessions selected")

    # ── Compute patch counts for ALL sessions (need cursor even for skipped) ─
    print("Computing row counts …")
    session_info: List[Tuple[Path, int, int, bool]] = []  # path, n_frames, n_patches, include
    all_files = _discover_sessions(data_path)  # full list for cursor
    selected_stems = {f.stem for f in session_files}
    for f in all_files:
        n_frames = _row_count(f)
        n_patches = _patches_for_session(n_frames, window_size, group_size, extraction_stride)
        include = f.stem in selected_stems
        session_info.append((f, n_frames, n_patches, include))
        marker = "→" if include else "  "
        print(f"  {marker} {f.name:40s}  frames={n_frames:6d}  patches={n_patches:7d}")

    total_expected = sum(s[2] for s in session_info)

    # ── Load latents (memory-mapped — never fully loaded into RAM) ────────────
    print(f"\nLoading {latents_path} (mmap) …")
    latents = np.load(latents_path, mmap_mode="r")
    print(f"  Latent shape  : {latents.shape}")
    print(f"  Expected total: {total_expected}  "
          f"({'MATCH ✓' if total_expected == len(latents) else f'MISMATCH (diff={len(latents)-total_expected:+d})'})")

    if total_expected != len(latents):
        warnings.warn(
            f"Patch count mismatch: latents has {len(latents)} rows, "
            f"Excel sum={total_expected}. Boundaries may be off.",
            stacklevel=2,
        )

    # ── Load saved K-means model ─────────────────────────────────────────────
    print(f"\nLoading reference K-means (K={k}) from {ref_dir} …")
    scaler, km = _load_model(ref_dir, k)

    # ── Per-session prediction ───────────────────────────────────────────────
    session_names: List[str] = []
    occupancies: List[np.ndarray] = []
    transition_mats: List[np.ndarray] = []
    entropies: List[float] = []
    summary: Dict[str, dict] = {}

    # For feature correlation (accumulated across selected sessions)
    all_feat_chunks: List[np.ndarray] = []
    all_lbl_chunks: List[np.ndarray] = []
    feature_names: Optional[List[str]] = None

    cursor = 0
    for (f, n_frames, n_patches, include) in session_info:
        name = f.stem
        if n_patches == 0:
            continue

        if not include:
            # Advance cursor without loading data into RAM
            cursor += n_patches
            continue

        print(f"\n── {name} ──")
        # Only read this session's slice from disk (mmap → minimal RAM)
        patch_slice = np.array(latents[cursor: cursor + n_patches], copy=True)
        cursor += n_patches

        # Pool patches → window-level vectors
        win_latents = pool_patches(patch_slice, group_size)
        del patch_slice  # free RAM immediately
        win_norm = scaler.transform(win_latents)
        del win_latents
        labels = km.predict(win_norm).astype(np.int64)

        if save_winlatents:
            wl_path = out / f"{name}_winlatents.npy"
            np.save(wl_path, win_norm)
            print(f"  winlatents saved: {wl_path}")

        del win_norm

        if save_labels:
            lbl_path = out / f"{name}_labels.npy"
            np.save(lbl_path, labels)
            print(f"  labels saved : {lbl_path}")

        occ = _occupancy(labels, k)
        dwell = _dwell_times(labels, k)
        T_mat = _transition_matrix(labels, k)
        ent = _entropy(occ)

        # Feature loading (one session at a time — memory safe)
        if raw_data_dir is not None:
            try:
                feat, feat_names = _load_session_features(
                    f, window_size, extraction_stride, len(labels)
                )
                n_align = min(len(feat), len(labels))
                all_feat_chunks.append(feat[:n_align])
                all_lbl_chunks.append(labels[:n_align])
                if feature_names is None:
                    feature_names = feat_names
                print(f"  features  : {feat.shape[1]} features, {n_align:,} windows aligned")
            except Exception as exc:
                warnings.warn(f"Feature loading failed for {name}: {exc}", stacklevel=2)

        print(f"  n_windows : {len(labels):,}")
        print(f"  entropy   : {ent:.4f}  (max {np.log(k):.4f})")
        print(f"  occupancy : " + "  ".join(f"S{s}={occ[s]:.3f}" for s in range(k)))
        print(f"  mean dwell: {dwell.mean():.1f} windows = {dwell.mean() * window_sec:.1f} s")

        session_names.append(name)
        occupancies.append(occ)
        transition_mats.append(T_mat)
        entropies.append(ent)
        summary[name] = {
            "n_frames": n_frames,
            "n_windows": int(len(labels)),
            "entropy": float(ent),
            "occupancy": occ.tolist(),
            "mean_dwell_windows": float(dwell.mean()),
            "mean_dwell_sec": float(dwell.mean() * window_sec),
            "per_state_dwell_sec": (dwell * window_sec).tolist(),
        }

    if not session_names:
        print("No sessions processed. Exiting.")
        return

    # ── Plots ────────────────────────────────────────────────────────────────
    occ_matrix = np.array(occupancies)
    _occupancy_heatmap(session_names, occ_matrix, k, out / "occupancy_heatmap.png")
    _transition_small_multiples(
        session_names, transition_mats, k, out / "transition_multiples.png"
    )
    _entropy_bar(session_names, entropies, k, out / "entropy_per_session.png")
    _trajectory_plot(session_names, occ_matrix, k, out / "session_trajectory.png")
    _cricket_vs_object_stats(session_names, occ_matrix, k, out / "cricket_vs_object_stats.png")

    # Feature correlation plots (pooled across all selected sessions)
    if all_feat_chunks and feature_names is not None:
        all_features = np.concatenate(all_feat_chunks, axis=0)
        all_labels   = np.concatenate(all_lbl_chunks,  axis=0)
        print(f"\nFeature correlation: {all_features.shape[0]:,} windows, "
              f"{all_features.shape[1]} features")
        _feature_heatmap_global(
            all_features, all_labels, feature_names, k, out / "feature_heatmap.png"
        )
        _feature_boxplots_global(
            all_features, all_labels, feature_names, k, out / "feature_boxplots.png"
        )

    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Saved summary JSON         → {out / 'summary.json'}")
    print(f"\nDone. Results in {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cross-session K-means stability (splits concatenated latents by session)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--latents", required=True,
        help="Path to concatenated session_latents.npy (patch-level, N×D)",
    )
    p.add_argument(
        "--data_dir", required=True,
        help="Directory containing m*_s*_cricket|object.xlsx files "
             "(in the same sorted order used during extraction)",
    )
    p.add_argument(
        "--model_dir", required=True,
        help="analyze_session.py output dir (must contain scaler.pkl and K_XX/km_KXX.pkl)",
    )
    p.add_argument("--k",          type=int, required=True,
                   help="K value (must have been run in analyze_session.py)")
    p.add_argument("--group_size", type=int, default=32,
                   help="Patches per window (= window_size // patch_len)")
    p.add_argument("--window_size", type=int, default=128,
                   help="Raw frames per window (transformer window_size)")
    p.add_argument("--patch_len",  type=int, default=4,
                   help="Frames per patch (from transformer config)")
    p.add_argument("--extraction_stride", type=int, default=None,
                   help="Stride used when building the preprocessed dataset. "
                        "Defaults to patch_len (e.g. 4 for the s16 preset). "
                        "Use window_size for non-overlapping extraction.")
    p.add_argument("--raw_data_dir", default=None,
                   help="Directory of Excel files to load raw behavioral features "
                        "for state characterisation. Only selected (filtered) sessions "
                        "are loaded — one at a time for memory efficiency.")
    p.add_argument("--filter",     default=None, dest="filter_pattern",
                   help="Only process sessions whose filename matches this regex "
                        "(e.g. 'm003' to process mouse 3 only). "
                        "Cursor still advances correctly through skipped sessions.")
    p.add_argument("--output_dir", default="compare_sessions_output",
                   help="Directory for output plots and summary JSON")
    p.add_argument("--fps",        type=float, default=62.4,
                   help="Camera frame rate (for converting windows to seconds)")
    p.add_argument("--save_labels", action="store_true",
                   help="Save per-session label arrays as {output_dir}/{session}_labels.npy. "
                        "Required input for plot_state_dynamics.py.")
    p.add_argument("--save_winlatents", action="store_true",
                   help="Save per-session normalised window-level latents as "
                        "{output_dir}/{session}_winlatents.npy. Required input for fit_hmm.py.")
    args = p.parse_args()

    run_compare(
        latents_path=args.latents,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        k=args.k,
        group_size=args.group_size,
        window_size=args.window_size,
        output_dir=args.output_dir,
        patch_len=args.patch_len,
        fps=args.fps,
        extraction_stride=args.extraction_stride,
        filter_pattern=args.filter_pattern,
        raw_data_dir=args.raw_data_dir,
        save_labels=args.save_labels,
        save_winlatents=args.save_winlatents,
    )


if __name__ == "__main__":
    main()
