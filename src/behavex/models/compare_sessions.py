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
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _patches_for_session(n_frames: int, window_size: int, group_size: int) -> int:
    """Number of patches a session with n_frames rows contributes to latents.npy."""
    if n_frames < window_size:
        return 0
    n_windows = (n_frames - window_size) // window_size + 1
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
    patch_len: int = 8,
    fps: float = 62.4,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir = Path(model_dir)
    data_path = Path(data_dir)
    window_sec = group_size * patch_len / fps

    # ── Discover sessions ────────────────────────────────────────────────────
    session_files = _discover_sessions(data_path)
    if not session_files:
        raise FileNotFoundError(
            f"No m*_s*_cricket|object.xlsx files found in {data_path}"
        )
    print(f"Found {len(session_files)} session files in {data_path}")

    # ── Compute expected patch counts per session ────────────────────────────
    print("Computing row counts (reading first column only) …")
    session_info: List[Tuple[Path, int, int]] = []  # (path, n_frames, n_patches)
    for f in session_files:
        n_frames = _row_count(f)
        n_patches = _patches_for_session(n_frames, window_size, group_size)
        session_info.append((f, n_frames, n_patches))
        print(f"  {f.name:40s}  frames={n_frames:6d}  patches={n_patches:6d}")

    total_expected = sum(s[2] for s in session_info)

    # ── Load latents & validate ──────────────────────────────────────────────
    print(f"\nLoading {latents_path} …")
    latents = np.load(latents_path, mmap_mode="r")
    print(f"  Latent shape : {latents.shape}")
    print(f"  Expected sum : {total_expected}  "
          f"({'MATCH' if total_expected == len(latents) else 'MISMATCH — check args'})")

    if total_expected != len(latents):
        diff = len(latents) - total_expected
        warnings.warn(
            f"Patch count mismatch: latents has {len(latents)} rows but Excel files "
            f"sum to {total_expected} (diff={diff:+d}).\n"
            "Check --window_size, --patch_len, --group_size match the extraction run.\n"
            "Proceeding anyway — boundaries may be off.",
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

    cursor = 0
    for (f, n_frames, n_patches) in session_info:
        name = f.stem  # e.g. m1_s1_cricket
        if n_patches == 0:
            print(f"\n  Skipping {name} — too few frames ({n_frames})")
            cursor += n_patches
            continue

        print(f"\n── {name} ──")
        patch_slice = np.asarray(latents[cursor: cursor + n_patches])
        cursor += n_patches

        # Pool patches → window-level vectors
        win_latents = pool_patches(patch_slice, group_size)
        win_norm = scaler.transform(win_latents)
        labels = km.predict(win_norm).astype(np.int64)

        occ = _occupancy(labels, k)
        dwell = _dwell_times(labels, k)
        T_mat = _transition_matrix(labels, k)
        ent = _entropy(occ)

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
    p.add_argument("--group_size", type=int, default=16,
                   help="Patches per window (= window_size // patch_len)")
    p.add_argument("--window_size", type=int, default=128,
                   help="Raw frames per window (transformer window_size)")
    p.add_argument("--patch_len",  type=int, default=8,
                   help="Frames per patch (from transformer config)")
    p.add_argument("--output_dir", default="compare_sessions_output",
                   help="Directory for output plots and summary JSON")
    p.add_argument("--fps",        type=float, default=62.4,
                   help="Camera frame rate (for converting windows to seconds)")
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
    )


if __name__ == "__main__":
    main()
