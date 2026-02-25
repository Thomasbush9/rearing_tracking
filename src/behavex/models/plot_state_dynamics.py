"""State dynamics analysis from saved per-session label arrays.

Generates four figures:
  A  ethogram_comparison.png       — raw ethogram rows: early / late / control
  B  bout_duration_violin.png      — S<focal> bout duration violins per session
  C  within_session_timecourse.png — rolling S<focal> occupancy within each session
  D  transition_early_vs_late.png  — pooled transition matrices early vs late cricket

Usage
-----
python -m behavex.models.plot_state_dynamics \\
    --labels_dir /path/to/compare_output/ \\
    --k 8 --focal_state 6 \\
    --group_size 32 --patch_len 4 --fps 62.4 \\
    --sessions_early m003_s003_cricket \\
    --sessions_late  m003_s007_cricket \\
    --sessions_control m003_s003_object \\
    --output_dir /path/to/compare_output/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavex.models.analyze_session import _transition_matrix
from behavex.models.compare_sessions import _parse_session_name


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_labels(labels_dir: Path) -> Dict[str, np.ndarray]:
    """Load all *_labels.npy files from directory.

    Returns a dict mapping session stem → label array.
    File naming convention: ``{session_stem}_labels.npy``.
    """
    result: Dict[str, np.ndarray] = {}
    for p in sorted(labels_dir.glob("*_labels.npy")):
        key = p.stem[: -len("_labels")]  # strip trailing "_labels"
        result[key] = np.load(p)
    return result


def _extract_bouts(labels: np.ndarray, state: int) -> List[int]:
    """Return list of run-lengths (in windows) for *state* in *labels*."""
    bouts: List[int] = []
    i = 0
    while i < len(labels):
        s = int(labels[i])
        j = i + 1
        while j < len(labels) and labels[j] == s:
            j += 1
        if s == state:
            bouts.append(j - i)
        i = j
    return bouts


def _rolling_occupancy(labels: np.ndarray, state: int, win: int) -> np.ndarray:
    """Sliding-window mean occupancy of *state* over *win* consecutive windows."""
    indicator = (labels == state).astype(float)
    kernel = np.ones(win) / win
    return np.convolve(indicator, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Ethogram comparison (early / late / control)
# ─────────────────────────────────────────────────────────────────────────────


def fig_ethogram_comparison(
    all_labels: Dict[str, np.ndarray],
    sessions_early: List[str],
    sessions_late: List[str],
    sessions_control: List[str],
    K: int,
    window_sec: float,
    out_path: Path,
    n_show_min: float = 15.0,
) -> None:
    """Figure A: one ethogram row per session, stacked vertically."""
    ordered: List[Tuple[str, str]] = (
        [(s, "early")   for s in sessions_early]
        + [(s, "late")    for s in sessions_late]
        + [(s, "control") for s in sessions_control]
    )
    ordered = [(s, tag) for s, tag in ordered if s in all_labels]
    if not ordered:
        print("  [Fig A] No matching sessions found — skipping")
        return

    n_rows = len(ordered)
    cmap = plt.get_cmap("tab10", K)
    n_show = int(n_show_min * 60 / window_sec)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, 1.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row, (name, tag) in enumerate(ordered):
        ax = axes[row, 0]
        labels = all_labels[name]
        subset = labels[:n_show]
        T = len(subset)
        time = np.arange(T) * window_sec / 60.0  # minutes

        for t in range(T - 1):
            ax.axvspan(time[t], time[t + 1],
                       facecolor=cmap(int(subset[t])), alpha=0.9, linewidth=0)
        ax.axvspan(time[-1], time[-1] + window_sec / 60.0,
                   facecolor=cmap(int(subset[-1])), alpha=0.9, linewidth=0)

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        short = name.replace("_cricket", " [C]").replace("_object", " [O]")
        ax.set_ylabel(f"{short}\n({tag})", fontsize=8, rotation=0,
                      ha="right", va="center", labelpad=100)

    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(s)) for s in range(K)]
    axes[0, 0].legend(handles, [f"S{s}" for s in range(K)],
                      loc="upper right", ncol=K, fontsize=7, framealpha=0.8)
    axes[-1, 0].set_xlabel("Time (minutes)", fontsize=9)
    plt.suptitle(
        f"Ethogram comparison: early vs late session  (K={K}, first {n_show_min:.0f} min)",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ethogram comparison  → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Bout duration distribution per session
# ─────────────────────────────────────────────────────────────────────────────


def fig_bout_violin(
    all_labels: Dict[str, np.ndarray],
    focal_state: int,
    window_sec: float,
    out_path: Path,
) -> None:
    """Figure B: violin plots of bout durations for focal state, per session."""
    cricket_sessions: List[Tuple[int, str, np.ndarray]] = []
    object_sessions:  List[Tuple[int, str, np.ndarray]] = []

    for name, labels in all_labels.items():
        parsed = _parse_session_name(name)
        if parsed is None:
            continue
        sess_num, cond = parsed
        entry = (sess_num, name, labels)
        (cricket_sessions if cond == "cricket" else object_sessions).append(entry)

    cricket_sessions.sort(key=lambda x: x[0])
    object_sessions.sort(key=lambda x: x[0])

    def _panel(ax: plt.Axes, sessions: List[Tuple[int, str, np.ndarray]],
               title: str, color: str) -> None:
        if not sessions:
            ax.set_visible(False)
            return
        data_plot: List[List[float]] = []
        xtick_labels: List[str] = []
        for sess_num, name, labels in sessions:
            bouts_win = _extract_bouts(labels, focal_state)
            bouts_sec = [b * window_sec for b in bouts_win]
            data_plot.append(bouts_sec if bouts_sec else [0.0])
            n_bouts = len(bouts_win)
            mean_dur = float(np.mean(bouts_sec)) if bouts_sec else 0.0
            xtick_labels.append(f"s{sess_num:02d}")
            print(f"  [Fig B] {name}: n_bouts={n_bouts:4d}, "
                  f"mean_bout_dur={mean_dur:.2f}s")

        parts = ax.violinplot(data_plot, positions=range(len(sessions)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(sessions)))
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_ylabel("Bout duration (s)", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _panel(axes[0], cricket_sessions,
           f"Cricket — S{focal_state} bout durations", "C1")
    _panel(axes[1], object_sessions,
           f"Object — S{focal_state} bout durations",  "C0")
    plt.suptitle(
        f"S{focal_state} bout duration distribution per session",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bout violin          → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure C — Within-session rolling occupancy
# ─────────────────────────────────────────────────────────────────────────────


def fig_rolling_occupancy(
    all_labels: Dict[str, np.ndarray],
    focal_state: int,
    window_sec: float,
    rolling_win: int,
    out_path: Path,
) -> None:
    """Figure C: rolling mean occupancy of focal state within each session."""
    cricket_sessions: List[Tuple[int, str, np.ndarray]] = []
    object_sessions:  List[Tuple[int, str, np.ndarray]] = []

    for name, labels in all_labels.items():
        parsed = _parse_session_name(name)
        if parsed is None:
            continue
        sess_num, cond = parsed
        entry = (sess_num, name, labels)
        (cricket_sessions if cond == "cricket" else object_sessions).append(entry)

    cricket_sessions.sort(key=lambda x: x[0])
    object_sessions.sort(key=lambda x: x[0])

    if not cricket_sessions:
        print("  [Fig C] No cricket sessions found — skipping")
        return

    n_cricket = len(cricket_sessions)
    cmap_c = plt.get_cmap("YlOrRd", max(n_cricket, 2))

    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, (sess_num, name, labels) in enumerate(cricket_sessions):
        rolling = _rolling_occupancy(labels, focal_state, rolling_win)
        # Place the rolling mean at the centre of each sliding window
        time_min = (np.arange(len(rolling)) + rolling_win / 2.0) * window_sec / 60.0
        color = cmap_c(idx / max(n_cricket - 1, 1))
        ax.plot(time_min, rolling, lw=1.8, color=color, label=f"s{sess_num:02d} (cricket)")

    for sess_num, name, labels in object_sessions:
        rolling = _rolling_occupancy(labels, focal_state, rolling_win)
        time_min = (np.arange(len(rolling)) + rolling_win / 2.0) * window_sec / 60.0
        ax.plot(time_min, rolling, lw=0.8, color="grey", alpha=0.4,
                label=f"s{sess_num:02d} (object)")

    win_min = rolling_win * window_sec / 60.0
    ax.set_xlabel("Time in session (minutes)", fontsize=9)
    ax.set_ylabel(f"S{focal_state} occupancy\n(rolling {rolling_win}-window mean)", fontsize=9)
    ax.set_title(
        f"Within-session S{focal_state} dynamics  "
        f"(rolling window = {rolling_win} windows = {win_min:.1f} min)",
        fontsize=10,
    )
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved rolling occupancy    → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure D — Transition matrix: early vs late cricket
# ─────────────────────────────────────────────────────────────────────────────


def fig_transition_early_vs_late(
    all_labels: Dict[str, np.ndarray],
    K: int,
    early_threshold: int,
    out_path: Path,
) -> None:
    """Figure D: pooled transition matrices for early vs late cricket sessions."""
    early_arrs: List[np.ndarray] = []
    late_arrs:  List[np.ndarray] = []

    for name, labels in all_labels.items():
        parsed = _parse_session_name(name)
        if parsed is None or parsed[1] != "cricket":
            continue
        sess_num, _ = parsed
        (early_arrs if sess_num <= early_threshold else late_arrs).append(labels)

    if not early_arrs or not late_arrs:
        print(
            f"  [Fig D] Need both early (≤s{early_threshold:02d}) and "
            f"late (>s{early_threshold:02d}) cricket sessions — skipping"
        )
        return

    T_early = _transition_matrix(np.concatenate(early_arrs), K)
    T_late  = _transition_matrix(np.concatenate(late_arrs),  K)
    T_diff  = T_late - T_early

    tick_labels = [f"S{s}" for s in range(K)]
    panel_size = max(3.0, K * 0.65)
    fig, axes = plt.subplots(1, 3, figsize=(panel_size * 3 + 1.0, panel_size + 1.5))

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
    _draw(axes[0], T_early, f"Early cricket (n={n_e} sessions)",  "Blues",  0,    1)
    _draw(axes[1], T_late,  f"Late cricket  (n={n_l} sessions)",  "Blues",  0,    1)
    _draw(axes[2], T_diff,  "Diff: late − early",                 "RdBu_r", -0.3, 0.3)

    plt.suptitle(
        f"Transition matrices — early (≤s{early_threshold:02d}) vs "
        f"late (>s{early_threshold:02d}) cricket  (K={K})",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition early/late → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def run_dynamics(
    labels_dir: str,
    k: int,
    focal_state: int,
    group_size: int,
    patch_len: int,
    fps: float,
    output_dir: str,
    sessions_early: Optional[List[str]] = None,
    sessions_late: Optional[List[str]] = None,
    sessions_control: Optional[List[str]] = None,
    rolling_win: int = 200,
    early_threshold: int = 4,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    window_sec = group_size * patch_len / fps
    print(f"Window duration   : {window_sec:.3f} s  "
          f"({group_size} patches × {patch_len} frames / {fps} fps)")

    print(f"\nLoading label arrays from {labels_dir} …")
    all_labels = _load_labels(Path(labels_dir))
    if not all_labels:
        print("No *_labels.npy files found. "
              "Run compare_sessions.py with --save_labels first.")
        return
    print(f"  Found {len(all_labels)} sessions: {sorted(all_labels)}")

    # Figure A — ethogram comparison
    early   = sessions_early   or []
    late    = sessions_late    or []
    control = sessions_control or []
    if early or late or control:
        fig_ethogram_comparison(
            all_labels, early, late, control,
            K=k, window_sec=window_sec,
            out_path=out / "ethogram_comparison.png",
        )
    else:
        print("  [Fig A] --sessions_early/late/control not specified — skipping")

    # Figure B — bout duration violin
    fig_bout_violin(
        all_labels, focal_state, window_sec,
        out_path=out / "bout_duration_violin.png",
    )

    # Figure C — within-session rolling occupancy
    fig_rolling_occupancy(
        all_labels, focal_state, window_sec, rolling_win,
        out_path=out / "within_session_timecourse.png",
    )

    # Figure D — transition matrix early vs late
    fig_transition_early_vs_late(
        all_labels, K=k, early_threshold=early_threshold,
        out_path=out / "transition_early_vs_late.png",
    )

    print(f"\nDone. Results in {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="State dynamics figures from saved per-session label arrays",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--labels_dir", required=True,
                   help="Directory containing *_labels.npy files "
                        "(produced by compare_sessions.py --save_labels)")
    p.add_argument("--k",          type=int, required=True,
                   help="Number of K-means states")
    p.add_argument("--focal_state", type=int, default=6,
                   help="Focal state index for bout / rolling analyses")
    p.add_argument("--group_size",  type=int, default=32,
                   help="Patches per window (= window_size // patch_len)")
    p.add_argument("--patch_len",   type=int, default=4,
                   help="Frames per patch")
    p.add_argument("--fps",         type=float, default=62.4,
                   help="Camera frame rate")
    p.add_argument("--output_dir",  default="dynamics_output",
                   help="Output directory for figures")
    p.add_argument("--sessions_early", nargs="+", default=None,
                   help="Session stem(s) for the 'early' row in Figure A "
                        "(e.g. m003_s003_cricket)")
    p.add_argument("--sessions_late", nargs="+", default=None,
                   help="Session stem(s) for the 'late' row in Figure A")
    p.add_argument("--sessions_control", nargs="+", default=None,
                   help="Session stem(s) for the control row in Figure A "
                        "(typically an object session)")
    p.add_argument("--rolling_win", type=int, default=200,
                   help="Sliding-window size (windows) for Figure C rolling occupancy")
    p.add_argument("--early_threshold", type=int, default=4,
                   help="Session-number threshold for Figure D: "
                        "sessions ≤ this are 'early', > this are 'late'")
    args = p.parse_args()

    run_dynamics(
        labels_dir=args.labels_dir,
        k=args.k,
        focal_state=args.focal_state,
        group_size=args.group_size,
        patch_len=args.patch_len,
        fps=args.fps,
        output_dir=args.output_dir,
        sessions_early=args.sessions_early,
        sessions_late=args.sessions_late,
        sessions_control=args.sessions_control,
        rolling_win=args.rolling_win,
        early_threshold=args.early_threshold,
    )


if __name__ == "__main__":
    main()
