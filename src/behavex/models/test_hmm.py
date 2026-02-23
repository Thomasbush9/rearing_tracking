from behavex.models.extract_latents import extract_session_latents_for_hmm
from behavex.models.hmm_trainer import HiddenMarkovModelTrainer
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_transition_matrix(trainer: HiddenMarkovModelTrainer, out_path: Path) -> None:
    """Heatmap of the HMM transition probability matrix."""
    A = trainer.get_transition_matrix()
    K = A.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, K * 0.7), max(4, K * 0.7)))
    im = ax.imshow(A, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Transition probability")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_yticklabels([f"S{i}" for i in range(K)])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("HMM Transition Matrix")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if A[i, j] > 0.5 else "black")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved transition matrix → {out_path}")


def plot_ethogram(states: np.ndarray, K: int, out_path: Path,
                  n_frames: int = 2000, fps: float = 62.4) -> None:
    """Horizontal ethogram (state per timestep) for a contiguous subset."""
    subset = states[:n_frames]
    T = len(subset)
    time = np.arange(T) / fps  # seconds

    cmap = plt.get_cmap("tab10", K)
    colors = [cmap(i) for i in range(K)]

    fig, ax = plt.subplots(figsize=(14, 2.5))
    for t in range(T - 1):
        ax.axvspan(time[t], time[t + 1], facecolor=colors[subset[t]], alpha=0.9, linewidth=0)
    # Last frame
    ax.axvspan(time[-1], time[-1] + 1 / fps, facecolor=colors[subset[-1]], alpha=0.9, linewidth=0)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in range(K)]
    ax.legend(handles, [f"State {s}" for s in range(K)],
              loc="upper right", ncol=K, fontsize=7, framealpha=0.8)
    ax.set_xlim(0, time[-1] + 1 / fps)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Ethogram — first {T} timesteps ({T / fps:.1f} s)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ethogram          → {out_path}")


def plot_state_occupancy(states: np.ndarray, K: int, out_path: Path) -> None:
    """Bar chart of overall state occupancy fractions."""
    counts = np.array([(states == s).sum() for s in range(K)])
    fractions = counts / len(states)
    cmap = plt.get_cmap("tab10", K)

    fig, ax = plt.subplots(figsize=(max(4, K * 0.8), 3.5))
    bars = ax.bar(range(K), fractions, color=[cmap(i) for i in range(K)], edgecolor="white")
    ax.set_xticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_ylabel("Occupancy fraction")
    ax.set_title("State Occupancy")
    ax.set_ylim(0, min(1.0, fractions.max() * 1.3))
    for bar, frac in zip(bars, fractions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{frac:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved state occupancy   → {out_path}")


if __name__ == "__main__":

    parser = ArgumentParser(description="Single-session HMM sanity check")
    parser.add_argument("--session_path", required=True, type=str)
    parser.add_argument("--model_path",   required=True, type=str)
    parser.add_argument("--output_dir",   default="hmm_sanity_check", type=str,
                        help="Directory to save latents, model, and plots")
    parser.add_argument("--n_states",     default=8, type=int,
                        help="Number of HMM states")
    parser.add_argument("--ethogram_frames", default=2000, type=int,
                        help="Number of timesteps shown in the ethogram")
    parser.add_argument("--fps",          default=62.4, type=float,
                        help="Recording frame rate (for time axis)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract latents ────────────────────────────────────────────────────
    print("Extracting latent space …")
    latents = extract_session_latents_for_hmm(args.session_path, args.model_path)
    print(f"  Latents shape: {latents.shape}")

    # ── 2. Fit HMM ────────────────────────────────────────────────────────────
    print(f"Training HMM (K={args.n_states}) …")
    # sticky_prior: initial self-transition probability fed to EM.
    # 0.95 → each state expects to stay put for ~1/0.05 = 20 patches ≈ 1.3 s at 62.4 fps
    trainer = HiddenMarkovModelTrainer(n_states=args.n_states, covariance_type="full")
    trainer.fit([latents], sticky_prior=0.95)
    states, log_probs = trainer.decode_states([latents])
    print(f"  State occupancy: { {s: int((states == s).sum()) for s in range(args.n_states)} }")

    # ── 3. Save latents + model ───────────────────────────────────────────────
    print("Saving artefacts …")
    np.save(out_dir / "session_latents.npy", latents)
    np.save(out_dir / "session_states.npy",  states)
    trainer.save(out_dir / "hmm_session.pkl")
    print(f"  Saved latents           → {out_dir / 'session_latents.npy'}")
    print(f"  Saved states            → {out_dir / 'session_states.npy'}")
    print(f"  Saved HMM model         → {out_dir / 'hmm_session.pkl'}")

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("Generating plots …")
    plot_transition_matrix(trainer, out_dir / "transition_matrix.png")
    plot_ethogram(states, args.n_states, out_dir / "ethogram.png",
                  n_frames=args.ethogram_frames, fps=args.fps)
    plot_state_occupancy(states, args.n_states, out_dir / "state_occupancy.png")

    print(f"\nDone. All outputs in: {out_dir.resolve()}")
