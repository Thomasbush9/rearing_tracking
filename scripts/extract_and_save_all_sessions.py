"""
Extract transformer latents from every session and (optionally) fit an HMM.

Usage — extraction only:
    python scripts/extract_and_save_all_sessions.py \\
        --checkpoint path/to/best.pt \\
        --data-dir   path/to/sessions/ \\
        --output     outputs/all_sessions_latents.npz

Usage — extraction + HMM with automatic state selection (BIC):
    python scripts/extract_and_save_all_sessions.py \\
        --checkpoint  path/to/best.pt \\
        --data-dir    path/to/sessions/ \\
        --output      outputs/all_sessions_latents.npz \\
        --fit-hmm \\
        --hmm-output  outputs/all_sessions_hmm.npz

Usage — extraction + HMM with fixed number of states:
    ... --fit-hmm --n-states 8

Output files:
    all_sessions_latents.npz
        latents             (N_total, d_model)   concatenated latents, all sessions
        session_names       (n_sessions,)         file stem per session
        session_conditions  (n_sessions,)         "cricket" or "object"
        session_starts      (n_sessions,)         first row index in latents[]
        session_lengths     (n_sessions,)         number of rows per session
        checkpoint          str
        window_size         int
        d_model             int

    all_sessions_hmm.npz   (only with --fit-hmm)
        latents             (N_total, d_model)
        states              (N_total,)            Viterbi state per timestep
        session_names       (n_sessions,)
        session_conditions  (n_sessions,)
        session_starts      (n_sessions,)
        session_lengths     (n_sessions,)
        transition_matrix   (K, K)
        stationary          (K,)
        n_states            int
        bic_scores          dict  key=n_states, value=BIC   (only if model selection ran)

    all_sessions_hmm.pkl   (only with --fit-hmm)  — full HiddenMarkovModelTrainer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
# Ensure the src tree is on the path when running as a plain script
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from behavex.models.extract_latents import extract_session_latents_for_hmm
from behavex.models.train_transformer import _get_session_paths


# ── helpers ───────────────────────────────────────────────────────────────────

def _condition_from_path(path: Path) -> str:
    """Return 'cricket' or 'object' from a session filename stem."""
    stem = path.stem.lower()
    if stem.endswith("_cricket"):
        return "cricket"
    if stem.endswith("_object"):
        return "object"
    return "unknown"


def extract_all_sessions(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    window_size: int = 128,
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """Extract latents from every session file and save to ``output_path``.

    Args:
        checkpoint_path: Path to ``best.pt`` or ``last.pt``.
        data_dir:        Directory containing session ``.xlsx`` files.
        output_path:     Destination ``.npz`` file.
        window_size:     Must match the value used during training.
        batch_size:      Inference batch size.
        device:          ``"mps"``, ``"cuda"``, ``"cpu"``, or ``None`` for auto.

    Returns:
        Dictionary with keys ``latents``, ``session_names``,
        ``session_conditions``, ``session_starts``, ``session_lengths``,
        ``d_model``.
    """
    import torch

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    data_dir = Path(data_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if device is not None:
        _device = torch.device(device)
    else:
        _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    session_paths = _get_session_paths(str(data_dir))
    print(f"Found {len(session_paths)} session files in {data_dir}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {_device}")
    print(f"Window size: {window_size}")
    print()

    all_latents: list[np.ndarray] = []
    session_names: list[str] = []
    session_conditions: list[str] = []
    session_starts: list[int] = []
    session_lengths: list[int] = []
    cursor = 0

    for i, sp in enumerate(session_paths):
        print(f"[{i+1}/{len(session_paths)}] {sp.name}", end=" ", flush=True)
        try:
            lat = extract_session_latents_for_hmm(
                session_path=sp,
                checkpoint_path=checkpoint_path,
                window_size=window_size,
                batch_size=batch_size,
                device=_device,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        n = len(lat)
        all_latents.append(lat)
        session_names.append(sp.stem)
        session_conditions.append(_condition_from_path(sp))
        session_starts.append(cursor)
        session_lengths.append(n)
        cursor += n
        print(f"→ {lat.shape}")

    if not all_latents:
        print("No latents extracted — check data directory and checkpoint.", file=sys.stderr)
        sys.exit(1)

    latents = np.concatenate(all_latents, axis=0)
    d_model = latents.shape[-1]
    print(f"\nTotal latents: {latents.shape}  ({len(session_names)} sessions)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        latents=latents,
        session_names=np.array(session_names, dtype=object),
        session_conditions=np.array(session_conditions, dtype=object),
        session_starts=np.array(session_starts, dtype=np.int64),
        session_lengths=np.array(session_lengths, dtype=np.int64),
        checkpoint=str(checkpoint_path),
        window_size=window_size,
        d_model=d_model,
    )
    print(f"Saved latents → {output_path}")

    return dict(
        latents=latents,
        session_names=session_names,
        session_conditions=session_conditions,
        session_starts=np.array(session_starts, dtype=np.int64),
        session_lengths=np.array(session_lengths, dtype=np.int64),
        d_model=d_model,
    )


def fit_hmm(
    extraction_result: dict,
    hmm_output_path: str | Path,
    n_states: int | None = None,
    state_range: range = range(4, 17),
    covariance_type: str = "full",
    sticky_prior: float = 0.95,
    random_state: int = 42,
) -> None:
    """Fit a Gaussian HMM on the extracted latents and save results.

    If ``n_states`` is ``None``, BIC model selection is run over
    ``state_range``. Otherwise the specified number of states is used directly.

    Args:
        extraction_result: Dict returned by :func:`extract_all_sessions`.
        hmm_output_path:   Destination ``.npz`` (the ``.pkl`` is saved alongside).
        n_states:          Fixed number of states; ``None`` → BIC selection.
        state_range:       Candidate state counts for BIC selection.
        covariance_type:   Gaussian covariance type (``"full"`` recommended).
        sticky_prior:      Self-transition bias; ``0.95`` → ~20-frame dwell times.
        random_state:      Random seed for reproducibility.
    """
    from behavex.models.hmm_trainer import HiddenMarkovModelTrainer, model_selection

    latents = extraction_result["latents"]
    session_starts = extraction_result["session_starts"]
    session_lengths = extraction_result["session_lengths"]

    # Build per-session list — hmmlearn handles session boundaries correctly
    sequences = [
        latents[s : s + l]
        for s, l in zip(session_starts, session_lengths)
    ]

    bic_scores: dict[int, float] = {}

    if n_states is None:
        print(f"\nRunning BIC model selection over {list(state_range)} states …")
        results = model_selection(
            sequences=sequences,
            state_range=state_range,
            covariance_type=covariance_type,
            criterion="bic",
            random_state=random_state,
            verbose=True,
        )
        best_k = results["best_n_states"]
        trainer = results[best_k]["trainer"]
        bic_scores = {k: results[k]["score"] for k in state_range if isinstance(results.get(k), dict) and "score" in results[k]}
        print(f"Selected K = {best_k}  (BIC = {results['best_score']:.2f})")
    else:
        best_k = n_states
        print(f"\nFitting HMM with K = {best_k} states …")
        trainer = HiddenMarkovModelTrainer(
            n_states=best_k, covariance_type=covariance_type
        )
        trainer.fit(sequences, sticky_prior=sticky_prior, random_state=random_state)

    # Decode: pass full concatenated array with per-session lengths so
    # Viterbi respects session boundaries.
    print("Decoding states (Viterbi) …")
    states, log_probs = trainer.decode_states(sequences)

    transition_matrix = trainer.get_transition_matrix()
    stationary = trainer.get_stationary_distribution()

    print(f"State occupancy: { {s: int(np.sum(states == s)) for s in range(best_k)} }")

    # Save .npz
    hmm_output_path = Path(hmm_output_path).expanduser().resolve()
    hmm_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        hmm_output_path,
        latents=latents,
        states=states.astype(np.int32),
        session_names=np.array(extraction_result["session_names"], dtype=object),
        session_conditions=np.array(extraction_result["session_conditions"], dtype=object),
        session_starts=session_starts,
        session_lengths=session_lengths,
        transition_matrix=transition_matrix,
        stationary=stationary,
        n_states=best_k,
        log_probs=log_probs,
        bic_scores=np.array(list(bic_scores.items()), dtype=object) if bic_scores else np.array([]),
    )
    print(f"Saved HMM results → {hmm_output_path}")

    # Save .pkl alongside
    pkl_path = hmm_output_path.with_suffix(".pkl")
    trainer.save(pkl_path)
    print(f"Saved HMM model   → {pkl_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract transformer latents from all sessions and optionally fit an HMM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True,  help="Path to best.pt or last.pt")
    p.add_argument("--data-dir",     required=True,  help="Directory with session .xlsx files")
    p.add_argument("--output",       required=True,  help="Output path for latents .npz")
    p.add_argument("--window-size",  type=int, default=128)
    p.add_argument("--batch-size",   type=int, default=64)
    p.add_argument("--device",       default=None,   help="mps | cuda | cpu (auto if omitted)")
    # HMM options
    p.add_argument("--fit-hmm",      action="store_true", help="Fit HMM after extraction")
    p.add_argument("--hmm-output",   default=None,   help="Output path for HMM .npz (default: <output>_hmm.npz)")
    p.add_argument("--n-states",     type=int, default=None, help="Fixed K; omit for BIC selection")
    p.add_argument("--state-min",    type=int, default=4,  help="Min states for BIC search")
    p.add_argument("--state-max",    type=int, default=16, help="Max states for BIC search (inclusive)")
    p.add_argument("--covariance",   default="full", choices=["full", "diag", "tied", "spherical"])
    p.add_argument("--sticky-prior", type=float, default=0.95)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    result = extract_all_sessions(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_path=args.output,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    if args.fit_hmm:
        hmm_out = args.hmm_output or str(Path(args.output).with_suffix("")) + "_hmm.npz"
        fit_hmm(
            extraction_result=result,
            hmm_output_path=hmm_out,
            n_states=args.n_states,
            state_range=range(args.state_min, args.state_max + 1),
            covariance_type=args.covariance,
            sticky_prior=args.sticky_prior,
            random_state=args.seed,
        )


if __name__ == "__main__":
    main()
