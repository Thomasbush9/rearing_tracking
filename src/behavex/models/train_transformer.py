from behavex.models.transformer import (
    MaskedTemporalTransformer,
    MaskedPredictiveTemporalTransformer,
    reconstruction_loss,
    predictive_loss,
    create_timestep_mask,
)
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataclasses import dataclass, field
from behavex.models.data import MaskedTemporalDataset
from typing import Union, Optional
from pathlib import Path
import re
import torch
import numpy as np
import datetime
import random
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


def normalize_feature(x: np.ndarray) -> np.ndarray:
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)


def make_windows(X: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows. X: (T, F) -> (N, window_size, F)."""
    windows = [X[i : i + window_size] for i in range(0, len(X) - window_size + 1, stride)]
    return np.stack(windows).astype(np.float32)


def prepare_masked_transformer_data(
    data: pd.DataFrame,
    angular_cols: list[str] | None = None,
    window_size: int = 128,
    stride: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Preprocess: normalize, angular→sin/cos, interpolate, window. Returns (windows, feature_names)."""
    if angular_cols is None:
        angular_cols = [
            "angle_head_body_axis",
            "angle_head_body_l",
            "angle_head_body_r",
            "ori_allBody",
            "ori_trunk",
            "ori_head",
        ]
    df = data.dropna(axis=1, how="all").copy()

    # Normalize non-angular numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_norm = [c for c in numeric_cols if c not in angular_cols and c != "timestamp"]
    df[cols_to_norm] = df[cols_to_norm].apply(normalize_feature)

    # Angular -> sin/cos
    for c in angular_cols:
        if c in df.columns:
            rad = np.deg2rad(df[c])
            df[f"{c}_sin"] = np.sin(rad)
            df[f"{c}_cos"] = np.cos(rad)
            df = df.drop(columns=[c])

    numeric = df.select_dtypes(include=[np.number]).drop(columns=["timestamp"], errors="ignore")
    numeric = numeric.interpolate(method="linear", limit_direction="both")
    feature_names = numeric.columns.tolist()
    return make_windows(numeric.values, window_size, stride), feature_names


def temporal_train_val_test_split(
    windows: np.ndarray, val_ratio: float, test_ratio: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split windows into contiguous train/val/test segments (preserves temporal order)."""
    n = len(windows)
    n_test = min(max(1, int(round(n * test_ratio))), max(0, n - 2))
    n_val = min(max(1, int(round(n * val_ratio))), n - n_test - 1)
    n_train = n - n_val - n_test
    return (
        windows[:n_train],
        windows[n_train : n_train + n_val],
        windows[n_train + n_val :],
    )


# Filename pattern: m{mouse}_s{session}_{cricket|object}.xlsx / .xls
_SESSION_PATTERN = re.compile(r"^m\d+_s\d+_(cricket|object)\.(xlsx|xls)$", re.IGNORECASE)


def _session_key_from_path(path: Path) -> tuple[str, str] | None:
    """From m001_s001_cricket.xlsx return (experiment='m001_s001', type='cricket')."""
    m = _SESSION_PATTERN.match(path.name)
    if not m:
        return None
    # stem is m001_s001_cricket, we want experiment m001_s001 and type cricket
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    experiment = "_".join(parts[:-1])  # m001_s001
    session_type = parts[-1].lower()  # cricket | object
    return (experiment, session_type)


def _timestamp_column(data: pd.DataFrame) -> str | None:
    """First column whose name (strip, lower) is 'timestamp'; None if not found."""
    for c in data.columns:
        if str(c).strip().lower() == "timestamp":
            return str(c)
    return None


def _load_one_session(
    path: Path,
    trim_before_dist_head: bool = True,
    start_time: float | None = None,
) -> pd.DataFrame:
    """
    Load one Excel file. Trim logic (first applicable):
    - If start_time is not None and a 'timestamp' column exists: keep rows with timestamp < start_time.
    - Elif trim_before_dist_head: keep rows before first valid dist_head (notebook style).
    - Else: return full file.
    """
    data = pd.read_excel(path)
    time_col = _timestamp_column(data)
    if start_time is not None and time_col is not None:
        return data[data[time_col] < start_time].copy()
    if not trim_before_dist_head:
        return data
    if "dist_head" not in data.columns:
        return data
    first_valid = data["dist_head"].first_valid_index()
    if first_valid is None:
        return data
    if first_valid == 0:
        return data.iloc[[]].copy()
    return data.loc[: first_valid - 1].copy()


def _load_start_times(start_times_path: Path) -> pd.DataFrame:
    """Load startTimes.xlsx with columns Experiment, CricketEntersTime, ObjectEntersTime."""
    df = pd.read_excel(start_times_path)
    for col in ("Experiment", "CricketEntersTime", "ObjectEntersTime"):
        if col not in df.columns:
            raise ValueError(f"startTimes file must have column '{col}'")
    return df


def _get_start_time_for_file(path: Path, start_times_df: pd.DataFrame) -> float | None:
    """Look up start time (seconds) for this session file; None if not found."""
    key = _session_key_from_path(path)
    if not key:
        return None
    experiment, session_type = key
    exp_norm = experiment.strip().lower()
    exp_col = start_times_df["Experiment"].astype(str).str.strip().str.lower()
    row = start_times_df[exp_col == exp_norm]
    if row.empty:
        return None
    col = "CricketEntersTime" if session_type == "cricket" else "ObjectEntersTime"
    return float(row[col].iloc[0])


def load_data_path(
    data_path: str,
    trim_before_dist_head: bool = True,
    start_times_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load from a single file or a directory. For a directory, finds all files matching
    m{mouse}_s{session}_{cricket|object}.xlsx (or .xls) and concatenates them.
    trim_before_dist_head: if True (default), when not using start times, keep rows before first valid dist_head.
    start_times_path: path to startTimes.xlsx (Experiment, CricketEntersTime, ObjectEntersTime).
      If None and data_path is a dir, uses data_path/startTimes.xlsx when present.
    """
    path = Path(data_path)
    if path.is_file():
        start_path = Path(start_times_path) if start_times_path else path.parent / "startTimes.xlsx"
        if start_path.is_file():
            start_df = _load_start_times(start_path)
            t = _get_start_time_for_file(path, start_df)
            return _load_one_session(path, trim_before_dist_head=(t is None), start_time=t)
        return _load_one_session(path, trim_before_dist_head)
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    start_path = Path(start_times_path) if start_times_path else path / "startTimes.xlsx"
    start_times_df: pd.DataFrame | None = None
    if start_path.is_file():
        start_times_df = _load_start_times(start_path)
    frames: list[pd.DataFrame] = []
    columns_ref: list[str] | None = None
    matching = sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in (".xlsx", ".xls") and _SESSION_PATTERN.match(f.name)
    )
    if not matching:
        raise FileNotFoundError(f"No session files (m*_s*_cricket|object.xlsx) in {path}")
    for f in tqdm(matching, desc="Loading sessions", unit="file"):
        start_time = _get_start_time_for_file(f, start_times_df) if start_times_df is not None else None
        if start_times_df is not None and start_time is None:
            print(f"  [start_times] No match for {f.name}, using fallback trim.")
        use_start = start_time is not None
        df = _load_one_session(
            f,
            trim_before_dist_head=not use_start and trim_before_dist_head,
            start_time=start_time,
        )
        if df.empty:
            continue
        if columns_ref is None:
            columns_ref = df.columns.tolist()
            frames.append(df)
        else:
            frames.append(df.reindex(columns=columns_ref))
    if not frames:
        raise FileNotFoundError(f"No session files (m*_s*_cricket|object.xlsx) in {path}")
    return pd.concat(frames, ignore_index=True)


@dataclass
class MaskedTemporalTrainingArgs:
    """Arguments for training a masked temporal transformer."""
    batch_size: int
    device: str
    learning_rate: float
    f_in: int
    train_windows: Union[np.ndarray, torch.Tensor]
    val_windows: Union[np.ndarray, torch.Tensor]
    test_windows: Union[np.ndarray, torch.Tensor] | None = None
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 4
    mask_ratio: float = 0.15
    epochs: int = 25
    weight_decay: float = 1e-5
    patience: int = 5
    verbose: bool = False
    save_model: bool = True
    save_path: str = field(default_factory=lambda: f"/Users/thomasbush/Downloads/models/masked_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    eval_every_n_epochs: int = 1
    feature_names: Optional[list[str]] = None
    unmasked_weight: float = 0.3
    n_predict_steps: int | None = None
    pred_loss_weight: float = 0.5
    checkpoint_every: int = 0
    resume_from: str | None = None
    save_last: bool = True
    save_best: bool = True
    save_optimizer: bool = True
    save_rng_state: bool = True
    use_compile: bool = False
    compile_backend: str | None = "inductor"
    compile_mode: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] | None = None
    wandb_mode: str | None = None
    wandb_entity: str | None = None


class MaskedTemporalTrainer:

    def __init__(self, training_args: MaskedTemporalTrainingArgs):
        self.training_args = training_args
        self.device = torch.device(training_args.device)
        self.is_predictive = training_args.n_predict_steps is not None
        if self.is_predictive:
            self.model = MaskedPredictiveTemporalTransformer(
                f_in=training_args.f_in,
                d_model=training_args.d_model,
                nhead=training_args.nhead,
                num_layers=training_args.num_layers,
                mask_ratio=training_args.mask_ratio,
                n_predict_steps=training_args.n_predict_steps,
            ).to(self.device)
        else:
            self.model = MaskedTemporalTransformer(
                f_in=training_args.f_in,
                d_model=training_args.d_model,
                nhead=training_args.nhead,
                num_layers=training_args.num_layers,
                mask_ratio=training_args.mask_ratio,
            ).to(self.device)
        self.train_loader = DataLoader(
            MaskedTemporalDataset(training_args.train_windows),
            batch_size=training_args.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            MaskedTemporalDataset(training_args.val_windows),
            batch_size=training_args.batch_size,
            shuffle=False,
        )
        self.test_loader = (
            DataLoader(
                MaskedTemporalDataset(training_args.test_windows),
                batch_size=training_args.batch_size,
                shuffle=False,
            )
            if training_args.test_windows is not None
            else None
        )
        self.optimizer = Adam(
            self.model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.save_path = Path(training_args.save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.save_path / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.save_path))
        self.val_seed = 42  # Fixed seed for reproducible val masks
        self.checkpoints_dir = self.save_path / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._wandb_run = self._init_wandb()
        self.start_epoch = 1
        if training_args.resume_from:
            resume_path = Path(training_args.resume_from).expanduser().resolve()
            if resume_path.is_file():
                try:
                    last_epoch = self._load_checkpoint(resume_path)
                    self.start_epoch = max(1, last_epoch + 1)
                    if training_args.verbose:
                        print(f"[trainer] Resumed from {resume_path} at epoch {last_epoch}")
                except Exception as exc:
                    warnings.warn(f"Failed to resume from {resume_path}: {exc}")
            else:
                warnings.warn(f"Resume checkpoint not found at {resume_path}")
        self._maybe_compile_model()

    def _eval(self, loader: DataLoader) -> float:
        """Eval with fixed seed per batch so val masks are identical across epochs (comparable val loss)."""
        self.model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch_idx, X in enumerate(loader):
                X = X.to(self.device)
                B, T, _ = X.shape
                torch.manual_seed(self.val_seed + batch_idx)
                mask = create_timestep_mask(B, T, self.training_args.mask_ratio, X.device)
                out = self.model(X, mask=mask)
                recon, mask = out[0], out[2]
                batch_loss = reconstruction_loss(
                    recon, X, mask, self.training_args.unmasked_weight
                ).item()
                if self.is_predictive:
                    pred_next = out[3]
                    batch_loss += self.training_args.pred_loss_weight * predictive_loss(
                        pred_next, X, self.training_args.n_predict_steps
                    ).item()
                total += batch_loss * X.size(0)
                n += X.size(0)
        self.model.train()
        return total / n if n else float("inf")

    def _plot_reconstruction(
        self,
        loader: DataLoader,
        save_name: str,
        title: str,
        seed: int,
        tb_tag: str | None = None,
        tb_step: int | None = None,
    ) -> None:
        """Plot GT vs reconstruction for one batch. Fixed seed for comparable masks."""
        if len(loader.dataset) == 0:
            return
        self.model.eval()
        X = next(iter(loader)).to(self.device)
        B, T, _ = X.shape
        torch.manual_seed(seed)
        mask = create_timestep_mask(B, T, self.training_args.mask_ratio, X.device)
        with torch.no_grad():
            out = self.model(X, mask=mask)
            recon, mask = out[0], out[2]
        self.model.train()

        gt = X[0].cpu().numpy()
        pred = recon[0].cpu().numpy()
        m = mask[0].cpu().numpy()
        pred_next = out[3][0].cpu().numpy() if self.is_predictive else None

        n_plot = gt.shape[1]
        names = self.training_args.feature_names
        if names is None or len(names) != n_plot:
            names = [f"Feat {i}" for i in range(n_plot)]

        row_h = 1.0
        fig, axes = plt.subplots(n_plot, 2, figsize=(12, row_h * n_plot), sharex="col")
        if n_plot == 1:
            axes = axes.reshape(1, -1)
        t = np.arange(gt.shape[0])
        masked_idx = np.where(m)[0]

        for i in range(n_plot):
            ax = axes[i, 0]
            ax.plot(t, gt[:, i], label="GT", alpha=0.8)
            ax.plot(t, pred[:, i], label="Recon", alpha=0.8, linestyle="--")
            for ti in masked_idx:
                ax.axvline(ti, color="orange", alpha=0.4, linewidth=0.8)
            ax.set_ylabel(names[i], fontsize=6)
            ax.legend(loc="upper right", fontsize=5)
            ax.grid(True, alpha=0.3)
            ax.set_title("Full seq (| = masked)" if i == 0 else None, fontsize=7)

            ax = axes[i, 1]
            if len(masked_idx) > 0:
                gt_m = gt[masked_idx, i]
                pred_m = pred[masked_idx, i]
                ax.scatter(gt_m, pred_m, alpha=0.6, s=15)
                mi, mx = min(gt_m.min(), pred_m.min()), max(gt_m.max(), pred_m.max())
                ax.plot([mi, mx], [mi, mx], "k--", alpha=0.5, label="y=x")
            ax.set_xlabel("GT", fontsize=6)
            ax.set_ylabel("Pred", fontsize=6)
            ax.legend(loc="upper right", fontsize=5)
            ax.grid(True, alpha=0.3)
            ax.set_title("Masked only" if i == 0 else None, fontsize=7)

        axes[-1, 0].set_xlabel("Time")
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(self.plots_dir / save_name, dpi=150, bbox_inches="tight")
        if tb_tag is not None and tb_step is not None:
            self.writer.add_figure(tb_tag, fig, tb_step)
            self._log_wandb_image(tb_tag, fig, tb_step)
        plt.close(fig)

        if self.is_predictive and pred_next is not None:
            k_max = min(self.training_args.n_predict_steps, 3)
            fig, pred_axes = plt.subplots(
                n_plot, 1, figsize=(12, 1.5 * n_plot), sharex=True
            )
            if n_plot == 1:
                pred_axes = [pred_axes]
            t = np.arange(gt.shape[0])
            for i, ax in enumerate(pred_axes):
                ax.plot(t, gt[:, i], label="GT", alpha=0.7)
                for k in range(k_max):
                    valid_len = gt.shape[0] - (k + 1)
                    if valid_len <= 0:
                        continue
                    ax.plot(
                        t[:valid_len],
                        pred_next[:valid_len, k, i],
                        linestyle="--",
                        alpha=0.8,
                        label=f"Pred t+{k+1}" if i == 0 else None,
                    )
                ax.set_ylabel(names[i], fontsize=6)
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=6)
            pred_axes[-1].set_xlabel("Time")
            fig.suptitle("Predictive head (multi-step ahead)")
            pred_name = save_name.replace(".png", "_pred.png")
            plt.tight_layout()
            plt.savefig(self.plots_dir / pred_name, dpi=150, bbox_inches="tight")
            if tb_tag is not None and tb_step is not None:
                pred_tag = f"{tb_tag}_pred"
                self.writer.add_figure(pred_tag, fig, tb_step)
                self._log_wandb_image(pred_tag, fig, tb_step)
            plt.close(fig)

    def _plot_validation(self, epoch: int) -> None:
        """Plot GT vs reconstruction on validation set."""
        self._plot_reconstruction(
            self.val_loader,
            f"val_recon_epoch{epoch}.png",
            f"Validation epoch {epoch}: ~15% timesteps masked (orange bars)",
            self.val_seed,
            tb_tag="val/gt_vs_recon",
            tb_step=epoch,
        )

    def _init_wandb(self):
        args = self.training_args
        if args.wandb_project is None:
            return None
        if wandb is None:
            warnings.warn("wandb is not installed; skipping wandb logging.")
            return None
        init_kwargs = {
            "project": args.wandb_project,
        }
        if args.wandb_run_name:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_tags:
            init_kwargs["tags"] = args.wandb_tags
        if args.wandb_mode:
            init_kwargs["mode"] = args.wandb_mode
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity
        config = {
            "model": "masked_predictive_transformer" if self.is_predictive else "masked_transformer",
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "mask_ratio": args.mask_ratio,
            "unmasked_weight": args.unmasked_weight,
            "n_predict_steps": args.n_predict_steps,
            "pred_loss_weight": args.pred_loss_weight,
            "device": args.device,
            "checkpoint_every": args.checkpoint_every,
            "use_compile": args.use_compile,
        }
        init_kwargs["config"] = config
        try:
            return wandb.init(**init_kwargs)
        except Exception as exc:
            warnings.warn(f"Failed to initialize wandb: {exc}")
            return None

    def _log_wandb(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self._wandb_run is None:
            return
        payload = dict(metrics)
        if step is not None and "epoch" not in payload:
            payload["epoch"] = step
        self._wandb_run.log(payload, step=step)

    def _log_wandb_image(self, tag: str, fig: plt.Figure, step: int | None) -> None:
        if self._wandb_run is None or wandb is None or step is None:
            return
        try:
            self._wandb_run.log({tag: wandb.Image(fig)}, step=step)
        except Exception as exc:
            warnings.warn(f"Failed to log image to wandb ({tag}): {exc}")

    def _maybe_compile_model(self) -> None:
        if not self.training_args.use_compile:
            return
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            warnings.warn("torch.compile is unavailable in this PyTorch build; skipping.")
            return
        try:
            self.model = compile_fn(
                self.model,
                backend=self.training_args.compile_backend,
                mode=self.training_args.compile_mode,
            )
            if self.training_args.verbose:
                print(
                    f"[trainer] torch.compile enabled "
                    f"(backend={self.training_args.compile_backend}, mode={self.training_args.compile_mode})"
                )
        except Exception as exc:
            warnings.warn(f"torch.compile failed; continuing without compile. Error: {exc}")

    def _capture_rng_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, object]) -> None:
        torch_state = state.get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state)
        cuda_state = state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        numpy_state = state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        python_state = state.get("python")
        if python_state is not None:
            random.setstate(python_state)

    def _checkpoint_state(self, epoch: int, val_loss: float | None) -> dict[str, object]:
        args = self.training_args
        state: dict[str, object] = {
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
            "val_loss": val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_state": self.model.state_dict(),
            "training_params": {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "mask_ratio": args.mask_ratio,
                "epochs": args.epochs,
                "unmasked_weight": args.unmasked_weight,
                "n_predict_steps": args.n_predict_steps,
                "pred_loss_weight": args.pred_loss_weight,
                "device": args.device,
            },
        }
        if args.save_optimizer:
            state["optimizer_state"] = self.optimizer.state_dict()
        if args.save_rng_state:
            state["rng_state"] = self._capture_rng_state()
        return state

    def _save_checkpoint(
        self, path: Path, epoch: int, val_loss: float | None, is_best: bool = False
    ) -> None:
        try:
            payload = self._checkpoint_state(epoch, val_loss)
            payload["is_best"] = is_best
            torch.save(payload, path)
            if self.training_args.verbose:
                label = "best" if is_best else path.name
                print(f"[trainer] Saved checkpoint ({label}) at epoch {epoch} -> {path}")
        except Exception as exc:
            warnings.warn(f"Failed to save checkpoint at {path}: {exc}")

    def _load_checkpoint(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device)
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            # Backward compatibility: checkpoint is raw state dict.
            self.model.load_state_dict(checkpoint)
            warnings.warn(
                f"Checkpoint {path} lacks trainer metadata; loaded weights only without optimizer state."
            )
            return 0
        self.model.load_state_dict(checkpoint["model_state"])
        if self.training_args.save_optimizer and "optimizer_state" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except Exception as exc:
                warnings.warn(f"Could not load optimizer state from {path}: {exc}")
        self.best_loss = checkpoint.get("best_loss", self.best_loss)
        self.best_epoch = checkpoint.get("best_epoch", self.best_epoch)
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.train_losses = checkpoint.get("train_losses", self.train_losses)
        self.val_losses = checkpoint.get("val_losses", self.val_losses)
        rng_state = checkpoint.get("rng_state")
        if rng_state and self.training_args.save_rng_state:
            self._restore_rng_state(rng_state)
        return int(checkpoint.get("epoch", 0))

    def learning_step(self, X: torch.Tensor) -> float:
        X = X.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(X)
        recon, mask = out[0], out[2]
        loss = reconstruction_loss(recon, X, mask, self.training_args.unmasked_weight)
        if self.is_predictive:
            loss = loss + self.training_args.pred_loss_weight * predictive_loss(
                out[3], X, self.training_args.n_predict_steps
            )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> float:
        args = self.training_args
        eval_every = max(1, args.eval_every_n_epochs)
        stop_training = False
        final_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, args.epochs + 1):
            final_epoch = epoch
            self.model.train()
            running_loss = 0.0
            n_train = len(self.train_loader.dataset)
            if n_train == 0:
                raise ValueError("Train dataset is empty; cannot train.")
            for X in self.train_loader:
                running_loss += self.learning_step(X) * X.size(0)
            avg_train = running_loss / n_train
            self.train_losses.append(avg_train)
            self.writer.add_scalar("train/loss", avg_train, epoch)
            self._log_wandb({"train/loss": avg_train}, step=epoch)

            epoch_val_loss: float | None = None
            if epoch % eval_every == 0 or epoch == args.epochs:
                epoch_val_loss = self._eval(self.val_loader)
                self.val_losses.append(epoch_val_loss)
                self.writer.add_scalar("val/loss", epoch_val_loss, epoch)
                self._log_wandb({"val/loss": epoch_val_loss}, step=epoch)
                self._plot_validation(epoch)
                if args.verbose:
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Train: {avg_train:.4f} | Val: {epoch_val_loss:.4f}"
                    )

                if epoch_val_loss < self.best_loss:
                    self.best_loss = epoch_val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    if args.save_model and args.save_best:
                        self._save_checkpoint(self.save_path / "best.pt", epoch, epoch_val_loss, is_best=True)
                    self._log_wandb({"best/val_loss": self.best_loss}, step=epoch)
                else:
                    self.patience_counter += 1

                if self.patience_counter >= args.patience:
                    if args.verbose:
                        print(f"Early stop at epoch {epoch}")
                    stop_training = True

            checkpoint_metric = epoch_val_loss if epoch_val_loss is not None else avg_train
            if args.save_model and args.save_last:
                self._save_checkpoint(self.save_path / "last.pt", epoch, checkpoint_metric)
            if args.save_model and args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
                ckpt_name = self.checkpoints_dir / f"checkpoint_epoch{epoch}.pt"
                self._save_checkpoint(ckpt_name, epoch, checkpoint_metric)

            if stop_training:
                break

        if self.test_loader is not None:
            test_loss = self._eval(self.test_loader)
            self.writer.add_scalar("test/loss", test_loss, final_epoch)
            self._log_wandb({"test/loss": test_loss}, step=final_epoch)
            self._plot_reconstruction(
                self.test_loader,
                "test_recon.png",
                "Test set: ~15% timesteps masked (orange bars)",
                self.val_seed + 1,
                tb_tag="test/gt_vs_recon",
                tb_step=final_epoch,
            )
            if args.verbose:
                print(f"Final test loss: {test_loss:.4f}")
        self.writer.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception as exc:
                warnings.warn(f"wandb finish failed: {exc}")
        return self.best_loss


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/thomasbush/Downloads/shared WithTWB/m001_s001_cricket.xlsx")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split and training (if set before load).")
    parser.add_argument("--test_loading", action="store_true", help="Only load data and print summary; do not train.")
    parser.add_argument("--full_file", action="store_true", help="Load full file(s); no trim.")
    parser.add_argument("--start_times", type=str, default=None, help="Path to startTimes.xlsx (Experiment, CricketEntersTime, ObjectEntersTime). Default: data dir/startTimes.xlsx when loading a dir.")
    parser.add_argument("--model", choices=["masked", "predictive"], default="masked", help="Model variant: masked (recon only) or predictive (recon + multi-step prediction).")
    parser.add_argument("--n_predict_steps", type=int, default=3, help="Number of next steps to predict (used when --model predictive).")
    parser.add_argument("--pred_loss_weight", type=float, default=0.5, help="Weight for predictive loss (used when --model predictive).")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save an additional checkpoint every N epochs (0 disables).")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint to resume training from.")
    parser.add_argument("--no-save-last", action="store_true", help="Disable saving last.pt after every epoch.")
    parser.add_argument("--no-save-best", action="store_true", help="Disable saving best.pt when validation improves.")
    parser.add_argument("--no-save-optimizer", action="store_true", help="Do not include optimizer state in checkpoints.")
    parser.add_argument("--no-save-rng", action="store_true", help="Do not persist RNG state (PyTorch/Numpy/Python).")
    parser.add_argument("--use-compile", action="store_true", help="Enable torch.compile for faster training (PyTorch 2.0+).")
    parser.add_argument("--compile-backend", type=str, default="inductor", help="Backend to use with torch.compile (default: inductor).")
    parser.add_argument("--compile-mode", type=str, default=None, help="Compilation mode for torch.compile (e.g., 'default', 'reduce-overhead').")
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name to enable logging.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run display name.")
    parser.add_argument("--wandb-tags", nargs="+", default=None, help="Optional list of W&B tags to attach to the run.")
    parser.add_argument("--wandb-mode", type=str, default=None, help="W&B mode: online, offline, disabled.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team/user) to log under.")
    args_cli = parser.parse_args()

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)

    # Load and preprocess (single file or dir of m*_s*_cricket|object.xlsx)
    try:
        data = load_data_path(
            args_cli.data,
            trim_before_dist_head=not args_cli.full_file,
            start_times_path=None if args_cli.full_file else args_cli.start_times,
        )
        windows, feature_names = prepare_masked_transformer_data(data, window_size=128, stride=1)
    except Exception as e:
        if args_cli.test_loading:
            print(f"Load failed: {e}")
            sys.exit(1)
        print(f"Could not load {args_cli.data}: {e}. Using dummy data.")
        windows = np.random.randn(1000, 128, 24).astype(np.float32)
        feature_names = None
    f_in = windows.shape[-1]

    if args_cli.test_loading:
        path = Path(args_cli.data)
        kind = "file" if path.is_file() else "dir"
        print(f"Load OK: {kind} -> {args_cli.data}")
        print(f"  data: {data.shape[0]} rows, {data.shape[1]} cols")
        print(f"  windows: {windows.shape} (N, T, F)")
        print(f"  features: {len(feature_names) if feature_names else f_in}")
        sys.exit(0)

    # Temporal split: contiguous train / val / test (preserves temporal dependencies)
    train_w, val_w, test_w = temporal_train_val_test_split(
        windows, args_cli.val_ratio, args_cli.test_ratio
    )

    n_predict_steps = args_cli.n_predict_steps if args_cli.model == "predictive" else None
    pred_loss_weight = args_cli.pred_loss_weight if args_cli.model == "predictive" else 0.5

    args = MaskedTemporalTrainingArgs(
        batch_size=32,
        device="mps",
        learning_rate=1e-3,
        f_in=f_in,
        train_windows=train_w,
        val_windows=val_w,
        test_windows=test_w,
        epochs=25,
        eval_every_n_epochs=1,
        verbose=True,
        feature_names=feature_names,
        n_predict_steps=n_predict_steps,
        pred_loss_weight=pred_loss_weight,
        checkpoint_every=args_cli.checkpoint_every,
        resume_from=args_cli.resume_from,
        save_last=not args_cli.no_save_last,
        save_best=not args_cli.no_save_best,
        save_optimizer=not args_cli.no_save_optimizer,
        save_rng_state=not args_cli.no_save_rng,
        use_compile=args_cli.use_compile,
        compile_backend=args_cli.compile_backend,
        compile_mode=args_cli.compile_mode,
        wandb_project=args_cli.wandb_project,
        wandb_run_name=args_cli.wandb_run_name,
        wandb_tags=args_cli.wandb_tags,
        wandb_mode=args_cli.wandb_mode,
        wandb_entity=args_cli.wandb_entity,
    )
    trainer = MaskedTemporalTrainer(args)
    best = trainer.train()
    print(f"Best val loss: {best:.4f}")


