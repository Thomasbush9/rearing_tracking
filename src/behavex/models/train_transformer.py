from behavex.models.transformer import (
    MaskedTemporalTransformer,
    reconstruction_loss,
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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


# Filename pattern: m{mouse}_s{session}_{cricket|object}.xlsx / .xls
_SESSION_PATTERN = re.compile(r"^m\d+_s\d+_(cricket|object)\.(xlsx|xls)$", re.IGNORECASE)


def _load_one_session(path: Path, trim_before_dist_head: bool = True) -> pd.DataFrame:
    """
    Load one Excel file. If trim_before_dist_head (default): return rows before first valid dist_head
    (data.loc[:first_valid_index-1]); if first_valid_index==0 return empty; else full file.
    If trim_before_dist_head=False, return full file (no trim).
    """
    data = pd.read_excel(path)
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


def load_data_path(data_path: str, trim_before_dist_head: bool = True) -> pd.DataFrame:
    """
    Load from a single file or a directory. For a directory, finds all files matching
    m{mouse}_s{session}_{cricket|object}.xlsx (or .xls) and concatenates them.
    trim_before_dist_head: if True (default), keep only rows before first valid dist_head (notebook style).
    """
    path = Path(data_path)
    if path.is_file():
        return _load_one_session(path, trim_before_dist_head)
    if not path.is_dir():
        raise FileNotFoundError(f"Not a file or directory: {path}")
    matching = sorted(
        f for f in path.iterdir()
        if f.is_file() and f.suffix.lower() in (".xlsx", ".xls") and _SESSION_PATTERN.match(f.name)
    )
    if not matching:
        raise FileNotFoundError(f"No session files (m*_s*_cricket|object.xlsx) in {path}")
    frames: list[pd.DataFrame] = []
    columns_ref: list[str] | None = None
    for f in tqdm(matching, desc="Loading sessions", unit="file"):
        df = _load_one_session(f, trim_before_dist_head)
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


class MaskedTemporalTrainer:

    def __init__(self, training_args: MaskedTemporalTrainingArgs):
        self.training_args = training_args
        self.device = torch.device(training_args.device)
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
                recon, _, mask = self.model(X, mask=mask)
                total += reconstruction_loss(
                    recon, X, mask, self.training_args.unmasked_weight
                ).item() * X.size(0)
                n += X.size(0)
        self.model.train()
        return total / n if n else 0.0

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
        self.model.eval()
        X = next(iter(loader)).to(self.device)
        B, T, _ = X.shape
        torch.manual_seed(seed)
        mask = create_timestep_mask(B, T, self.training_args.mask_ratio, X.device)
        with torch.no_grad():
            recon, _, mask = self.model(X, mask=mask)
        self.model.train()

        gt = X[0].cpu().numpy()
        pred = recon[0].cpu().numpy()
        m = mask[0].cpu().numpy()

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

    def learning_step(self, X: torch.Tensor) -> float:
        X = X.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        recon, _, mask = self.model(X)
        loss = reconstruction_loss(recon, X, mask, self.training_args.unmasked_weight)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self) -> float:
        args = self.training_args
        for epoch in range(1, args.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for X in self.train_loader:
                running_loss += self.learning_step(X) * X.size(0)
            avg_train = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(avg_train)
            self.writer.add_scalar("train/loss", avg_train, epoch)

            if epoch % args.eval_every_n_epochs == 0:
                val_loss = self._eval(self.val_loader)
                self.val_losses.append(val_loss)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self._plot_validation(epoch)
                if args.verbose:
                    print(f"Epoch {epoch}/{args.epochs} | Train: {avg_train:.4f} | Val: {val_loss:.4f}")

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    if args.save_model:
                        torch.save(self.model.state_dict(), self.save_path / "best.pt")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= args.patience:
                    if args.verbose:
                        print(f"Early stop at epoch {epoch}")
                    break

        self.writer.close()
        if self.test_loader is not None:
            test_loss = self._eval(self.test_loader)
            self._plot_reconstruction(
                self.test_loader,
                "test_recon.png",
                "Test set: ~15% timesteps masked (orange bars)",
                self.val_seed + 1,
            )
            if args.verbose:
                print(f"Final test loss: {test_loss:.4f}")
        return self.best_loss


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/Users/thomasbush/Downloads/shared WithTWB/m001_s001_cricket.xlsx")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_loading", action="store_true", help="Only load data and print summary; do not train.")
    parser.add_argument("--full_file", action="store_true", help="Load full file(s); do not trim at first valid dist_head (default: trim to rows before first valid dist_head, notebook style).")
    args_cli = parser.parse_args()

    # Load and preprocess (single file or dir of m*_s*_cricket|object.xlsx)
    try:
        data = load_data_path(args_cli.data, trim_before_dist_head=not args_cli.full_file)
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

    # Split: train / val / test
    train_w, temp_w = train_test_split(
        windows, test_size=args_cli.val_ratio + args_cli.test_ratio, random_state=args_cli.seed, shuffle=True
    )
    val_ratio_adj = args_cli.val_ratio / (args_cli.val_ratio + args_cli.test_ratio)
    val_w, test_w = train_test_split(temp_w, test_size=1 - val_ratio_adj, random_state=args_cli.seed)

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
    )
    trainer = MaskedTemporalTrainer(args)
    best = trainer.train()
    print(f"Best val loss: {best:.4f}")


