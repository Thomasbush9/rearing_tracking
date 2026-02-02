from behavex.models.transformer import MaskedTemporalTransformer, masked_reconstruction_loss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataclasses import dataclass, field
from behavex.models.data import MaskedTemporalDataset
from typing import Union
from pathlib import Path
import torch
import numpy as np
import datetime


@dataclass
class MaskedTemporalTrainingArgs:
    """Arguments for training a masked temporal transformer."""
    batch_size: int
    device: str
    learning_rate: float
    f_in: int
    train_windows: Union[np.ndarray, torch.Tensor]
    test_windows: Union[np.ndarray, torch.Tensor]
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 4
    mask_ratio: float = 0.15
    epochs: int = 25
    weight_decay: float = 1e-5
    patience: int = 5
    verbose: bool = False
    save_model: bool = True
    save_path: str = field(default_factory=lambda: f"models/masked_transformer_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    test_every_n_epochs: int = 1


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
        self.test_loader = DataLoader(
            MaskedTemporalDataset(training_args.test_windows),
            batch_size=training_args.batch_size,
            shuffle=False,
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
        self.test_losses: list[float] = []
        Path(training_args.save_path).mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=training_args.save_path)

    def learning_step(self, X: torch.Tensor) -> float:
        X = X.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        recon, _, mask = self.model(X)
        loss = masked_reconstruction_loss(recon, X, mask)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for X in self.test_loader:
                X = X.to(self.device)
                recon, _, mask = self.model(X)
                loss = masked_reconstruction_loss(recon, X, mask)
                total_loss += loss.item() * X.size(0)
                n += X.size(0)
        self.model.train()
        return total_loss / n if n else 0.0

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

            if epoch % args.test_every_n_epochs == 0:
                test_loss = self.test()
                self.test_losses.append(test_loss)
                self.writer.add_scalar("test/loss", test_loss, epoch)
                if args.verbose:
                    print(f"Epoch {epoch}/{args.epochs} | Train: {avg_train:.4f} | Test: {test_loss:.4f}")

                if test_loss < self.best_loss:
                    self.best_loss = test_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    if args.save_model:
                        torch.save(self.model.state_dict(), Path(args.save_path) / "best.pt")
                else:
                    self.patience_counter += 1

                if self.patience_counter >= args.patience:
                    if args.verbose:
                        print(f"Early stop at epoch {epoch}")
                    break

        self.writer.close()
        return self.best_loss


if __name__ == "__main__":
    train_w = np.random.randn(500, 128, 24).astype(np.float32)
    test_w = np.random.randn(100, 128, 24).astype(np.float32)
    args = MaskedTemporalTrainingArgs(
        batch_size=32, device="cpu", learning_rate=1e-3,
        f_in=24, train_windows=train_w, test_windows=test_w,
        epochs=3, test_every_n_epochs=1, verbose=True,
    )
    trainer = MaskedTemporalTrainer(args)
    best = trainer.train()
    print(f"Best test loss: {best:.4f}")


