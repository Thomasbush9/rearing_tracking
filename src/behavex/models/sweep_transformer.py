"""WandB sweep framework for masked temporal transformer fine-tuning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError:
    wandb = None

from behavex.models.train_transformer import run_train

DEFAULT_SWEEP_CONFIG: dict[str, Any] = {
    "method": "bayes",
    "metric": {"name": "val/loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-2},
        "d_model": {"values": [32, 64, 128]},
        "nhead": {"values": [2, 4]},
        "num_layers": {"values": [2, 4, 6]},
        "mask_ratio": {"values": [0.10, 0.15, 0.20]},
        "unmasked_weight": {"values": [0.1, 0.3, 0.5]},
        "batch_size": {"values": [16, 32, 64]},
        "pred_loss_weight": {"values": [0.3, 0.5, 0.7]},
        "data": {"value": None},
        "val_ratio": {"value": 0.15},
        "test_ratio": {"value": 0.15},
        "epochs": {"value": 25},
        "patience": {"value": 5},
        "device": {"value": "mps"},
        "model": {"value": "predictive"},
        "n_predict_steps": {"value": 3},
        "seed": {"value": 42},
    },
}


def train_fn() -> None:
    """WandB sweep train function: init run, call run_train, log final metrics."""
    if wandb is None:
        raise RuntimeError("wandb is required for sweeps. Install with: pip install wandb")
    with wandb.init() as run:
        config = dict(wandb.config)
        best_loss = run_train(config)
        wandb.log({"best/val_loss": best_loss})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wandb sweep for masked transformer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML sweep config")
    parser.add_argument("--count", type=int, default=None, help="Number of trials")
    parser.add_argument("--project", type=str, default="masked_transformer", help="WandB project")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--data", type=str, default=None, help="Raw data path (overrides config)")
    parser.add_argument("--preprocessed-data", type=str, default=None, help="Preprocessed .npz path (faster; overrides config)")
    parser.add_argument("--mmap", action="store_true", help="Load preprocessed .npz with memory-mapping (requires uncompressed file)")
    parser.add_argument("--name", type=str, default=None, help="Sweep name")
    args = parser.parse_args()

    if wandb is None:
        print("wandb is required. Install with: pip install wandb", file=sys.stderr)
        sys.exit(1)

    if args.config:
        import yaml

        with open(args.config) as f:
            sweep_config = yaml.safe_load(f)
    else:
        sweep_config = DEFAULT_SWEEP_CONFIG.copy()
        if args.name:
            sweep_config["name"] = args.name

    if args.preprocessed_data:
        if "parameters" not in sweep_config:
            sweep_config["parameters"] = {}
        sweep_config["parameters"]["preprocessed_data"] = {"value": args.preprocessed_data}
        sweep_config["parameters"]["data"] = {"value": None}
        sweep_config["parameters"]["mmap"] = {"value": args.mmap}
    elif args.data:
        if "parameters" not in sweep_config:
            sweep_config["parameters"] = {}
        sweep_config["parameters"]["data"] = {"value": args.data}

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    wandb.agent(sweep_id, function=train_fn, count=args.count)


if __name__ == "__main__":
    main()
