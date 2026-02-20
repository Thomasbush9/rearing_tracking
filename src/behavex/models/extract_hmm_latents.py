"""
Enhanced latent extraction module optimized for HMM training.

This extends extract_latents.py with features specifically designed for
Hidden Markov Model training, including full temporal sequences and
layer-wise representation extraction.
"""

from pathlib import Path
import argparse
import sys
from typing import Union, List, Dict, Optional, Tuple
import warnings

try:
    import torch
    import numpy as np

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from behavex.models.transformer import (
    MaskedTemporalTransformer,
    MaskedPredictiveTemporalTransformer,
    PatchedMaskedTemporalTransformer,
    PatchedMaskedPredictiveTemporalTransformer,
)
from behavex.models.data import MaskedTemporalDataset
from torch.utils.data import DataLoader

from .train_transformer import (
    load_preprocessed_dataset,
    prepare_masked_transformer_data,
    load_data_path,
    temporal_train_val_test_split,
)
from .extract_latents import _load_model_from_checkpoint, load_single_session_windows


class HmmLatentExtractor:
    """
    Extracts latent representations optimized for HMM training.

    Features:
    - Full temporal sequences (per-timestep latents)
    - Layer-wise representation extraction
    - Sequential data preservation (no window collapsing)
    - Uncertainty quantification
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
        return_layer_mean: bool = False,
    ):
        """
        Initialize extractor with trained model.

        Args:
            checkpoint_path: Path to trained transformer checkpoint
            device: Device to run extraction on
            return_layer_mean: If True, use mean of all layer outputs
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for latent extraction")

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.return_layer_mean = return_layer_mean

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Get model info
        self.d_model = self.model.d_model
        self.num_layers = len(self.model.layers)

    def _load_model(self):
        """Load model from checkpoint."""
        model, _ = _load_model_from_checkpoint(
            self.checkpoint_path, self.device, return_layer_mean=self.return_layer_mean
        )
        return model

    def extract_sequential_latents(
        self,
        data_path: Union[str, Path],
        split: str = "all",
        batch_size: int = 64,
        representation_type: str = "per_timestep",
        layer_idx: Optional[int] = None,
        no_mask: bool = True,
        flatten_windows: bool = True,
        return_uncertainty: bool = False,
        n_forward_passes: int = 10,
        window_size: int = 128,
        mmap_mode: Optional[str] = None,
        apply_zscore: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Extract latent representations for HMM training.

        Args:
            data_path: Path to raw data or preprocessed .npz
            split: Which data split to extract ('train', 'val', 'test', 'all')
            batch_size: Batch size for extraction
            representation_type: 'per_timestep' or 'per_window'
            layer_idx: Which transformer layer to extract from. None for final output.
                      Can be negative to index from end (-1 = last layer, -2 = second-to-last, etc.)
            no_mask: If True, use no masking for deterministic extraction
            flatten_windows: If False, returns list of sequences, one per window.
                           If True, concatenates all windows into single long sequence.
            return_uncertainty: If True, runs multiple forward passes with dropout ON
                              to estimate uncertainty in latent space.
            n_forward_passes: Number of forward passes for uncertainty estimation
            window_size: Window size for data loading
            mmap_mode: Memory map mode for preprocessed data
            apply_zscore: Apply z-score normalization to extracted latents

        Returns:
            Dictionary with:
            - 'latents': Extracted representations
                       If flatten_windows=True: (n_total_timesteps, d_model) or (n_total_timesteps, d_model, n_forward_passes)
                       If flatten_windows=False: List of arrays, each (window_len, d_model)
            - 'sequence_lengths': List of lengths for each window/sequence
            - 'timestamps': Optional timestamps if available
            - 'uncertainty_mean': If return_uncertainty=True, mean over forward passes
            - 'uncertainty_std': If return_uncertainty=True, std over forward passes
            - 'metadata': Additional info (file paths, window indices, etc.)
            - 'split': Which splits were extracted
            - 'd_model': Latent dimension
            - 'layer_idx': Which layer was extracted
        """
        # Load data
        windows_list, split_names, feature_names = self._load_data(
            data_path, split, window_size, mmap_mode
        )

        # Extract latents per split
        all_results = []
        for windows, split_name in zip(windows_list, split_names):
            if len(windows) == 0:
                continue

            results = self._extract_from_windows(
                windows=windows,
                batch_size=batch_size,
                representation_type=representation_type,
                layer_idx=layer_idx,
                no_mask=no_mask,
                flatten_windows=flatten_windows,
                return_uncertainty=return_uncertainty,
                n_forward_passes=n_forward_passes,
                apply_zscore=apply_zscore,
                split_name=split_name,
            )
            all_results.append(results)

        # Combine results from all splits
        combined_results = self._combine_results(all_results)
        combined_results["feature_names"] = feature_names

        return combined_results

    def _load_data(
        self,
        data_path: Union[str, Path],
        split: str,
        window_size: int,
        mmap_mode: Optional[str],
    ) -> Tuple[List[np.ndarray], List[str], List]:
        """Load data based on type and split."""
        data_path = Path(data_path)

        if data_path.suffix == ".npz":
            # Preprocessed data
            mmap_mode = mmap_mode or ("r" if mmap_mode else None)
            train_w, val_w, test_w, feature_names, _, _, _ = load_preprocessed_dataset(
                str(data_path), mmap_mode=mmap_mode
            )

            with np.load(data_path, allow_pickle=True) as npz:
                loaded_window_size = (
                    int(npz["window_size"]) if "window_size" in npz else window_size
                )
        else:
            # Raw data
            data, _ = load_data_path(str(data_path), trim_before_dist_head=True)
            windows, feature_names, _ = prepare_masked_transformer_data(
                data, window_size=window_size, stride=1
            )
            train_w, val_w, test_w = temporal_train_val_test_split(windows, 0.15, 0.15)
            loaded_window_size = window_size

        # Select split(s)
        if split == "train":
            windows_list = [train_w]
            split_names = ["train"]
        elif split == "val":
            windows_list = [val_w]
            split_names = ["val"]
        elif split == "test":
            windows_list = [test_w]
            split_names = ["test"]
        else:
            windows_list = [train_w, val_w, test_w]
            split_names = ["train", "val", "test"]

        return windows_list, split_names, feature_names

    def _extract_from_windows(
        self,
        windows: np.ndarray,
        batch_size: int,
        representation_type: str,
        layer_idx: Optional[int],
        no_mask: bool,
        flatten_windows: bool,
        return_uncertainty: bool,
        n_forward_passes: int,
        apply_zscore: bool,
        split_name: str,
    ) -> Dict:
        """Extract latents from a set of windows."""

        if representation_type == "per_timestep":
            return self._extract_per_timestep(
                windows=windows,
                batch_size=batch_size,
                layer_idx=layer_idx,
                no_mask=no_mask,
                flatten_windows=flatten_windows,
                return_uncertainty=return_uncertainty,
                n_forward_passes=n_forward_passes,
                apply_zscore=apply_zscore,
                split_name=split_name,
            )
        elif representation_type == "per_window":
            return self._extract_per_window(
                windows=windows,
                batch_size=batch_size,
                layer_idx=layer_idx,
                no_mask=no_mask,
                return_uncertainty=return_uncertainty,
                n_forward_passes=n_forward_passes,
                apply_zscore=apply_zscore,
                split_name=split_name,
            )
        else:
            raise ValueError(f"Unknown representation_type: {representation_type}")

    def _extract_per_timestep(
        self,
        windows: np.ndarray,
        batch_size: int,
        layer_idx: Optional[int],
        no_mask: bool,
        flatten_windows: bool,
        return_uncertainty: bool,
        n_forward_passes: int,
        apply_zscore: bool,
        split_name: str,
    ) -> Dict:
        """Extract latent for each timestep in each window."""

        loader = DataLoader(
            MaskedTemporalDataset(windows),
            batch_size=batch_size,
            shuffle=False,
        )

        is_patched = hasattr(self.model, "patch_embedding")

        all_latents = []
        sequence_lengths = []

        if return_uncertainty:
            all_latent_samples = []

        with torch.no_grad():
            for batch_idx, X in enumerate(loader):
                X = X.to(self.device)
                B, T, _ = X.shape

                # Create mask — patched models expect (B, N) patch-level, not (B, T)
                if no_mask:
                    if is_patched:
                        N = T // self.model.patch_len
                        mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
                    else:
                        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
                else:
                    mask = None

                if return_uncertainty:
                    # Multiple forward passes with dropout ON
                    latent_samples = []
                    self.model.train()  # Enable dropout

                    for _ in range(n_forward_passes):
                        out = self._forward_with_layer_selection(X, mask, layer_idx)
                        latent = out[1]  # (B, T, d_model) or (B, N_patches, d_model)
                        latent_samples.append(latent.cpu().numpy())

                    self.model.eval()  # Back to eval mode

                    # Stack samples and compute stats
                    latent_stack = np.stack(
                        latent_samples, axis=-1
                    )  # (B, L, d_model, n_forward_passes)
                    latent_mean = np.mean(latent_stack, axis=-1)  # (B, L, d_model)

                    # Store mean as primary representation
                    batch_latents = latent_mean
                    all_latent_samples.append(latent_stack)

                else:
                    # Single deterministic forward pass
                    out = self._forward_with_layer_selection(X, mask, layer_idx)
                    latent = out[1]  # (B, T, d_model) or (B, N_patches, d_model)
                    batch_latents = latent.cpu().numpy()

                # L = T for standard models, L = N_patches for patched models
                L = batch_latents.shape[1]

                # Store sequences
                for i in range(B):
                    seq_latent = batch_latents[i]  # (L, d_model)

                    if apply_zscore:
                        seq_latent = (seq_latent - np.mean(seq_latent, axis=0)) / (
                            np.std(seq_latent, axis=0) + 1e-8
                        )

                    all_latents.append(seq_latent)
                    sequence_lengths.append(L)

        if flatten_windows:
            # Concatenate all sequences into one long sequence
            if all_latents:
                combined_latents = np.concatenate(all_latents, axis=0)
            else:
                combined_latents = np.array([]).reshape(0, self.d_model)
        else:
            # Keep as list of sequences
            combined_latents = all_latents

        rep_type = "per_patch" if is_patched else "per_timestep"
        result = {
            "latents": combined_latents,
            "sequence_lengths": sequence_lengths,
            "split_name": split_name,
            "d_model": self.d_model,
            "layer_idx": layer_idx,
            "representation_type": rep_type,
            "flatten_windows": flatten_windows,
        }

        if return_uncertainty and all_latent_samples:
            # Concatenate uncertainty samples
            all_samples = np.concatenate(
                all_latent_samples, axis=0
            )  # (N, T, d_model, n_forward_passes)
            result["uncertainty_samples"] = all_samples

        return result

    def _extract_per_window(
        self,
        windows: np.ndarray,
        batch_size: int,
        layer_idx: Optional[int],
        no_mask: bool,
        return_uncertainty: bool,
        n_forward_passes: int,
        apply_zscore: bool,
        split_name: str,
    ) -> Dict:
        """Extract single latent vector per window."""

        loader = DataLoader(
            MaskedTemporalDataset(windows),
            batch_size=batch_size,
            shuffle=False,
        )

        all_latents = []

        if return_uncertainty:
            all_latent_samples = []

        is_patched_per_window = hasattr(self.model, "patch_embedding")
        with torch.no_grad():
            for X in loader:
                X = X.to(self.device)
                B, T, _ = X.shape
                if no_mask:
                    if is_patched_per_window:
                        N = T // self.model.patch_len
                        mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
                    else:
                        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
                else:
                    mask = None

                if return_uncertainty:
                    latent_samples = []
                    self.model.train()  # Enable dropout

                    for _ in range(n_forward_passes):
                        out = self._forward_with_layer_selection(X, mask, layer_idx)
                        latent = out[1]  # (B, T, d_model)
                        # Use last timestep as window representation
                        window_latent = latent[:, -1, :]  # (B, d_model)
                        latent_samples.append(window_latent.cpu().numpy())

                    self.model.eval()

                    latent_stack = np.stack(
                        latent_samples, axis=-1
                    )  # (B, d_model, n_forward_passes)
                    latent_mean = np.mean(latent_stack, axis=-1)  # (B, d_model)

                    all_latents.append(latent_mean)
                    all_latent_samples.append(latent_stack)

                else:
                    out = self._forward_with_layer_selection(X, mask, layer_idx)
                    latent = out[1]  # (B, T, d_model)
                    window_latent = latent[:, -1, :]  # (B, d_model)

                    if apply_zscore:
                        window_latent = (
                            window_latent.cpu().numpy()
                            - np.mean(window_latent.cpu().numpy(), axis=0)
                        ) / (np.std(window_latent.cpu().numpy(), axis=0) + 1e-8)
                    else:
                        window_latent = window_latent.cpu().numpy()

                    all_latents.append(window_latent)

        combined_latents = (
            np.concatenate(all_latents, axis=0) if all_latents else np.array([])
        )

        result = {
            "latents": combined_latents,
            "sequence_lengths": [1]
            * len(combined_latents),  # Each window is one "timestep"
            "split_name": split_name,
            "d_model": self.d_model,
            "layer_idx": layer_idx,
            "representation_type": "per_window",
        }

        if return_uncertainty and all_latent_samples:
            all_samples = np.concatenate(all_latent_samples, axis=0)
            result["uncertainty_samples"] = all_samples

        return result

    def _forward_with_layer_selection(
        self, X: torch.Tensor, mask: Optional[torch.Tensor], layer_idx: Optional[int]
    ):
        """
        Forward pass with optional layer-specific extraction.

        If layer_idx is specified, extracts representation from that layer.
        """
        if layer_idx is None:
            # Standard forward
            return self.model(X, mask=mask)

        # Extract from specific layer
        # This requires bypassing the normal forward pass
        return self._extract_from_specific_layer(X, mask, layer_idx)

    def _extract_from_specific_layer(
        self, X: torch.Tensor, mask: Optional[torch.Tensor], layer_idx: int
    ):
        """Extract representation from a specific transformer layer."""
        B, T, f_in = X.shape

        # Embedding
        if (
            hasattr(self.model, "patch_embedding")
            and self.model.patch_embedding is not None
        ):
            # Patched model
            h = self.model.patch_embedding(X)
        else:
            # Standard model
            h = self.model.embedding(X)

        # Timestep/value masking
        if mask is None and hasattr(self.model, "mask_ratio"):
            timestep_mask = torch.zeros(B, T, dtype=torch.bool, device=X.device)
            value_mask = torch.zeros(B, T, f_in, dtype=torch.bool, device=X.device)
        else:
            timestep_mask = mask
            value_mask = torch.zeros(B, T, f_in, dtype=torch.bool, device=X.device)

        # Apply first norm
        h = self.model.norm(h)

        # Pass through transformer layers
        selected_layer_output = None

        # Handle negative indices
        if layer_idx < 0:
            layer_idx = self.num_layers + layer_idx

        for i, layer in enumerate(self.model.layers):
            h = layer(h, is_causal=self.model.causal)

            if i == layer_idx:
                selected_layer_output = h
                break

        if selected_layer_output is None:
            raise ValueError(
                f"Invalid layer_idx: {layer_idx}. Model has {self.num_layers} layers."
            )

        # Apply final norm
        latent = self.model.norm(selected_layer_output)

        # Reconstruction head
        recon = self.model.reconstruction_head(latent)

        return (recon, latent, timestep_mask, value_mask)

    def _combine_results(self, all_results: List[Dict]) -> Dict:
        """Combine results from multiple splits into single output."""
        if not all_results:
            return {}

        # Check if latents should be flattened
        if all_results[0].get("flatten_windows", True):
            # Concatenate all latents
            all_latents = []
            all_sequence_lengths = []
            split_names = []

            for result in all_results:
                if len(result["latents"]) > 0:
                    all_latents.append(result["latents"])
                    all_sequence_lengths.extend(result["sequence_lengths"])
                    split_names.extend(
                        [result["split_name"]] * result["latents"].shape[0]
                    )

            if all_latents:
                latents = np.concatenate(all_latents, axis=0)
            else:
                latents = np.array([]).reshape(0, self.d_model)

            return {
                "latents": latents,
                "sequence_lengths": all_sequence_lengths,
                "splits": split_names,
                "d_model": self.d_model,
                "layer_idx": all_results[0]["layer_idx"],
                "representation_type": all_results[0]["representation_type"],
            }
        else:
            # Keep as list of sequences
            all_sequences = []
            all_sequence_lengths = []
            split_names = []

            for result in all_results:
                all_sequences.extend(result["latents"])
                all_sequence_lengths.extend(result["sequence_lengths"])
                split_names.extend(
                    [result["split_name"]] * len(result["sequence_lengths"])
                )

            return {
                "latents": all_sequences,
                "sequence_lengths": all_sequence_lengths,
                "splits": split_names,
                "d_model": self.d_model,
                "layer_idx": all_results[0]["layer_idx"],
                "representation_type": all_results[0]["representation_type"],
            }

    def save_results(self, results: Dict, output_path: Union[str, Path]):
        """Save extraction results to .npz file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "d_model": results["d_model"],
            "layer_idx": results.get("layer_idx", None),
            "representation_type": results["representation_type"],
            "sequence_lengths": results["sequence_lengths"],
            "splits": np.array(results["splits"], dtype=object),
        }

        # Handle different latent formats
        if isinstance(results["latents"], list):
            # List of sequences - save with length prefix for each
            flat_latents = []
            sequence_start_indices = []
            start_idx = 0

            for seq in results["latents"]:
                flat_latents.append(seq)
                sequence_start_indices.append(start_idx)
                start_idx += len(seq)

            if flat_latents:
                save_dict["latents"] = np.concatenate(flat_latents, axis=0)
            else:
                save_dict["latents"] = np.array([]).reshape(0, results["d_model"])

            save_dict["sequence_start_indices"] = sequence_start_indices
        else:
            # Already flat array
            save_dict["latents"] = results["latents"]

        # Save uncertainties if present
        if "uncertainty_samples" in results:
            save_dict["uncertainty_samples"] = results["uncertainty_samples"]

        np.savez_compressed(output_path, **save_dict)
        print(f"Saved latent extraction results to: {output_path}")


def main():
    """Command-line interface for latent extraction."""
    parser = argparse.ArgumentParser(
        description="Extract transformer latents optimized for HMM analysis"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained transformer checkpoint (best.pt or last.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to raw data file/directory or preprocessed .npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path for extracted latents",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Which data split to extract",
    )
    parser.add_argument(
        "--representation-type",
        type=str,
        default="per_timestep",
        choices=["per_timestep", "per_window"],
        help="Type of representation to extract",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Extract from specific layer (None for final, -1 for last, -2 for second-to-last, etc.)",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        default=True,
        help="Use unmasked forward pass for deterministic extraction (default: True)",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        dest="use_mask",
        help="Use random masking (non-deterministic)",
    )
    parser.add_argument(
        "--flatten-windows",
        action="store_true",
        default=True,
        help="Concatenate all windows into single sequence (default: True)",
    )
    parser.add_argument(
        "--keep-window-structure",
        action="store_false",
        dest="flatten_windows",
        help="Keep windows separate as list of sequences",
    )
    parser.add_argument(
        "--return-uncertainty",
        action="store_true",
        default=False,
        help="Estimate uncertainty with multiple forward passes (requires dropout)",
    )
    parser.add_argument(
        "--n-forward-passes",
        type=int,
        default=10,
        help="Number of forward passes for uncertainty estimation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (auto-detect if not specified)",
    )
    parser.add_argument(
        "--layer-mean",
        action="store_true",
        default=False,
        help="Use mean of all encoder layers as representation",
    )
    parser.add_argument(
        "--apply-zscore",
        action="store_true",
        default=False,
        help="Apply z-score normalization to extracted latents",
    )
    parser.add_argument(
        "--preprocessed-data",
        action="store_true",
        default=False,
        help="Input is preprocessed .npz data",
    )
    parser.add_argument(
        "--mmap",
        action="store_true",
        default=False,
        help="Use memory mapping for preprocessed data",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Window size for data loading (if not preprocessed)",
    )

    args = parser.parse_args()

    no_mask = not args.use_mask

    # Create extractor
    extractor = HmmLatentExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        return_layer_mean=args.layer_mean,
    )

    print(f"Extracting latents from checkpoint: {args.checkpoint}")
    print(
        f"Model config: d_model={extractor.d_model}, num_layers={extractor.num_layers}"
    )
    print(
        f"Layer selection: {'mean of all layers' if args.layer_mean else f'layer {args.layer_idx or 'final'}'}"
    )

    # Extract latents
    print(f"Extracting {args.representation_type} representations...")

    results = extractor.extract_sequential_latents(
        data_path=args.data,
        split=args.split,
        batch_size=args.batch_size,
        representation_type=args.representation_type,
        layer_idx=args.layer_idx,
        no_mask=no_mask,
        flatten_windows=args.flatten_windows,
        return_uncertainty=args.return_uncertainty,
        n_forward_passes=args.n_forward_passes,
        apply_zscore=args.apply_zscore,
        mmap_mode="r" if args.mmap else None,
        window_size=args.window_size,
    )

    # Save results
    extractor.save_results(results, args.output)

    # Print summary
    latents_shape = (
        results["latents"].shape
        if hasattr(results["latents"], "shape")
        else f"list of {len(results['latents'])} arrays"
    )
    print(f"\nExtraction complete!")
    print(f"Latents shape: {latents_shape}")
    print(f"d_model: {results['d_model']}")
    print(f"Number of sequences: {len(results['sequence_lengths'])}")
    print(f"Total timesteps: {sum(results['sequence_lengths'])}")
    print(f"Splits: {set(results['splits'])}")


if __name__ == "__main__":
    main()
