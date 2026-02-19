"""
End-to-end pipeline for transformer → HMM analysis.

This script demonstrates a complete workflow from trained transformer model
to Hidden Markov Model analysis of latent dynamics.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from behavex.models.extract_hmm_latents import HmmLatentExtractor
from behavex.models.hmm_data import LatentDataPreprocessor, select_optimal_clusters
from behavex.models.hmm_trainer import HiddenMarkovModelTrainer, model_selection


class TransformerHMMPipeline:
    """End-to-end pipeline for transformer latent extraction and HMM training."""

    def __init__(self, checkpoint_path: str):
        """Initialize pipeline with trained transformer checkpoint."""
        self.checkpoint_path = Path(checkpoint_path)
        self.extractor = None
        self.preprocessor = None
        self.hmm_trainer = None

    def extract_latents(
        self,
        data_path: str,
        output_path: str,
        representation_type: str = "per_timestep",
        layer_idx: int = None,
        split: str = "all",
        batch_size: int = 64,
        apply_zscore: bool = True,
    ):
        """
        Extract latents from transformer.

        Args:
            data_path: Path to raw data
            output_path: Where to save extracted latents
            representation_type: 'per_timestep' or 'per_window'
            layer_idx: Which transformer layer to extract from
            split: Which data split to use
            batch_size: Batch size for extraction
            apply_zscore: Apply z-score normalization
        """
        print("\n" + "=" * 60)
        print("Step 1: Extracting Latent Representations")
        print("=" * 60)

        self.extractor = HmmLatentExtractor(
            checkpoint_path=self.checkpoint_path, return_layer_mean=False
        )

        print(
            f"Model loaded: d_model={self.extractor.d_model}, num_layers={self.extractor.num_layers}"
        )
        if layer_idx is not None:
            print(f"Extracting from layer {layer_idx}")
        else:
            print("Extracting from final layer output")

        results = self.extractor.extract_sequential_latents(
            data_path=data_path,
            representation_type=representation_type,
            layer_idx=layer_idx,
            split=split,
            batch_size=batch_size,
            apply_zscore=apply_zscore,
            flatten_windows=True,  # Flatten for HMM
            no_mask=True,  # Deterministic
        )

        # Save results
        self.extractor.save_results(results, output_path)

        print(f"Extracted {results['latents'].shape[0]} latent vectors")
        print(f"Dimensionality: {results['latents'].shape[1]} (d_model)")
        print(f"Number of sequences: {len(results['sequence_lengths'])}")

        return results

    def preprocess_for_hmm(
        self,
        latents: np.ndarray,
        sequence_lengths: list,
        use_clustering: bool = True,
        n_symbols: int = 50,
        cluster_method: str = "kmeans",
        add_temporal_features: bool = False,
    ):
        """
        Preprocess latents for HMM training.

        Args:
            latents: Extracted latent representations
            sequence_lengths: Lengths of sequences
            use_clustering: Whether to cluster into discrete symbols
            n_symbols: Number of symbols for discrete HMM
            cluster_method: 'kmeans' or 'minibatch_kmeans'
            add_temporal_features: Whether to add velocity/acceleration features
        """
        print("\n" + "=" * 60)
        print("Step 2: Preprocessing for HMM")
        print("=" * 60)

        self.preprocessor = LatentDataPreprocessor(
            latents=latents, sequence_lengths=sequence_lengths
        )

        # Analyze sequences
        seq_stats = self.preprocessor.analyze_sequence_lengths()
        print(f"Sequence length statistics:")
        print(f"  Mean: {seq_stats['mean']:.1f} ± {seq_stats['std']:.1f}")
        print(f"  Range: [{seq_stats['min']:.0f}, {seq_stats['max']:.0f}]")
        print(f"  Total timesteps: {seq_stats['total_timesteps']:.0f}")

        # Prepare HMM data
        cluster_config = (
            {"n_symbols": n_symbols, "method": cluster_method}
            if use_clustering
            else None
        )

        hmm_data = self.preprocessor.prepare_for_hmm(
            normalize=True,
            use_velocity_features=add_temporal_features,
            cluster_config=cluster_config,
        )

        if use_clustering:
            print(
                f"Clustered observations into {n_symbols} discrete symbols using {cluster_method}"
            )
            emission_type = "discrete"
        else:
            print(
                f"Using continuous Gaussian emissions ({latents.shape[1]} dimensions)"
            )
            emission_type = "gaussian"

        return hmm_data

    def train_hmm(
        self,
        hmm_data: dict,
        n_states_range: range = range(5, 21),
        use_model_selection: bool = True,
        criterion: str = "bic",
    ):
        """
        Train HMM on processed data.

        Args:
            hmm_data: Preprocessed data from previous step
            n_states_range: Range of state counts to try
            use_model_selection: Whether to train multiple models and select best
            criterion: 'bic' or 'aic' for model selection
        """
        print("\n" + "=" * 60)
        print("Step 3: Training Hidden Markov Model")
        print("=" * 60)

        sequences = hmm_data["concatenated"]
        lengths = hmm_data["lengths"]
        emission_type = hmm_data["emission_type"]

        print(f"Training HMM with {emission_type} emissions")
        print(f"Data shape: {sequences.shape}")
        print(f"Number of sequences: {len(lengths)}")

        if use_model_selection:
            # Train multiple HMMs and select best
            print(
                f"\nPerforming model selection over state counts {list(n_states_range)}..."
            )

            results = model_selection(
                sequences=sequences,
                lengths=lengths,
                state_range=n_states_range,
                emission_type=emission_type,
                covariance_type="full",
                n_symbols=hmm_data.get("n_symbols"),
                criterion=criterion,
                verbose=True,
            )

            best_n_states = results["best_n_states"]
            best_score = results["best_score"]
            self.hmm_trainer = results[best_n_states]["trainer"]

            print(
                f"\nBest model: {best_n_states} states ({criterion} = {best_score:.2f})"
            )

            # Print scores table
            print("\nModel Selection Results:")
            print(
                f"{'States':<8} {'Log-Likelihood':<16} {'n_params':<10} {'BIC':<12} {'AIC':<12}"
            )
            print("-" * 60)
            for n_states in n_states_range:
                if n_states in results and "error" not in results[n_states]:
                    r = results[n_states]
                    aic = -2 * r["log_likelihood"] + 2 * r["n_params"]
                    print(
                        f"{n_states:<8} {r['log_likelihood']:<16.2f} {r['n_params']:<10} {r['score']:<12.2f} {aic:<12.2f}"
                    )

        else:
            # Train single HMM with fixed number of states
            best_n_states = n_states_range[0]
            print(f"\nTraining HMM with {best_n_states} states...")

            self.hmm_trainer = HiddenMarkovModelTrainer(
                n_states=best_n_states,
                emission_type=emission_type,
                covariance_type="full",
            )

            if emission_type == "discrete":
                # Use pre-computed symbols if available
                if "symbol_sequences" in hmm_data:
                    # Already have symbols, just fit transition structure
                    symbols_concat = np.concatenate(hmm_data["symbol_sequences"])
                    symbols_concat = symbols_concat.reshape(-1, 1)
                    self.hmm_trainer.model = self.hmm_trainer._create_model()
                    self.hmm_trainer.cluster_model = hmm_data["cluster_model"]
                    self.hmm_trainer.model.fit(symbols_concat, lengths)
                else:
                    # Perform clustering and HMM fitting in one step
                    self.hmm_trainer.fit_discrete(
                        sequences=sequences,
                        n_symbols=hmm_data["n_symbols"],
                        lengths=lengths,
                        cluster_method="kmeans",
                    )
            else:
                self.hmm_trainer.fit(
                    sequences=sequences,
                    lengths=lengths,
                    normalize=False,  # Already normalized
                )

        # Decode state sequences
        print("\nDecoding state sequences...")
        state_sequences, log_probs = self.hmm_trainer.decode_states(
            sequences=sequences, lengths=lengths
        )

        # Compute state statistics
        self.state_stats = self.hmm_trainer.get_state_statistics(
            self.preprocessor.sequences
        )

        print(f"\nState statistics:")
        print(f"{'State':<8} {'N_obs':<10} {'Prior':<10} {'Duration':<12}")
        print("-" * 50)
        for state, stats in self.state_stats.items():
            print(
                f"{state:<8} {stats['n_observations']:<10} {stats['prior_probability']:<10.4f} {'N/A':<12}"
            )

        return {
            "trainer": self.hmm_trainer,
            "state_sequences": state_sequences,
            "log_likelihoods": log_probs,
            "n_states": best_n_states,
            "state_stats": self.state_stats,
        }

    def analyze_hmm(self, hmm_results: dict, output_path: str):
        """
        Analyze trained HMM and save results.

        Args:
            hmm_results: Results from training step
            output_path: Where to save analysis results
        """
        print("\n" + "=" * 60)
        print("Step 4: Analyzing HMM Results")
        print("=" * 60)

        trainer = hmm_results["trainer"]
        state_sequences = hmm_results["state_sequences"]

        # Save HMM model
        model_path = Path(output_path).parent / f"{Path(output_path).stem}_model.pkl"
        trainer.save(model_path)
        print(f"Saved HMM model to: {model_path}")

        # Extract analysis metrics
        transition_matrix = trainer.get_transition_matrix()
        stationary_dist = trainer.get_stationary_distribution()

        print(f"\nTransition matrix shape: {transition_matrix.shape}")
        print(f"Stationary distribution: {stationary_dist}")

        # Save complete results
        results = {
            "n_states": hmm_results["n_states"],
            "state_sequences": state_sequences,
            "log_likelihoods": hmm_results["log_likelihoods"],
            "transition_matrix": transition_matrix,
            "stationary_distribution": stationary_dist,
            "sequence_lengths": self.preprocessor._lengths,
            "emission_type": trainer.emission_type,
            "state_statistics": hmm_results["state_stats"],
            "model_path": str(model_path),
        }

        if hasattr(trainer, "cluster_model") and trainer.cluster_model is not None:
            results["cluster_centers"] = trainer.cluster_model.cluster_centers_

        np.savez(Path(output_path), **results)
        print(f"Saved analysis results to: {output_path}")

        return results

    def run_full_pipeline(
        self,
        data_path: str,
        output_dir: str,
        layer_idx: int = None,
        use_clustering: bool = True,
        n_symbols: int = 50,
        n_states_range: range = range(5, 16),
        **kwargs,
    ):
        """
        Run complete pipeline from transformer to HMM analysis.

        Args:
            data_path: Input data path
            output_dir: Directory for outputs
            layer_idx: Which transformer layer to extract
            use_clustering: Whether to use discrete emissions
            n_symbols: Number of symbols for clustering
            n_states_range: Range of HMM state counts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "#" * 60)
        print("# TRANSFORMER → HMM PIPELINE")
        print("#" * 60)

        # Step 1: Extract latents
        latents_path = output_dir / "latents.npz"
        latent_results = self.extract_latents(
            data_path=data_path, output_path=latents_path, layer_idx=layer_idx, **kwargs
        )

        # Step 2: Preprocess
        hmm_data = self.preprocess_for_hmm(
            latents=latent_results["latents"],
            sequence_lengths=latent_results["sequence_lengths"],
            use_clustering=use_clustering,
            n_symbols=n_symbols,
        )

        # Step 3: Train HMM
        hmm_results = self.train_hmm(
            hmm_data=hmm_data, n_states_range=n_states_range, use_model_selection=True
        )

        # Step 4: Analyze
        analysis_path = output_dir / "hmm_analysis.npz"
        analysis_results = self.analyze_hmm(hmm_results, analysis_path)

        print("\n" + "#" * 60)
        print("# PIPELINE COMPLETE")
        print("#" * 60)
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - Latents: {latents_path.name}")
        print(f"  - HMM Analysis: {analysis_path.name}")
        print(f"  - HMM Model: {Path(analysis_results['model_path']).name}")

        return {
            "latent_results": latent_results,
            "hmm_data": hmm_data,
            "hmm_results": hmm_results,
            "analysis_results": analysis_results,
            "output_dir": output_dir,
        }


def demo_pipeline(
    checkpoint_path: str = None, data_path: str = None, output_dir: str = None
):
    """
    Demonstrate pipeline with example parameters.

    Args:
        checkpoint_path: Path to trained transformer (default: example path)
        data_path: Path to data (default: example path)
        output_dir: Output directory (default: ./hmm_results)
    """
    # Example defaults
    checkpoint_path = checkpoint_path or "models/best.pt"
    data_path = data_path or "data/processed/all"
    output_dir = output_dir or "hmm_results/demo"

    print("=" * 70)
    print("DEMO: Transformer → HMM Pipeline")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    # Create pipeline
    pipeline = TransformerHMMPipeline(checkpoint_path)

    # Run with example configuration
    results = pipeline.run_full_pipeline(
        data_path=data_path,
        output_dir=output_dir,
        layer_idx=None,  # Use final layer
        representation_type="per_timestep",  # Extract per-timestep latents
        split="all",  # Use all data
        batch_size=32,
        use_clustering=True,  # Use discrete emissions
        n_symbols=50,  # 50 discrete symbols
        n_states_range=range(5, 16),  # Try 5-15 states
        n_forward_passes=5,  # For uncertainty (if dropout)
    )

    return results


def main():
    """Command-line interface for full pipeline."""
    parser = argparse.ArgumentParser(
        description="Full pipeline: Extract transformer latents → Train HMM → Analyze"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained transformer checkpoint (.pt)",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Path to raw data or preprocessed .npz"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory for outputs"
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=None,
        help="Transformer layer to extract (None for final, negative for from end)",
    )
    parser.add_argument(
        "--representation-type",
        type=str,
        default="per_timestep",
        choices=["per_timestep", "per_window"],
        help="Type of representation to extract",
    )
    parser.add_argument(
        "--use-clustering",
        action="store_true",
        default=True,
        help="Use discrete emissions with clustering (default: True)",
    )
    parser.add_argument(
        "--no-clustering",
        action="store_false",
        dest="use_clustering",
        help="Use continuous Gaussian emissions",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=50,
        help="Number of discrete symbols for clustering",
    )
    parser.add_argument(
        "--n-states-min",
        type=int,
        default=5,
        help="Minimum number of HMM states to try",
    )
    parser.add_argument(
        "--n-states-max",
        type=int,
        default=15,
        help="Maximum number of HMM states to try",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for latent extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for extraction (auto if not specified)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Data split to use",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = TransformerHMMPipeline(args.checkpoint)

    # Run full pipeline
    results = pipeline.run_full_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        layer_idx=args.layer_idx,
        representation_type=args.representation_type,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        use_clustering=args.use_clustering,
        n_symbols=args.n_symbols,
        n_states_range=range(args.n_states_min, args.n_states_max + 1),
    )

    print("\n✓ Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
