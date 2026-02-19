"""
Data preprocessing module for HMM training.

This module provides tools for preparing extracted latent representations
for Hidden Markov Model training, including clustering and feature engineering.
"""

import numpy as np
from typing import Union, List, Dict, Optional, Tuple
from pathlib import Path
import warnings

try:
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn(
        "scikit-learn not installed. Install with: pip install scikit-learn",
        ImportWarning,
    )

from .hmm_trainer import model_selection


class LatentDataPreprocessor:
    """
    Preprocesses extracted latent representations for HMM training.

    Handles clustering, dimensionality reduction, temporal feature extraction,
    and preparation of sequence data.
    """

    def __init__(
        self,
        latents: Union[np.ndarray, List[np.ndarray]],
        sequence_lengths: Optional[List[int]] = None,
        scalers: Optional[Dict] = None,
    ):
        """
        Initialize preprocessor with extracted latents.

        Args:
            latents: Extracted representations
                     If np.ndarray: (n_samples_total, d_model) flattened
                     If List: List of arrays [(seq_len_i, d_model)]
            sequence_lengths: Required if latents is flattened array
            scalers: Optional pre-fit scalers {'latent': scaler, 'temporal': scaler}
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn required for preprocessing. "
                "Install with: pip install scikit-learn"
            )

        self.latents = latents
        self.sequence_lengths = sequence_lengths
        self.scalers = scalers or {}

        # Convert to canonical format
        self.sequences, self._lengths = self._convert_to_sequences()

        # Verify consistency
        if len(self.sequences) != len(self._lengths):
            raise ValueError("Number of sequences doesn't match sequence_lengths")

    def _convert_to_sequences(self) -> Tuple[List[np.ndarray], List[int]]:
        """Convert latents to list of sequences format."""
        if isinstance(self.latents, list):
            if self.sequence_lengths is None:
                lengths = [len(seq) for seq in self.latents]
                return self.latents, lengths
            else:
                return self.latents, self.sequence_lengths

        else:  # Numpy array - must be flattened
            if self.sequence_lengths is None:
                raise ValueError("sequence_lengths required for flattened array")

            # Split into sequences
            sequences = []
            start = 0
            for length in self.sequence_lengths:
                sequences.append(self.latents[start : start + length])
                start += length

            return sequences, self.sequence_lengths

    def flatten(self) -> Tuple[np.ndarray, List[int]]:
        """
        Flatten sequences back to (n_samples_total, d_model) format.

        Returns:
            concatenated: (n_samples_total, d_model)
            sequence_lengths: List of lengths
        """
        if len(self.sequences) == 0:
            return np.array([]).reshape(0, 0), []

        d_model = self.sequences[0].shape[1]
        concatenated = np.concatenate(self.sequences, axis=0)

        return concatenated, self._lengths

    def normalize(
        self, method: str = "standard", fit_scaler: bool = True
    ) -> Dict[str, StandardScaler]:
        """
        Apply normalization to latents.

        Args:
            method: 'standard' (z-score) or 'robust' (median/IQR)
            fit_scaler: If True, fit scaler on current data. If False, use existing scaler.

        Returns:
            Dictionary with fitted scaler(s)
        """
        if fit_scaler or "latent" not in self.scalers:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Fit on all data
            concatenated, _ = self.flatten()
            scaler.fit(concatenated)
            self.scalers["latent"] = scaler
        else:
            scaler = self.scalers["latent"]

        # Transform all sequences
        transformed_sequences = []
        for seq in self.sequences:
            transformed = scaler.transform(seq)
            transformed_sequences.append(transformed)

        self.sequences = transformed_sequences

        print(f"Applied {method} normalization to latents")
        print(f"Latent mean: {scaler.mean_[:3]}...")
        print(f"Latent std:  {np.sqrt(scaler.var_)[:3]}...")

        return self.scalers

    def reduce_dimensionality(
        self, n_components: int, method: str = "pca", **kwargs
    ) -> Dict:
        """
        Reduce dimensionality of latents.

        Args:
            n_components: Target dimensionality
            method: 'pca' (Principal Component Analysis)
            **kwargs: Additional arguments for method

        Returns:
            Dictionary with results, including explained variance for PCA
        """
        if method == "pca":
            reducer = PCA(n_components=n_components, **kwargs)

            # Fit on all data
            concatenated, lengths = self.flatten()
            reduced = reducer.fit_transform(concatenated)

            print(
                f"PCA explained variance ratio: {reducer.explained_variance_ratio_[:5]}..."
            )
            print(
                f"Total explained variance: {reducer.explained_variance_ratio_.sum():.4f}"
            )

            # Split back into sequences
            new_sequences = []
            start = 0
            for length in lengths:
                new_sequences.append(reduced[start : start + length])
                start += length

            self.sequences = new_sequences

            return {
                "reducer": reducer,
                "explained_variance_ratio": reducer.explained_variance_ratio_,
                "explained_variance": reducer.explained_variance_,
            }

        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

    def extract_temporal_features(
        self,
        add_velocity: bool = True,
        add_acceleration: bool = False,
        delta_t: int = 1,
    ) -> List[np.ndarray]:
        """
        Extract temporal derivatives as additional features.

        Adds velocity and acceleration based on finite differences.

        Args:
            add_velocity: If True, add first derivative
            add_acceleration: If True, add second derivative
            delta_t: Time step for finite differences

        Returns:
            List of sequences with temporal features appended
        """
        temporal_sequences = []

        for seq in self.sequences:
            T, d = seq.shape
            features = [seq]

            if add_velocity and T > delta_t:
                # Compute velocity (finite difference)
                velocity = np.zeros_like(seq)
                velocity[delta_t:] = (seq[delta_t:] - seq[:-delta_t]) / delta_t
                features.append(velocity)

            if add_acceleration and T > 2 * delta_t:
                # Compute acceleration (second derivative)
                acceleration = np.zeros_like(seq)
                if add_velocity:
                    # Use computed velocity
                    vel = features[1]
                    acceleration[2 * delta_t :] = (
                        vel[2 * delta_t :] - vel[delta_t:-delta_t]
                    ) / delta_t
                else:
                    # Compute directly from position
                    acceleration[2 * delta_t :] = (
                        seq[2 * delta_t :]
                        - 2 * seq[delta_t:-delta_t]
                        + seq[: -2 * delta_t]
                    ) / (delta_t**2)
                features.append(acceleration)

            # Concatenate features
            if len(features) > 1:
                seq_with_temporal = np.concatenate(features, axis=1)
            else:
                seq_with_temporal = seq

            temporal_sequences.append(seq_with_temporal)

        n_temporal = (
            len(self.sequences[0]) - temporal_sequences[0].shape[1]
            if len(temporal_sequences) > 0
            else 0
        )
        print(f"Added {abs(n_temporal)} temporal features per timestep")

        return temporal_sequences

    def cluster_observations(
        self, n_symbols: int, method: str = "kmeans", batch_size: int = 1000, **kwargs
    ) -> Dict:
        """
        Discretize continuous latents into symbols for discrete HMM.

        Args:
            n_symbols: Number of discrete symbols
            method: 'kmeans', 'minibatch_kmeans', or 'gmm'
            batch_size: Batch size for minibatch k-means
            **kwargs: Additional arguments for clustering

        Returns:
            Dictionary with:
            - 'symbol_sequences': List of integer symbol sequences
            - 'cluster_model': Fitted clustering model
            - 'cluster_centers': Symbol embeddings (continuous representation)
            - 'inertia': Clustering inertia (if applicable)
            - 'silhouette_score': Silhouette score (if applicable)
        """
        concatenated, _ = self.flatten()

        if method == "kmeans":
            cluster_model = KMeans(
                n_clusters=n_symbols, random_state=42, n_init=10, **kwargs
            )
            symbols = cluster_model.fit_predict(concatenated)

            print(
                f"KMeans clustering: n_symbols={n_symbols}, inertia={cluster_model.inertia_:.2f}"
            )

        elif method == "minibatch_kmeans":
            cluster_model = MiniBatchKMeans(
                n_clusters=n_symbols,
                random_state=42,
                n_init=3,
                batch_size=batch_size,
                **kwargs,
            )
            symbols = cluster_model.fit_predict(concatenated)

            print(
                f"Minibatch KMeans: n_symbols={n_symbols}, inertia={cluster_model.inertia_:.2f}"
            )

        elif method == "gmm":
            cluster_model = GaussianMixture(
                n_components=n_symbols,
                random_state=42,
                covariance_type="full",
                n_init=5,
                **kwargs,
            )
            symbols = cluster_model.fit_predict(concatenated)

            print(
                f"GMM clustering: n_symbols={n_symbols}, BIC={cluster_model.bic(concatenated):.2f}"
            )

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Split symbols back into sequences
        symbol_sequences = []
        start = 0
        for length in self._lengths:
            symbol_sequences.append(symbols[start : start + length])
            start += length

        # Prepare return dictionary
        result = {
            "symbol_sequences": symbol_sequences,
            "cluster_model": cluster_model,
        }

        if hasattr(cluster_model, "cluster_centers_"):
            result["cluster_centers"] = cluster_model.cluster_centers_
        elif hasattr(cluster_model, "means_"):
            result["cluster_centers"] = cluster_model.means_

        if hasattr(cluster_model, "inertia_"):
            result["inertia"] = cluster_model.inertia_

        return result

    def prepare_for_hmm(
        self,
        normalize: bool = True,
        use_velocity_features: bool = False,
        cluster_config: Optional[Dict] = None,
    ) -> Dict:
        """
        Full preprocessing pipeline for HMM training.

        Args:
            normalize: Whether to apply normalization
            use_velocity_features: Whether to add temporal derivatives
            cluster_config: Config for clustering into symbols
                         {'n_symbols': int, 'method': str, ...}
                         If None, keep continuous features

        Returns:
            Dictionary with preprocessed data ready for HMM
        """
        results = {}

        # 1. Normalize
        if normalize:
            scaler = self.normalize(method="standard", fit_scaler=True)
            results["scaler"] = scaler

        # 2. Add temporal features if requested
        if use_velocity_features:
            self.sequences = self.extract_temporal_features(
                add_velocity=True, add_acceleration=False
            )
            results["temporal_features_added"] = True

        # Convert to canonical format
        concatenated, lengths = self.flatten()
        results["concatenated"] = concatenated
        results["lengths"] = lengths

        # 3. Cluster if requested
        if cluster_config is not None:
            cluster_results = self.cluster_observations(**cluster_config)
            results.update(cluster_results)
            results["emission_type"] = "discrete"
        else:
            results["emission_type"] = "gaussian"

        print(
            f"Prepared data for HMM: {concatenated.shape[0]} samples, {concatenated.shape[1]} features"
        )
        print(f"Emission type: {results['emission_type']}")

        return results

    def analyze_sequence_lengths(self) -> Dict[str, float]:
        """Analyze distribution of sequence lengths."""
        if not self._lengths:
            return {}

        lengths = np.array(self._lengths)

        stats = {
            "mean": float(np.mean(lengths)),
            "std": float(np.std(lengths)),
            "min": float(np.min(lengths)),
            "max": float(np.max(lengths)),
            "median": float(np.median(lengths)),
            "n_sequences": len(lengths),
            "total_timesteps": int(np.sum(lengths)),
        }

        return stats

    def subsample_sequences(
        self, subsample_rate: int = 2, method: str = "decimate"
    ) -> List[np.ndarray]:
        """
        Subsample sequences in time for computational efficiency.

        Args:
            subsample_rate: Keep every nth timestep
            method: 'decimate' (simple downsampling) or 'mean' (average pooling)

        Returns:
            List of subsampled sequences
        """
        subsampled = []

        for seq in self.sequences:
            if method == "decimate":
                sub_seq = seq[::subsample_rate]
            elif method == "mean":
                # Average pooling
                T, d = seq.shape
                pooled_T = T // subsample_rate
                sub_seq = (
                    seq[: pooled_T * subsample_rate]
                    .reshape(pooled_T, subsample_rate, d)
                    .mean(axis=1)
                )
            else:
                raise ValueError(f"Unknown subsampling method: {method}")

            subsampled.append(sub_seq)

        print(
            f"Subsampled sequences at rate {subsample_rate}: "
            f"original mean length {np.mean([len(s) for s in self.sequences]):.1f}, "
            f"new mean length {np.mean([len(s) for s in subsampled]):.1f}"
        )

        return subsampled

    def save(self, filepath: Union[str, Path]):
        """Save preprocessed data to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        concatenated, lengths = self.flatten()

        save_dict = {
            "latents": concatenated,
            "sequence_lengths": lengths,
            "d_model": concatenated.shape[1],
        }

        if self.scalers:
            # Save scaler parameters (simple scalers only)
            scaler_dict = {}
            for name, scaler in self.scalers.items():
                if hasattr(scaler, "mean_"):
                    scaler_dict[name] = {
                        "type": type(scaler).__name__,
                        "mean_": scaler.mean_,
                        "scale_": scaler.scale_ if hasattr(scaler, "scale_") else None,
                        "var_": scaler.var_ if hasattr(scaler, "var_") else None,
                    }
            save_dict["scalers"] = scaler_dict

        np.savez_compressed(filepath, **save_dict)
        print(f"Saved preprocessed data to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LatentDataPreprocessor":
        """Load preprocessed data from disk."""
        with np.load(filepath, allow_pickle=True) as data:
            latents = data["latents"]
            lengths = data["sequence_lengths"]

            instance = cls(latents=latents, sequence_lengths=lengths)

            if "scalers" in data:
                # Note: This is simplified - full sklearn objects can't be serialized easily
                # In practice, you'd use pickle for the whole object or re-fit scalers
                print(
                    "Warning: Scalers loaded but not fully reconstructed. "
                    "Consider re-normalizing if needed."
                )

        return instance


def select_optimal_clusters(
    latents: Union[np.ndarray, List[np.ndarray]],
    sequence_lengths: Optional[List[int]] = None,
    max_symbols: int = 100,
    cluster_range: range = range(10, 101, 10),
    method: str = "kmeans",
    criterion: str = "inertia_ratio",
) -> int:
    """
    Find optimal number of clusters for discretization.

    Args:
        latents: Latent representations
        max_symbols: Maximum symbols to consider
        cluster_range: Range of cluster counts to evaluate
        method: Clustering method
        criterion: 'inertia_ratio' (elbow method) or 'silhouette'

    Returns:
        Optimal number of clusters
    """
    preprocessor = LatentDataPreprocessor(latents, sequence_lengths)
    concatenated, _ = preprocessor.flatten()

    scores = {}

    for n_clusters in cluster_range:
        if n_clusters > len(concatenated):
            break

        print(f"Evaluating {n_clusters} clusters...", end=" ")

        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusterer.fit(concatenated)

            if criterion == "inertia_ratio":
                scores[n_clusters] = clusterer.inertia_
            elif criterion == "silhouette":
                from sklearn.metrics import silhouette_score

                scores[n_clusters] = silhouette_score(concatenated, clusterer.labels_)

        print(f"score={scores[n_clusters]:.2f}")

    # Find elbow point (inertia)
    if criterion == "inertia_ratio":
        # Compute second derivative to find elbow
        cluster_counts = sorted(scores.keys())
        inertias = [scores[k] for k in cluster_counts]

        # Normalize
        inertias = np.array(inertias)
        inertias = inertias / inertias[0]

        # Compute derivative of normalized inertia
        derivatives = np.diff(inertias)

        # Find point of maximum curvature (largest second derivative)
        second_derivatives = np.diff(derivatives)
        elbow_idx = np.argmax(second_derivatives) + 1
        optimal_n = cluster_counts[elbow_idx]

        print(f"Optimal clusters via elbow method: {optimal_n}")

    elif criterion == "silhouette":
        # Max silhouette score
        optimal_n = max(scores.keys(), key=lambda k: scores[k])
        print(f"Optimal clusters via silhouette: {optimal_n}")

    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return optimal_n
