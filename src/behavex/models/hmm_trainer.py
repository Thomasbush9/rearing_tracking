"""
Hidden Markov Model training module for temporal sequence analysis.

This module provides tools for training HMMs on latent representations
extracted from transformer models.
"""

import numpy as np
import pickle
from typing import Union, List, Tuple, Dict, Optional
from pathlib import Path
import warnings

try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn(
        "hmmlearn not installed. Install with: pip install hmmlearn", ImportWarning
    )

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class HiddenMarkovModelTrainer:
    """
    Trains Hidden Markov Models on sequential latent representations.

    Supports both continuous (Gaussian) emissions and discrete emissions
    after clustering. Compatible with hmmlearn library.
    """

    def __init__(
        self,
        n_states: int,
        covariance_type: str = "full",
        emission_type: str = "gaussian",
        n_symbols: Optional[int] = None,
    ):
        """
        Initialize HMM trainer.

        Args:
            n_states: Number of hidden states
            covariance_type: Type of covariance for Gaussian emissions
                           ('full', 'tied', 'diag', 'spherical')
            emission_type: Type of emission distribution
                          ('gaussian', 'discrete', 'gmm')
            n_symbols: Number of discrete symbols (required for discrete emissions)
        """
        if not HMM_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for HMM training. "
                "Install with: pip install hmmlearn"
            )

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.emission_type = emission_type
        self.n_symbols = n_symbols

        self.model = None
        self.cluster_model = None
        self.is_fitted = False
        self.scaler = None

    def _create_model(self) -> hmm.BaseHMM:
        """Create appropriate HMM model based on emission type."""
        if self.emission_type == "gaussian":
            return hmm.GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=100,
                tol=1e-4,
                verbose=False,
            )
        elif self.emission_type == "discrete":
            if self.n_symbols is None:
                raise ValueError("n_symbols required for discrete emissions")
            return hmm.MultinomialHMM(
                n_components=self.n_states, n_iter=100, tol=1e-4, verbose=False
            )
        elif self.emission_type == "gmm":
            # Custom GMM-HMM could be implemented here
            raise NotImplementedError("GMM emissions not yet implemented")
        else:
            raise ValueError(f"Unknown emission type: {self.emission_type}")

    def preprocess_data(
        self, sequences: List[np.ndarray], normalize: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Preprocess sequence data for HMM training.

        Args:
            sequences: List of arrays, each (sequence_length, n_features)
            normalize: Whether to standardize features

        Returns:
            concatenated: Concatenated array (n_samples_total, n_features)
            lengths: List of sequence lengths
        """
        lengths = [len(seq) for seq in sequences]
        concatenated = np.concatenate(sequences, axis=0)

        # Standardize if requested
        if normalize:
            self.scaler = StandardScaler()
            concatenated = self.scaler.fit_transform(concatenated)

        return concatenated, lengths

    def fit(
        self,
        sequences: Union[np.ndarray, List[np.ndarray]],
        lengths: Optional[List[int]] = None,
        init_method: str = "kmeans",
        n_iter: int = 100,
        normalize: bool = True,
        random_state: Optional[int] = None,
    ) -> "HiddenMarkovModelTrainer":
        """
        Train HMM on sequence data.

        Args:
            sequences: Either concatenated array (n_samples, n_features)
                      or list of arrays [(seq_len_i, n_features)]
            lengths: Sequence lengths (required if sequences is concatenated)
            init_method: Initialization method ('kmeans', 'random')
            n_iter: Maximum iterations for EM
            normalize: Whether to standardize features
            random_state: Random seed

        Returns:
            self: Fitted trainer instance
        """
        # Handle input format
        if isinstance(sequences, list):
            concatenated, lengths = self.preprocess_data(sequences, normalize)
        else:
            concatenated = sequences
            if lengths is None:
                raise ValueError("lengths required when sequences is array")

        # Create and configure model
        self.model = self._create_model()
        self.model.n_iter = n_iter

        if random_state is not None:
            self.model.random_state = random_state

        # Fit model
        if self.emission_type == "gaussian":
            self.model.fit(concatenated, lengths)
        elif self.emission_type == "discrete":
            raise ValueError(
                "Use fit_discrete() for discrete emissions. "
                "Data must be discretized first."
            )

        self.is_fitted = True
        return self

    def fit_discrete(
        self,
        sequences: List[np.ndarray],
        n_symbols: int,
        lengths: Optional[List[int]] = None,
        cluster_method: str = "kmeans",
        init_method: str = "kmeans",
        n_iter: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Train discrete-emission HMM.

        First clusters continuous observations into discrete symbols,
        then trains HMM on resulting symbol sequences.

        Args:
            sequences: List of continuous latent sequences
            n_symbols: Number of discrete symbols
            lengths: Sequence lengths (None if sequences is list)
            cluster_method: 'kmeans' or 'gmm'
            init_method: HMM initialization ('kmeans' or 'random')
            n_iter: EM iterations
            random_state: Random seed

        Returns:
            Dictionary with trained components
        """
        # Concatenate all sequences for clustering
        if isinstance(sequences, list):
            concatenated = np.concatenate(sequences, axis=0)
            lengths = [len(seq) for seq in sequences]
        else:
            concatenated = sequences
            if lengths is None:
                raise ValueError("lengths required for concatenated array")

        # Cluster continuous observations into symbols
        if cluster_method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=n_symbols, random_state=random_state, n_init=10
            )
            symbols = self.cluster_model.fit_predict(concatenated)
            cluster_centers = self.cluster_model.cluster_centers_

        elif cluster_method == "gmm":
            self.cluster_model = GaussianMixture(
                n_components=n_symbols,
                random_state=random_state,
                covariance_type="full",
                n_init=5,
            )
            symbols = self.cluster_model.fit_predict(concatenated)
            cluster_centers = self.cluster_model.means_

        else:
            raise ValueError(f"Unknown cluster method: {cluster_method}")

        # Convert to column vector expected by hmmlearn
        symbols = symbols.reshape(-1, 1)

        # Create discrete HMM
        self.emission_type = "discrete"
        self.n_symbols = n_symbols
        self.model = self._create_model()
        self.model.n_iter = n_iter

        if random_state is not None:
            self.model.random_state = random_state

        # Train on symbol sequences
        self.model.fit(symbols, lengths)
        self.is_fitted = True

        return {
            "symbols": symbols,
            "cluster_centers": cluster_centers,
            "concatenated": concatenated,
            "lengths": lengths,
        }

    def decode_states(
        self,
        sequences: Union[np.ndarray, List[np.ndarray]],
        lengths: Optional[List[int]] = None,
        algorithm: str = "viterbi",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode most likely state sequences.

        Args:
            sequences: Observation sequences
            lengths: Sequence lengths (if concatenated)
            algorithm: 'viterbi' or 'map' (maximum a posteriori)

        Returns:
            state_sequences: Most likely state for each timestep
            log_probabilities: Log probability of each sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")

        # Handle input format
        if isinstance(sequences, list):
            concatenated = np.concatenate(sequences, axis=0)
            lengths = [len(seq) for seq in sequences]
        else:
            concatenated = sequences
            if lengths is None:
                raise ValueError("lengths required for concatenated array")

        # Apply scaler if used during training
        if self.scaler is not None:
            concatenated = self.scaler.transform(concatenated)

        # Convert to symbols if discrete HMM
        if self.emission_type == "discrete":
            symbols = self.cluster_model.predict(concatenated)
            concatenated = symbols.reshape(-1, 1)

        # Decode states
        if algorithm == "viterbi":
            state_sequences = self.model.predict(concatenated, lengths)
        elif algorithm == "map":
            _, state_sequences = self.model.decode(concatenated, lengths)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Compute log probabilities
        log_probs = []
        start = 0
        for length in lengths:
            seq = concatenated[start : start + length]
            log_prob = self.model.score(
                seq.reshape(1, -1) if len(seq.shape) == 1 else seq
            )
            log_probs.append(log_prob)
            start += length

        return state_sequences, np.array(log_probs)

    def score_sequences(
        self,
        sequences: Union[np.ndarray, List[np.ndarray]],
        lengths: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute log-likelihood of sequences under the model.

        Args:
            sequences: Observation sequences
            lengths: Sequence lengths

        Returns:
            log_likelihoods: Log-likelihood for each sequence
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")

        # Handle input format
        if isinstance(sequences, list):
            concatenated = np.concatenate(sequences, axis=0)
            lengths = [len(seq) for seq in sequences]
        else:
            concatenated = sequences
            if lengths is None:
                raise ValueError("lengths required for concatenated array")

        # Apply scaler if used
        if self.scaler is not None:
            concatenated = self.scaler.transform(concatenated)

        # Convert to symbols if discrete HMM
        if self.emission_type == "discrete":
            symbols = self.cluster_model.predict(concatenated)
            concatenated = symbols.reshape(-1, 1)

        # Score sequences
        log_likelihoods = []
        start = 0
        for length in lengths:
            seq = concatenated[start : start + length]
            log_prob = self.model.score(
                seq.reshape(1, -1) if len(seq.shape) == 1 else seq
            )
            log_likelihoods.append(log_prob)
            start += length

        return np.array(log_likelihoods)

    def get_transition_matrix(self) -> np.ndarray:
        """Get HMM state transition matrix."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.transmat_

    def get_stationary_distribution(self) -> np.ndarray:
        """Get stationary distribution of Markov chain."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Compute eigenvector of transition matrix
        eigvals, eigvecs = np.linalg.eig(self.model.transmat_.T)
        stationary = eigvecs[:, np.isclose(eigvals, 1)].real
        stationary = stationary / stationary.sum()
        return stationary.ravel()

    def get_state_statistics(self, sequences: List[np.ndarray]) -> Dict:
        """
        Compute statistics for each HMM state.

        Args:
            sequences: Original observation sequences

        Returns:
            Dictionary with state-wise statistics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Decode states
        state_seqs, _ = self.decode_states(sequences)

        # Concatenate all sequences
        all_seqs = np.concatenate(sequences, axis=0)

        stats = {}
        for state in range(self.n_states):
            mask = state_seqs == state
            state_observations = all_seqs[mask]

            if len(state_observations) > 0:
                stats[f"state_{state}"] = {
                    "n_observations": len(state_observations),
                    "mean_duration": None,  # Compute from state_seqs
                    "feature_means": np.mean(state_observations, axis=0),
                    "feature_stds": np.std(state_observations, axis=0),
                    "prior_probability": np.mean(mask),
                }

        return stats

    def save(self, filepath: Union[str, Path]):
        """Save trained HMM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        save_dict = {
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "emission_type": self.emission_type,
            "n_symbols": self.n_symbols,
            "model_params": {
                "transmat_": self.model.transmat_,
                "startprob_": self.model.startprob_,
            },
            "scaler": self.scaler,
            "is_fitted": self.is_fitted,
        }

        if self.emission_type == "gaussian":
            save_dict["model_params"]["means_"] = self.model.means_
            save_dict["model_params"]["covars_"] = self.model.covars_
        elif self.emission_type == "discrete":
            save_dict["model_params"]["emissionprob_"] = self.model.emissionprob_
            save_dict["cluster_model"] = self.cluster_model

        with open(filepath, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HiddenMarkovModelTrainer":
        """Load trained HMM model."""
        with open(filepath, "rb") as f:
            save_dict = pickle.load(f)

        # Create instance
        instance = cls(
            n_states=save_dict["n_states"],
            covariance_type=save_dict["covariance_type"],
            emission_type=save_dict["emission_type"],
            n_symbols=save_dict["n_symbols"],
        )

        # Create model and restore parameters
        instance.model = instance._create_model()
        for key, value in save_dict["model_params"].items():
            setattr(instance.model, key, value)

        instance.scaler = save_dict["scaler"]
        instance.is_fitted = save_dict["is_fitted"]

        if instance.emission_type == "discrete":
            instance.cluster_model = save_dict["cluster_model"]

        return instance


def model_selection(
    sequences: List[np.ndarray],
    state_range: range = range(3, 11),
    emission_type: str = "gaussian",
    covariance_type: str = "full",
    n_symbols: Optional[int] = None,
    criterion: str = "bic",
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, Dict]:
    """
    Train HMMs with different numbers of states and select best.

    Args:
        sequences: Observation sequences
        state_range: Range of n_states to try
        emission_type: 'gaussian' or 'discrete'
        covariance_type: For Gaussian emissions
        n_symbols: For discrete emissions
        criterion: 'bic' or 'aic' for model selection
        random_state: Random seed
        verbose: Print progress

    Returns:
        Dictionary with results for each n_states
    """
    results = {}
    best_score = np.inf
    best_n_states = None

    for n_states in state_range:
        if verbose:
            print(f"Training HMM with {n_states} states...")

        # Create and train model
        trainer = HiddenMarkovModelTrainer(
            n_states=n_states,
            covariance_type=covariance_type,
            emission_type=emission_type,
            n_symbols=n_symbols,
        )

        try:
            if emission_type == "discrete":
                trainer.fit_discrete(
                    sequences,
                    n_symbols=n_symbols,
                    cluster_method="kmeans",
                    random_state=random_state,
                )
            else:
                trainer.fit(sequences, random_state=random_state)

            # Score model
            log_likelihood = np.sum(trainer.score_sequences(sequences))
            n_params = estimate_n_params(trainer.model, emission_type)
            n_samples = sum(len(seq) for seq in sequences)

            if criterion == "bic":
                score = -2 * log_likelihood + n_params * np.log(n_samples)
            elif criterion == "aic":
                score = -2 * log_likelihood + 2 * n_params
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            results[n_states] = {
                "trainer": trainer,
                "log_likelihood": log_likelihood,
                "n_params": n_params,
                "score": score,
            }

            if score < best_score:
                best_score = score
                best_n_states = n_states

        except Exception as e:
            if verbose:
                print(f"Failed for {n_states} states: {e}")
            results[n_states] = {"error": str(e)}

    results["best_n_states"] = best_n_states
    results["best_score"] = best_score
    results["criterion"] = criterion

    if verbose:
        print(
            f"\nBest number of states: {best_n_states} ({criterion} = {best_score:.2f})"
        )

    return results


def estimate_n_params(model, emission_type: str) -> int:
    """Estimate number of parameters in HMM model."""
    n_states = model.n_components

    # Transition matrix params
    n_params = n_states * (n_states - 1)  # Sum-to-1 constraint

    # Initial state distribution
    n_params += n_states - 1  # Sum-to-1 constraint

    if emission_type == "gaussian":
        # Mean vectors
        n_features = model.means_.shape[1]
        n_params += n_states * n_features

        # Covariance matrices
        if model.covariance_type == "full":
            n_params += n_states * n_features * (n_features + 1) // 2
        elif model.covariance_type == "tied":
            n_params += n_features * (n_features + 1) // 2
        elif model.covariance_type == "diag":
            n_params += n_states * n_features
        elif model.covariance_type == "spherical":
            n_params += n_states

    elif emission_type == "discrete":
        # Emission probabilities
        n_symbols = model.emissionprob_.shape[1]
        n_params += n_states * (n_symbols - 1)  # Sum-to-1 constraint per state

    return n_params
