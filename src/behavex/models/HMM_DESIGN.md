# Latent Representation Extraction for HMM Training - Design Document

## Overview
This document outlines the design for efficiently extracting latent representations from trained transformer models and using them to train Hidden Markov Models (HMMs) for temporal sequence analysis.

## Architecture

### 1. Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Transformer Model                        │
│              (Already trained - checkpoint)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Forward Pass (no masking)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         Latent Extraction Module                           │
│  - extract_hmm_latents.py (extends extract_latents.py)     │
│  - Efficient batch processing                              │
│  - Multiple representation strategies                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Latent Sequences (B, T, d_model)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│         HMM Data Preparation Module                        │
│  - hmm_trainer.py                                          │
│  - Sequence segmentation & clustering                      │
│  - Discretization strategies                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Discrete State Sequences
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              HMM Training Module                           │
│  - Baum-Welch / EM algorithm                               │
│  - Multiple initialization strategies                      │
│  - Model selection (AIC/BIC)                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ Trained HMM Model
                           ▼
┌─────────────────────────────────────────────────────────────┐
│          Analysis & Visualization Module                   │
│  - State decoding (Viterbi)                                │
│  - Transition matrices                                     │
│  - State persistence analysis                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Latent Extraction Strategies

#### Strategy 1: Per-Timestep Representations (Recommended for HMM)
- Extracts full latent sequence: `(batch, timesteps, d_model)`
- Preserves temporal structure essential for HMM
- Each timestep becomes an observation in the HMM

#### Strategy 2: Per-Window Representations
- Extracts single vector per window: `(batch, d_model)`
- Options: first timestep, last timestep, or mean pooling
- Useful for window-level state analysis

#### Strategy 3: Layer-Wise Representations
- Extracts outputs from all transformer layers
- Can select specific layer depths for different granularity
- Middle layers often capture best temporal dynamics

### 3. HMM Training Approaches

#### Approach A: Direct Gaussian HMM
- Treat continuous latents as emissions from Gaussian distributions
- Each HMM state emits from a multivariate Gaussian
- Pros: No discretization loss, end-to-end differentiable
- Cons: Computationally expensive, assumes Gaussian emissions

#### Approach B: Discretized HMM
- Cluster latent vectors into discrete symbols (k-means, GMM)
- Train discrete-emission HMM on symbol sequences
- Pros: Faster, flexible emission models
- Cons: Discretization may lose information

#### Approach C: Deep HMM
- Neural network parameterizes emission distributions
- Combines power of deep representations with HMM structure
- Pros: Flexible, captures complex patterns
- Cons: More complex training

## Implementation Design

### 1. Enhanced Latent Extraction (extract_hmm_latents.py)

```python
class HmmLatentExtractor:
    """Extracts latent representations optimized for HMM training."""
    
    def __init__(self, checkpoint_path, device='auto'):
        """Load trained transformer model."""
        pass
    
    def extract_sequential_latents(
        self,
        data_path,
        split='all',
        batch_size=64,
        representation_type='per_timestep',
        layer_idx=None,  # None for final, int for specific layer
        use_layer_mean=False,
        flatten_windows=False,  # Concatenate all windows into single sequence
    ):
        """
        Extract latent representations for HMM training.
        
        Args:
            representation_type: 'per_timestep' or 'per_window'
            layer_idx: Which layer to extract from. None for final output.
            flatten_windows: If True, concatenate all windows into (N_total, d_model)
                           If False, keep as list of (T, d_model) sequences
        
        Returns:
            Dictionary with:
            - 'latents': np.array of shape depends on representation_type
            - 'sequence_lengths': List of lengths for each sequence
            - 'timestamps': Original timestamps if available
            - 'metadata': Additional info (file paths, window indices)
        """
        pass
    
    def extract_with_uncertainty(
        self,
        data_path,
        n_forward_passes=10,
        **kwargs
    ):
        """
        Extract multiple forward passes with dropout ON
        to estimate uncertainty in latent space.
        
        Returns:
            Mean and variance of latents for Bayesian HMM.
        """
        pass
```

### 2. HMM Data Preprocessor (hmm_data.py)

```python
class HmmDataPreprocessor:
    """Prepares latent representations for HMM training."""
    
    def __init__(self, latents, sequence_lengths):
        """Initialize with extracted latents."""
        pass
    
    def cluster_observations(
        self,
        n_symbols=50,
        method='kmeans',  # or 'gmm', 'hierarchical'
        covariance_type='full'  # for GMM
    ):
        """
        Discretize continuous latents into symbols.
        
        Returns:
            - symbol_sequences: List of integer sequences
            - cluster_model: Fitted clustering model
            - cluster_centers: Symbol embeddings
        """
        pass
    
    def extract_temporal_features(self):
        """
        Add temporal features to latents (velocity, acceleration).
        May improve HMM state capture.
        """
        pass
    
    def create_padded_batch(self, pad_value=-1):
        """
        Create padded array for batch HMM training.
        shape: (n_sequences, max_length, n_features)
        """
        pass
```

### 3. HMM Trainer (hmm_trainer.py)

```python
class HiddenMarkovModelTrainer:
    """Trains HMM on extracted latent sequences."""
    
    def __init__(self, n_states, covariance_type='full'):
        """Initialize HMM with number of states."""
        pass
    
    def fit(
        self,
        sequences,
        lengths,
        init_method='kmeans',  # or 'random'
        n_iter=100,
        tol=1e-4
    ):
        """
        Train HMM using Baum-Welch EM algorithm.
        
        Args:
            sequences: Concatenated sequence array
            lengths: Length of each individual sequence
        """
        pass
    
    def decode_states(self, sequences, lengths):
        """
        Find most likely state sequence (Viterbi decoding).
        
        Returns:
            state_sequences: Most likely state for each timestep
            state_probabilities: Probability of each state
        """
        pass
    
    def score_sequences(self, sequences, lengths):
        """
        Compute log-likelihood of sequences under model.
        """
        pass
    
    def model_selection(
        self,
        sequences,
        lengths,
        state_range=range(3, 21),
        criterion='bic'  # or 'aic'
    ):
        """
        Train HMMs with different numbers of states
        and select best using information criterion.
        """
        pass
```

### 4. Efficient Extraction Pipeline

```python
class EfficientHmmPipeline:
    """End-to-end pipeline for transformer → HMM."""
    
    def __init__(self, transformer_checkpoint, device='auto'):
        """Initialize with trained transformer."""
        self.extractor = HmmLatentExtractor(transformer_checkpoint, device)
        self.trainer = None
        self.preprocessor = None
    
    def run_with_symbol_clustering(
        self,
        data_path,
        n_hmm_states=10,
        n_symbols=50,
        batch_size=64
    ):
        """
        Run full pipeline: extract → cluster → train HMM.
        
        Returns:
            Dictionary with trained HMM, state sequences, metrics
        """
        # 1. Extract latents
        results = self.extractor.extract_sequential_latents(
            data_path=data_path,
            batch_size=batch_size,
            representation_type='per_timestep',
            flatten_windows=False
        )
        
        # 2. Preprocess: cluster into symbols
        self.preprocessor = HmmDataPreprocessor(
            results['latents'],
            results['sequence_lengths']
        )
        
        symbol_data = self.preprocessor.cluster_observations(
            n_symbols=n_symbols,
            method='kmeans'
        )
        
        # 3. Train discrete HMM
        self.trainer = HiddenMarkovModelTrainer(n_states=n_hmm_states)
        
        self.trainer.fit(
            sequences=symbol_data['symbol_sequences'],
            lengths=results['sequence_lengths'],
            init_method='kmeans'
        )
        
        # 4. Decode states
        state_sequences = self.trainer.decode_states(
            sequences=symbol_data['symbol_sequences'],
            lengths=results['sequence_lengths']
        )
        
        return {
            'hmm_model': self.trainer,
            'state_sequences': state_sequences,
            'latent_data': results,
            'symbol_data': symbol_data
        }
    
    def run_with_gaussian_hmm(
        self,
        data_path,
        n_hmm_states=10,
        batch_size=64
    ):
        """Run pipeline using continuous Gaussian emissions."""
        # Extract latents (same as above)
        # Train Gaussian HMM directly on continuous latents
        pass
```

## Key Design Decisions

### 1. Representation Type Selection
- **per_timestep**: Best for capturing fine-grained temporal dynamics
- **per_window**: Useful for high-level behavior segmentation
- **layer-specific**: Middle layers (2-3 of 4) balance local/global patterns

### 2. Flattening Windows
When windows overlap, flattening creates contiguous sequences:
- Handle overlapping windows by averaging or selecting primary window
- Track original timestamps to maintain temporal ordering

### 3. Discretization Trade-offs
- **N_symbols**: 50-200 typically works well
- Balance between: too few (information loss) vs too many (sparse transitions)
- Can adjust based on dataset size and complexity

### 4. HMM State Count
- Model selection (AIC/BIC) helps find optimal N_states
- Typical range: 5-20 states for behavior analysis
- More states = finer granularity but risk overfitting

### 5. Uncertainty Quantification
- Multiple forward passes with dropout ON estimates posterior variance
- Bayesian HMM can incorporate uncertainty in latent space
- Useful for identifying ambiguous regions

## Integration Points

### With Existing Code
```python
# Load from existing extract_latents.py
from extract_latents import load_model, create_dataloader

# Extend with HMM-specific features
class HmmLatentExtractor:
    def __init__(self, checkpoint_path):
        self.model, self.config = load_model(checkpoint_path)
        # ... add HMM-specific setup
```

### Output Format
```python
# Saved .npz file contains:
{
    'latents': np.ndarray,  # Primary data
    'sequence_lengths': list,  # Lengths for HMM training
    'state_sequences': np.ndarray,  # Decoded HMM states
    'timestamps': np.ndarray,  # Original timestamps
    'transition_matrix': np.ndarray,  # A_ij from HMM
    'emission_params': dict,  # Gaussian parameters or cluster centers
    'hmm_metrics': {  # Model quality
        'log_likelihood': float,
        'aic': float,
        'bic': float,
        'mean_state_duration': float
    }
}
```

## Performance Considerations

### Memory Efficiency
- Stream extraction for large datasets
- Process in chunks to avoid OOM
- Use memmap for large latent arrays

### Computational Speed
- Batch extraction (already in extract_latents.py)
- Pre-compute latents once, train HMM multiple times
- GPU acceleration for extraction, CPU/GPU for HMM

### Scalability
- Parallel extraction across multiple GPUs
- Distributed HMM training for very large datasets
- Online learning for streaming data

## Next Steps

1. **Implement HmmLatentExtractor** - Enhance existing extraction
2. **Implement HmmDataPreprocessor** - Clustering and preparation
3. **Implement HiddenMarkovModelTrainer** - Core HMM training
4. **Create pipeline script** - End-to-end usage example
5. **Add evaluation metrics** - State persistence, transition analysis
6. **Visualization tools** - State sequences, transition matrices

## Dependencies

- `hmmlearn` or `pomegranate` for HMM implementation
- `scikit-learn` for clustering (k-means, GMM)
- Existing dependencies: torch, numpy, scipy
- Optional: `ruptures` for changepoint detection (alternative to HMM)

## Usage Example (Target)

```bash
# Extract latents and train HMM
python run_hmm_pipeline.py \\
  --checkpoint models/best.pt \\
  --data data/processed/all \\
  --n-states 10 \\
  --n-symbols 100 \\
  --representation per_timestep \\
  --output hmm_model.npz

# Analyze results
python analyze_hmm.py \\
  --hmm-model hmm_model.npz \\
  --plot-states \\
  --plot-transitions
```