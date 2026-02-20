
== Model Building and Training

To study mouse behavior and uncover latent structure underlying spontaneous actions, we build upon classical approaches to modeling naturalistic behavior, such as Hidden Markov Models (HMMs), autoregressive HMMs (AR-HMMs), and keypoint-MoSeq @daboraMotionSequencingMoSeq2024, which segment continuous behavior into discrete syllables from pose-derived features. Rather than replacing these frameworks, we adopt a complementary modeling perspective that integrates them with self-supervised representation learning techniques from natural language processing.

In this view, behavior is interpreted as a temporally ordered sequence of actions, where elementary behavioral units recur and combine over time in a manner analogous to tokens in language. This formulation allows us to leverage sequence modeling architectures developed for natural language processing to learn rich, context-dependent representations of behavioral dynamics, while remaining grounded in established pose-derived feature spaces. Crucially, by training a model to reconstruct and predict behavioral time series in an unsupervised manner, we obtain a learned latent space that can subsequently be used to identify discrete behavioral states, measure their temporal structure, and compare behavioral repertoires across experimental conditions.

=== Input Representation and Preprocessing

The input to the model is a multivariate time series of pose-derived features extracted from tracked keypoints. Rather than operating on raw keypoint coordinates, which are sensitive to the animal's absolute position in the arena, we use a set of derived kinematic and geometric features including distances (e.g. inter-keypoint distances, distance to objects), body and head orientations, heading angles, relative bearing, and angular velocities. Angular features (e.g. `angle_head_body_axis`, `ori_trunk`, `facing_angle`, `rel_bearing`) are transformed into sine and cosine components to preserve their circular geometry and avoid discontinuities at $plus.minus 180 degree$. All non-angular numeric features are z-scored (zero mean, unit variance). Missing values are filled via linear interpolation. The resulting feature vector at each frame has dimensionality $F$ (determined by the number of tracked features after angular expansion and column selection).

=== Windowing and Patching

A key difference between language and behavioral time series lies in the information density per token: words carry rich semantic content, whereas individual time points in a continuous signal encode comparatively little information. Transformer models designed for language therefore struggle when applied naively to time series at single-frame resolution @zengAreTransformersEffective2022. To address this, we adopt a sliding-window approach: the full recording is segmented into overlapping windows of $w = 128$ frames with stride $s = 1$, yielding training samples of shape $(w, F)$. The window size was selected based on the autocorrelation structure of the behavioral features to ensure that each window captures sufficient temporal context for the model to learn meaningful dynamics.

In addition to the per-timestep (non-patched) architecture, we train a patched variant inspired by recent work demonstrating that treating time series as sequences of patches substantially improves Transformer performance @nieTimeSeriesWorth2023. In the patched model, each window of $w$ frames is divided into $N = w slash P$ non-overlapping patches, where $P$ is the patch length. Each patch groups $P$ consecutive timesteps across all $F$ features into a single vector of dimension $P times F$, which is then linearly projected to the model dimension $d_"model"$. This reduces the sequence length seen by the Transformer from $w$ to $N$, lowering the quadratic cost of self-attention while simultaneously enriching the information content of each token.

We trained both architectures — timestep-level and patch-level — and evaluated their performance to determine which representation yields the most informative latent space for downstream behavioral analysis.

=== Transformer Architecture

Both model variants are built on an encoder-only Transformer architecture with the following design choices:

- *Pre-normalization*: each encoder layer applies RMSNorm @zhangRootMeanSquare2019 before the self-attention and feed-forward sub-layers, rather than the post-normalization scheme of the original Transformer. This improves training stability, particularly for deeper models.

- *Rotary Positional Embeddings (RoPE)*: instead of fixed sinusoidal or learned absolute positional encodings, we apply Rotary Positional Embeddings @suRoFormerEnhancedTransformer2023 to the query and key projections within each attention head. RoPE encodes relative position through rotation of the embedding vectors, enabling the model to generalize to different sequence positions and capture relative temporal relationships between behavioral events.

- *Multi-head self-attention*: queries, keys, and values are computed via a single fused linear projection, split across $H$ attention heads with head dimension $d_h = d_"model" slash H$. When available, the implementation uses Flash Attention via PyTorch's `scaled_dot_product_attention` for memory-efficient computation.

- *Feed-forward network*: a two-layer MLP with hidden dimension $2 times d_"model"$, ReLU activation, and dropout.

The input features are first linearly projected from $F$ (or $P times F$ for the patched model) to $d_"model"$. After passing through $L$ encoder layers, the output is normalized via a final RMSNorm layer, producing a latent representation of shape $(B, T, d_"model")$ for the non-patched model or $(B, N, d_"model")$ for the patched model, where each vector captures the local behavioral state enriched by attention over the full temporal context of the window.

=== Self-Supervised Training Objectives <training_objectives>

The models are trained in a fully self-supervised manner, without behavioral labels, using a combination of three complementary objectives designed to encourage the latent space to capture both local feature structure and temporal dynamics:

+ *Masked reconstruction loss.* At each training step, a random subset of timesteps (or patches, in the patched model) is selected for masking. The masked positions are replaced with a learnable mask token $bold(m) in bb(R)^(d_"model")$, and the model is trained to reconstruct the original input at those positions from the surrounding unmasked context. The loss is the mean squared error (MSE) between the predicted and true feature vectors at masked positions. A weighted pass-through regularization term is added on unmasked positions to encourage faithful encoding of observed frames:
  $ cal(L)_"recon" = "MSE"(hat(x)_"masked", x_"masked") + lambda_u dot "MSE"(hat(x)_"unmasked", x_"unmasked") $
  where $lambda_u$ controls the unmasked reconstruction weight.

+ *Value masking loss* (optional). At non-masked timesteps, a fraction of individual feature dimensions is randomly zeroed out, and the model must recover them. This encourages the model to learn cross-feature dependencies and produces more robust representations:
  $ cal(L)_"value" = "MSE"(hat(x)_(b,t,f), x_(b,t,f)) quad "for value-masked" (b,t,f) $

+ *Multi-step-ahead predictive loss.* A separate linear prediction head maps each latent vector to the next $K$ future timesteps (or patches). For the non-patched model, each timestep $t$ predicts $x_(t+1), x_(t+2), dots, x_(t+K)$; for the patched model, each patch $n$ predicts the content of patches $n+1, dots, n+K$. This causal prediction objective encourages the latent representations to encode forward-looking temporal dynamics:
  $ cal(L)_"pred" = 1/(T-K) sum_(t=1)^(T-K) sum_(k=1)^K "MSE"(hat(x)_(t+k), x_(t+k)) $

The total training loss is a weighted combination:
$ cal(L)_"total" = cal(L)_"recon" + cal(L)_"value" + lambda_p dot cal(L)_"pred" $
where $lambda_p$ is the predictive loss weight.

This multi-objective formulation was designed so that the masked reconstruction forces the model to build rich contextual representations of each timestep, while the predictive objective shapes the latent space to be informative about future behavioral trajectories — a property particularly useful for identifying behavioral states that are defined by their temporal evolution.

=== Hyperparameter Optimization

Model hyperparameters were optimized via Bayesian hyperparameter sweeps using Weights & Biases @biewaldExperimentTrackingWeights2020, minimizing the validation loss. The data was split temporally into training (70%), validation (15%), and test (15%) sets, preserving the temporal order of windows to prevent data leakage. The model was trained using the Adam optimizer with weight decay $10^(-5)$ for up to 25 epochs with early stopping (patience = 5 epochs).

The following hyperparameters were searched:

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    table.header(
      [*Hyperparameter*], [*Non-patched*], [*Patched*],
    ),
    [$d_"model"$], [{32, 64, 128}], [{64, 128}],
    [Attention heads $H$], [{2, 4}], [{2, 4}],
    [Encoder layers $L$], [{2, 4, 6}], [{2, 4, 6}],
    [Learning rate], [log-uniform $[10^(-4), 10^(-2)]$], [log-uniform $[10^(-4), 10^(-2)]$],
    [Mask ratio], [{0.10, 0.15, 0.20}], [{0.25, 0.35, 0.40}],
    [Unmasked weight $lambda_u$], [{0.1, 0.3, 0.5}], [0.4],
    [Batch size], [{16, 32, 64}], [{32, 64, 128}],
    [Predictive weight $lambda_p$], [{0.3, 0.5, 0.7}], [{0.3, 0.5, 0.7}],
    [Patch length $P$], [---], [{4, 16}],
    [Prediction steps $K$], [3], [{5, 10, 15}],
  ),
  caption: [Hyperparameter search space for the non-patched and patched transformer models. Fixed parameters: Adam optimizer, weight decay $= 10^(-5)$, window size $w = 128$, early stopping patience $= 5$ epochs, maximum 25 epochs.]
)

=== Latent Representation Extraction

After training, the model is used in inference mode to extract latent representations for all windows in the dataset. At inference time, no masking is applied (the mask is set to all zeros), so the model processes the full, unperturbed input and produces a deterministic latent vector for each timestep (or patch). For the non-patched model, this yields a latent time series of shape $(w, d_"model")$ per window; for the patched model, the output is $(N, d_"model")$ at patch resolution. These latent representations encode the behavioral state at each moment, contextualized by the surrounding temporal dynamics within the window, and serve as the input for downstream analyses including dimensionality reduction, clustering, and discrete state identification.

#figure(
  image("figs/model_architecture.pdf", width: 100%),
  caption: [
    *Model Architecture.*
    *(A)* Overview of the encoder-only Transformer with RoPE attention and masked input. The input time series is linearly projected (or patch-embedded) to $d_"model"$, masked positions are replaced with a learnable token, and the encoder produces latent representations used for reconstruction and prediction.
    *(B)* Patching: consecutive frames are grouped into non-overlapping patches of length $P$, reducing the token sequence from $w$ to $N = w slash P$.
    *(C)* Training objectives: masked reconstruction, value masking, and multi-step prediction are combined into a single loss.
  ],
)

