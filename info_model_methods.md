Quick clarification questions (answer briefly)
1. Behavioral representation (very important)
What exactly is your input time series?
2D keypoints?
3D keypoints?
Pose-derived features (angles, velocities, distances)?
Dimensionality per frame (roughly)? e.g. F = 36, F = 78, etc.
2. Temporal resolution
Frame rate of the videos (e.g. 30 Hz, 60 Hz)?
Typical sequence length fed to the model (e.g. 10k frames, 2k frames, sliding windows)?
3. Models you actually trained
For the Transformer-based models:
Encoder-only Transformer? (sounds like yes)
Masked modeling? Next-step prediction? Both? (Fig C suggests multiple losses)
Did you include:
positional encodings (sinusoidal / RoPE)?
causal or bidirectional attention?
For the classical baseline, which ones do you actually use?
HMM?
AR-HMM?
MoSeq-style HMM on PCA features?
(We should only name what you truly ran.)
4. Window vs patch details
Window size w (in frames)?
Patch size P and stride S?
Were patches overlapping?
5. Goal of the model
This affects wording a lot:
Is the goal representation learning (latent behavioral structure)?
Or prediction?
Or segmentation into syllables?
(You can say “unsupervised representation learning” if appropriate.)
