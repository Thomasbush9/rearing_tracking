
**Goals**: Update the script to run multiple test from the hmm, save the data (results) in subdirectories to be able to test compare things. Run multiple HMM in parallele if possible with differen params. 


Bellow there are some info: 

Big picture principle
You’re not “fitting an HMM”.
You’re testing whether a Markovian latent-state model is a stable, generalizable, and interpretable description of dynamics in a learned representation.
Everything below serves one of those three words:
stable – generalizable – interpretable.
Phase 0 — Representation sanity (before HMMs)
Question: Is the latent space even worth modeling with discrete states?
Runs
Take a small uniform subsample (e.g. 50k points).
Inspect:
variance spectrum (PCA)
autocorrelation timescales
smoothness over time
Decision gate
If latents are white-noise-like → HMM won’t help.
If there’s temporal structure → proceed.
What you report
“Latent representations exhibit strong temporal autocorrelation and low-dimensional structure, motivating a state-based temporal model.”
Phase 1 — Feasibility & scale discovery
Question: How much data is “enough” for this representation?
Runs
Fix a simple HMM (diag Gaussian, modest K like 10–20).
Fit on increasing data budgets:
50k → 150k → 300k → 500k
Use identical subsampling strategy and multiple seeds.
What you look for
Held-out log-likelihood plateauing
Transition matrices stabilizing
Dwell-time distributions converging
Decision gate
Pick the smallest budget where gains flatten.
This becomes your standard training budget.
What you report
“Model performance and inferred dynamics stabilized beyond ~X×10⁵ samples; all subsequent analyses use this budget.”
This justifies subsampling rigorously.
Phase 2 — Model capacity selection (K states)
Question: How many states are justified by the data?
Runs
Fix:
subsample budget (from Phase 1)
emission type
Sweep K (e.g. 5 → 60).
Multiple seeds per K.
Evaluation axes (all equally important)
Predictive fit (held-out LL)
Stability (seed-to-seed consistency)
Dynamics realism
dwell times
switching rates
State utilization
occupancy entropy
unused / duplicated states
Decision gate
Reject K where:
states split into near-duplicates
switching becomes implausibly fast
stability collapses
Choose a range, not a single magic K.
What you report
“We observe an optimal regime around K≈…, balancing predictive performance, stability, and realistic temporal structure.”
This avoids “BIC says 47, trust me bro”.
Phase 3 — Robustness to data selection
Question: Are the states an artifact of how we sampled?
Runs
Repeat Phase 2’s best K range with:
different subsample seeds
different chunk placements
different stride values
What you look for
Similar states reappear (up to permutation)
Similar transition structure
Similar dwell distributions
Decision gate
If states drift wildly → overfitting or representation too noisy.
If they’re robust → you’ve got something real.
What you report
“Inferred states and transitions were robust to subsampling strategy and random initialization.”
This is a very strong validation claim.
Phase 4 — Null models & controls
Question: Does the HMM actually model dynamics, or just clusters?
Runs
Time-shuffled control
Same data, order randomized
IID mixture baseline
Same emissions, no transitions
Expected outcomes
Shuffling destroys structure (flat transitions, no dwell)
HMM outperforms mixture on held-out LL
Decision gate
If HMM doesn’t beat these → rethink assumptions.
What you report
“Temporal structure is essential: shuffled and IID baselines fail to capture the observed dynamics.”
This is often missing in theses — adding it makes yours stronger.
Phase 5 — Posterior predictive realism
Question: Does the model generate sequences that look like the real ones (statistically)?
Runs
Sample synthetic sequences from the fitted HMM.
Compare to real data on:
dwell times
state occupancy
transition frequencies
low-order temporal statistics
Decision gate
Model must reproduce summary statistics, not raw frames.
What you report
“The fitted HMM reproduces key temporal statistics of the latent dynamics.”
This bridges fit quality and interpretability.
Phase 6 — Experimental meaning (if applicable)
Question: Do states relate to experimental conditions?
Runs
Keep HMM fixed.
Compare:
occupancy
transition rates
across conditions (solo vs interaction, etc.).
Control
Condition-label shuffle.
What you report
“State usage and transitions differ systematically across conditions, beyond chance.”
Now the HMM isn’t just a compression tool — it’s explanatory.
