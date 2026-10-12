# Design and scientific scope

## 1. The hypothesis

Use Climate Manifold spatial statistics to initialize a latent state and its
history, then evolve a conditional stochastic latent trajectory jointly with
an existing NeuralGCM forecast. Its current latent continually conditions a
temperature closure; the latent model does not replace the atmospheric solver.

Hasselmann (1976) motivates the response of slow variables to fast fluctuations
and the need for stabilizing feedback. It does **not** derive the particular
learned memory kernel or guarantee that adding noise improves forecasts.
The exponential-memory augmentation here is a modeling hypothesis related to
reduced stochastic closures; no exact Mori–Zwanzig kernel or
fluctuation–dissipation relation is claimed.

The original NeuralGCM paper reports multidecadal climate metrics under
prescribed SST, while identifying difficulties extrapolating to substantially
different climates. Thus “long-term failure” is too broad: evaluate the
specific target, such as drift, variability, calibration, or changed-SST
generalization.

## 2. State and equations

The physical state `X` remains NeuralGCM's nondimensional, spectral,
sigma-coordinate state. The separate closure state contains memory `m_r`, an
OU state `Z` and its own PRNG key. Recurrent mode additionally carries source
latent `z`, separate spatial memory `M_s` and a separate PRNG key. Time below is
in hours; `Z` is in Kelvin and source latents are dimensionless.

\[
u = \tanh(W f+b),\qquad
dm_r = (u-m_r)dt/\tau_r,
\]
\[
dZ=-Zdt/\tau_\eta+\sigma(f)dW_t,
\]
\[
\Delta T=\int_t^{t+h}\left[A_su+\sum_r A_rm_r+Z/\tau_\eta\right]ds.
\]

The deterministic readouts are bounded. Diffusion amplitude is nonnegative
and bounded, with units K/sqrt(hour). Timescales are positive fixed
hyperparameters in this first version. The auxiliary memory and OU subsystem
is stable for frozen inputs; this does not establish stability of the coupled
atmospheric model.

For each accepted internal step:

1. Extract normalized, current-state features.
2. Call the unchanged `base.advance` once.
3. Integrate memory and the OU endpoint/integral exactly conditional on those
   frozen features. Sample the correlated endpoint and integral together.
4. Convert `Delta T` from K into internal units, transform **only the increment**
   to spectral coefficients, and add to `temperature_variation`.
5. In recurrent mode, update the source latent and spatial memory from the
   current latent, current temperature features and previous spatial memory;
   carry them with the corrected physical state into the next step. The new
   latent first enters the closure at the next step.

This is a first-order operator split around the pretrained model. There is no
claim of exact integration of the complete nonlinear coupled system. The
temperature sees colored forcing via the auxiliary SDE, not an independently
sampled `sqrt(dt)` temperature kick. In particular, an OU state initialized at
zero yields O(dt^3) variance for its short-step integrated forcing.

Independent nodal OU drivers followed by spectral projection are a
resolution-specific covariance baseline. They are not an isotropic learned
spherical covariance model and should not be compared across resolution
without recalibration.

## 3. Recurrent spatial latent and memory

The source encoder's actual architecture and trained statistics are exported
to an array-only JAX artifact. The schema includes physical field names/units,
source and latent grids, checkpoint hash, source revision and normalization.
The exact port covers cyclic longitude/replicated latitude padding, SiLU,
ceil-mode average pooling, optional information encoder, variable-specific
fusion, and latent normalization.

The source spatial loss choice (W2/KL/signed, where recorded) remains the
choice used to train that source checkpoint. These losses are not retrained
or silently substituted in this frozen-encoder forecast integration.

Recurrent training is the default when `--spatial-history` is supplied. Past
observations strictly before origin seed `M_s`; an exact origin observation
provides `z_0`. Origin is available information, consistent with physical-state
initialization. A last pre-origin observation is not silently renamed `z_0`.

For normalized sigma-temperature features `T`, a pointwise network gives a
bounded relaxation target `g_theta(z,T,M_s)` and a nonnegative bounded diagonal
diffusion `sigma_theta(z,T,M_s)`. Its frozen-coefficient step is

\[
dz=(g_\theta-z)dt/\tau_z+\sigma_\theta dW,
\qquad dM^s_r=(z-M^s_r)dt/\tau^s_r,
\]
\[
z_{t+h}=a_z z_t+(1-a_z)g_\theta+
\sigma_\theta\sqrt{\tau_z(1-a_z^2)/2}\,\epsilon_t,
\quad a_z=e^{-h/\tau_z},\quad\epsilon_t\sim\mathcal N(0,I).
\]

The conditional law is `p(z_next | z, T, M_s)`. The spatial-memory update holds
the current `z` over this step. It is not the integral of a continuously varying
latent Brownian path. The augmented coupled algorithm is a split,
frozen-coefficient approximation, not an exact nonlinear SDE solution.

Current `[T,z]` features drive the existing closure. The sampled next latent is
stored in `AugmentedState.latent_state` and becomes the next transition/closure
input. Independent innovations accumulate through this recurrence and the
weighted spatial history. Neither `z` nor `M_s` is reset from future labels.
Default latent relaxation is 24 h; spatial memories use 24/120/720 h. Closure
memory retains 6/24/120 h. All are configurable fixed positive timescales.

Observed physical warmup uses `[T,0]` and leaves the latent state unchanged.
Spatial initialization seeds only `z,M_s` and preserves the warmed closure
memory/OU state. The two memories are distinct, although forecast-time closure
memory subsequently summarizes both thermal and latent features.

Reject historical internal gaps and stale tails beyond `max_history_gap_hours`.
Declare that limit for production/irregular data; legacy artifacts infer a
median interval, which cannot diagnose a uniformly undersampled archive.
Train/evaluation reuse the temporal checks and report exclusions. Training
also requires exact future label times when latent supervision is enabled.
Evaluation requires source observations only through each forecast origin.

`--spatial-mode initialization` retains the older alternative: history replaces
closure memory, online source channels are zero and there is no latent
transition. Old checkpoints default `latent_dim=0` and retain their original
parameter schema. Omitting history selects native thermal memory alone.

Latent-grid interpolation is periodic in longitude and linear in latitude,
with polar edge extension. It interpolates dimensionless features; it is not
a conservative regridding method for mass or energy.

## 4. Training and causality

The pretrained base and source encoder are frozen. JAX differentiates closure
and latent-transition parameters through the corrected forecast recurrence and
pretrained decoder. Field and latent objectives use fair CRPS (the unbiased iid-ensemble
estimator) plus an optional variance-corrected estimator of predictive-mean
squared error. A finite-sample unbiased loss can be negative. Evaluation
reports empirical and fair CRPS separately. In recurrent mode the total loss is
`L_fields + lambda_z * L_latent` (default `lambda_z=0.1`). Freely rolled-out
latents are scored against frozen source-encoder labels at actual observation
times with model-grid area weights. Field loss uses the decoded data grid.
Daily labels are not reused as hourly conditional-likelihood targets. Future
latent labels are stripped from initialization context and never teacher-force
the forecast. Explicit `--latent-loss-weight 0` removes this supervision for an
ablation; validation uses the same total objective as training.
Checkpoints contain closure/transition arrays, config
and experiment metadata in NPZ, not pickled executable objects.

The temporal split precedes window construction. Native observation histories
and targets from a validation window remain within the validation partition.
Optional source history may predate that partition, but must strictly predate
its forecast origin, with the exact origin latent additionally allowed in
recurrent mode. This source conditioning need not lie inside the native
warmup partition. Independent evaluation of native windows starts after the observations used
for both training and validation-based checkpoint selection. Future verification
states are used for loss only, not encoded into forecast state or source
memory; the source encoder may encode future data only to produce loss labels.
Source encoder/normalization training dates must separately be audited;
source split indices alone are insufficient evidence of a calendar cutoff.

SST and sea ice persist from origin by default. Observed future forcing is an
explicit, differently labeled hindcast option. Forcing timestamps still
advance at each internal step.

## 5. Known limits

- Only temperature is corrected. No new humidity, wind, pressure or precipitation
  tendency is claimed; these can respond indirectly in later dynamics steps.
- Subtracting each sigma-layer's Gaussian area-mean correction is an optional
  drift control, not conservation of atmospheric moist energy.
- Diagnostics stored by the base step are not recomputed after the correction.
- No ocean/land/cryosphere slow-state model has been introduced.
- The conditional latent network uses temperature, latent and spatial memory;
  it does not directly condition on wind, humidity or pressure. Gaussian noise
  is pointwise/diagonal before subsequent processing, not a learned spatial
  covariance or a geometric manifold-preserving diffusion.
- Finite exponential memories are not a fractional power-law kernel. An 8-day
  historical sequence does not sufficiently initialize a 30-day memory; tune
  history length and forecast horizon together with timescales.
- This is supervised stochastic state-space modeling, not reinforcement
  learning. Longer memory and persistent latent recurrence are mechanisms to
  test, not evidence of better climate skill.
- Initial-condition stochasticity from a stochastic NeuralGCM checkpoint and
  new OU stochasticity coexist. Compare deterministic-base and stochastic-base
  experiments separately to detect redundant noise.
- The bundled fixture, synthetic numerical tests and gradient checks establish
  implementation behavior, not weather skill or long-run climate stability.

## References

- Hasselmann, K. (1976). *Stochastic climate models. Part I. Theory.*
  [doi:10.1111/j.2153-3490.1976.tb00696.x](https://doi.org/10.1111/j.2153-3490.1976.tb00696.x)
- Kochkov et al. (2024). *Neural general circulation models for weather and
  climate.* [doi:10.1038/s41586-024-07744-y](https://doi.org/10.1038/s41586-024-07744-y)
- [Official NeuralGCM stable API](https://neuralgcm.readthedocs.io/en/stable/api.html)
- [Official checkpoint descriptions](https://neuralgcm.readthedocs.io/en/stable/checkpoints.html)
- [Climate Manifold pinned source](https://github.com/nayehyeon61-glitch/climate_manifold/tree/bd1592fe9283f650b5b1f62deb7e196743b9db9a)
- Yuan et al. (2022). *The impact of long-term memory on the climate response to
  greenhouse gas emissions.* [doi:10.1038/s41612-022-00298-8](https://doi.org/10.1038/s41612-022-00298-8).
  This global-mean fractional-memory analysis motivates testing kernel shape;
  its power-law response and estimated order are not implemented by the finite
  exponential memories. See [the recurrent implementation](RECURRENT_LATENT_KO.md)
  and [the previous initialization-only audit](MEMORY_AUDIT_KO.md).
