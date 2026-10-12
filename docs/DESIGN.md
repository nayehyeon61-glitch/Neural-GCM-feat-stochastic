# Design and scientific scope

## 1. The hypothesis

Use spatial statistics learned with Climate Manifold to provide a causal
initial condition for unresolved-process memory. Evolve that memory jointly
with an existing NeuralGCM forecast, rather than asking a latent forecast
model to replace the entire atmospheric solver.

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
OU state `Z` and its own PRNG key. Time below is in hours; `Z` is in Kelvin.

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

This is a first-order operator split around the pretrained model. There is no
claim of exact integration of the complete nonlinear coupled system. The
temperature sees colored forcing via the auxiliary SDE, not an independently
sampled `sqrt(dt)` temperature kick. In particular, an OU state initialized at
zero yields O(dt^3) variance for its short-step integrated forcing.

Independent nodal OU drivers followed by spectral projection are a
resolution-specific covariance baseline. They are not an isotropic learned
spherical covariance model and should not be compared across resolution
without recalibration.

## 3. What “spatial manifold memory” means here

The source encoder's actual architecture and trained statistics are exported
to an array-only JAX artifact. The schema includes physical field names/units,
source and latent grids, checkpoint hash, source revision and normalization.
The exact port covers cyclic longitude/replicated latitude padding, SiLU,
ceil-mode average pooling, optional information encoder, variable-specific
fusion, and latent normalization.

The source spatial loss choice (W2/KL/signed, where recorded) remains the
choice used to train that source checkpoint. These losses are not retrained
or silently substituted in this frozen-encoder forecast integration.

The forecast initialization uses two channel groups:

- Historical stage: `[zeros_for_sigma_temperature, E_s(observed_surface)]`.
  Select a fixed number of observations strictly before origin, integrate
  their following intervals up to origin, and retain the deterministic memory.
  Reject internal gaps and a stale last observation beyond the artifact's
  `max_history_gap_hours`. Declare that limit when preparing irregular data;
  older artifacts infer the median interval, which cannot diagnose a uniformly
  undersampled archive. Training/evaluation use the same temporal check and
  report excluded windows. New checkpoints record the resolved limit and
  evaluation requires that same limit.
- Forecast stage: `[current_native_temperature_variation / scale, zeros_for_E_s]`.
  The physical forecast drives subsequent memory. The source-history signal
  remains only through the initialized memory and decays on its timescales.

This is **initial-condition statistical memory**, not continuously refreshed
surface-manifold guidance. With a longest timescale of 120 h its direct
contribution decays strongly after several weeks. Longer timescales can be
tested, but should be compared against current-state and no-source controls.
Ongoing spatial guidance would require source-compatible diagnosed fields or
an independently validated learned diagnostic bridge.

When spatial initialization is enabled, it replaces the deterministic
thermal-only warmup memory. The base forecast state and auxiliary OU state
remain unchanged. This ensures the distinction is explicit and testable.

Latent-grid interpolation is periodic in longitude and linear in latitude,
with polar edge extension. It interpolates dimensionless features; it is not
a conservative regridding method for mass or energy.

## 4. Training and causality

The pretrained base parameters are frozen. JAX differentiates closure
parameters through the actual corrected forecast recurrence and the pretrained
decoder. The objective is normalized fair CRPS (the unbiased iid-ensemble
estimator) plus an optional variance-corrected estimator of predictive-mean
squared error. A finite-sample unbiased loss can be negative. Evaluation
reports empirical and fair CRPS separately. Checkpoints contain closure arrays, config
and experiment metadata in NPZ, not pickled executable objects.

The temporal split precedes window construction. Native observation histories
and targets from a validation window remain within the validation partition.
Optional source history may predate that partition, but must strictly predate
its forecast origin. Independent evaluation starts after the observations used
for both training and validation-based checkpoint selection. Future verification
states are used for loss only, not encoded into forecast state or source
memory. Source encoder/normalization training dates must separately be audited;
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
  its power-law response and estimated order are not implemented by the default
  three-timescale closure. See [the Korean memory audit](MEMORY_AUDIT_KO.md).
