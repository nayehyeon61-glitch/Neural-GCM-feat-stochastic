# Implementation validation

Validated in the implementation session of 2026-10-12. These results establish
code behavior, not weather/climate skill.

## Environment

Linux CPU; Python 3.12.14, NeuralGCM 1.2.2, Dinosaur 1.5.0, JAX/JAXlib 0.11.2,
Optax 0.2.8, NumPy 2.5.4, SciPy 1.18.1, Xarray 2026.9.0, Pytest 9.1.1.
Offline source equivalence additionally used Torch 2.14.1+cpu.

## Automated tests

After recurrent latent integration, `python -m pytest -q`: **104 passed** in
30.79 seconds, with two occurrences of an upstream Optax deprecation warning.
The previous initialization-only audit had 76 passing tests. New coverage
includes latent-chain moments, causal sequence supervision and compatibility.

| Area | Checks |
|---|---|
| Auxiliary integration | Exact memory partition invariance, OU joint endpoint/integral covariance, small-step variance scaling, member independence, JIT and gradients |
| NeuralGCM boundary | Internal-hour and Kelvin conversions, exact disabled passthrough, Gaussian mean option, state preservation and corrected recurrence |
| Source encoder | Four parity cases against actual pinned source classes, optional information/fusion, odd pooling, cyclic/regional padding, normalization and schema refusal |
| Source history | Real prepare/load contract, past-only selection, future/origin perturbation invariance, periodic regrid, nonzero history influence, stale-tail/internal-gap rejection, explicit irregular cadence, single-sample contract and shared origin filtering |
| Data preparation | Actual Dinosaur conservative regrid, correct pressure/units, missing fields/levels/forcings and timestamp rejection |
| Training/scoring | Causal rollout, corrected multi-step derivatives, optimizer update, safe checkpoint roundtrip, validation/test exclusion, empirical/fair CRPS and drift metrics |
| Recurrent latent | Conditional Gaussian moments, small-step variance, accumulated two-step innovations/covariance, exact past-memory sum, source-latent influence on later temperature, independent RNG and disabled passthrough |
| Latent training | Future label changes affect loss but not predictions, exact origin/lead matching, joint objective and model-grid quadrature, warmup path, old/new checkpoint schemas, evaluation without future source latents |

## Actual NeuralGCM integration

`python scripts/smoke_neuralgcm.py` passed with the official bundled
`tl63_stochastic_mini.pkl` toy fixture:

- 32 sigma levels; 128 x 64 internal nodal grid; 1-hour timestep.
- Actual encode, base advance, closure advance and corrected decode completed.
- Disabled closure matched the complete base state bitwise, including RNG.
- Decoded fields were finite.
- Nonzero finite corrected-temperature gradient norm: approximately
  `5.2381e-4` for the script's artificial target.

The real data-preparation CLI was also run using the same checkpoint and the
bundled field sample. The saved NetCDF passed the public input/forcing APIs
with 37 pressure levels and a 128 x 64 data grid.

## Training and evaluation CLI integration

The bundled sample was repeated across synthetic hourly timestamps solely to
exercise I/O, causality, compilation and optimizer flow. An explicitly
synthetic two-channel history artifact exercised the source-memory interface;
it was not a trained source encoder output.

The current training CLI completed one optimizer update and one validation
window with two ensemble members and a one-step forecast. Fair-CRPS/mean-MSE
loss and gradient were finite; gradient norm was approximately `4.4282e-4`.
The safe NPZ checkpoint was then restored by the evaluation CLI on later
synthetic timestamps. Both corrected and `--baseline` runs completed with
matched dates/seeds and finite JSON metrics.

These synthetic inputs/checkpoints/results are ignored under `runs/` and are
not published as trained models or forecast benchmarks.

## Recurrent latent integration (v0.2)

`python scripts/smoke_neuralgcm.py --recurrent-latent` completed two actual
internal NeuralGCM steps (32 sigma levels, 128 x 64 nodes, 1 hour per step),
using an explicitly synthetic two-channel initial latent:

- Disabled adapter remained bitwise identical to the unmodified base forecast.
- Corrected decoded fields were finite.
- Total physical-field-loss gradient norm: approximately `1.10451e-3`.
- The latent-transition subtree's gradient norm **from the later physical
  temperature loss alone** was approximately `2.37628e-6`. No latent-label loss
  was used in that smoke check, so the test establishes the actual path from
  generated latent to subsequent physical prediction.

The actual training CLI also completed one optimizer update and validation
with two members and two hourly leads on the existing synthetic repeated-field
dataset. Joint physical/latent objective and gradients were finite; gradient
norm was approximately `3.95489e-3`. Validation reported its physical and latent
components separately and saved a recurrent NPZ checkpoint. This is a CLI
connectivity check, not learned Climate Manifold weights or a skill estimate.

The evaluation CLI restored that checkpoint on later synthetic test times.
Its source-history artifact was truncated **exactly at the tested origin**;
the subsequent two forecast leads still completed with finite metrics.
This checks that future source labels are not an inference requirement.

`scripts/render_architecture.py` generated the architecture PNG and SVG.
The rendered figure was inspected and its initialization, recurrent-state and
training-label routes were checked against the implemented code.

## Not validated

Production checkpoint training, GPU performance, real ERA5 forecast skill,
multimonth/multiyear stability, conserved moist-energy/water budgets, climate
response under changed SST, or a trained source checkpoint's generalization.
Those require the actual pretrained checkpoints and a chronological
training/validation/test experiment.

## Memory audit correction

The previous context selector accepted arbitrarily stale source history: two
daily samples from 2001-01-01/02 were allowed to initialize a 2002-01-01 forecast,
holding the last latent for 8,736 hours. This is now rejected before regridding
or memory integration. The selected internal intervals and final interval to
origin must not exceed `max_history_gap_hours`. Preparation can declare the
bound; legacy artifacts infer a strict median interval. A single sample needs
an explicit bound. Inference cannot identify a uniformly undersampled archive,
so production experiments should declare the intended limit.

Training/evaluation reuse that temporal check when selecting forecast windows,
report exclusions, and reject an empty eligible partition. Resolved training
limits are stored and must match at evaluation. This fixes an input-validity
error; it does not establish long-run skill. The audit also documents remaining
initialization-only conditioning, short default memory/horizon, source training
date provenance, physical budget and data-scaling limitations in
`MEMORY_AUDIT_KO.md`.
