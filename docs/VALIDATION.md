# Implementation validation

Validated in the implementation session of 2026-10-12. These results establish
code behavior, not weather/climate skill.

## Environment

Linux CPU; Python 3.12.14, NeuralGCM 1.2.2, Dinosaur 1.5.0, JAX/JAXlib 0.11.2,
Optax 0.2.8, NumPy 2.5.4, SciPy 1.18.1, Xarray 2026.9.0, Pytest 9.1.1.
Offline source equivalence additionally used Torch 2.14.1+cpu.

## Automated tests

After the memory audit and stale-history fix, `python -m pytest -q`:
**76 passed** in 25.88 seconds, with one upstream Optax deprecation warning.
The targeted history/source suite also passed: **23 passed**.

| Area | Checks |
|---|---|
| Auxiliary integration | Exact memory partition invariance, OU joint endpoint/integral covariance, small-step variance scaling, member independence, JIT and gradients |
| NeuralGCM boundary | Internal-hour and Kelvin conversions, exact disabled passthrough, Gaussian mean option, state preservation and corrected recurrence |
| Source encoder | Four parity cases against actual pinned source classes, optional information/fusion, odd pooling, cyclic/regional padding, normalization and schema refusal |
| Source history | Real prepare/load contract, past-only selection, future/origin perturbation invariance, periodic regrid, nonzero history influence, stale-tail/internal-gap rejection, explicit irregular cadence, single-sample contract and shared origin filtering |
| Data preparation | Actual Dinosaur conservative regrid, correct pressure/units, missing fields/levels/forcings and timestamp rejection |
| Training/scoring | Causal rollout, corrected multi-step derivatives, optimizer update, safe checkpoint roundtrip, validation/test exclusion, empirical/fair CRPS and drift metrics |

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
