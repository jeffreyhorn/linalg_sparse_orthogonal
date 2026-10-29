# Sprint 221 Day 8: Synthetic Observed And Measured Fixtures

## Evidence Boundary

These fixtures exercise validator behavior only. `synthetic_observed()` and
`synthetic_measured()` in `tests/test_validate_external_baseline_evidence.py`
are the canonical builders. Every generated run ID, host label and command
is visibly synthetic; no CHOLMOD build, solve or benchmark was performed.
Passing the validator in a temporary test bundle does not establish a real
external-baseline result. The CLI and optional pilot gate remain fail-closed
with `implementation_pending`.

## Two-Sided Correctness Fixture

The `correctness_observed` builder supplies the selected matrix and full
project/baseline identity, including baseline ABI, loaded-library and
dependency metadata, thread counts and canonical cell ID. Both sides carry
synthetic successful phase statuses, factor controls, four-value solutions
and recomputed Day 4 residual, backward-error and oracle gates. The paired
solution difference and limit are also recomputed. Test staging hashes the
frozen methodology files and writes each side's statuses, diagnostics and
solution as separate JSON bytes in a temporary bundle. The validator checks
those hashes, roles and raw values against the reported side before accepting
the status.

## Measured Fixture

The `measured` builder retains the two-sided correctness result and adds the
planned monotonic clock identity and one requested/observed BLAS and OpenMP
thread. Its temporary raw timing JSON contains `factor_ready`, `one_solve`
and `factor_plus_first_solve`. Each reported window has five warmup and 50
timed pairs, alternating AB/BA order, successful pair statuses and positive
integer nanosecond durations. The fixture computes minimum, maximum, median,
quartiles, IQR and MAD directly from the raw timed pairs. Its primary median
comes from `factor_plus_first_solve`. The validator requires the measured
clock to match the frozen method, verifies the thread controls, hashes the
raw JSON before parsing it, and recomputes every reported summary.

## Paired Rejections

Tests reject a one-sided pass, a missing solver raw file, a solver raw
solution changed even after its digest is updated, an incorrect timing
median, an incorrect window pair count, a missing clock identity, a changed
thread count and a synthetic winner token. They also verify the canonical
cell ID, methodology hashes, and each bundle artifact digest against bytes.
These are adversarial fixture checks, not performance comparisons or release
claims.
