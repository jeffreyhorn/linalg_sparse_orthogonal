# Sprint 70 Day 7: Public-Surface and Proof-Surface Audit I

Date: 2026-06-15
Branch: `sprint-70`

## Purpose

Reduce the broad Epic 7 public-surface and proof-surface cleanup question to
one ranked contradiction map so later cleanup work starts from the strongest
remaining drift instead of from generic documentation or test churn.

## Surfaces Audited

Public/product and support surfaces:

- `README.md`
- `docs/tutorial.md`
- `INSTALL.md`
- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`
- `include/sparse_cholesky.h`

Proof-owner and assurance surfaces:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`
- `tests/test_reorder_nd.c`
- `tests/test_fuzz.c`

## Ranked Contradictions

### 1. Strongest public front-door contradiction: `README.md`

`README.md` remains the strongest public cleanup target because it still mixes:

- compact product front door
- workflow chooser
- benchmark/reporting summary
- platform/install quality summary
- deep algorithm/performance history
- giant test inventory and maintainer-routing notes

The contradiction is no longer about missing ownership. It is about residual
density and chronology overexposure in the repo's highest-signal public entry
surface.

### 2. Strongest header/reference contradiction: `include/sparse_cholesky.h`

`include/sparse_cholesky.h` is the strongest current header-side cleanup
candidate because it combines:

- valid API-local caveats
- retained sprint-history and ABI-history explanation
- benchmark references
- dispatch/telemetry interpretation that partly overlaps public docs and
  maintainer policy

It therefore reads as the most mixed-ownership high-signal header among the
surfaces audited for Day 7.

### 3. Strongest support-surface contradiction: `INSTALL.md`

`INSTALL.md` is now the strongest support-surface cleanup target because:

- the install/package/platform contract is largely truthful and coherent
- but the platform notes still preserve older sprint-history explanation more
  directly than a mature operator-facing install guide should

This makes it a stronger current support-surface candidate than:

- `examples/README.md`
- `benchmarks/README.md`
- `docs/maintainer_guide.md`

which are now denser by design or already more bounded.

### 4. Strongest proof-surface contradiction: `tests/test_reorder_nd.c`

`tests/test_reorder_nd.c` is the strongest remaining proof-surface cleanup
candidate because it still carries the densest concentration of:

- sprint-day chronology
- tuning-history commentary
- preserved planning archaeology
- live proof mixed with large volumes of historical rationale

This is the clearest remaining permanent proof owner that still reads partly
like an archive.

### 5. Strongest second proof-surface contradiction: `tests/test_chol_csc.c`

`tests/test_chol_csc.c` remains a major proof-surface burden because it mixes:

- family-local CSC correctness
- supernodal helper proof
- dispatch proof
- large fixture proof
- substantial retained sprint chronology

It ranks behind `tests/test_reorder_nd.c` because its current ownership split
is somewhat clearer after Epic 6 and 7 cleanup, even though the file remains
large and history-heavy.

## Lower-Priority or Support-Only Surfaces

The following do not rank as the strongest current contradiction centers:

- `docs/tutorial.md`
  - dense, but primarily in service of the teaching flow rather than drift
- `examples/README.md`
  - bounded and relatively clean after earlier adoption/proof alignment work
- `benchmarks/README.md`
  - dense, but much of that density is benchmark-local truth rather than
    accidental contradiction
- `docs/maintainer_guide.md`
  - intentionally policy-heavy; it is the policy home, not the main cleanup
    contradiction center
- `tests/test_integration.c`
  - important, but more coherent than the two largest remaining giant-test
    hotspots
- `tests/test_fuzz.c`
  - bounded and comparatively readable

## Exit State

Sprint 70 Day 7 closes with one ranked contradiction map:

1. strongest public front-door contradiction:
   - `README.md`
2. strongest header/reference contradiction:
   - `include/sparse_cholesky.h`
3. strongest support-surface contradiction:
   - `INSTALL.md`
4. strongest proof-surface contradiction:
   - `tests/test_reorder_nd.c`
5. strongest second proof-surface contradiction:
   - `tests/test_chol_csc.c`

That gives Day 8 one exact job:

- separate the highest-value future cleanup lanes from lower-value surface
  churn and freeze the support-surface fence before Sprint 70 moves on to the
  validation/platform contract audit
