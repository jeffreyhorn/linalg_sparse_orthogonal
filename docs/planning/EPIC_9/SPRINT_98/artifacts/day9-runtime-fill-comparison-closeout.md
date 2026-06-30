# Day 9 Runtime/Fill Comparison Closeout

## Purpose

Complete the Sprint 98 runtime/fill comparison lane by reconciling the Day 8
artifact with benchmark-governance language, rerunning the focused validation
command, and recording the final claim boundary.

## Completed Lane

The completed Sprint 98 runtime/fill lane is:

```sh
make bench-reorder-sprint86
```

This expands to:

```sh
build/bench_reorder --sprint86-slice --skip-factor
```

The lane covers:

- fixtures:
  - `bcsstk14`
  - `Pres_Poisson`
- reorder rows:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`
- primary fill field:
  - `nnz_L`
- local timing context:
  - `reorder_ms`
- fixed contextual labels:
  - `factor_ms=skip`
  - `reorder_path=direct`
  - `fixture_slice=sprint86`
  - `nd_base_threshold=160`

## Artifact Ownership

The Day 8 artifact owns the Sprint 98 observed evidence:

- `artifacts/day8-runtime-fill-comparison-batch1.md`

The benchmark binary owns the emitted CSV schema and field semantics:

- `benchmarks/bench_reorder.c`

Benchmark-local command documentation remains in:

- `benchmarks/README.md`

The maintainer-guide guardrail now names the Sprint 98 artifact as a bounded
two-fixture calibration slice and explicitly keeps it out of the canonical
maintained performance surface.

## Guardrail Added

`docs/maintainer_guide.md` now states that the Sprint 98 reorder/fill artifact:

- uses `make bench-reorder-sprint86`
- is a bounded two-fixture calibration slice
- treats `nnz_L` as the primary fill field
- treats `reorder_ms` as local timing context only
- does not replace canonical maintained performance reporting
- does not create portable timing claims

No benchmark code, benchmark schema, Makefile target, workflow, or canonical
reporting surface changed on Day 9.

## Validation

Focused runtime/fill validation passed:

```sh
make bench-reorder-sprint86
```

Observed output preserved the expected structure:

```text
=== Running bench_reorder --sprint86-slice --skip-factor ===
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,6.6,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,60.1,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,92.1,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,286.3,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,80.5,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,3928.3,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,8157.1,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,3705.7,skip,direct,sprint86,160
```

The timing values are from this local run and remain non-threshold evidence.

## Claim Boundary

Allowed:

- Sprint 98 records bounded reorder/fill calibration evidence for the selected
  two-fixture slice.
- `nnz_L` can be used to compare fill behavior within the selected artifact.
- `reorder_ms` can help maintainers understand local run context.

Not allowed:

- universal reorder superiority claims
- portable timing claims
- cross-platform performance claims
- claims that the artifact replaces `make bench-canonical-report`
- claims that broad `bench_reorder`, `bench_fillin`, or `bench_amd_qg` output
  is included in this Sprint 98 lane

## Residual Queue

Deferred runtime/fill work remains:

- generated report automation if repeated Sprint 98-style artifacts become
  common
- optional benchmark-doc refinements if future artifacts widen beyond this
  slice
- broader runtime/fill corpus selection
- canonical report expansion, only after a separate boundary proves it remains
  cheap and stable
- workflow artifact capture, only after deciding whether the lane should become
  reviewed, supplemental, or local-only

## Closeout

The Sprint 98 runtime/fill lane is complete. It leaves a bounded, reproducible
artifact and a maintainer-facing guardrail without changing benchmark code,
workflow behavior, canonical reporting, or timing thresholds.
