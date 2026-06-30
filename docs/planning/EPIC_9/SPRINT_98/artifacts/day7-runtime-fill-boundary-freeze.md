# Day 7 Runtime/Fill Boundary Freeze

## Purpose

Freeze the Sprint 98 runtime/fill comparison workload before implementation.
The goal is to capture one bounded, repeatable comparison lane for fill and
runtime calibration without turning benchmark output into a broad performance
claim.

## Inputs Re-Read

- `artifacts/day2-comparison-surface-rerank.md`
- `artifacts/day3-proof-comparison-architecture-design.md`
- `benchmarks/README.md`
- `benchmarks/bench_reorder.c`
- `benchmarks/bench_amd_qg.c`
- `Makefile` benchmark targets
- `docs/maintainer_guide.md` benchmark-governance ownership section

## Selected Workload

The Sprint 98 runtime/fill implementation lane is:

```sh
make bench-reorder-sprint86
```

This expands to:

```sh
build/bench_reorder --sprint86-slice --skip-factor
```

The selected workload is the bounded two-fixture reorder/fill slice:

- `bcsstk14`
- `Pres_Poisson`

The selected reorder rows are:

- `none`
- `rcm`
- `amd`
- `colamd`
- `nd`

The selected path is the direct reorder path, not `--reorder-via-analyze`.
Numeric factorization remains skipped so the lane stays focused on reorder
time and symbolic fill.

## Metric Contract

The Sprint 98 runtime/fill lane may preserve and report these fields:

- `matrix`
- `n`
- `reorder`
- `nnz_L`
- `reorder_ms`
- `factor_ms`
- `reorder_path`
- `fixture_slice`
- `nd_base_threshold`

Metric interpretation:

- `nnz_L` is the primary fill-quality comparison field.
- `reorder_ms` is branch-local timing context, not portable performance proof.
- `factor_ms` must remain `skip` for this lane unless a later boundary
  explicitly widens the workload.
- `reorder_path` must remain `direct` for this lane.
- `fixture_slice` must remain `sprint86` for this lane.
- `nd_base_threshold` records the active ND base-threshold setting so reruns
  can be lined up.

## Artifact Shape

Day 8/9 implementation should create a generated Sprint 98 planning artifact
from this lane rather than changing canonical benchmark reporting first.

Preferred artifact shape:

- location:
  - `docs/planning/EPIC_9/SPRINT_98/artifacts/`
- content:
  - exact command
  - raw CSV output or a copied raw-output block
  - small interpretation table keyed by fixture and reorder
  - claim-boundary notes
  - follow-up queue

The artifact can be produced manually from the focused command or by a small
script only if the script removes meaningful repetition. Do not expand
`make bench-canonical-report` for this lane unless a later day explicitly
reopens canonical reporting.

## Adjacent Surfaces

Allowed Day 8/9 touch points:

- `docs/planning/EPIC_9/SPRINT_98/artifacts/`
- `docs/planning/EPIC_9/SPRINT_98/WORKING_NOTES.md`
- `benchmarks/README.md` only if implementation changes command or schema
  documentation
- `docs/maintainer_guide.md` only if the maintained benchmark-governance
  interpretation changes
- `Makefile` only if a small report target is justified by repeated use

Not selected for Day 8/9:

- broad `make bench`
- full `bench_reorder` all-fixture runs
- `bench_reorder --reorder-via-analyze`
- `bench_fillin` as a separate maintained Sprint 98 lane
- `bench_amd_qg` as a primary Sprint 98 artifact lane
- canonical report surface expansion
- timing thresholds
- workflow or CI changes

## Claim Boundary

Allowed:

- Sprint 98 records bounded reorder/fill calibration evidence for the selected
  two-fixture slice.
- The lane is useful for branch-local before/after comparison.
- `nnz_L` is the primary fill-quality field and `reorder_ms` is contextual
  timing information.

Not allowed:

- claims of universal reorder superiority
- cross-platform or portable timing claims
- claims that this replaces canonical maintained benchmark reporting
- package or platform maturity claims
- interpretation of `bench_amd_qg`, `bench_fillin`, or full `bench_reorder`
  output as part of this selected lane without a new boundary

## Validation Commands

Focused command:

```sh
make bench-reorder-sprint86
```

Expected structural checks:

- output includes the comment line with:
  - `nd_base_threshold=160`
  - `factor=no`
  - `via_analyze=no`
  - `slice=sprint86`
- output includes the stable CSV header:
  - `matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold`
- output has rows for `bcsstk14` and `Pres_Poisson`
- output has rows for `none`, `rcm`, `amd`, `colamd`, and `nd`
- `factor_ms` is `skip`
- `reorder_path` is `direct`
- `fixture_slice` is `sprint86`

Docs-only Day 7 hygiene:

```sh
git diff --check
rg -n "[ \t]+$" docs/planning/EPIC_9/SPRINT_98
```

If Day 8/9 changes benchmark C files or headers, rerun:

```sh
make format && make lint && make test
```

## Day 7 Validation Snapshot

`make bench-reorder-sprint86` passed locally and emitted the selected two-fixture
slice:

```text
=== Running bench_reorder --sprint86-slice --skip-factor ===
# nd_base_threshold=160, factor=no, via_analyze=no, slice=sprint86
matrix,n,reorder,nnz_L,reorder_ms,factor_ms,reorder_path,fixture_slice,nd_base_threshold
bcsstk14,1806,none,190791,0.0,skip,direct,sprint86,160
bcsstk14,1806,rcm,178311,7.2,skip,direct,sprint86,160
bcsstk14,1806,amd,116071,62.0,skip,direct,sprint86,160
bcsstk14,1806,colamd,146037,95.7,skip,direct,sprint86,160
bcsstk14,1806,nd,132634,297.4,skip,direct,sprint86,160
Pres_Poisson,14822,none,5061932,0.0,skip,direct,sprint86,160
Pres_Poisson,14822,rcm,3187081,81.8,skip,direct,sprint86,160
Pres_Poisson,14822,amd,2668793,4000.8,skip,direct,sprint86,160
Pres_Poisson,14822,colamd,3415793,8116.8,skip,direct,sprint86,160
Pres_Poisson,14822,nd,2474435,3699.7,skip,direct,sprint86,160
```

Timing values are local to this run and should not be treated as portable
thresholds.

## Day 8 Entry Criteria

Day 8 can implement the runtime/fill artifact from this exact command and
metric contract. If implementation pressure requires a different fixture,
metric, path, benchmark, or reporting surface, stop and write a new boundary
before editing benchmark code or workflow files.
