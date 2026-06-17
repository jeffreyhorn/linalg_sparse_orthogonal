# Sprint 76 Day 5 Artifact: Longitudinal Report Design

Date: 2026-06-17
Branch: sprint-76

## Purpose

Define the bounded implementation contract for Sprint 76's first canonical
reporting landing before any code or workflow edits begin.

## Main Result

Sprint 76 now has one explicit first implementation contract:

- required implementation center:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- support only if the first batch truly forces it:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `README.md`

## Ownership Split

### Canonical report script owner

- `scripts/bench_canonical_report.sh`

Owns:

- report-directory layout
- exact emitted artifact set for one canonical report bundle
- stable metadata capture for cross-run and cross-branch comparison
- exact canonical command mapping
- explicit artifact inventory inside the bundle

The script should remain a lightweight artifact packager, not a benchmark
policy engine.

### Public workflow owner

- `Makefile`

Owns:

- the public `make bench-canonical-report` entry point
- the default report output location
- any bounded override seam for report destination or report label
- keeping the canonical report command cheap, local-friendly, and CI-artifact
  friendly

The Makefile should keep workflow ownership, not row-schema ownership.

## Preserved Guarantees

The first batch must preserve:

- the canonical maintained benchmark surface stays:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- one CSV per canonical emitter remains the main numeric artifact surface
- benchmark binaries still own CSV row semantics and proof fields
- `make bench-canonical-report` remains threshold-free
- the report bundle remains a comparison aid, not a portability or pass/fail
  gate

## Bounded Metadata Contract

The safe first-batch metadata lane is now fixed:

- generated timestamp
- report surface identity:
  - canonical
  - proof
- exact command mapping
- explicit artifact inventory
- git commit or branch identity when locally available
- optional bounded user-supplied label for before/after or branch comparison

This is enough to improve longitudinal readability without pretending the
report bundle can prove portable performance.

## Explicit Non-Goals

The first Sprint 76 batch explicitly does not include:

- timing thresholds or pass/fail benchmark gates
- machine-specific verdict logic
- widening the report bundle to runtime or exploratory benchmark surfaces
- canonical benchmark driver rewrites
- benchmark-row schema rewrites inside:
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- broad docs/governance/platform cleanup

## Support-Surface Reading

Support surfaces should move only if the implementation truly forces them:

- `benchmarks/README.md`
  - only if the benchmark-local interpretation of canonical reporting becomes
    clearer after the landed metadata/report-bundle contract
- `docs/maintainer_guide.md`
  - only if the canonical/runtime/exploratory policy or output ownership needs
    a sharper policy statement
- `README.md`
  - only if the compact top-level benchmark summary becomes inaccurate

## Day 6 Implication

The Day 6 implementation batch should therefore start from:

- exact implementation center:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
- support only if truly forced:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `README.md`
- explicitly deferred:
  - threshold-policy widening
  - runtime-lane expansion
  - canonical benchmark driver edits
  - broad benchmark/docs cleanup
