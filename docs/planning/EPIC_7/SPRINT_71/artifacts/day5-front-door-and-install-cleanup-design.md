# Sprint 71 Day 5: Front-Door & Install Cleanup Design

Date: 2026-06-16
Branch: `sprint-71`

## Purpose

Define the exact first cleanup-batch contract for `README.md` and
`INSTALL.md` before any edits land.

## First-Batch Design

### `README.md`

The first batch should make `README.md` read more clearly as the compact
product-story front door.

That means:

- keep workflow choice concise
- keep examples / benchmarks / tests ownership explicit but shorter
- keep the benchmark-report reading explicit but compact
- keep the install/package summary stable and bounded

That does not mean:

- turning `README.md` into the full maintainer-policy authority
- turning it into the full install runbook
- widening product, capability, or platform claims

### `INSTALL.md`

The first batch should make `INSTALL.md` read more clearly as the operator and
install-contract surface.

That means:

- keep quick-start build/install flow concise
- keep the static-first release shape explicit
- keep the reviewed-platform summary narrow and truthful
- keep the local install/package proof scripts explicit

That does not mean:

- widening local proof into a broad reviewed install-validation claim
- implying shared-library or dynamic-ABI maturity
- re-centering workflow-choice or benchmark-governance explanation here

## Preserved Claim Checklist

The first batch must preserve the following claims:

### `README.md`

- the orthogonal linked-list public center remains the shipped current product
  reading
- examples teach workflow, benchmarks prove retained workflow/performance, and
  tests own regression/oracle/property guarantees
- `make bench-canonical-report` remains threshold-free artifact reporting, not
  a timing gate
- the current platform-confidence summary remains intact

### `INSTALL.md`

- the maintained package surface remains static-first
- reviewed Linux/macOS/Windows asymmetry remains explicit
- `tests/test_install.sh` and `tests/test_cmake_install.sh` remain maintained
  supplemental proof surfaces
- Windows remains the reviewed CMake-first consumer story rather than a
  separate reviewed install-validation lane

### Cross-surface

- no capability widening
- no shared-library or dynamic-ABI promise
- no benchmark or example wording that steals test-owned guarantees

## Support-Surface Authority Split

The first batch should leave the following responsibilities where they already
belong:

- `docs/tutorial.md`
  - step-by-step teaching flow
- `examples/README.md`
  - adoption and workflow teaching
- `benchmarks/README.md`
  - workflow/performance proof interpretation
- `docs/maintainer_guide.md`
  - deeper policy authority and deferred-queue reading

These surfaces move only if the landed `README.md` / `INSTALL.md` wording
forces follow-through.

## First-Batch Non-Touch Set

The exact non-touch set for the landing is:

- `include/sparse_cholesky.h`
- `include/sparse_analysis.h`
- `include/sparse_iterative.h`
- `include/sparse_eigs.h`
- implementation `src/` files
- permanent proof-owner test files
- platform workflow files
- support surfaces unless the first landing truly forces them

## Exit State

Sprint 71 Day 5 closes with one exact Day 6 design:

1. `README.md` becomes a more compact product front door
2. `INSTALL.md` becomes a cleaner operator/install contract surface
3. support surfaces retain their existing teaching, proof, and policy
   authority
4. no header, implementation, test, or workflow widening is allowed in the
   first batch
