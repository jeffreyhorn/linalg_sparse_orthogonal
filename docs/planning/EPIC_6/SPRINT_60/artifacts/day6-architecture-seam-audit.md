# Sprint 60 Day 6: Architecture Seam Audit

Date: 2026-06-08
Branch: `sprint-60`


## Purpose

Map the highest-leverage architectural seams that later Epic 6 implementation
work must preserve so usability, configuration, backend, benchmark, platform,
and packaging changes can land against a stable contract.

## High-Risk Architecture Seams

### 1. Public repeated-run direct ownership seam

The explicit repeated-run direct lifecycle is already centered on one stable
public seam:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`

This seam should remain the only repeated-run direct ownership model. Later
Epic 6 work may improve usability around it, but should not replace it with a
generic universal direct handle.

### 2. Public repeated-run iterative/eigensolver handle seams

The repeated-run iterative and eigensolver paths are already explicit and
opaque:

- iterative:
  - `sparse_iter_handle_t`
  - `prepare_*`
  - `sparse_solve_*_with_handle(...)`
- eigensolver:
  - `sparse_eigs_handle_t`
  - `sparse_eigs_handle_prepare(...)`
  - `sparse_eigs_sym_with_handle(...)`

Reuse preserves allocation capacity only. Support-family boundaries remain
intentional and should not widen implicitly.

### 3. Configuration/control placement seam

This is the highest-risk architectural seam in the repo because control is
currently split across:

- public option structs
- compile-time build switches
- process-global env vars
- internal debug/profile toggles

Later Epic 6 work needs an explicit placement rule for every control surface:

- public typed option
- internal typed policy
- compile-time build switch
- legacy compatibility override

### 4. Benchmark proof vs governance seam

The existing workflow-proof benchmarks should remain proof surfaces:

- `bench_refactor`
- `bench_refactor_csc`
- `bench_iterative_reuse`
- `bench_eigs_reuse`

Performance-governance work should sit above them and define:

- canonical baselines
- regression-sensitive tiers
- machine-readable conventions

That is a different architecture layer than the driver binaries themselves.

### 5. Validation/platform contract seam

The current validation/platform shape is already architectural:

- `quality-review-full`
- `quality-review`
- `quality-review-cmake`
- `deadcode-check`
- reviewed CMake parity
- explicit Linux/macOS/Windows dispositions

Later Epic 6 platform work must preserve or deliberately revise this contract
from measured evidence, not from convenience-driven CI churn.

### 6. Packaging/build seam

The current packaging/build surface is bounded around:

- one primary `STATIC` library target
- install/export support
- optional but bounded build switches
- explicit Windows gating in the build graph

Later backend or packaging work must distinguish carefully between:

- internal implementation change
- optional build/runtime surface
- public product promise

## Highest-Risk Seam Interactions

1. direct usability improvements vs preserved analysis/factors lifecycle
2. typed configuration modernization vs legacy env-var compatibility
3. backend abstraction vs self-contained default build
4. performance governance vs existing workflow-proof benchmark drivers
5. packaging/platform ambition vs reviewed truthfulness contract
6. assurance expansion vs already-large test and hotspot surfaces

## Candidate Architecture-Contract Topics for Days 7-10

- control-placement rules
- compatibility-preservation rules
- bounded backend-widening rules
- benchmark-proof versus benchmark-governance rules
- validation/platform/packaging truthfulness rules
- maintainability cleanup rules tied to real ownership seams

## Day 6 Exit State

Sprint 60 now has an architecture seam map that is concrete enough to support
the Day 7-8 configuration/performance audits and the later architecture and
validation contract freeze.
