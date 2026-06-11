# Sprint 63 Day 4: Lifecycle Uniformity Design and Safety Contract

Date: 2026-06-10
Branch: sprint-63

## Purpose

Turn the Day 3 ranked seam map into an explicit Sprint 63 design contract that
defines:

- what direct lifecycle behavior stays unchanged
- what internal direct behavior becomes more uniform
- which family lands first
- where the first implementation fence must stop

## Preserved Workflow Fence

Sprint 63 does not redesign the direct API split.

The preserved public workflow model remains:

- one-shot direct wrappers remain first-class/default peer entry points
- the explicit repeated-run direct lifecycle remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`

Implication:

- Sprint 63 should reduce internal lifecycle surprise and unevenness
- Sprint 63 should not blur away the difference between one-shot wrappers and
  the explicit reuse path

## First Landing: LU Lifecycle Follow-Through

### Why LU goes first

LU still owns the strongest remaining lifecycle crossover:

- reordered one-shot LU already preserves the caller matrix on cancel/failure
- the default-compatible reordered wrapper already crosses into the shared
  lifecycle machinery
- the remaining highest-value problem is lifecycle/result/factor-state
  coherence, not basic one-shot mutation surprise

### What Sprint 63 should improve in LU

- factor publication semantics after successful wrapper-driven factorization
- rejection/preservation semantics when callers try to re-enter the wrapper on
  an already reordered/factored matrix
- consistency between the wrapper path and the shared repeated-run machinery
  for solve/refactor-like outcomes

### What Sprint 63 should not do in LU

- no new top-level direct API
- no hidden copy-to-succeed behavior that masks real ownership
- no broad docs/examples/benchmark spillover in the same first landing

### Primary surfaces

- `include/sparse_lu.h`
- `src/sparse_lu.c`
- `tests/test_integration.c`
- optional only if needed:
  - `tests/test_sparse_lu.c`

## Second Landing: Cholesky CSC Repeated-Run Uniformity

### Why Cholesky goes second

The public Cholesky one-shot story is already much cleaner after Sprint 62.
The strongest remaining seam is now internal:

- CSC conversion/write-back behavior
- CSC dispatch and working-format asymmetry
- analysis-aware repeated-run state retention/publication discipline

### What Sprint 63 should improve in Cholesky

- CSC repeated-run publication discipline
- CSC versus linked-list lifecycle coherence where the public repeated-run story
  already promises a stable explicit lifecycle
- solve/refactor state-retention semantics where CSC behavior still diverges
  more than justified

### What Sprint 63 should not do in Cholesky

- no broad rerun of the Sprint 62 one-shot preservation work
- no full cancellation-model rewrite for the linked-list no-reorder lane
- no fake convergence between all Cholesky internal paths if the ownership
  models are still meaningfully different

### Primary surfaces

- `include/sparse_cholesky.h`
- `src/sparse_cholesky.c`
- `src/sparse_chol_csc.c`
- `tests/test_integration.c`
- `tests/test_chol_csc.c`

## Compatibility Contract

### Semantics that stay unchanged

- one-shot wrappers stay one-shot and caller-owned
- the explicit repeated-run lifecycle stays the canonical reuse path
- reordered LU one-shot calls preserve the caller matrix on cancel/failure
- reordered Cholesky one-shot calls preserve the caller matrix on cancel/failure
- family-local cancellation differences preserved intentionally in Sprint 62
  remain family-local unless a later concrete contradiction justifies change

### Semantics that should become more uniform

- factor-state publication and old-factor preservation behavior
- solve/refactor interpretation where LU and CSC-backed repeated-run paths meet
- internal CSC dispatch/state-retention discipline

### Semantics explicitly deferred

- no broad direct-family identity across LU, Cholesky, LDL^T, and QR
- no direct API redesign
- no packaging/platform/configuration overlap
- no broad benchmark-governance or backend-policy work

## Ownership Split

### Public behavior lane

Own only:

- bounded wording clarifications
- small header truthfulness follow-through if implementation landing needs it

Do not own:

- broad redesign
- cross-family abstraction work

### Internal hardening lane

Own primarily:

- factor-state/result semantics
- CSC publication/write-back discipline
- solve/refactor coherence
- wrapper/shared-lifecycle alignment

### Proof lane

Primary:

- `tests/test_integration.c`
- `tests/test_chol_csc.c`

Secondary only if needed:

- `tests/test_sparse_lu.c`

### Later workflow-proof lane

Only after semantics land:

- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

## Exit State

Sprint 63 now has an explicit lifecycle-uniformity design contract:

- LU is fixed as the first implementation target
- Cholesky/CSC is fixed as the second implementation target
- preserved compatibility behavior is explicit
- the direct API split stays intact
- Day 5 can now reduce this contract to an exact touched-file landing fence
