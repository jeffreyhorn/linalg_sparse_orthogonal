# Sprint 63 Day 8: Post-Landing Audit and Residual Re-Rank

Date: 2026-06-10
Branch: sprint-63

## Purpose

Re-rank the remaining Sprint 63 lifecycle queue from the landed Day 6-7 branch
state instead of the pre-landing audit, and fix the exact Day 9-10 target from
the live code and proof surface.

## Reviewed Surfaces

Implementation and proof surfaces reviewed:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_ldlt.c`
- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_ldlt_csc.c`
- `tests/test_integration.c`
- `tests/test_sparse_lu.c`
- `tests/test_chol_csc.c`
- `tests/test_ldlt.c`
- `tests/test_ldlt_csc.c`
- `examples/example_analysis.c`
- `benchmarks/bench_refactor.c`

Public truth surfaces rechecked:

- `include/sparse_analysis.h`
- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `include/sparse_ldlt.h`
- `README.md`
- `docs/tutorial.md`
- `docs/maintainer_guide.md`

## Main Finding

After the Day 6-7 landings, Sprint 63 no longer has a broad direct-wrapper
problem.

The strongest remaining queue now reduces to:

1. shared direct lifecycle solve/refactor semantics
2. large-`n` CSC-backed Cholesky failure-path / retention proof
3. bounded docs/example/benchmark follow-through after the semantics lane lands
4. deferred LDL^T comparison work only if a later contradiction appears

## What Changed Since the Opening Audit

### 1. LU is no longer the strongest remaining seam

The Day 6 LU batch closed the highest-value wrapper contradiction:

- invalid pivot now rejects deterministically
- rejection happens before reorder/factor mutation
- preserved-state retry behavior is explicit and tested

That leaves LU in a follow-through role, not as the next primary target.

### 2. The Cholesky CSC wrapper seam is materially smaller

The Day 7 Cholesky batch closed the highest-value CSC dispatch asymmetry:

- invalid backend now rejects explicitly
- CSC dispatch is selected once
- `used_csc_path` is published before later failure returns

That moves Cholesky out of the “wrapper-entry inconsistency” bucket and into a
smaller shared-lifecycle semantics/proof bucket.

### 3. The strongest remaining hole is now in the shared lifecycle layer

The live branch now points to `src/sparse_analysis.c` as the highest-leverage
remaining Sprint 63 seam.

Already true on the landed branch:

- `sparse_factor_numeric(...)` factors into temporary storage and only replaces
  the caller `factors` object after success
- `sparse_refactor_numeric(...)` validates existing factors, factors into a
  temporary, and preserves old factors on error
- `tests/test_integration.c` already proves:
  - zeroed-factor solve rejection
  - mismatched-analysis solve rejection with preserved factors
  - zeroed-factor refactor acceptance
  - mismatched-existing-factor refactor rejection
  - old-factor preservation on refactor failure
  - same-pattern Cholesky public-lifecycle parity against one-shot factorization

Still uneven after Day 7:

- the strongest refactor-failure preservation proof is still concentrated in:
  - sub-threshold linked-list Cholesky (`n = 40`)
  - large-`n` LDL^T (`n = 150`) on the indefinite KKT path
- the large-`n` Cholesky public lifecycle path is already proven on success
  (`n = 120`), but its CSC-backed failure/retention semantics are not yet
  pinned with the same strength
- `example_analysis.c` and `bench_refactor.c` correctly describe successful
  same-pattern reuse, but they are not proof surfaces for failure-path factor
  retention semantics

## Updated Rank Order

### 1. Strongest next target

Shared direct lifecycle semantics on the large-`n` CSC-backed Cholesky lane.

Why it is now first:

- it sits directly on the public repeated-run direct lifecycle
- it is the strongest remaining CSC-sensitive direct path
- it is already close to fully proved, so a bounded batch can finish the seam
- it avoids reopening the direct one-shot surface unnecessarily

### 2. Secondary target

Bounded docs/header/example/benchmark follow-through.

This should only move after the next semantics slice lands. The remaining issue
is no longer broad workflow wording; it is precision around reuse versus
failure-path retention.

### 3. Deferred target

LDL^T follow-through stays deferred:

- it already has large-`n` public-lifecycle same-pattern proof
- it already has large-`n` failure-preserve proof
- its CSC dispatch/result semantics were already tighter than Cholesky before
  the Day 7 landing

## Exact Day 9 Target

The next design batch should answer one bounded question:

- how should Sprint 63 pin large-`n` CSC-backed Cholesky
  `factor` / `refactor` / `solve` retention semantics so the public repeated-run
  direct lifecycle reads as one coherent contract?

Likely touched-file fence:

- required:
  - `src/sparse_analysis.c`
  - `tests/test_integration.c`
- likely header truth follow-through only if the landed semantics move it:
  - `include/sparse_analysis.h`
- optional only if the proof burden forces it:
  - `tests/test_chol_csc.c`
  - `examples/example_analysis.c`
  - `benchmarks/bench_refactor.c`

## Explicit Non-Targets

- no reopening LU one-shot semantics unless the shared lifecycle pass exposes a
  real regression
- no broad LDL^T widening for symmetry
- no benchmark-governance or packaging/platform spillover
- no general docs cleanup while the remaining semantics lane is still moving

## Exit State

Sprint 63 Day 8 closes with a materially smaller and more concrete queue:

- the remaining work is no longer “more lifecycle uniformity” in the abstract
- the strongest remaining seam is now shared direct lifecycle
  solve/refactor semantics on the large-`n` CSC-backed Cholesky lane
- Day 9 can proceed from an exact touched-file fence and a consciously smaller
  deferred queue
