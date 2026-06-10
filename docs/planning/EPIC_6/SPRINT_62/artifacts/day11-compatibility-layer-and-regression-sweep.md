# Sprint 62 Day 11: Compatibility Layer And Regression Sweep

Date: 2026-06-10
Branch: `sprint-62`

## Purpose

Tighten the post-Day-10 direct-lifecycle proof surface, remove stale one-shot
cancel/mutation wording, and explicitly verify the refined LU/Cholesky
usability contract without widening Sprint 62 into a broader direct-family
rewrite.

## Main Result

### 1. The public LU and Cholesky progress/cancel comments now match shipped behavior

The Day 11 patch aligned the family-local callback commentary with the actual
landed implementation:

- `include/sparse_lu.h`
  - no-reorder cancel-at-step-0 restores the pre-entry compatibility mirrors
  - reordered one-shot cancellation preserves the caller-owned matrix because
    the factor path runs on a temporary reordered working copy and only
    publishes on success
- `include/sparse_cholesky.h`
  - no-reorder linked-list cancel-at-step-0 remains non-bit-identical because
    upper-triangle entries are stripped before the first emission
  - reordered one-shot cancellation/failure can still preserve the caller-owned
    matrix because the reordered work happens on a temporary working copy

This removes the remaining stale implication that reordered one-shot
cancel/failure necessarily leaves the caller matrix reordered or partially
mutated.

### 2. The integration proof now covers the remaining high-value Cholesky compatibility cases

`tests/test_integration.c` gained two bounded regressions:

- `test_cholesky_refactor_attempt_rejects_existing_reordered_factor_and_preserves_old_factor`
- `test_cholesky_reordered_not_spd_preserves_original_matrix`

They prove:

- incompatible one-shot Cholesky retry on an already reordered/factored matrix
  returns `SPARSE_ERR_BADARG`
- the previously built factor remains valid for later solves after that
  rejected retry
- reordered non-SPD failure preserves identity row/col perms and original
  caller-owned matrix entries
- the failed reordered matrix remains unfactored afterwards

### 3. The top-level direct cancel commentary now reflects family-local semantics

The large integration comment block above the direct progress/cancel tests was
updated so it no longer implies one generic direct-family guarantee.

It now states the exact shipped split:

- LU no-reorder: bit-identical at step 0
- LU reordered one-shot: caller matrix preserved via temporary working copy
- Cholesky no-reorder linked-list: not bit-identical
- Cholesky reordered one-shot: caller matrix preserved via temporary working
  copy
- LDL^T: input matrix bit-identical because factor state is separately owned

That keeps the proof surface honest about the remaining intentional differences
between direct families.

## Touched Surface

Touched:

- `include/sparse_lu.h`
- `include/sparse_cholesky.h`
- `tests/test_integration.c`

Not widened into:

- `src/sparse_lu.c`
- `src/sparse_cholesky.c`
- `src/sparse_analysis.c`
- `tests/test_cholesky.c`
- docs/example/benchmark surfaces

## Preserved Compatibility

Preserved:

- one-shot LU and one-shot Cholesky remain first-class/default direct paths
- the explicit repeated-run direct lifecycle remains:
  - `sparse_analyze()`
  - `sparse_factor_numeric()`
  - `sparse_factor_solve()`
  - `sparse_refactor_numeric()`
- no-reorder linked-list Cholesky cancel-at-step-0 remains compatibility-only
  and non-bit-identical
- family-local ownership models remain unchanged

Strengthened:

- reordered LU/Cholesky one-shot cancellation wording now matches the shipped
  preservation behavior
- rejected one-shot Cholesky reuse now has explicit regression proof
- reordered Cholesky non-SPD failure preservation now has explicit regression
  proof

Still deferred:

- broader direct-family docs/examples simplification
- LDL^T follow-through beyond the already-coherent current state
- QR convergence work

## Validation

Required gate run:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

Result:

- all passed

Reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 285.48 sec`

Non-blocking validation note:

- the reviewed CMake rebuild again emitted the ordinary
  `bench_eigs_reuse.c` double-promotion warnings while rebuilding that bench
  binary, but the full reviewed path still completed cleanly and passed all
  parity gates

## Exit State

Sprint 62 now hands off a tighter direct-usability proof story across the
landed LU and Cholesky one-shot hardening work:

- the public wording matches the shipped mutation/cancel behavior
- the highest-value remaining compatibility seams are explicitly proven
- the remaining Sprint 62 queue is narrower and primarily adoption/docs
  follow-through rather than unproven lifecycle behavior
