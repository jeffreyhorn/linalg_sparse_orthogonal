# Sprint 67 Day 11 Artifact: Large-n Cholesky analysis/CSC handoff batch

Date: 2026-06-13
Branch: `sprint-67`

## Scope

Bounded Sprint 67 maintainability landing on the large-`n` Cholesky
analysis-to-CSC handoff across:

- `src/sparse_analysis.c`
- `src/sparse_chol_csc.c`
- `src/sparse_chol_csc_internal.h`
- `tests/test_chol_csc.c`
- `tests/test_integration.c`

Explicit non-goals for this batch:

- LDL^T CSC follow-through
- graph / ND / reorder residuals
- iterative / eigensolver residuals
- public API redesign
- packaging / platform / build-surface churn

## Problem

After the Day 9 shared ND-policy convergence batch, the strongest remaining
bounded ownership seam moved to the large-`n` Cholesky explicit-analysis CSC
handoff.

The public repeated-run lifecycle already routed large-`n` Cholesky through:

1. analysis-aware CSC conversion
2. supernodal CSC elimination
3. CSC writeback to the shared `SparseMatrix` factor payload

But that lane still kept a second orchestration shell in
`src/sparse_analysis.c`, while the family-local CSC helper surface in
`src/sparse_chol_csc.c` exposed a narrower factor shim that still used scalar
`chol_csc_eliminate(...)` even when an explicit Cholesky analysis was already
present.

That left two partially parallel large-`n` analysis-backed Cholesky CSC paths:

- the public repeated-run lifecycle
- the family-local CSC factor shim

## Landing

### 1. Family-local `chol_csc_factor(...)` now owns the large-n analysis-backed factor route

`src/sparse_chol_csc.c` now resolves factorization like this:

- `analysis == NULL`:
  - heuristic CSC conversion
  - scalar `chol_csc_eliminate(...)`
- `analysis != NULL && A->rows < SPARSE_CSC_THRESHOLD`:
  - analysis-aware CSC conversion
  - scalar `chol_csc_eliminate(...)`
- `analysis != NULL && A->rows >= SPARSE_CSC_THRESHOLD`:
  - analysis-aware CSC conversion
  - supernodal `chol_csc_eliminate_supernodal(...)`
  - shared `SPARSE_CSC_SUPERNODE_MIN_SIZE` cutoff

That makes the family-local helper match the maintained large-`n` public
repeated-run Cholesky lane instead of carrying a weaker scalar-only analysis
route.

### 2. `factor_cholesky_with_analysis_csc(...)` now reuses the family-local helper

`src/sparse_analysis.c` no longer duplicates the CSC elimination dispatch for
the large-`n` explicit-analysis Cholesky lane.

It now:

1. calls `chol_csc_factor(A, analysis, &L_csc)`
2. creates the shared `SparseMatrix` factor payload
3. starts Cholesky factor state
4. writes back with `chol_csc_writeback_to_sparse(L_csc, L, NULL)`

The repeated-run public lifecycle still publishes factors in analysis
coordinate space with `analysis->perm` as the authoritative symmetric
permutation, but it no longer owns a second copy of the CSC elimination choice.

### 3. The internal helper contract is now stated directly

`src/sparse_chol_csc_internal.h` now documents that:

- analysis-backed large-`n` factor calls mirror the public repeated-run
  Cholesky lifecycle
- the supernodal path is selected there with the shared cutoff
- smaller or analysis-free helper calls keep the scalar path

## Proof

### Family-local proof

`tests/test_chol_csc.c` now adds:

- `test_factor_with_analysis_large_n_matches_explicit_supernodal_route`

It builds a large SPD tridiagonal matrix (`n = 120`), computes a Cholesky
analysis, then proves:

- `chol_csc_factor(A, &analysis, &L_helper)` and
- `chol_csc_from_sparse_with_analysis(A, &analysis, &L_explicit)` followed by
  `chol_csc_eliminate_supernodal(L_explicit, SPARSE_CSC_SUPERNODE_MIN_SIZE)`

produce structurally and numerically matching `CholCsc` factors.

### Public proof

`tests/test_integration.c` strengthens the existing large-`n` public path
comparison:

- `test_cholesky_factor_opts_matches_explicit_analysis_path`

It now also asserts:

- the one-shot Cholesky side resolved to the CSC lane with `used_csc_path == 1`

while keeping the existing solve-equivalence comparison between:

- one-shot reordered Cholesky
- explicit public `sparse_analyze(...)` + `sparse_factor_numeric(...)`

## Validation

Executed:

- `make format`
- `make lint`
- `make test`
- `make quality-review-full`

All passed.

Maintained anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 468.96 sec`

## Result

Sprint 67 Day 11 closes one real maintainability seam:

- large-`n` analysis-backed Cholesky CSC factor routing now has one shared
  family-local owner
- the repeated-run public lifecycle now reuses that owner instead of carrying
  a second copy of the elimination dispatch
- the proof stays bounded to the intended Cholesky CSC and public integration
  surfaces
