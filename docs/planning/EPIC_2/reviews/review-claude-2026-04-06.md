# Sparse Linear Algebra Library Review

**Date:** 2026-04-06
**Reviewer:** Claude
**Scope:** Full `linalg_sparse_orthogonal` repository at commit `0551e18` (post-Sprint 10 merge)

## Methodology

This review examines source code, headers, tests, build system, and documentation. It focuses on findings NOT already covered by the Codex review of the same date, which identified the missing LDL^T path, symbolic/numeric split, ordering limitations, storage format strategy, QR least-squares gaps, dense-in-disguise SVD paths, and iterative solver coverage.

## Findings

### 1. High: Numerical tolerance handling is inconsistent across solvers

The library uses a mix of absolute and relative tolerances without consistent documentation or scaling:

- `sparse_lu.c` uses `DROP_TOL * matrix_norm` (relative to infinity norm) for singularity checks
- `sparse_cholesky.c` uses hardcoded `1e-30` (absolute) for diagonal checks
- `sparse_ilu.c` uses hardcoded `1e-30` (absolute) for zero-pivot detection
- `sparse_lu_csr.c` uses `1e-300` in solve paths and caller-supplied `tol` in elimination
- GMRES Hessenberg entries use hardcoded `1e-30`

A matrix with all entries ~1e-50 will succeed in LU (norm-relative) but fail in Cholesky/ILU (absolute 1e-30 threshold). This makes solver behavior unpredictable on scaled systems. All factorizations should use a consistent relative tolerance strategy.

### 2. High: CMakeLists.txt is missing 6 test targets

The Makefile defines 29 test targets, but CMakeLists.txt only has 23. Missing:
- `test_cholesky`
- `test_csr`
- `test_matmul`
- `test_reorder`
- `test_sprint4_integration`
- `test_threads`

Users building with CMake (including the documented Windows workflow) silently skip these tests. CI uses Makefile so the gap is invisible.

### 3. High: Thread safety claims are overstated

README states "Thread-safe -- concurrent solves on shared factored matrices, per-matrix pool allocators" and lists concurrent read-only access (nnz, get, matvec) as thread-safe. However:

- `sparse_norminf()` reads and writes `cached_norm` and `cached_norm_valid` without synchronization. Two threads calling `sparse_norminf()` on the same matrix have a data race.
- The `-DSPARSE_MUTEX` option only protects `sparse_insert()` and `sparse_remove()`, not norm caching, factorization, or any read path.
- No `_Atomic` or `volatile` qualifiers on cached fields.

The README should either fix the cache races (e.g., `_Atomic` on cached_norm fields) or narrow the thread-safety claims.

### 4. Medium: No factored-state flag prevents use-before-factor bugs

`sparse_lu_solve()` operates on a matrix that was previously factored by `sparse_lu_factor()`, but there is no runtime flag to distinguish a factored matrix from an unfactored one. Calling `sparse_lu_solve()` on an unfactored matrix produces garbage silently. Similarly for Cholesky. A `factored` flag in the matrix struct (set by factor, checked by solve) would catch this class of bugs.

### 5. Medium: Accessor functions cannot report errors

`sparse_get_phys()`, `sparse_get()`, `sparse_rows()`, `sparse_cols()`, `sparse_nnz()` all return values directly (double or idx_t). For NULL inputs or out-of-bounds indices, they silently return 0 or 0.0, making it impossible for callers to distinguish "element is zero" from "invalid input." This is a design choice that trades ergonomics for safety.

### 6. Medium: ILU factorization requires identity permutations but validation is incomplete

`sparse_ilu_factor()` documents (in source comments) that the input must have identity permutations, and returns `SPARSE_ERR_BADARG` if not. However, the public header `sparse_ilu.h` does not document this precondition. A user reading only the header would not know that passing a previously-factored or reordered matrix will fail. The same applies to ILUT.

### 7. Medium: QR column-pivoted factorization stores dense Householder vectors

`sparse_qr_factor()` allocates dense `m x n` workspace for Householder reflectors (`qr->dense_v` and the `tau` array). For large sparse rectangular matrices where `m >> n`, this is acceptable, but for `m ~ n` on large sparse problems, the dense Q storage dominates memory. The sparse-mode QR mitigates this but is a separate code path. This architectural limitation should be documented.

### 8. Medium: SVD lowrank-sparse allocates dense m x n accumulator

`sparse_svd_lowrank_sparse()` is documented in the header as allocating an `m*n` dense accumulator internally. For the "sparse" low-rank output variant, this defeats the purpose on large matrices. The implementation should use rank-k outer products directly into the sparse output without the dense intermediate.

### 9. Low: Version macros in sparse_types.h can drift from VERSION file

The VERSION file is the single source of truth for CMake and Makefile, but `sparse_types.h` has manually-maintained `SPARSE_VERSION_MAJOR/MINOR/PATCH` macros with only a comment ("keep in sync with VERSION file") preventing drift. A build-time generation step (e.g., `configure_file` in CMake, `sed` in Makefile) would eliminate this risk.

### 10. Low: 32-bit index type limits matrix scale

`idx_t` is `int32_t`, limiting matrices to ~2 billion nonzeros. This is not documented as a known limitation. Modern large-scale sparse work often exceeds this. The limitation should be documented, and a future migration path to configurable index width (int32/int64) should be considered.

### 11. Low: Makefile xcrun dependency is macOS-specific

`Makefile` line 5 calls `/usr/bin/xcrun --show-sdk-path` unconditionally on non-Linux platforms. This fails silently (returns empty) on Linux and errors on Windows. The `ifneq ($(SYSROOT),)` guard handles it, but the subprocess call is unnecessary overhead and confusing on non-macOS.

### 12. Low: No CI testing on Windows or with CMake

CI runs Makefile-based build, UBSan, benchmarks, and coverage on Ubuntu only. There is no CMake CI job, no Windows CI, and no macOS CI. Given that INSTALL.md documents all three platforms and CMake is a supported build system, this is a coverage gap.

## Comparison with Codex Review

The Codex review correctly identified the major architectural gaps:
- Missing LDL^T factorization
- No symbolic/numeric factorization split
- Limited ordering stack
- Linked-list as primary numeric format
- QR minimum-norm gap
- Dense SVD internals
- Missing iterative solvers (IC, MINRES, BiCGSTAB)

This review adds findings on:
- Tolerance inconsistency across solvers (not mentioned by Codex)
- CMake test target drift (not mentioned)
- Thread safety overstatement (not mentioned)
- Missing factored-state flag (not mentioned)
- Accessor error reporting limitation (not mentioned)
- ILU precondition documentation gap (not mentioned)
- Build system and CI gaps (not mentioned)
- Version macro drift risk (not mentioned)

## Priority Assessment

Combining both reviews, the priority order for the next development cycle should be:

1. **Build system / CI fixes** (CMake test sync, CI for CMake/Windows) -- low effort, high ROI
2. **Tolerance standardization** -- affects correctness on scaled systems
3. **Thread safety fixes** -- either fix races or narrow claims
4. **LDL^T factorization** -- biggest capability gap (Codex finding #1)
5. **Symbolic/numeric split** -- architectural foundation for serious sparse work (Codex #2)
6. **Incomplete Cholesky + MINRES** -- highest-value iterative additions (Codex #7)
7. **Ordering improvements** (COLAMD, better AMD) -- needed for production scale (Codex #3)
8. **QR minimum-norm least-squares** -- completeness gap (Codex #5)
9. **Compressed numeric backend strategy** -- long-term architecture (Codex #4)
10. **BiCGSTAB + sparse eigensolvers** -- nice-to-have iterative extensions (Codex #7)

## Validation

All tests pass. `make test` runs 774 tests across 29 suites with zero failures. No correctness regressions detected. The issues identified are architectural, completeness, and robustness concerns rather than functional breakage.
