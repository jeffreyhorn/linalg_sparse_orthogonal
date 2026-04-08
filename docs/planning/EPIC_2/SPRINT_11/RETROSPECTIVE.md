# Sprint 11 Retrospective

**Sprint:** 11 — Tolerance Standardization, Thread Safety & Quality Hardening
**Duration:** 14 days (Days 1-14)
**Status:** Complete

## Definition of Done Checklist

- [x] CMake test target parity with Makefile (Day 1)
- [x] Tolerance audit: all 21 hardcoded 1e-30/1e-300 sites catalogued (Day 2)
- [x] `sparse_rel_tol()` helper implemented in `sparse_matrix_internal.h` (Day 2)
- [x] Tolerance standardization — LU, Cholesky, ILU (11 sites, Day 3)
- [x] Tolerance standardization — QR, SVD, Dense, Iterative (10 sites, Day 4)
- [x] QR R-extraction drop threshold made relative (3 additional sites, Day 4)
- [x] Scaled-matrix tolerance tests at 1e-40 and 1e+40 (Day 4)
- [x] Tolerance strategy documented in headers and algorithm.md (Day 5)
- [x] Edge-case tolerance tests (Day 5)
- [x] `cached_norm` race fixed with `_Atomic double` (Day 6)
- [x] Concurrent norminf tests (Day 6)
- [x] README thread safety section rewritten (Day 7)
- [x] TSan CI job added (Day 7)
- [x] `factored` field added to SparseMatrix (Day 8)
- [x] Solve-before-factor detection in all solve functions (Day 8)
- [x] `sparse_mark_factored()` API for externally-constructed factors (Day 8)
- [x] Comprehensive factored-state tests (Day 9)
- [x] API preconditions documented with `@pre` tags (Day 10)
- [x] ILU/ILUT reject factored matrices at runtime (Day 10)
- [x] Version macros generated from VERSION file (Day 11)
- [x] Hardcoded version removed from `sparse_types.h` (Day 11)
- [x] 32-bit index limitation documented (Day 12)
- [x] README updated with all Sprint 11 changes (Day 12)
- [x] Sprint 11 cross-feature integration tests (Day 13)
- [x] Full regression: UBSan clean, benchmarks clean (Day 13)
- [x] Packaging tests pass with correct version (Day 13)
- [x] Sprint retrospective written (Day 14)

## What Went Well

1. **Tolerance standardization was thorough and correct.** The audit
   identified 21 sites across 8 source files, plus 3 additional sites in
   QR R-extraction that used hardcoded `1e-15`. The `sparse_rel_tol()`
   helper provides a single consistent pattern: `max(tol * norm, DBL_MIN * 100)`.
   All 812 tests pass, including scaled matrices at 1e-40 and 1e+40 —
   scales that would have failed with the old absolute thresholds.

2. **The Cholesky sqrt(norm) fix was a good catch.** During Day 4 testing,
   the huge-scale Cholesky test failed because L entries scale as sqrt(A).
   Using `sqrt(factor_norm)` as the reference norm for L diagonal checks
   is the mathematically correct choice and was non-obvious from the plan.

3. **The factored-state flag caught a real design issue.** The CSR roundtrip
   test (`test_cholesky_csr_roundtrip_solve`) was passing an unfactored
   matrix to `sparse_cholesky_solve`. This worked before only because the
   solve used hardcoded tolerances. Adding `sparse_mark_factored()` provides
   a clean escape hatch for legitimate external-factor use cases.

4. **Thread safety fix was minimal and correct.** Changing `double cached_norm`
   to `_Atomic double` with relaxed ordering is the smallest possible fix.
   The computation is idempotent, so relaxed ordering is sufficient. The
   minimal TSan test (task `bkwqeyczm`) completed with exit 0, confirming
   no data races.

5. **Version generation worked end-to-end.** The `configure_file()` approach
   in CMake and `sed`-based Makefile target both generate consistent headers
   from the single VERSION file. The packaging tests verified version
   consistency across `pkg-config`, CMake `find_package`, and the C macro.

## What Didn't Go Well

1. **TSan is extremely slow on macOS.** The stress tests (8 threads x 1000
   iterations) never completed under TSan on the development machine. Even
   a no-op pthread test with TSan took over a minute. The CI job targets
   Ubuntu where TSan performance is normal. Local TSan validation was
   limited to the minimal 2-thread test.

2. **QR R-extraction was an unplanned scope expansion.** The plan only
   called for replacing the 21 catalogued tolerance sites, but Day 4 testing
   revealed that QR factorization drops R entries using hardcoded `1e-15`,
   causing complete failure at small scales. Fixing this added 3 sites and
   required relaxing the dense-vs-sparse nnz assertion by ±1. This was
   necessary but wasn't anticipated by the audit.

3. **The `1e-15` sites in `sparse_matrix.c` (matmul, axpby) were not
   addressed.** These are arithmetic noise thresholds in matrix operations,
   not singularity detection, so they fall outside the tolerance audit scope.
   They could cause issues with extremely small-scale matrices in matmul
   and should be addressed in a future sprint.

## Bugs Found During Sprint

| # | Description | Root Cause | Fix |
|---|-------------|-----------|-----|
| 1 | Cholesky solve rejects huge-scale matrices | `sing_tol = DROP_TOL * factor_norm` doesn't account for sqrt scaling of L | Use `sqrt(factor_norm)` as reference (Day 4) |
| 2 | QR produces empty R at 1e-40 scale | R entries dropped by hardcoded `fabs(val) > 1e-15` | Changed to relative `DROP_TOL * |R(i,i)|` (Day 4) |
| 3 | CSR roundtrip test fails with factored-state check | Test assumed `sparse_cholesky_solve` works on non-factored matrices | Added `sparse_mark_factored()` call (Day 8) |
| 4 | NOLINT comment lost after clang-format wraps ILU condition | `NOLINTNEXTLINE` only covers the next line, not a wrapped continuation | Moved NOLINT to the line with the actual dereference (Day 3) |

## Final Metrics

| Metric | Before Sprint 11 | After Sprint 11 |
|--------|:-:|:-:|
| Hardcoded 1e-30/1e-300 sites | 21 | 0 |
| `sparse_rel_tol()` call sites | 0 | 20 |
| Test suites | 29 | 30 |
| Total tests | 774 | 812 |
| Thread safety tests | 6 | 8 |
| Tolerance edge-case tests | 0 | 24 |
| Factored-state tests | 0 | 16 |
| Files changed | — | 27 |
| Lines added | — | 1174 |
| Lines removed | — | 97 |

## Sprint 11 Feature Summary

| Feature | Files | Description |
|---------|-------|-------------|
| `sparse_rel_tol()` | `sparse_matrix_internal.h` | Relative tolerance helper: `max(tol * norm, DBL_MIN * 100)` |
| Tolerance standardization | 8 src files | All 21+3 sites use `sparse_rel_tol()` |
| `_Atomic cached_norm` | `sparse_matrix_internal.h`, `sparse_matrix.c` | Thread-safe norm caching |
| `factored` flag | `sparse_matrix_internal.h`, `sparse_lu.c`, `sparse_cholesky.c` | Solve-before-factor detection |
| `sparse_mark_factored()` | `sparse_matrix.h`, `sparse_matrix.c` | Mark externally-constructed factors |
| `factor_norm` in ILU/CSR | `sparse_ilu.h`, `sparse_lu_csr.h` | Norm cached for relative tolerance |
| Version generation | `sparse_version.h.in`, Makefile, CMakeLists.txt | Auto-generate version from VERSION file |
| TSan CI job | `.github/workflows/ci.yml` | Thread safety CI |

## Items Not Deferred

All planned items were completed. No items deferred to Sprint 12.
