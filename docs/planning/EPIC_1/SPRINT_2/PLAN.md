# Sprint 2 Plan: Hardening & Arithmetic Extensions

**Sprint Duration:** 14 days
**Goal:** Shore up robustness gaps from Sprint 1, add fundamental matrix arithmetic, and establish a larger test corpus for validating future features.

**Starting Point:** A clean sparse LU library with 4 source files, 4 public headers, 139 tests, 8 reference matrices, and 3 benchmarks. All tests pass under UBSan. ASan has not been validated (macOS sandbox hang).

**End State:** Robust I/O error reporting with errno context, relative drop tolerance replacing the absolute `1e-14` threshold, full ASan validation, `sparse_add()`/`sparse_scale()` in the public API, and ≥5 real-world SuiteSparse matrices with benchmark results.

---

## Day 1: errno Preservation — Design & Implementation

**Theme:** Map system errno to library error codes in I/O paths

**Time estimate:** 4 hours

### Tasks
1. Add `SPARSE_ERR_IO` error code to `sparse_err_t` enum in `include/sparse_types.h` (value 10) for general I/O errors that carry errno context
2. Add `sparse_errno()` function to `sparse_types.h` that returns the last captured system errno from a failed I/O operation (stored in a file-scoped static in `sparse_types.c`)
3. Add internal helper `sparse_set_errno(int errnum)` in `src/sparse_matrix_internal.h` for I/O functions to call before returning an error
4. Update `sparse_strerror()` to handle `SPARSE_ERR_IO`
5. Update `sparse_save_mm()` in `src/sparse_matrix.c`: capture `errno` after failed `fopen()` and `fprintf()` calls, call `sparse_set_errno()`, return `SPARSE_ERR_IO`
6. Update `sparse_load_mm()` in `src/sparse_matrix.c`: capture `errno` after failed `fopen()`, `fgets()`, `fscanf()` calls

### Deliverables
- `SPARSE_ERR_IO` error code in enum
- `sparse_errno()` public API function
- All `fopen`/`fgets`/`fscanf`/`fprintf` failure paths capture errno before returning

### Completion Criteria
- Code compiles with zero warnings under `-Wall -Wextra -Wpedantic`
- Existing `make test` passes (no regressions)

---

## Day 2: errno Preservation — Tests & Documentation

**Theme:** Validate errno capture and update docs

**Time estimate:** 4 hours

### Tasks
1. Add errno tests to `tests/test_sparse_io.c`:
   - Test `sparse_load_mm()` with nonexistent file → verify `SPARSE_ERR_IO` returned and `sparse_errno() == ENOENT`
   - Test `sparse_save_mm()` to unwritable path (e.g., `/dev/null/bad`) → verify `SPARSE_ERR_IO` and `sparse_errno() != 0`
   - Test that `sparse_errno()` returns 0 after a successful I/O operation
2. Decide whether existing `SPARSE_ERR_FOPEN`/`SPARSE_ERR_FREAD`/`SPARSE_ERR_FWRITE` should be kept alongside `SPARSE_ERR_IO` or consolidated. If consolidating, update all call sites and tests to use the new code
3. Update API doc comments in `include/sparse_matrix.h` for `sparse_save_mm()` and `sparse_load_mm()` to document errno behavior
4. Run `make test` and `make sanitize` — all tests pass

### Deliverables
- ≥3 new errno-specific test cases in `test_sparse_io.c`
- Updated API documentation in header
- Clean `make test` and `make sanitize` runs

### Completion Criteria
- All new errno tests pass
- Existing 139+ tests still pass
- `make sanitize` (UBSan) clean

---

## Day 3: Relative Drop Tolerance — Norm Computation & Design

**Theme:** Compute and cache infinity norm for relative tolerance

**Time estimate:** 5 hours

### Tasks
1. Add `sparse_norminf()` function to public API in `include/sparse_matrix.h`:
   - Computes `||A||_inf = max_i (sum_j |a_ij|)` by walking row linked lists
   - Returns norm via output parameter, `sparse_err_t` return for error handling
2. Implement `sparse_norminf()` in `src/sparse_matrix.c`:
   - Walk each row's linked list, accumulate `fabs(node->val)`, track max across rows
   - Handle empty matrix (norm = 0.0)
3. Add `double cached_norm` field to `SparseMatrix` internal struct in `src/sparse_matrix_internal.h`:
   - Initialize to `-1.0` (uncached sentinel) in `sparse_create()`
   - Invalidate (reset to `-1.0`) in `sparse_insert()` and `sparse_remove()` so it recomputes after modification
4. Write unit tests for `sparse_norminf()`:
   - Identity matrix → norm = 1.0
   - Known tridiagonal → manually computed norm
   - Empty matrix → norm = 0.0
   - Rectangular matrix → correct result
5. Verify `make test` passes with new tests

### Deliverables
- `sparse_norminf()` in public API, implemented and tested
- `cached_norm` field in internal struct with invalidation on modification
- ≥4 new norm tests

### Completion Criteria
- `sparse_norminf()` returns correct values for all test cases
- `make test` and `make sanitize` clean

---

## Day 4: Relative Drop Tolerance — LU Integration

**Theme:** Replace absolute DROP_TOL check in backward substitution with relative threshold

**Time estimate:** 5 hours

### Tasks
1. Modify `sparse_lu_factor()` in `src/sparse_lu.c`:
   - Compute `||A||_inf` before elimination begins (call `sparse_norminf()` on the original matrix, or accept a pre-computed norm)
   - Store the norm in a new `double factor_norm` field in `SparseMatrix` (set during factorization, used during solve)
2. Modify `sparse_backward_sub()` in `src/sparse_lu.c`:
   - Replace `if (fabs(u_ii) < DROP_TOL)` with `if (fabs(u_ii) < DROP_TOL * mat->factor_norm)` where `factor_norm` is the cached norm from factorization
   - Fallback: if `factor_norm` is 0 or uncached, use absolute DROP_TOL (backward compatible)
3. Review the fill-in drop in `sparse_lu_factor()`: the existing `DROP_TOL * max_val` (relative to pivot) is already relative — no change needed there
4. Add tests in `test_sparse_lu.c`:
   - Construct a well-conditioned matrix with very small entries (e.g., scale identity by 1e-16) — should solve correctly with relative tolerance, would fail with absolute
   - Verify a genuinely singular matrix still returns `SPARSE_ERR_SINGULAR`
5. Run full test suite — verify no regressions in existing factorization tests

### Deliverables
- Backward substitution uses relative tolerance `DROP_TOL * ||A||_inf`
- `factor_norm` field cached during `sparse_lu_factor()`
- ≥2 new tests validating relative tolerance behavior

### Completion Criteria
- Scaled-identity test passes (was failing with absolute tolerance)
- Singular matrix detection still works
- All 139+ existing tests pass
- `make sanitize` clean

---

## Day 5: Relative Drop Tolerance — Edge Cases & Refinement

**Theme:** Harden relative tolerance against edge cases

**Time estimate:** 4 hours

### Tasks
1. Add edge-case tests in `test_edge_cases.c`:
   - Zero matrix (norm = 0) → factorization should return `SPARSE_ERR_SINGULAR`, not divide-by-zero
   - Single-element 1×1 matrix with tiny value → should solve correctly
   - Matrix with mixed large and small entries → backward sub should not false-trigger singularity
   - Matrix where `||A||_inf` is very large (entries ~1e15) → relative tolerance should not be too lax
2. Handle the edge case where `factor_norm == 0.0`: if the norm is zero, the matrix is all-zero and should have already been caught as singular during pivot selection — add a guard
3. Consider adding `sparse_lu_factor_opts()` or a field in the factor call that accepts a user-supplied tolerance, overriding the default `DROP_TOL * norm`. Decide and document the approach (implementation can be deferred if not essential)
4. Update `docs/algorithm.md` with a note about the relative tolerance change and its implications
5. Run `make test`, `make sanitize`, `make bench` — all clean

### Deliverables
- ≥4 new edge-case tests for relative tolerance
- Zero-norm guard in backward substitution
- Updated algorithm documentation
- Clean bench run confirming no performance regression

### Completion Criteria
- All edge-case tests pass
- `make bench` output shows no significant timing regression vs Sprint 1 baseline
- `make sanitize` clean
- Algorithm doc updated

---

## Day 6: ASan Validation — Build Targets & Initial Run

**Theme:** Add AddressSanitizer build support and identify issues

**Time estimate:** 5 hours

### Tasks
1. Add `asan` target to `Makefile`:
   - Compile with `-fsanitize=address -fno-omit-frame-pointer -g`
   - Link with `-fsanitize=address`
   - Separate build directory (e.g., `build/asan/`) to avoid mixing with release objects
2. Update `CMakeLists.txt` `SANITIZE` option to support `SANITIZE=asan`, `SANITIZE=ubsan`, `SANITIZE=all` (currently only does ubsan)
3. Run `make asan` on macOS — observe whether the hang from Sprint 1 reproduces
4. If hang occurs: investigate root cause (likely macOS SIP/sandbox interaction with ASan runtime). Try:
   - Setting `ASAN_OPTIONS=detect_leaks=0` (leak detector often hangs on macOS)
   - Using `MallocNanoZone=0` environment variable
   - Building with Homebrew clang instead of Apple clang
5. Document findings — what works, what doesn't, and platform-specific workarounds
6. Run whatever ASan configuration works against the full test suite, capture output

### Deliverables
- `make asan` target in Makefile
- Updated CMake sanitizer support
- ASan run attempted on macOS with findings documented

### Completion Criteria
- `make asan` target compiles successfully
- If ASan runs: capture output (pass or list of findings)
- If ASan hangs: root cause identified, workaround documented, fallback plan for Day 7

---

## Day 7: ASan Validation — Fix Findings & CI Integration

**Theme:** Fix any ASan-reported issues, add to CI

**Time estimate:** 5 hours

### Tasks
1. Triage and fix any ASan findings from Day 6:
   - Heap-buffer-overflow: fix bounds
   - Use-after-free: fix lifetime management
   - Memory leaks: add missing `sparse_free()` or `fclose()` calls
   - Stack-buffer-overflow: fix array sizing
2. If macOS ASan is unreliable, add a Linux CI path:
   - Update `scripts/ci.sh` to detect platform and choose appropriate sanitizer flags
   - Document that ASan validation requires Linux or `ASAN_OPTIONS=detect_leaks=0` on macOS
3. Run `make asan` (or equivalent) against full test suite — all tests must pass clean
4. Run `make sanitize` (UBSan) — confirm still clean
5. Update `scripts/ci.sh` to include ASan run as a CI step (conditional on platform)

### Deliverables
- All ASan findings fixed (if any)
- `scripts/ci.sh` updated with ASan step
- Documented platform-specific sanitizer guidance

### Completion Criteria
- Full test suite passes under ASan (0 findings)
- Full test suite passes under UBSan (0 findings)
- CI script runs both sanitizers
- No regressions in `make test`

---

## Day 8: Sparse Matrix Scaling — Implementation

**Theme:** Implement `sparse_scale()` for scalar multiplication

**Time estimate:** 5 hours

### Tasks
1. Add `sparse_scale()` declaration to `include/sparse_matrix.h`:
   - Signature: `sparse_err_t sparse_scale(SparseMatrix *mat, double alpha)`
   - In-place: multiplies every nonzero entry by `alpha`
   - If `alpha == 0.0`: clear the matrix (remove all entries)
   - Invalidate cached norm after scaling
2. Implement `sparse_scale()` in `src/sparse_matrix.c`:
   - Walk all row headers, traverse each row's linked list, multiply `node->val *= alpha`
   - Handle `alpha == 0.0` by removing all entries (or calling `sparse_remove` for each)
   - Update `nnz` if zeroing
3. Write tests in a new `tests/test_sparse_arith.c`:
   - Scale identity by 3.0 → all diagonal entries are 3.0
   - Scale by 1.0 → matrix unchanged
   - Scale by 0.0 → nnz becomes 0
   - Scale by -1.0 → all entries negated
   - Scale rectangular matrix
   - NULL matrix → `SPARSE_ERR_NULL`
4. Add `test_sparse_arith` to Makefile and CMakeLists.txt build targets
5. `make test` passes

### Deliverables
- `sparse_scale()` in public API, fully implemented
- New `test_sparse_arith.c` test file with ≥6 scaling tests
- Build system updated

### Completion Criteria
- All scaling tests pass
- Existing tests unaffected
- Zero compiler warnings

---

## Day 9: Sparse Matrix Addition — Implementation

**Theme:** Implement `sparse_add()` for matrix addition with scaling

**Time estimate:** 6 hours

### Tasks
1. Add `sparse_add()` declaration to `include/sparse_matrix.h`:
   - Signature: `sparse_err_t sparse_add(const SparseMatrix *A, const SparseMatrix *B, double alpha, double beta, SparseMatrix **C_out)`
   - Computes `C = alpha*A + beta*B`
   - A and B must have the same dimensions → `SPARSE_ERR_SHAPE` if mismatch
   - C is a newly allocated matrix
2. Implement `sparse_add()` in `src/sparse_matrix.c`:
   - Create output matrix C with same dimensions
   - Walk A's entries: insert `alpha * val` into C
   - Walk B's entries: for each (i,j), get existing C(i,j), add `beta * val`, insert sum
   - Drop entries that cancel to zero (within DROP_TOL)
3. Consider an in-place variant `sparse_add_inplace()`:
   - Signature: `sparse_err_t sparse_add_inplace(SparseMatrix *A, const SparseMatrix *B, double alpha, double beta)`
   - Computes `A = alpha*A + beta*B` in-place
   - More memory-efficient for large matrices
4. Write tests in `tests/test_sparse_arith.c`:
   - A + zero matrix = A
   - A + (-1)*A = zero matrix
   - A + B with disjoint sparsity patterns
   - A + B with overlapping entries
   - Dimension mismatch → `SPARSE_ERR_SHAPE`
   - Rectangular matrices

### Deliverables
- `sparse_add()` in public API
- In-place variant if design is clean
- ≥6 new addition tests in `test_sparse_arith.c`

### Completion Criteria
- All addition tests pass
- `make test` and `make sanitize` clean
- Zero compiler warnings

---

## Day 10: Sparse Arithmetic — Integration Testing & Polish

**Theme:** End-to-end testing of arithmetic ops with solver, polish API

**Time estimate:** 5 hours

### Tasks
1. Integration tests combining arithmetic with solver:
   - Construct `A = alpha*I + beta*T` (scaled identity plus tridiagonal) using `sparse_add()`, then factor and solve — verify correct residual
   - Use `sparse_scale()` to precondition a matrix before factorization
   - Verify that `sparse_norminf()` gives correct results on matrices produced by `sparse_add()`
2. Test arithmetic ops under sanitizers:
   - Run `test_sparse_arith` under UBSan and ASan
3. Performance sanity check:
   - Benchmark `sparse_add()` on 1000×1000 sparse matrices — should complete in <1 second
   - Benchmark `sparse_scale()` — should be O(nnz)
4. API consistency review:
   - Ensure all new functions follow existing patterns (`sparse_err_t` returns, NULL checks, const correctness)
   - Ensure `sparse_strerror()` handles all error codes
5. Update header doc comments for all new functions

### Deliverables
- ≥3 integration tests combining arithmetic with solver
- Sanitizer-clean arithmetic tests
- Performance sanity data
- Consistent, documented API

### Completion Criteria
- All tests pass (existing + new)
- `make sanitize` (UBSan + ASan) clean
- `sparse_add()` on 1000×1000 completes in <1s
- API docs complete in headers

---

## Day 11: Larger Reference Matrices — Selection & Download

**Theme:** Select and acquire real SuiteSparse matrices

**Time estimate:** 4 hours

### Tasks
1. Select ≥5 SuiteSparse matrices covering different characteristics:
   - **Small unsymmetric** (~100–500 rows): e.g., `west0067` or `orsirr_1` — general sparse, tests solver robustness
   - **Medium symmetric** (~500–1000 rows): e.g., `bcsstk14` or `nos4` — structural engineering, SPD
   - **Medium unsymmetric** (~500–1000 rows): e.g., `west0479` or `fs_541_1` — chemical engineering
   - **Banded/structured** (~200–500 rows): e.g., `steam1` or `pores_1` — known sparsity pattern
   - **Challenging** (~100–500 rows): e.g., `fidap007` or `orsreg_1` — tests pivoting and fill-in
2. Write `scripts/download_matrices.sh`:
   - Download .mtx files from SuiteSparse Matrix Collection (sparse.tamu.edu)
   - Validate downloaded files (check header format)
   - Place in `tests/data/suitesparse/` subdirectory
3. Add `tests/data/suitesparse/` to `.gitignore` (large files, downloaded on demand) or commit small ones (<50KB)
4. Run `sparse_load_mm()` on each downloaded matrix to verify they parse correctly

### Deliverables
- ≥5 SuiteSparse matrices downloaded and parseable
- `scripts/download_matrices.sh` download script
- Matrix files in `tests/data/suitesparse/`

### Completion Criteria
- All selected matrices load without error via `sparse_load_mm()`
- Download script runs successfully and is idempotent
- Matrix selection documented (name, size, nnz, source, purpose)

---

## Day 12: Larger Reference Matrices — Solver Validation

**Theme:** Validate existing solver against real-world matrices

**Time estimate:** 5 hours

### Tasks
1. Add `test_suitesparse.c` test file:
   - For each downloaded matrix: load, factor (partial pivoting), generate random RHS `b`, solve, compute residual `||b - Ax|| / (||A|| * ||x||)`
   - Assert relative residual < threshold (e.g., `1e-10` for well-conditioned, larger for ill-conditioned)
   - Test with both complete and partial pivoting
   - Skip non-square matrices for LU (test only load and `sparse_norminf()`)
2. Handle expected failures gracefully:
   - Some matrices may be singular or nearly singular — detect and report, don't crash
   - Document which matrices are expected to succeed/fail and why
3. Add `test_suitesparse` to build system (only runs if matrix files are present — conditional test)
4. Run tests — capture results

### Deliverables
- `test_suitesparse.c` with validation tests for each matrix
- Results table: matrix name, size, nnz, pivoting mode, residual norm, pass/fail
- Build system integration (conditional on data availability)

### Completion Criteria
- ≥5 matrices tested with LU factorize+solve
- Relative residual < `1e-10` for well-conditioned matrices
- No crashes or sanitizer violations on any matrix
- Results documented

---

## Day 13: Larger Reference Matrices — Benchmarking

**Theme:** Benchmark solver performance on real-world matrices

**Time estimate:** 5 hours

### Tasks
1. Update `benchmarks/bench_main.c` to accept a directory of .mtx files:
   - Add `--dir PATH` option to benchmark all .mtx files in a directory
   - Report per-matrix: name, dimensions, nnz, factor time, solve time, fill-in ratio, memory, residual
   - Output results in a parseable format (CSV or tab-separated)
2. Run benchmarks on all SuiteSparse matrices:
   - Both complete and partial pivoting
   - Record timing data
3. Add a `bench-suitesparse` target to Makefile that runs benchmarks on the downloaded matrices
4. Analyze results:
   - Which matrices cause significant fill-in?
   - How does partial vs complete pivoting compare on real matrices?
   - Any matrices where performance is unexpectedly poor?
5. Write `docs/planning/EPIC_1/SPRINT_2/benchmark_results.md` with findings

### Deliverables
- `bench_main.c` supports directory benchmarking
- Benchmark results for all SuiteSparse matrices (both pivoting modes)
- Results analysis in `benchmark_results.md`
- `make bench-suitesparse` target

### Completion Criteria
- Benchmark runs to completion on all matrices without errors
- Results captured in structured format
- Analysis identifies fill-in patterns and pivoting tradeoffs
- No performance anomalies left uninvestigated

---

## Day 14: Sprint Review & Cleanup

**Theme:** Final validation, cleanup, and retrospective

**Time estimate:** 5 hours

### Tasks
1. Full regression run:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make asan` — ASan clean
   - `make bench` — benchmarks run, no crashes
   - `make bench-suitesparse` — SuiteSparse benchmarks run (if matrices present)
2. Code review pass:
   - Check all new public API functions have doc comments
   - Check all new error codes are handled in `sparse_strerror()`
   - Check `const` correctness on all new functions
   - Check no compiler warnings with strict flags
3. Update `README.md`:
   - Add new API functions to the API summary
   - Document `sparse_errno()` usage
   - Note ASan support and platform requirements
4. Write `docs/planning/EPIC_1/SPRINT_2/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics (test count, assertion count, etc.)
   - Items deferred to Sprint 3

### Deliverables
- All tests pass under all sanitizers
- Updated README with new API surface
- Sprint retrospective document
- Clean git history with meaningful commits

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 UBSan findings
- `make asan` passes — 0 ASan findings (or documented platform limitation)
- `make bench` and `make bench-suitesparse` complete without error
- README reflects current API
- Retrospective written with honest assessment

---

## Sprint Summary

| Day | Theme | Hours | Key Output |
|-----|-------|-------|------------|
| 1 | errno — implementation | 4 | `SPARSE_ERR_IO`, `sparse_errno()`, I/O paths updated |
| 2 | errno — tests & docs | 4 | ≥3 errno tests, API docs |
| 3 | Relative tolerance — norm | 5 | `sparse_norminf()`, cached norm field |
| 4 | Relative tolerance — LU integration | 5 | Backward sub uses relative threshold |
| 5 | Relative tolerance — edge cases | 4 | Edge-case hardening, algorithm doc |
| 6 | ASan — build & initial run | 5 | `make asan` target, initial findings |
| 7 | ASan — fixes & CI | 5 | All findings fixed, CI updated |
| 8 | Sparse scaling | 5 | `sparse_scale()`, `test_sparse_arith.c` |
| 9 | Sparse addition | 6 | `sparse_add()`, in-place variant |
| 10 | Arithmetic integration | 5 | Integration tests, performance check |
| 11 | SuiteSparse — download | 4 | ≥5 matrices, download script |
| 12 | SuiteSparse — validation | 5 | Solver correctness on real matrices |
| 13 | SuiteSparse — benchmarking | 5 | Performance data, analysis |
| 14 | Sprint review & cleanup | 5 | Retrospective, README, full regression |

**Total estimate:** 67 hours (avg ~4.8 hrs/day, max 6 hrs/day)
