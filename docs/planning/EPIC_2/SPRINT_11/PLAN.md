# Sprint 11 Plan: Build System, Tolerance Standardization & Thread Safety Fixes

**Sprint Duration:** 14 days
**Goal:** Fix the most urgent quality and correctness issues identified in the Codex and Claude reviews before adding new features. Synchronize build systems, standardize numerical tolerances across all solvers, resolve thread safety issues, add factored-state validation, improve API documentation, and automate version management.

**Starting Point:** A sparse linear algebra library with 14 headers, ~108 public API functions, 774 tests across 29 suites, CSR LU acceleration, block solvers, packaging (make install, CMake find_package, pkg-config). Code reviews identified: 22 hardcoded absolute tolerance sites across 8 source files, CMakeLists.txt missing 6 test targets, thread safety claims overstated due to cache races, no factored-state flag, undocumented API preconditions, and version macro drift risk.

**End State:** All solvers use consistent norm-relative tolerance strategy, CMake and Makefile produce identical test coverage with CI verification, thread safety guarantees are accurate and documented, solve-before-factor is detected at runtime, all public API preconditions are documented, and version is managed from a single source file.

---

## Day 1: CMake Test Target Sync

**Theme:** Bring CMakeLists.txt test targets in sync with Makefile and add CI enforcement

**Time estimate:** 8 hours

### Tasks
1. Add the 6 missing test targets to CMakeLists.txt:
   - `test_cholesky`
   - `test_csr`
   - `test_matmul`
   - `test_reorder`
   - `test_sprint4_integration`
   - `test_threads` (needs `-pthread` link flag, handle with `find_package(Threads)`)
2. Verify all 29 test suites build and pass under CMake:
   - `mkdir build && cd build && cmake .. && cmake --build . && ctest`
   - Compare test count with `make test` output
3. Add a CMake CI job to `.github/workflows/ci.yml`:
   - Build with CMake on Ubuntu
   - Run `ctest` and fail the job if any test fails
   - Run alongside existing Makefile-based jobs
4. Add a script or CI check that validates test target counts match between Makefile and CMakeLists.txt (e.g., count `add_sparse_test` calls vs `TEST_SRCS` entries)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- CMakeLists.txt with all 29 test targets
- CMake CI job in GitHub Actions
- Test count validation between build systems

### Completion Criteria
- `cmake --build . && ctest` runs all 29 test suites and passes
- `make test` and `ctest` produce identical test counts
- CI passes with both Makefile and CMake jobs
- `make format && make lint && make test` clean

---

## Day 2: Tolerance Audit & Helper Design

**Theme:** Audit all 22 hardcoded tolerance sites and design the standardized tolerance strategy

**Time estimate:** 8 hours

### Tasks
1. Catalog every hardcoded tolerance in the codebase:
   - `sparse_cholesky.c`: 2 sites using `1e-30` absolute for diagonal checks
   - `sparse_ilu.c`: 1 site using `1e-30` for zero-pivot detection
   - `sparse_lu_csr.c`: 3 sites using `1e-300` in solve paths
   - `sparse_iterative.c`: 2 sites using `1e-30` in GMRES Hessenberg
   - `sparse_svd.c`: 4 sites using `1e-30` as absolute floor
   - `sparse_qr.c`: 2 sites using `1e-30` for diagonal checks
   - `sparse_dense.c`: 2 sites using `1e-30` in tridiag QR deflation
   - `sparse_lu.c`: uses `DROP_TOL * norm` (already relative — reference pattern)
2. Classify each site:
   - **Singularity detection** (Cholesky diagonal, ILU pivot, CSR U diagonal, QR diagonal) — should use relative tolerance
   - **Convergence/deflation** (GMRES lucky breakdown, SVD convergence, tridiag QR deflation) — may keep small absolute floor
   - **Drop tolerance** (LU fill-in) — already relative, good pattern
3. Design `sparse_rel_tol()` internal helper:
   - Signature: `static inline double sparse_rel_tol(double matrix_norm, double user_tol)`
   - Returns `max(user_tol * matrix_norm, DBL_MIN * 100)` to avoid underflow
   - Place in `sparse_matrix_internal.h` for use by all source files
4. Document the tolerance strategy in a code comment block in `sparse_matrix_internal.h`
5. No code changes to tolerance sites yet — audit and design only

### Deliverables
- Complete audit document (can be in-code comments or notes)
- `sparse_rel_tol()` helper function implemented
- Tolerance strategy documented

### Completion Criteria
- Every hardcoded 1e-30 and 1e-300 site identified and classified
- Helper function compiles and is usable from all source files
- Strategy decision made for each site category

---

## Day 3: Tolerance Standardization — LU, Cholesky, ILU

**Theme:** Replace hardcoded tolerances in the three direct factorization paths

**Time estimate:** 10 hours

### Tasks
1. Standardize `sparse_cholesky.c` (2 sites):
   - Replace `1e-30` diagonal checks with `sparse_rel_tol(factor_norm, tol)` where `factor_norm` is the infinity norm computed at factorization start
   - Compute and cache `factor_norm` in Cholesky like LU already does
2. Standardize `sparse_ilu.c` (1 site + ILUT):
   - Replace `1e-30` zero-pivot detection with relative tolerance
   - ILU operates on an unfactored matrix, so compute norm before factorization
   - Update ILUT path similarly
3. Standardize `sparse_lu_csr.c` solve paths (3 sites):
   - Replace `1e-300` absolute checks in `lu_csr_solve()`, `lu_csr_solve_block()`, and compaction with relative tolerance
   - The CSR elimination already takes `tol` from caller — propagate to solve
4. Verify no behavior change on existing tests:
   - All 774 tests must still pass
   - Run on SuiteSparse matrices to check no false-singular rejections
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Cholesky, ILU/ILUT, and CSR LU solve use norm-relative tolerances
- No hardcoded 1e-30 or 1e-300 remaining in these files

### Completion Criteria
- `grep -rn '1e-30\|1e-300' src/sparse_cholesky.c src/sparse_ilu.c src/sparse_lu_csr.c` returns nothing
- All 774 tests pass
- `make format && make lint && make test` clean

---

## Day 4: Tolerance Standardization — QR, SVD, Dense, Iterative

**Theme:** Replace hardcoded tolerances in the remaining source files

**Time estimate:** 10 hours

### Tasks
1. Standardize `sparse_qr.c` (2 sites):
   - Replace `1e-30` diagonal checks with tolerance relative to R[0,0] or input norm
   - QR already uses `1e-14 * R[0,0]` for rank detection — make singularity checks consistent
2. Standardize `sparse_svd.c` (4 sites):
   - `abs_tol` floor in bidiag SVD iteration: keep small absolute floor but compute from matrix norm
   - Convergence checks: relative to largest singular value
   - Lanczos reorthogonalization threshold: relative to current norm
3. Standardize `sparse_dense.c` (2 sites):
   - Tridiag QR deflation checks: relative to diagonal sum
   - These are already partially relative — make fully consistent
4. Standardize `sparse_iterative.c` (2 sites):
   - GMRES Hessenberg lucky breakdown: relative to current Krylov vector norm
   - GMRES back-substitution: relative to Hessenberg diagonal
5. Write tolerance consistency tests:
   - Create a scaled matrix (all entries * 1e-40) and verify all solvers still work
   - Create a large-entry matrix (all entries * 1e+40) and verify no false singularity
6. Run `make format && make lint && make test` — all clean

### Deliverables
- All remaining hardcoded tolerances replaced with relative strategy
- Scaled-matrix tolerance tests added
- Zero hardcoded 1e-30 or 1e-300 remaining in src/

### Completion Criteria
- `grep -rn '1e-30\|1e-300' src/` returns zero matches (excluding comments)
- Scaled-matrix tests pass for 1e-40 and 1e+40 scaling factors
- All existing tests pass
- `make format && make lint && make test` clean

---

## Day 5: Tolerance Documentation & Edge Case Tests

**Theme:** Document the new tolerance strategy and add edge-case tests

**Time estimate:** 8 hours

### Tasks
1. Document tolerance strategy in each public header:
   - Add `@note` blocks explaining tolerance semantics
   - `sparse_lu.h`: explain `tol` parameter is absolute pivot threshold, but singularity detection uses norm-relative
   - `sparse_cholesky.h`: explain relative diagonal check
   - `sparse_ilu.h`: explain relative pivot tolerance
   - `sparse_lu_csr.h`: explain `tol` and `drop_tol` semantics
2. Update `docs/algorithm.md` with tolerance strategy section
3. Add tolerance edge-case tests:
   - Near-singular matrix (condition number ~1e15): verify solve succeeds with refinement
   - Exactly singular matrix: verify all solvers detect and report singularity
   - Matrix with entries spanning many magnitudes (1e-20 to 1e+20): verify factorization
   - Zero matrix: verify appropriate error returned
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Tolerance semantics documented in all factorization headers
- Algorithm documentation updated
- Edge-case tolerance tests added

### Completion Criteria
- Every factorization header documents its tolerance semantics
- Edge-case tests pass
- `make format && make lint && make test` clean

---

## Day 6: Thread Safety — Audit & cached_norm Fix

**Theme:** Fix the cached_norm data race and audit all shared mutable state

**Time estimate:** 10 hours

### Tasks
1. Audit all mutable fields in SparseMatrix struct that could be accessed concurrently:
   - `cached_norm` and `cached_norm_valid` — written by `sparse_norminf()`, read by solve paths
   - `factor_norm` — written by factorization, read by solve
   - Pool allocator fields — mutated by insert/remove only
   - Permutation arrays — mutated by factorization only
2. Fix `cached_norm` race using `_Atomic`:
   - Make `cached_norm_valid` an `_Atomic int` (C11 atomics)
   - Use `atomic_load`/`atomic_store` with relaxed ordering for the flag
   - Protect `cached_norm` write with a compare-and-swap pattern or accept benign races with documentation
3. Alternative: if `_Atomic` is not portable enough, use a simple mutex for norm caching (under `SPARSE_MUTEX`) or document the race
4. Test under TSan:
   - Write a test that calls `sparse_norminf()` from two threads on the same matrix
   - Verify no TSan warnings
5. Run `make format && make lint && make test` and `make tsan` — all clean

### Deliverables
- `cached_norm` race fixed with C11 atomics or documented
- TSan-clean concurrent norm test

### Completion Criteria
- `make tsan` passes with zero findings
- Concurrent norminf test passes
- `make format && make lint && make test` clean

---

## Day 7: Thread Safety — README Update & TSan CI

**Theme:** Update thread safety documentation and add TSan to CI

**Time estimate:** 8 hours

### Tasks
1. Rewrite README thread safety section:
   - Update the thread safety table to accurately reflect guarantees after Day 6 fixes
   - Document which operations are safe for concurrent access on the same matrix
   - Document which operations require external synchronization
   - Clarify that `SPARSE_MUTEX` protects insert/remove only (not factorization or norm caching)
2. Add TSan CI job to `.github/workflows/ci.yml`:
   - Build with `-fsanitize=thread`
   - Run the thread-safety test suite
   - Fail the job on any TSan finding
3. Review and update `sparse_matrix_internal.h` comments about thread safety invariants
4. Run `make format && make lint && make test` and `make tsan` — all clean

### Deliverables
- Accurate README thread safety documentation
- TSan CI job

### Completion Criteria
- README thread safety claims match implementation
- TSan CI job passes
- `make format && make lint && make test` clean

---

## Day 8: Factored-State Flag — Design & Implementation

**Theme:** Add runtime detection of solve-before-factor bugs

**Time estimate:** 10 hours

### Tasks
1. Add `factored` field to SparseMatrix struct:
   - Add `int factored;` field to the struct in `sparse_matrix_internal.h`
   - Initialize to 0 in `sparse_create()`
   - Set to 1 in `sparse_lu_factor()`, `sparse_lu_factor_opts()`, `sparse_cholesky_factor()`, `sparse_cholesky_factor_opts()`
   - Reset to 0 in `sparse_insert()`, `sparse_remove()` (matrix modified after factorization)
2. Check factored flag in solve functions:
   - `sparse_lu_solve()`: return `SPARSE_ERR_BADARG` if `!mat->factored`
   - `sparse_lu_solve_block()`: same check
   - `sparse_cholesky_solve()`: same check
   - `sparse_lu_refine()`: check that LU argument is factored
   - `sparse_lu_condest()`: check that LU argument is factored
   - `sparse_lu_solve_transpose()`: check that matrix is factored
3. Add `SPARSE_ERR_BADARG` documentation for solve-before-factor in all affected headers
4. Ensure `sparse_copy()` copies the factored flag
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `factored` field in SparseMatrix
- All solve functions check factored state
- Solve-before-factor returns `SPARSE_ERR_BADARG`

### Completion Criteria
- Calling `sparse_lu_solve()` on an unfactored matrix returns `SPARSE_ERR_BADARG`
- Calling solve after factor succeeds
- Modifying a factored matrix resets the flag
- All existing tests pass (they all factor before solving)
- `make format && make lint && make test` clean

---

## Day 9: Factored-State Flag — Tests & Edge Cases

**Theme:** Test the factored-state flag thoroughly

**Time estimate:** 8 hours

### Tasks
1. Write solve-before-factor tests:
   - Create a matrix, call `sparse_lu_solve()` without factoring — expect `SPARSE_ERR_BADARG`
   - Same for Cholesky solve
   - Same for LU condest, refine, transpose solve, block solve
2. Write factor-then-modify-then-solve tests:
   - Factor a matrix, insert a new entry, call solve — expect `SPARSE_ERR_BADARG`
   - Factor, remove an entry, solve — expect `SPARSE_ERR_BADARG`
3. Write positive tests:
   - Factor and solve — expect `SPARSE_OK`
   - Factor, solve, solve again — expect `SPARSE_OK` (solve doesn't clear flag)
   - Copy a factored matrix, solve on copy — expect `SPARSE_OK`
4. Test interaction with CSR path:
   - `lu_csr_factor_solve()` does not use the flag (standalone path) — verify
5. Run `make format && make lint && make test` — all clean

### Deliverables
- Comprehensive factored-state tests
- All edge cases covered

### Completion Criteria
- All new tests pass
- All existing tests still pass
- `make format && make lint && make test` clean

---

## Day 10: ILU/ILUT & API Precondition Documentation

**Theme:** Document all undocumented preconditions across the public API

**Time estimate:** 8 hours

### Tasks
1. Document ILU/ILUT preconditions in `sparse_ilu.h`:
   - `sparse_ilu_factor()`: document that input must have identity permutations (not previously reordered or factored)
   - `sparse_ilut_factor()`: same precondition
   - Document that passing a factored matrix will return `SPARSE_ERR_BADARG`
   - Add `@pre` tags in Doxygen comments
2. Audit and document preconditions in other headers:
   - `sparse_qr.h`: document that input should be unfactored
   - `sparse_lu.h`: document that `sparse_lu_factor()` is destructive (modifies input)
   - `sparse_cholesky.h`: document destructive factorization and SPD requirement
   - `sparse_svd.h`: document that input is not modified (read-only)
   - `sparse_lu_csr.h`: document that `lu_csr_from_sparse()` is non-destructive
3. Add precondition violation tests:
   - Pass a factored matrix to `sparse_ilu_factor()` — expect `SPARSE_ERR_BADARG` (now detectable via factored flag)
   - Pass a reordered matrix to ILU — expect `SPARSE_ERR_BADARG`
4. Run `make format && make lint && make test` — all clean

### Deliverables
- All public API preconditions documented in headers
- Precondition violation tests

### Completion Criteria
- Every factorization header documents input requirements
- Every solve header documents factored-state requirement
- Precondition violation tests pass
- `make format && make lint && make test` clean

---

## Day 11: Version Macro Generation

**Theme:** Automate version macro generation from the VERSION file

**Time estimate:** 10 hours

### Tasks
1. Create `include/sparse_version.h.in` template:
   - `#define SPARSE_VERSION_MAJOR @SPARSE_VERSION_MAJOR@`
   - `#define SPARSE_VERSION_MINOR @SPARSE_VERSION_MINOR@`
   - `#define SPARSE_VERSION_PATCH @SPARSE_VERSION_PATCH@`
   - `#define SPARSE_VERSION_STRING "@SPARSE_VERSION_STRING@"`
   - Include the `SPARSE_VERSION_ENCODE` macro and `SPARSE_VERSION` definition
2. CMake: use `configure_file()` to generate `sparse_version.h` from template and VERSION file
   - Parse VERSION file into components
   - Generate into build directory
   - Install the generated header
3. Makefile: add a `generate-version` target that uses `sed` to produce `sparse_version.h`
   - Parse VERSION file
   - Substitute into template
   - Output to `include/sparse_version.h` (or build directory)
4. Update `sparse_types.h`:
   - Remove hardcoded version macros
   - `#include "sparse_version.h"` instead
5. Update `sparse.pc.in` and install targets if needed
6. Verify version is consistent across:
   - `pkg-config --modversion sparse`
   - CMake `find_package` version
   - `SPARSE_VERSION_STRING` macro
7. Run `make format && make lint && make test` — all clean

### Deliverables
- Version header generated from VERSION file at build time
- No hardcoded version macros in source
- Consistent version across all build/install paths

### Completion Criteria
- Changing VERSION file and rebuilding produces correct version everywhere
- `make test` passes
- `make install` produces correct pkg-config version
- CMake install produces correct package version
- `make format && make lint && make test` clean

---

## Day 12: 32-bit Index Documentation & README Updates

**Theme:** Document idx_t limitations and update README for all Sprint 11 changes

**Time estimate:** 6 hours

### Tasks
1. Document 32-bit index limitation:
   - Add to README "Known Limitations" section: `idx_t` is `int32_t`, limiting matrices to ~2 billion nonzeros
   - Add rationale comment in `sparse_types.h` explaining the choice and future migration path
   - Note in `INSTALL.md` if relevant
2. Update README for Sprint 11 changes:
   - Update thread safety table
   - Update tolerance documentation references
   - Note factored-state validation
   - Update any outdated test counts
3. Update `docs/algorithm.md` if tolerance strategy section was added on Day 5
4. Run `make format && make lint && make test` — all clean

### Deliverables
- 32-bit index limitation documented
- README current with Sprint 11 features
- All documentation consistent

### Completion Criteria
- README accurately reflects current library state
- No stale test counts or feature claims
- `make format && make lint && make test` clean

---

## Day 13: Integration Testing & Final Validation

**Theme:** Cross-feature integration testing and full regression

**Time estimate:** 10 hours

### Tasks
1. Full regression:
   - `make clean && make test` — all tests pass
   - `make sanitize` — UBSan clean
   - `make tsan` — TSan clean (thread safety)
   - `make bench` — benchmarks run without crashes
   - Packaging tests: `bash tests/test_install.sh` and `bash tests/test_cmake_install.sh`
2. Cross-feature integration tests:
   - Tolerance tests: verify all solvers work on scaled matrices (1e-40 and 1e+40)
   - Factored-state tests: verify solve-before-factor detection across all solve paths
   - Thread safety tests: concurrent norminf + concurrent solve on same factored matrix
   - CMake build: verify all 29 tests pass under `ctest`
3. Version consistency test:
   - Build, install, verify `pkg-config --modversion sparse` matches VERSION file
   - Build CMake example, verify `SPARSE_VERSION_STRING` matches VERSION file
4. Run `make format && make lint && make test` — all clean

### Deliverables
- Full regression pass under all sanitizers
- Cross-feature integration tests
- Version consistency verified

### Completion Criteria
- `make test` passes — 0 failures
- `make sanitize` passes — 0 findings
- `make tsan` passes — 0 findings
- All packaging tests pass
- Version consistent across all paths
- `make format && make lint && make test` clean

---

## Day 14: Sprint Review & Retrospective

**Theme:** Final documentation, sprint review, and retrospective

**Time estimate:** 4 hours

### Tasks
1. Final metrics collection:
   - Total test count (should be > 774 with new tests)
   - Hardcoded tolerance count (should be 0)
   - CMake vs Makefile test target parity
   - Thread safety test coverage
2. Write `docs/planning/EPIC_2/SPRINT_11/RETROSPECTIVE.md`:
   - Definition of Done checklist
   - What went well / what didn't
   - Bugs found during sprint
   - Final metrics
   - Items deferred (if any)
3. Update project plan if any Sprint 11 items were deferred to Sprint 12+
4. Run `make format && make lint && make test` — final clean build

### Deliverables
- Sprint retrospective document
- Updated metrics
- Clean final build

### Completion Criteria
- All Sprint 11 items complete or explicitly deferred
- Retrospective written with honest assessment
- `make format && make lint && make test` clean
