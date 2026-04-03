# Sprint 10 Retrospective

**Sprint:** 10 — CSR Acceleration, Block Operations & Packaging
**Duration:** 14 days (Days 1-14)
**Status:** Complete

## Definition of Done Checklist

- [x] CSR working format with conversion to/from linked-list
- [x] Scatter-gather LU elimination in CSR (>=2x speedup on orsirr_1)
- [x] CSR LU solve with forward/backward substitution
- [x] Dense subblock detection (supernodal column grouping)
- [x] Dense LU kernels (dgetrf-style factor + solve)
- [x] Block-aware CSR elimination with dense block optimization
- [x] Block LU hardening — edge cases, near-singular blocks, boundary blocks
- [x] Block LU solve for multiple RHS vectors
- [x] Block CG solver with per-column convergence
- [x] Block GMRES solver with per-column deflation
- [x] Coverage gap tests and >=95% threshold enforcement in CI
- [x] `make install` / `make uninstall` with pkg-config
- [x] CMake `find_package(Sparse)` integration
- [x] INSTALL.md with cross-platform instructions
- [x] Cross-feature integration tests
- [x] Full regression under UBSan — 0 findings
- [x] Benchmarks run without crashes
- [x] README updated with Sprint 10 features
- [x] Sprint retrospective written

## What Went Well

1. **CSR acceleration exceeded targets.** The scatter-gather elimination
   achieved 12x speedup on orsirr_1, well above the 2x target. The CSR
   format eliminates linked-list pointer chasing entirely during elimination,
   and pre-allocated fill-factor capacity avoids most reallocations.

2. **Block solvers composed cleanly.** The block CG and block GMRES
   implementations reused the existing single-RHS solver logic with
   per-column convergence tracking. The `sparse_matvec_block` function
   amortizes matrix traversal across all RHS vectors.

3. **Dense block detection and in-place factoring** worked on real-world
   matrices. Supernodal column grouping correctly identified dense regions
   in structured matrices, and the dense LU path avoided scatter-gather
   overhead for those blocks.

4. **Packaging targets (Days 11-12) were straightforward.** The `make
   install` target, pkg-config template, CMake `find_package` config, and
   validation scripts all worked on the first or second attempt. The
   scripted validation (24 total checks across both tests) catches
   regressions automatically.

5. **Coverage enforcement in CI.** Adding the 95% threshold gate to the
   Makefile and GitHub Actions ensures that coverage cannot silently
   regress in future sprints.

## What Didn't Go Well

1. **The block LU solve API required pre-factoring.** The
   `sparse_lu_solve_block` function requires the matrix to already be
   factored, which is consistent with `sparse_lu_solve` but was easy to
   forget in the integration test. A one-shot `sparse_lu_factor_solve_block`
   convenience function would have been a good addition.

2. **Build directory conflicts with parallel make targets.** Running
   `make sanitize`, `make bench`, and packaging tests concurrently caused
   build directory conflicts because they all start with `make clean`. This
   is a known limitation of the Makefile structure — each target assumes
   exclusive build directory access.

3. **CMakeLists.txt was out of date.** The CMake build was missing
   `sparse_lu_csr.c` and several test targets (`test_fuzz`, `test_lu_csr`,
   `test_block_solvers`). This was caught and fixed in Day 12, but ideally
   new source files would be added to both Makefile and CMakeLists.txt in
   the same commit.

## Bugs Found During Sprint

- **None shipped.** Integration test Day 13 caught a test bug (calling
  `sparse_lu_solve_block` without prior factoring), but this was a test
  authoring error, not a library bug.
- **No UBSan or ASan findings** on any Sprint 10 code.

## Performance Improvements

| Metric | Before Sprint 10 | After Sprint 10 |
|--------|------------------|-----------------|
| LU factor+solve on orsirr_1 | 1.38 s (linked-list) | 0.11 s (CSR) |
| Speedup | — | **12x** |
| Block solve (5 RHS) | 5 × single solve | 1 block solve (shared SpMV) |

## Final Metrics

| Metric | Value |
|--------|-------|
| Public headers | 14 |
| Public API functions | ~108 |
| Source lines (src/) | ~9,300 |
| Header lines (include/) | ~2,700 |
| Test suites | 29 |
| Total unit tests | 774 |
| Test lines (tests/) | ~26,000 |
| Line coverage | >=95% (CI-enforced) |
| Sanitizer findings | 0 (UBSan clean) |
| SuiteSparse reference matrices | 6 |

## New APIs Added in Sprint 10

**sparse_lu_csr.h (new header — 13 functions):**
- `lu_csr_from_sparse`, `lu_csr_to_sparse`, `lu_csr_free`
- `lu_csr_eliminate`, `lu_csr_eliminate_block`
- `lu_csr_solve`, `lu_csr_solve_block`
- `lu_csr_factor_solve`
- `lu_detect_dense_blocks`, `lu_extract_dense_block`, `lu_insert_dense_block`
- `lu_dense_factor`, `lu_dense_solve`

**sparse_lu.h (1 new function):**
- `sparse_lu_solve_block`

**sparse_matrix.h (1 new function):**
- `sparse_matvec_block`

**sparse_iterative.h (2 new functions):**
- `sparse_cg_solve_block`
- `sparse_gmres_solve_block`

**sparse_types.h (new macros):**
- `SPARSE_VERSION_MAJOR`, `SPARSE_VERSION_MINOR`, `SPARSE_VERSION_PATCH`
- `SPARSE_VERSION`, `SPARSE_VERSION_ENCODE`, `SPARSE_VERSION_STRING`

## Items Deferred

- **Shared library support.** Only static library (`libsparse_lu_ortho.a`)
  is built and installed. Shared library with soname versioning
  (`libsparse.so.1.0.0`) was in the original plan but deferred — the static
  library serves current needs and avoids platform-specific shared library
  complexity (macOS dylib vs Linux so vs Windows DLL).

- **One-shot block factor+solve.** A `sparse_lu_factor_solve_block`
  convenience function (factor + block solve in one call) would reduce
  API friction. Can be added in a future sprint.

## Day-by-Day Summary

| Day | Theme | Key Deliverables |
|-----|-------|-----------------|
| 1 | CSR data structures | `LuCsr` struct, `lu_csr_from_sparse`, `lu_csr_to_sparse` |
| 2 | CSR elimination | `lu_csr_eliminate` — scatter-gather kernel |
| 3 | CSR solve + benchmarks | `lu_csr_solve`, `lu_csr_factor_solve`, >=2x speedup confirmed |
| 4 | Dense block detection | `lu_detect_dense_blocks` — supernodal column grouping |
| 5 | Dense LU kernels | `lu_dense_factor`, `lu_dense_solve`, `lu_csr_eliminate_block` |
| 6 | Block LU hardening | Edge cases, near-singular fallback, SuiteSparse validation |
| 7 | Block LU solve | `sparse_lu_solve_block`, `lu_csr_solve_block` |
| 8 | Block CG | `sparse_cg_solve_block`, `sparse_matvec_block` |
| 9 | Block GMRES | `sparse_gmres_solve_block` with per-column deflation |
| 10 | Coverage + CI | Coverage gap tests, 95% threshold in Makefile + GitHub Actions |
| 11 | Makefile packaging | `make install`/`uninstall`, pkg-config, version macros |
| 12 | CMake packaging | `find_package(Sparse)`, cmake_example, INSTALL.md |
| 13 | Integration testing | 14 cross-feature tests, UBSan, benchmarks, packaging validation |
| 14 | Review + retrospective | README update, metrics, retrospective |
