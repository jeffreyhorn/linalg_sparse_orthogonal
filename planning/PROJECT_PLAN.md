# Project Plan: sparse_lu_orthogonal — Sprints 2–4

Items deferred from Sprint 1 (see `SPRINT_1/RETROSPECTIVE.md`), organized into sprints with logical dependency ordering.

---

## Sprint 2: Hardening & Arithmetic Extensions

**Duration:** 14 days (~60 hours)

**Goal:** Shore up robustness gaps from Sprint 1, add fundamental matrix arithmetic, and establish a larger test corpus for validating future features.

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | errno preservation | Map system `errno` to `sparse_err_t` codes in all I/O paths (`fopen`, `fread`, `fscanf` failures). Add `SPARSE_ERR_IO` with optional errno snapshot. | 6 hrs |
| 2 | Relative drop tolerance | Replace absolute `DROP_TOL` in backward substitution's singular-check with a tolerance relative to the matrix norm (e.g., `||A||_inf * eps`). Compute and cache the norm during factorization. | 10 hrs |
| 3 | ASan validation | Resolve macOS sandbox ASan hang. Test on Linux or native macOS. Add ASan build target to Makefile/CMake. Run full test suite under ASan and fix any findings. | 10 hrs |
| 4 | Sparse matrix addition/scaling | Implement `sparse_add(A, B, alpha, beta)` for `C = alpha*A + beta*B`. Support in-place variant. Add scalar scaling `sparse_scale(A, alpha)`. Unit tests with identity, zero, and rectangular matrices. | 16 hrs |
| 5 | Larger reference matrices | Download real SuiteSparse matrices (e.g., west0479, bcsstk14, fidap007). Write download/conversion script. Add benchmark runs against these matrices. Validate existing solver on matrices with ≥1000 nonzeros. | 12 hrs |

### Deliverables

- All I/O errors report meaningful `sparse_err_t` codes with errno context
- Backward substitution uses relative tolerance; ill-conditioned matrices no longer false-trigger singularity
- Full test suite passes under both ASan and UBSan
- `sparse_add()` and `sparse_scale()` in public API with tests
- ≥5 real-world SuiteSparse matrices in test corpus with benchmark results

**Total estimate:** ~54 hours

---

## Sprint 3: Numerical Robustness & Fill-Reducing Reordering

**Duration:** 14 days (~70 hours)

**Goal:** Add condition number estimation for diagnostics and implement fill-reducing reordering (AMD/RCM) to improve factorization performance on larger matrices.

### Prerequisites from Sprint 2

- Relative drop tolerance (needed for robust condition number estimation)
- Sparse matrix addition/scaling (needed for norm computations in condition estimation)
- Larger reference matrices (needed for validating reordering effectiveness)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Condition number estimation | Estimate `cond(A)` from LU factors using Hager's algorithm (1-norm estimator). Return `sparse_lu_condest()` alongside factorization. Warn user when condition number exceeds threshold. | 24 hrs |
| 2 | Fill-reducing reordering (AMD/RCM) | Implement Approximate Minimum Degree (AMD) or Reverse Cuthill-McKee (RCM) ordering. Integrate as optional pre-processing step in `sparse_lu_factor()` via options struct. Benchmark fill-in reduction on SuiteSparse matrices. | 40 hrs |

### Deliverables

- `sparse_lu_condest()` returns 1-norm condition estimate from existing LU factors
- Solver warns on ill-conditioned systems (condition number > user-configurable threshold)
- `sparse_reorder_amd()` and/or `sparse_reorder_rcm()` in public API
- Factorization options struct supports `SPARSE_REORDER_NONE`, `SPARSE_REORDER_AMD`, `SPARSE_REORDER_RCM`
- Benchmark data showing fill-in reduction on ≥3 SuiteSparse matrices
- All existing tests continue to pass; new tests for reordering correctness

**Total estimate:** ~64 hours

---

## Sprint 4: Cholesky Factorization & Thread Safety

**Duration:** 14 days (~70 hours)

**Goal:** Add Cholesky factorization for SPD matrices (exploiting symmetry for half storage and no pivoting) and make the library safe for concurrent use.

### Prerequisites from Sprint 3

- Fill-reducing reordering (Cholesky on sparse SPD matrices benefits significantly from AMD/RCM)
- Condition number estimation (useful for detecting non-SPD or near-singular matrices before Cholesky)

### Items

| # | Item | Description | Estimate |
|---|------|-------------|----------|
| 1 | Cholesky factorization | Implement `sparse_cholesky_factor()` for symmetric positive-definite matrices. Store only lower triangle. Implement `sparse_cholesky_solve()` (forward/back-sub on L and L^T). Detect non-SPD during factorization (negative/zero pivot). Integrate with fill-reducing reordering. Test against SPD SuiteSparse matrices. | 40 hrs |
| 2 | Thread safety | Make pool allocator thread-safe via thread-local pools or mutex protection. Document thread-safety guarantees (e.g., read-only sharing of factored matrices is safe, concurrent mutation is not). Add concurrent-access stress tests. Review all global/static state. | 28 hrs |

### Deliverables

- `sparse_cholesky_factor()` and `sparse_cholesky_solve()` in public API
- Cholesky exploits symmetry: stores only lower triangle, ~2x memory savings
- Non-SPD detection with clear error code (`SPARSE_ERR_NOT_SPD`)
- Cholesky integrates with AMD/RCM reordering from Sprint 3
- Thread-local or mutex-protected pool allocator
- Documented thread-safety contract in API headers
- Concurrent stress tests pass under TSan (Thread Sanitizer)
- All existing LU tests remain passing

**Total estimate:** ~68 hours

---

## Summary

| Sprint | Theme | Duration | Estimate | Key Outputs |
|--------|-------|----------|----------|-------------|
| 2 | Hardening & Arithmetic | 14 days | ~54 hrs | ASan, relative tolerance, sparse add/scale, SuiteSparse matrices |
| 3 | Numerics & Reordering | 14 days | ~64 hrs | Condition estimation, AMD/RCM reordering |
| 4 | Cholesky & Thread Safety | 14 days | ~68 hrs | Cholesky for SPD, thread-safe allocator |

**Total across Sprints 2–4:** 42 days, ~186 hours
