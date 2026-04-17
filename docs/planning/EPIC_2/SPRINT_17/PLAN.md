# Sprint 17 Plan: CSR/CSC Numeric Backend for Cholesky and LDL^T

**Sprint Duration:** 14 days
**Goal:** Extend the CSR working-format strategy (proven in Sprint 10 for LU) to Cholesky and LDL^T, making compressed formats the primary numeric backend for all direct solvers while keeping the linked list as the mutable front end.

**Starting Point:** A sparse linear algebra library with ~20 headers, ~150+ public API functions, ~1250+ tests across ~38 test suites. Direct solvers include LU (with both linked-list and CSR working-format paths from Sprint 10, ~12x speedup on orsirr_1), Cholesky (linked-list only, `src/sparse_cholesky.c`, left-looking column-by-column with dense column accumulator), and LDL^T (linked-list only, `src/sparse_ldlt.c`, Bunch-Kaufman 1x1/2x2 pivoting). Sprint 14 delivered an analyze/factor split (`sparse_analyze()`, `sparse_factor_numeric()`, `sparse_refactor_numeric()`) with elimination tree, postorder, and column counts available for all three factorizations. Sprint 15-16 extended the iterative solver framework (BiCGSTAB, stagnation detection, residual history). Cholesky and LDL^T inner loops are bottlenecked by linked-list pointer chasing on large SPD/symmetric-indefinite SuiteSparse matrices.

**End State:** A CSC (compressed sparse column) working-format backend for both Cholesky and LDL^T, mirroring Sprint 10's CSR strategy for LU. `chol_csc_from_sparse()`, `chol_csc_eliminate()`, and `chol_csc_solve()` provide the column-oriented kernel for `L*L^T`. `ldlt_csc_eliminate()` extends that kernel with symmetric Bunch-Kaufman pivoting (1x1 and 2x2 blocks) in CSC. Supernodal detection identifies groups of columns with identical nonzero structure and factors them with dense kernels (reusing `lu_dense_factor()` patterns from Sprint 10). The linked-list `SparseMatrix` remains the mutable input format; CSC becomes the numeric backend, with conversion at the boundary. `sparse_cholesky_factor()` and `sparse_ldlt_factor()` keep their public APIs and dispatch to the CSC path by default for matrices above a size threshold. Benchmarks demonstrate >= 2x speedup over the linked-list path on SuiteSparse SPD and symmetric indefinite matrices, with bit-exact (or tolerance-equal) residuals.

---

## Day 1: CSC Working Format — Design & Data Structures

**Theme:** Design the CSC working format for Cholesky and stand up scaffolding

**Time estimate:** 10 hours

### Tasks
1. Design the `CholCsc` struct in a new internal header `src/sparse_chol_csc_internal.h`:
   - `idx_t *col_ptr` — length n+1, column pointers
   - `idx_t *row_idx` — row indices (length nnz, sorted within each column)
   - `double *values` — numeric values (length nnz)
   - `idx_t n`, `idx_t nnz`, `idx_t capacity`
   - Cached `factor_norm` (||A||_inf or ||A||_1) for tolerance-relative drop decisions
   - Optional `idx_t *colcount_hint` from Sprint 14 symbolic analysis for pre-allocation
2. Decide storage layout for L:
   - Strictly lower triangular L (diagonal stored separately, since L[i,i] = 1 is *not* the case here — Cholesky has nonunit diagonal)
   - For Cholesky, store full L including diagonal in CSC (each column lists rows i >= j with L[i,j] != 0, starting with i = j)
   - Document the invariant in the header
3. Justify CSC over CSR for Cholesky:
   - L*L^T factorization is naturally column-oriented (left-looking and up-looking)
   - cdiv(j) divides column j by sqrt of diagonal; cmod(j,k) modifies column j by column k
   - Compare with Sprint 10's CSR choice for LU (row-oriented elimination)
   - Add a design comment to `src/sparse_chol_csc_internal.h` explaining the trade-off
4. Create function skeletons:
   - `chol_csc_alloc(idx_t n, idx_t initial_nnz, CholCsc *out)`
   - `chol_csc_free(CholCsc *m)`
   - `chol_csc_grow(CholCsc *m, idx_t needed)` — capacity growth helper
5. Add new files to `Makefile` and `CMakeLists.txt`:
   - `src/sparse_chol_csc.c` (new)
   - `src/sparse_chol_csc_internal.h` (new)
   - `tests/test_chol_csc.c` (new, empty for now)
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `CholCsc` struct definition and internal header
- Allocation/free/grow helpers
- Build system updated; empty test file compiles
- Design comment justifying CSC vs CSR for Cholesky

### Completion Criteria
- Library builds with new files included
- All existing 1250+ tests still pass
- `make format && make lint && make test` clean

---

## Day 2: CSC Working Format — Conversion Routines

**Theme:** Implement linked-list <-> CSC conversion respecting permutations

**Time estimate:** 12 hours

### Tasks
1. Implement `chol_csc_from_sparse(const SparseMatrix *A, const idx_t *perm, CholCsc *out)`:
   - Input: linked-list `SparseMatrix` representing the lower triangle of A (or full symmetric A with extraction)
   - Apply optional permutation `perm` (AMD or RCM) — `perm[i] = j` means new index i maps to old index j
   - Build `col_ptr` via two-pass counting (first pass: per-column nnz counts; cumulative sum; second pass: scatter row indices and values)
   - Sort row indices within each column ascending (required for correctness of cmod traversal)
   - Pre-size to `2 * nnz(A)` initial capacity by default; allow caller-supplied `colcount_hint` from `sparse_analyze()` to size exactly
2. Implement `chol_csc_to_sparse(const CholCsc *L, const idx_t *perm, SparseMatrix *out)`:
   - Reconstruct the linked-list lower-triangular `SparseMatrix` from CSC L
   - Reverse the permutation so the result is in the user's coordinate system
   - Preserve permutation metadata on the output `SparseMatrix`
3. Write round-trip tests in `tests/test_chol_csc.c`:
   - Identity matrix: convert -> convert back -> exact match (values and structure)
   - Diagonal SPD matrix
   - Tridiagonal SPD matrix
   - Dense lower triangle 5x5
   - SuiteSparse SPD matrix (e.g., bcsstk01) — full structural and value preservation
   - Empty matrix (n = 0) and 1x1 edge cases
4. Permutation tests:
   - Apply identity permutation -> result matches no-permutation version
   - Apply AMD permutation -> verify nonzero pattern is permuted correctly
5. Memory leak check: run new tests under ASan
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `chol_csc_from_sparse()` and `chol_csc_to_sparse()` conversion routines
- >= 6 round-trip and permutation tests
- ASan-clean conversions

### Completion Criteria
- Round-trip preserves all matrix entries exactly (no value drift)
- Permutations applied correctly in both directions
- No memory leaks under ASan
- `make format && make lint && make test` clean

---

## Day 3: CSC Working Format — Symbolic Integration & Hardening

**Theme:** Wire CSC working format into Sprint 14's symbolic analysis pipeline

**Time estimate:** 10 hours

### Tasks
1. Extend `sparse_analyze()` (or add a CSC-specific path in `src/sparse_analysis.c`):
   - When the analysis is for `SPARSE_CHOLESKY`, compute the per-column nnz of L (`colcount`) and expose it as a `colcount_hint` consumable by `chol_csc_from_sparse()`
   - Verify that the existing etree/postorder/colcount outputs from Sprint 14 are sufficient (no new symbolic kernels needed yet)
2. Implement `chol_csc_from_sparse_with_analysis(const SparseMatrix *A, const sparse_analysis_t *analysis, CholCsc *out)`:
   - Allocate `values`/`row_idx` exactly to sum(colcount) (exact predicted nnz of L)
   - Use the symbolic structure to lay out `col_ptr` before any numeric work
   - Removes capacity-growth churn during elimination
3. Add tests:
   - Verify exact-allocation path produces identical CSC output to the dynamically-grown path
   - Verify `nnz(L)` matches sum of colcount predictions on bcsstk01, mesh1e1, and a small random SPD test matrix
4. Implement `chol_csc_validate(const CholCsc *L)` — internal sanity-check helper:
   - `col_ptr[0] == 0`, `col_ptr[n] == nnz`
   - Row indices sorted within each column
   - Diagonal entry present and is the first row in each column
   - All row indices `>= col` (lower triangular invariant)
   - Use this from tests and DEBUG asserts (not in release builds)
5. Hardening edge cases:
   - Matrix with structurally zero columns (should still produce valid CSC)
   - Matrix with pre-existing fill in input that exceeds initial allocation (capacity growth path exercised)
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Symbolic-aware CSC allocation path
- `chol_csc_validate()` invariant checker
- Hardened conversion against edge cases
- >= 4 new tests covering symbolic integration

### Completion Criteria
- Exact-allocation path produces structurally identical CSC to dynamic-growth path
- All invariants hold on test matrices
- `nnz(L)` predictions from Sprint 14 colcounts match actual conversion output
- `make format && make lint && make test` clean

---

## Day 4: CSC Cholesky Elimination — Kernel Design & Scaffolding

**Theme:** Design the column-oriented elimination kernel and scaffold it

**Time estimate:** 12 hours

### Tasks
1. Design `chol_csc_eliminate(CholCsc *L)` — left-looking column-by-column Cholesky on CSC arrays:
   - For column `j = 0..n-1`:
     - **cmod(j, k)**: for each previously computed column `k < j` with `L[j,k] != 0`, update column `j`: `L[i,j] -= L[i,k] * L[j,k]` for all `i >= j` with `L[i,k] != 0`
     - **cdiv(j)**: `L[j,j] = sqrt(L[j,j])`, then `L[i,j] /= L[j,j]` for all `i > j`
   - Use a dense workspace vector of length n (pattern from Sprint 10 CSR LU scatter-gather):
     - Scatter column j into dense workspace at start
     - Apply all cmod updates in dense form
     - Gather nonzeros back into CSC at end of column
   - Document the algorithm with citations (George/Liu, "Computer Solution of Large Sparse Positive Definite Systems")
2. Design supporting workspace struct `CholCscWorkspace`:
   - `double *dense_col` (length n) — scatter target
   - `idx_t *dense_pattern` (length n) — list of nonzero indices in current column
   - `int8_t *dense_marker` (length n) — bit-vector "is this row in current column's pattern"
   - `idx_t pattern_count`
3. Implement workspace allocation/free helpers
4. Implement the cdiv step (square root + scaling) as a standalone helper:
   - `chol_csc_cdiv(CholCsc *L, idx_t j)` — operates on the dense workspace before gather
   - Detect non-positive diagonal -> return `SPARSE_ERR_NOT_SPD`
5. Stub out cmod step (Day 5 implements full cmod loop)
6. Initial single-column tests:
   - Diagonal matrix: cdiv only, no cmod -> verify L = sqrt(D)
   - 2x2 SPD matrix: one cmod + one cdiv -> verify exact match with hand calculation
7. Run `make format && make lint && make test` — all clean

### Deliverables
- `chol_csc_eliminate()` skeleton with cdiv implemented
- `CholCscWorkspace` struct and allocator
- Algorithm design documented in code comments
- 2+ initial cdiv tests passing

### Completion Criteria
- cdiv produces correct results on diagonal and trivial 2x2 cases
- Non-SPD detection (negative diagonal) returns the right error code
- `make format && make lint && make test` clean

---

## Day 5: CSC Cholesky Elimination — Full Scatter-Gather Kernel

**Theme:** Implement the cmod inner loop and complete the elimination kernel

**Time estimate:** 12 hours

### Tasks
1. Implement the cmod inner loop in `chol_csc_eliminate()`:
   - Identify which prior columns `k < j` contribute: those with at least one row index `>= j` (i.e., `L[i,k]` for some `i >= j`)
   - For efficiency, precompute for each column the row position of the first row index `>= j` (linear scan within the sorted column)
   - Scatter each contributing column's update into dense workspace, then perform cmod update
   - Use Sprint 10's scatter-gather pattern: scatter -> compute -> gather, with `dense_marker` for fill detection
2. Handle fill-in correctly:
   - When a cmod update introduces a nonzero at row `i` not previously in column `j`'s pattern, append `i` to `dense_pattern` and mark in `dense_marker`
   - On gather, sort `dense_pattern` ascending before writing back to CSC
   - If gather requires more space than the column's pre-allocated slot, trigger `chol_csc_grow()` (or fail if symbolic-allocation path was used and prediction was wrong — assert)
3. Apply drop tolerance (relative to column diagonal magnitude): `if (|val| < SPARSE_DROP_TOL * |L[j,j]|) drop`
   - Match the threshold strategy used by `sparse_cholesky.c` to keep numerical behavior consistent
4. Tests on small SPD matrices:
   - 3x3, 4x4, 5x5 hand-verified Cholesky factorizations
   - Tridiagonal SPD (n = 10)
   - Block-diagonal SPD (two 3x3 blocks)
   - Random SPD via `A = M*M^T + n*I`
5. Verify `L * L^T == A` to tolerance for all cases
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Complete `chol_csc_eliminate()` kernel (cmod + cdiv)
- Fill-in handling with optional capacity growth
- Drop tolerance applied consistently with linked-list path
- 6+ correctness tests on small SPD matrices

### Completion Criteria
- `L * L^T` matches A to `1e-10` relative residual on all test matrices
- Fill-in produces same nnz pattern as Sprint 14 symbolic prediction
- Non-SPD inputs detected and reported
- `make format && make lint && make test` clean

---

## Day 6: CSC Cholesky Solve & SuiteSparse Validation

**Theme:** Implement triangular solves on CSC L and validate against SuiteSparse SPD matrices

**Time estimate:** 12 hours

### Tasks
1. Implement `chol_csc_solve(const CholCsc *L, const double *b, double *x, idx_t n)`:
   - Forward solve: `L*y = b` (column-oriented sweep, since L is in CSC)
   - Backward solve: `L^T*x = y` (row-oriented sweep on CSC = column-oriented on L^T)
   - Operate in place on `x` after copying `b`
   - Document why CSC is well-suited for both sweeps in Cholesky (no transpose needed; L^T solve uses the same column structure)
2. Implement permutation handling:
   - `chol_csc_solve_perm(const CholCsc *L, const idx_t *perm, const double *b, double *x, idx_t n)`
   - Apply `P*b`, solve in permuted system, then unpermute solution
3. Write a public API shim — internal-only for now:
   - `sparse_cholesky_factor_csc()` — full pipeline: convert linked-list -> CSC -> eliminate -> store CSC L for later solve
   - `sparse_cholesky_solve_csc()` — solve using stored CSC L
   - These will become the default backend in Day 12; keep them internal until benchmarks confirm
4. SuiteSparse SPD validation:
   - bcsstk01 (small structural mechanics SPD)
   - mesh1e1 (FEM mesh)
   - bcsstk09 (medium SPD if available)
   - For each: factor via CSC path, solve `Ax = b` for known x, verify `||A*x - b|| / ||b|| < 1e-10`
5. Wire numeric edge cases:
   - Singular matrix (zero diagonal after cmod) -> `SPARSE_ERR_NOT_SPD`
   - Indefinite matrix (negative diagonal after cmod) -> `SPARSE_ERR_NOT_SPD`
   - Very small diagonal (below `SPARSE_TINY_PIVOT`) -> documented behavior
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `chol_csc_solve()` and `chol_csc_solve_perm()`
- Internal public-API shims
- SuiteSparse SPD residual validation tests
- Documented numeric edge case behavior

### Completion Criteria
- Solve residuals `< 1e-10` on all SuiteSparse SPD test matrices
- Indefinite/singular detection returns correct error codes
- `make format && make lint && make test` clean

---

## Day 7: CSC LDL^T — Design & Kernel Adaptation

**Theme:** Design the CSC LDL^T storage and adapt the Cholesky kernel for symmetric indefinite factorization

**Time estimate:** 10 hours

### Tasks
1. Design the `LdltCsc` storage:
   - `CholCsc L` — same column storage as Cholesky (unit lower triangular this time, so diagonal is 1 and not stored — or stored as-is for uniformity, document the choice)
   - `double *D` — diagonal entries (length n, but for 2x2 pivots two consecutive entries form a 2x2 block)
   - `idx_t *pivot_size` — length n, each entry is 1 (1x1 pivot) or 2 (start of 2x2 pivot)
   - `idx_t *perm` — Bunch-Kaufman pivot permutation (separate from fill-reducing permutation)
   - Reuse `sparse_ldlt_t`'s pivot conventions from `src/sparse_ldlt.c` so the public API stays unchanged
2. Add `LdltCsc` struct + allocation helpers in `src/sparse_ldlt_csc_internal.h` (new)
3. Add `src/sparse_ldlt_csc.c` (new) and `tests/test_ldlt_csc.c` (new) to Makefile and CMakeLists.txt
4. Design the elimination kernel adaptation:
   - Same cmod/cdiv structure as Cholesky, but cdiv writes to `D` (no sqrt) and L gets unit diagonal
   - cmod uses `D[k]` as the divisor: `L[i,j] -= L[i,k] * D[k] * L[j,k]`
   - For 2x2 pivot blocks: cmod uses 2x2 block matrix multiply; cdiv inverts the 2x2 block via factored LDL of the 2x2
5. Implement `ldlt_csc_from_sparse()` — analogous to `chol_csc_from_sparse()`, with extra allocation for `D` and `pivot_size`
6. Round-trip and validation tests:
   - Identity matrix
   - Diagonal indefinite matrix (negative entries)
   - Conversion preserves all values
7. Run `make format && make lint && make test` — all clean

### Deliverables
- `LdltCsc` struct and internal header
- Conversion routines
- Build system updated
- 3+ round-trip tests

### Completion Criteria
- Library builds with new files
- Round-trip conversion preserves D, L, and pivot_size
- `make format && make lint && make test` clean

---

## Day 8: CSC LDL^T — Bunch-Kaufman Pivoting in CSC

**Theme:** Implement symmetric pivoting and 1x1/2x2 block elimination in CSC

**Time estimate:** 10 hours

### Tasks
1. Implement `ldlt_csc_eliminate(LdltCsc *F)`:
   - Column-by-column with pivot selection at each column j:
     - Bunch-Kaufman criterion: compare `|L[j,j]|` against the largest off-diagonal magnitude in column j (after cmod into dense workspace)
     - If `|L[j,j]| >= alpha * max_off_diag` (alpha = (1+sqrt(17))/8 ≈ 0.6404), use 1x1 pivot
     - Otherwise consider 2x2 block pivot with column k that has the largest off-diagonal entry; verify 2x2 pivot stability
   - For 1x1 pivot: cdiv writes `D[j] = L[j,j]`, then `L[i,j] /= D[j]`, then `L[j,j] = 1`
   - For 2x2 pivot at columns j,j+1: factor the 2x2 block, store inverse in `D[j..j+1]`, scale columns j and j+1 by the inverse 2x2
2. Symmetric permutation handling:
   - When 2x2 pivot swaps column k with column j+1, swap rows AND columns symmetrically
   - In CSC this requires touching both column k and any column with row index k (similar to LU pivoting in CSR but symmetric)
   - Maintain `perm` array tracking the cumulative permutation
3. Mirror the existing Bunch-Kaufman logic in `src/sparse_ldlt.c` to keep numerical behavior identical (port the inner pivot-selection function rather than reimplement)
4. Tests on small symmetric indefinite matrices:
   - Pure 1x1 pivot case (well-conditioned diagonal)
   - Forced 2x2 pivot case (zero diagonal at first column)
   - Mixed 1x1 and 2x2 pivots
   - Compare factorization output element-by-element with the linked-list `sparse_ldlt_factor()` on the same input (must produce identical D, pivot_size, and L up to ordering)
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `ldlt_csc_eliminate()` with Bunch-Kaufman 1x1 and 2x2 pivoting
- Symmetric permutation maintenance
- 4+ tests comparing CSC vs linked-list LDL^T output

### Completion Criteria
- CSC LDL^T produces identical (or numerically equivalent) factorization to linked-list path on hand-crafted test matrices
- 2x2 pivots correctly chosen and applied
- `make format && make lint && make test` clean

---

## Day 9: CSC LDL^T Solve & SuiteSparse Indefinite Validation

**Theme:** Implement LDL^T triangular/block solves and validate against symmetric indefinite SuiteSparse matrices

**Time estimate:** 8 hours

### Tasks
1. Implement `ldlt_csc_solve(const LdltCsc *F, const double *b, double *x, idx_t n)`:
   - Apply pivot permutation to b
   - Forward solve `L*y = P*b` (unit lower triangular)
   - Diagonal solve `D*z = y`:
     - For 1x1 blocks: `z[i] = y[i] / D[i]`
     - For 2x2 blocks at i,i+1: solve the 2x2 system using stored inverse
   - Backward solve `L^T*w = z`
   - Apply inverse permutation to get x
2. Edge cases:
   - Zero diagonal in 1x1 pivot -> `SPARSE_ERR_SINGULAR`
   - Singular 2x2 block -> `SPARSE_ERR_SINGULAR`
3. SuiteSparse symmetric indefinite validation:
   - Pick 2-3 small symmetric indefinite matrices from existing test corpus (e.g., reuse fixtures from `tests/test_ldlt.c`)
   - Compare residual `||A*x - b|| / ||b||` between CSC and linked-list LDL^T paths
   - Verify they match to tolerance (`< 1e-10`)
4. Inertia consistency check:
   - The number of positive, negative, and zero eigenvalues (Sylvester's law of inertia from `D`) must match between the two paths on each test matrix
5. Run `make format && make lint && make test` — all clean

### Deliverables
- `ldlt_csc_solve()` with full pivot/diagonal/triangular handling
- SuiteSparse symmetric indefinite residual tests
- Inertia consistency tests

### Completion Criteria
- Solve residuals `< 1e-10` on all test matrices
- Inertia matches linked-list path on every test matrix
- `make format && make lint && make test` clean

---

## Day 10: Supernodal Detection — Column Structure Analysis

**Theme:** Identify supernodes (groups of columns with identical nonzero structure below the diagonal)

**Time estimate:** 12 hours

### Tasks
1. Design the supernode detection algorithm:
   - A supernode is a contiguous group of columns `j, j+1, ..., j+s-1` such that the nonzero pattern of column `j+i` (for `i = 0..s-1`) is exactly `{j, j+1, ..., j+s-1} ∪ S` for some shared set `S` of rows below the supernode
   - Use the fundamental supernode definition (Liu, Ng, Peyton): consecutive columns with identical structure below the supernode block
   - Detection uses Sprint 14's etree and column counts: a supernode boundary occurs when colcount changes or the etree branches
2. Implement `chol_csc_detect_supernodes(const CholCsc *L_symbolic, idx_t *super_starts, idx_t *super_count, idx_t min_size)`:
   - Input: symbolic structure of L (from analysis or from CSC itself)
   - Output: array of supernode start columns; total count of supernodes
   - `min_size` parameter (default 4): only group columns into a supernode if size >= min_size, else treat as 1x1 columns (matches Sprint 10's `lu_detect_dense_blocks` threshold)
3. Reuse the dense block detection patterns from `src/sparse_lu_csr.c` (`lu_detect_dense_blocks`) where applicable; adapt for the column-oriented (rather than diagonal-block) supernodal structure
4. Tests:
   - Diagonal matrix: every column is its own (size-1) "supernode"; with `min_size = 4`, no supernodes detected
   - Dense matrix: a single supernode covering all columns
   - Block-diagonal matrix with two 5x5 dense blocks: two supernodes
   - Tridiagonal SPD: no supernodes (each column has different structure)
   - Real SuiteSparse SPD (bcsstk01): inspect detected supernodes manually, document
5. Add a tiny printf-based debug helper (compiled out in release) to dump supernode partitions for visual inspection during development
6. Run `make format && make lint && make test` — all clean

### Deliverables
- `chol_csc_detect_supernodes()` implementation
- 5+ tests covering canonical structures
- Debug visualization helper (DEBUG-only)

### Completion Criteria
- Detected supernodes correct on canonical structures
- On bcsstk01, detected supernode count is reasonable (matches expected from CHOLMOD-style analysis to within a small margin)
- `make format && make lint && make test` clean

---

## Day 11: Supernodal Dense Kernels & Integration

**Theme:** Wire dense block kernels into the CSC Cholesky elimination via supernodes

**Time estimate:** 12 hours

### Tasks
1. Implement supernodal cmod/cdiv:
   - For a supernode of size `s` starting at column `j`, factor the diagonal `s x s` block in dense form (Cholesky on a small dense matrix)
   - Below the diagonal block, apply the dense triangular solve to update the off-diagonal rows: `L_below /= L_diag^T`
   - These two steps replace `s` separate cdiv calls with one dense factor + one dense triangular solve
   - For cmod across supernodes: treat the contributing supernode as a dense block multiplier (`L_target -= L_contrib * D_block * L_contrib^T` for the supernodal panel)
2. Reuse Sprint 10 dense kernels where possible:
   - `lu_dense_factor()` for the diagonal block (or a Cholesky variant `chol_dense_factor()` if needed)
   - Add `chol_dense_factor()` if `lu_dense_factor()` doesn't fit (Cholesky needs sqrt; LU does not)
   - Add `chol_dense_solve()` for the triangular solve
3. Modify `chol_csc_eliminate()` to use the supernodal path when supernodes are detected:
   - Detect supernodes once at the start
   - For each supernode, call `chol_csc_supernode_eliminate(L, super_start, super_size, workspace)`
   - For non-supernodal columns (size 1 or below `min_size`), use the existing column-by-column path
4. Tests:
   - Dense 8x8 SPD matrix: should detect a single supernode and factor with dense kernel; verify `L*L^T = A`
   - Block-diagonal SPD with 5x5 blocks: each block is its own supernode; verify factorization correctness
   - SuiteSparse bcsstk01: factor with and without supernodal kernel; verify identical residuals
5. Apply same supernodal optimization to `ldlt_csc_eliminate()`:
   - Adapt for 1x1/2x2 pivots within supernodes (a 2x2 pivot inside a supernode is straightforward; a 2x2 pivot crossing a supernode boundary forces a supernode split)
   - Conservative initial implementation: disable supernodal kernel when 2x2 pivot is selected; use scalar path
6. Run `make format && make lint && make test` — all clean

### Deliverables
- Supernodal cmod/cdiv kernels
- `chol_dense_factor()` and `chol_dense_solve()` helpers
- Supernodal path integrated into `chol_csc_eliminate()`
- Conservative supernodal support in `ldlt_csc_eliminate()`
- 5+ correctness tests

### Completion Criteria
- Dense and block-diagonal matrices factored correctly via supernodal kernel
- Residuals match scalar path on SuiteSparse SPD matrices
- `make format && make lint && make test` clean

---

## Day 12: Benchmarks — CSC vs Linked-List Comparison

**Theme:** Benchmark CSC Cholesky and LDL^T against the linked-list path on SuiteSparse matrices

**Time estimate:** 10 hours

### Tasks
1. Add `benchmarks/bench_chol_csc.c`:
   - For each SuiteSparse SPD matrix in the existing benchmark corpus: factor via linked-list, factor via CSC (with and without supernodal), measure factor time and solve time
   - Report speedup ratio (linked-list / CSC) and (linked-list / CSC supernodal)
   - Verify residuals match between paths (`||A*x - b|| / ||b||`)
   - Output CSV-ready format for inclusion in README performance table
2. Add `benchmarks/bench_ldlt_csc.c` with the analogous structure for symmetric indefinite matrices
3. Update Makefile target `make bench` to include the new benchmarks
4. Run benchmarks and capture results:
   - Target: >= 2x speedup over linked-list path on at least the larger SuiteSparse matrices (small matrices may show no speedup or even slowdown due to conversion overhead)
   - If a matrix shows < 2x speedup, investigate (profile with `perf` or instrument time per phase: convert vs eliminate vs solve)
5. Wire CSC as the default backend (with size threshold) in `sparse_cholesky_factor()` and `sparse_ldlt_factor()`:
   - For matrices with `n >= SPARSE_CSC_THRESHOLD` (default: 100, tunable), dispatch to CSC path
   - For smaller matrices, keep linked-list path (avoid conversion overhead)
   - Add an `opts` field to override the default for benchmarking and testing
6. Document the threshold choice in code comments referring to benchmark data
7. Begin documentation pass (2 hrs):
   - Draft README performance section update with new speedup numbers
8. Run `make format && make lint && make test && make bench` — all clean

### Deliverables
- `benchmarks/bench_chol_csc.c` and `benchmarks/bench_ldlt_csc.c`
- CSV-formatted speedup report
- Default-backend dispatch with size threshold
- README performance section draft

### Completion Criteria
- >= 2x speedup measured on SPD matrices in the SuiteSparse corpus (excluding very small matrices)
- Residuals match between paths to tolerance
- Default dispatch works transparently — existing tests unaffected
- `make format && make lint && make test && make bench` clean

---

## Day 13: Benchmark Validation, Tuning & Documentation

**Theme:** Finalize benchmarks, validate residuals at scale, and write user-facing documentation

**Time estimate:** 10 hours

### Tasks
1. Comprehensive validation pass (8 hrs):
   - Run full test suite under ASan and UBSan: `make sanitize`
   - Run benchmarks under `make bench` and capture final numbers
   - For each SuiteSparse SPD/indefinite matrix in the corpus:
     - Factor via CSC, factor via linked-list, compare residuals (`< 1e-10` agreement)
     - Verify inertia matches for indefinite matrices
     - Verify supernodal vs scalar path produce equivalent results
   - Tune `SPARSE_CSC_THRESHOLD` based on measured crossover point (the matrix size where CSC overtakes linked-list)
   - If any matrix shows worse residual or wrong inertia in CSC, debug and fix
2. Documentation update (2 hrs):
   - Update README:
     - Add CSC Cholesky/LDL^T to feature list
     - Update performance section with new speedup numbers
     - Note the size threshold for default dispatch
   - Update `docs/algorithm.md` (or equivalent):
     - Add section on CSC working format for Cholesky
     - Add section on Bunch-Kaufman pivoting in CSC
     - Add section on supernodal detection and dense kernels
   - Add new public-API entries (if any) to API docs
3. Run `make format && make lint && make test && make sanitize` — all clean

### Deliverables
- ASan/UBSan-clean test suite with new code paths
- Final benchmark numbers
- Tuned `SPARSE_CSC_THRESHOLD`
- Updated README and algorithm docs

### Completion Criteria
- All tests pass under ASan and UBSan
- Documentation is accurate and reflects the new backend
- Final benchmark report shows >= 2x speedup target met
- `make format && make lint && make test && make sanitize` clean

---

## Day 14: Documentation Wrap-Up, Sprint Review & Retrospective

**Theme:** Final documentation, metrics collection, sprint retrospective

**Time estimate:** 12 hours

### Tasks
1. Documentation finalization (8 hrs):
   - Comprehensive Doxygen pass on `chol_csc_*` and `ldlt_csc_*` API (even though most are internal, document for future maintainers)
   - Add header-level design comment to `src/sparse_chol_csc.c` and `src/sparse_ldlt_csc.c` explaining:
     - Why CSC over linked-list for the numeric kernel
     - The cdiv/cmod algorithm in plain English with a small worked example
     - The supernodal extension and when it kicks in
     - The role of Sprint 14 symbolic analysis in pre-allocating CSC capacity
   - Update `docs/planning/EPIC_2/PROJECT_PLAN.md` Sprint 17 section: mark items complete or note any deferrals
   - Update API function count in README
   - Cross-link from `sparse_lu_csr.h` design comment to the new CSC headers (showing the parallel)
2. Final metrics collection (1 hr):
   - Total test count (target: ~1300+ from current ~1250)
   - CSC Cholesky test count
   - CSC LDL^T test count
   - Supernodal detection test count
   - Benchmark numbers (CSC vs linked-list speedup per matrix)
3. Full regression (1 hr):
   - `make clean && make format && make lint && make test`
   - `make sanitize`
   - `make examples` and run each
   - `make bench`
4. Write `docs/planning/EPIC_2/SPRINT_17/RETROSPECTIVE.md` (2 hrs):
   - Definition of Done checklist (all 6 items from PROJECT_PLAN.md Sprint 17)
   - What went well (e.g., reuse of Sprint 10 patterns, Sprint 14 symbolic infrastructure)
   - What didn't (e.g., which speedups missed the 2x target, what supernodal cases were deferred)
   - Final metrics
   - Items deferred (if any) — with rationale and link to follow-up sprint
   - Lessons for future "linked-list -> compressed" refactors

### Deliverables
- Sprint 17 retrospective document
- Final updated metrics in README
- Clean final regression build
- All Sprint 17 items complete or explicitly deferred

### Completion Criteria
- All Sprint 17 items from PROJECT_PLAN.md addressed
- Retrospective written with honest assessment of what hit and missed targets
- `make format && make lint && make test && make sanitize && make bench` clean
- README, algorithm docs, and project plan accurately reflect the new backend
