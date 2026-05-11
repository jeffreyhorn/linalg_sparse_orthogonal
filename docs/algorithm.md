# Algorithm Description

## Orthogonal Linked-List Data Structure

The library stores sparse matrices in an **orthogonal linked-list** (also called a cross-linked list) representation. Each non-zero entry is a `Node` containing:

```
struct Node {
    idx_t   row, col;     // physical position
    double  value;
    Node   *right;        // next non-zero in this row (by column)
    Node   *down;         // next non-zero in this column (by row)
};
```

Each node belongs to two sorted linked lists simultaneously:
- A **row list** linking all non-zeros in the same row, sorted by column index
- A **column list** linking all non-zeros in the same column, sorted by row index

The matrix struct maintains header arrays `row_headers[0..m-1]` and `col_headers[0..n-1]` pointing to the first node in each row/column list.

### Advantages

- **Efficient row and column traversal** — both are O(nnz_row) or O(nnz_col)
- **Efficient element insertion and deletion** — O(nnz_row + nnz_col) worst case
- **Natural for LU factorization** — elimination needs both row and column access
- **No reallocation on fill-in** — new nodes are allocated from the pool

### Trade-offs

- Higher per-element overhead than CSR/CSC (two pointers per node vs. one index)
- Pointer chasing limits cache performance for SpMV compared to compressed formats
- Not suitable for extremely large matrices where CSR/CSC memory density matters

## Memory Management: Slab Pool Allocator

Nodes are allocated from a slab pool with a free-list:

1. **Slabs**: Fixed-size arrays of `NODES_PER_SLAB` (default 4096) nodes, allocated via `malloc` and chained together.

2. **Allocation**: `pool_alloc()` first checks the free-list; if empty, it bumps the `used` counter in the current slab; if the slab is full, a new slab is allocated.

3. **Deallocation**: `pool_release()` pushes the node onto the free-list (using the `right` pointer as the free-list link). The node is not zeroed — it is simply reused on the next allocation.

4. **Bulk free**: `pool_free_all()` walks the slab chain and frees each slab. Called by `sparse_free()`.

This design avoids per-node `malloc`/`free` calls, which is critical during LU factorization where thousands of nodes may be created and destroyed.

## Permutation Arrays

The matrix maintains four permutation arrays of length n (for square matrices):

| Array | Mapping | Purpose |
|-------|---------|---------|
| `row_perm[i]` | logical row i → physical row | Encodes P in P·A·Q = L·U |
| `inv_row_perm[j]` | physical row j → logical row | Inverse of row_perm |
| `col_perm[i]` | logical col i → physical col | Encodes Q in P·A·Q = L·U |
| `inv_col_perm[j]` | physical col j → logical col | Inverse of col_perm |

Invariant: `row_perm[inv_row_perm[j]] == j` and `inv_row_perm[row_perm[i]] == i` for all i, j.

Initially all permutations are the identity. During LU factorization, pivoting swaps entries in these arrays.

## LU Factorization Algorithm

The factorization computes P·A·Q = L·U where:
- P is a row permutation (always applied)
- Q is a column permutation (identity for partial pivoting, non-trivial for complete pivoting)
- L is unit lower triangular (1s on diagonal, stored below diagonal)
- U is upper triangular (stored on and above diagonal)

### Pseudocode

```
for k = 0 to n-1:
    // 1. Pivot selection
    if COMPLETE pivoting:
        find (p, q) = argmax |A(i, j)| for i >= k, j >= k  (in logical space)
        swap logical rows k ↔ p
        swap logical cols k ↔ q
    else (PARTIAL pivoting):
        find p = argmax |A(i, k)| for i >= k  (in logical space, column k only)
        swap logical rows k ↔ p

    // 2. Check pivot
    pivot = A(k, k)   // logical (k,k) = physical (row_perm[k], col_perm[k])
    if |pivot| < tol:
        return SINGULAR

    // 3. Snapshot: collect physical row indices that need elimination
    snapshot = [physical rows in column col_perm[k] with logical row > k]

    // 4. Elimination
    for each physical row pr in snapshot:
        logical_row = inv_row_perm[pr]
        multiplier = A(logical_row, k) / pivot
        A(logical_row, k) = multiplier       // store L entry

        // Subtract multiplier * (pivot row) from this row
        for each entry (k, j) in logical pivot row where j > k:
            A(logical_row, j) -= multiplier * A(k, j)
            // Drop if |value| < DROP_TOL * |pivot|
```

### Snapshot Mechanism (Bug 3.1 Fix)

The elimination loop modifies entries in the pivot column, which changes the column list being traversed. To avoid corrupted traversal, the algorithm **snapshots** all physical row indices that need elimination before the loop begins. This is stored in a temporary array allocated once and reused for each pivot step.

### Forward Substitution (Bug 3.3 Fix)

Forward substitution solves L·y = b where L is unit lower triangular:

```
for i = 0 to n-1:
    y[i] = b[i]
    // Walk entire row i, accumulating L entries where col < i
    for each entry (i, j) in logical row i:
        if j < i:
            y[i] -= L(i, j) * y[j]
```

The key fix: the row is traversed completely without early termination. In the orthogonal list, entries are stored in physical column order, which may not correspond to logical column order after pivoting. An early `break` on `j >= i` would miss L entries stored after U entries in the physical list.

## Solve Procedure

Given the factored matrix (containing L and U) with permutations P and Q:

```
Solve A*x = b:
    1. pb = P * b              // pb[i] = b[row_perm[i]]
    2. Solve L * y = pb        // forward substitution
    3. Solve U * z = y         // backward substitution
    4. x = Q * z               // x[i] = z[inv_col_perm[i]]
```

Step 4 uses `inv_col_perm` (not `col_perm`) because Q maps logical to physical columns, and we need the inverse to go from the factored column space back to the original.

## Iterative Refinement

Given the original matrix A, the LU factorization, and an initial solution x:

```
for iter = 1 to max_iters:
    r = b - A * x              // residual (using original A, not LU)
    if ||r|| / ||b|| < tol:
        break
    Solve A * d = r using LU   // correction
    x = x + d                  // update
```

This exploits the fact that the LU factorization can be reused for multiple solves. The SpMV with the original matrix provides a more accurate residual than working with the L and U factors.

## Complexity Analysis

### Space

| Component | Cost |
|-----------|------|
| Node storage | 2 × sizeof(pointer) + sizeof(double) + 2 × sizeof(idx_t) per non-zero ≈ 32 bytes/nnz |
| Row/column headers | 2 × n × sizeof(pointer) |
| Permutation arrays | 4 × n × sizeof(idx_t) |
| Pool slabs | Rounded up to NODES_PER_SLAB granularity |

### Time

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Insert/remove | O(nnz_row + nnz_col) | List traversal to find position |
| Get (physical) | O(nnz_row) | Row list scan |
| SpMV (y = A*x) | O(nnz) | One pass over all entries |
| LU factor (partial) | O(nnz × n) worst case | Depends on fill-in; O(nnz) for banded |
| LU factor (complete) | O(n^2 × nnz) worst case | Submatrix search at each step |
| Forward/backward sub | O(nnz_LU) | Traversal of L or U entries |
| Iterative refinement | O(k × (nnz + nnz_LU)) | k iterations, one SpMV + one solve each |

### Fill-in Behavior

Fill-in during LU factorization depends heavily on matrix structure and pivoting:

| Matrix type | Partial pivoting fill-in | Complete pivoting fill-in |
|-------------|-------------------------|--------------------------|
| Tridiagonal | 1.0x (no fill-in) | ~1.6x |
| Pentadiagonal | ~1.5x | ~2-3x |
| Arrow | Catastrophic (→ 100% dense) | Catastrophic |
| Random sparse (k=5/row) | ~2-5x | ~3-8x |

Fill-reducing reordering (AMD, RCM) is available — see the **Fill-Reducing Reordering** section below.

## Drop Tolerance

During elimination, new fill-in entries with |value| < DROP_TOL × |pivot| are dropped (not inserted). This controls memory growth at the expense of factorization accuracy. The default DROP_TOL of 1e-14 is conservative — it drops only entries that are negligible relative to the pivot, preserving near-exact arithmetic.

For matrices where fill-in is a concern, a larger drop tolerance (e.g., 1e-8) can significantly reduce memory usage while maintaining acceptable accuracy, especially when followed by iterative refinement.

## Tolerance Strategy

All numerical tolerance checks across the library use the `sparse_rel_tol()` helper (defined in `sparse_matrix_internal.h`), which computes:

```
threshold = max(user_tol × reference_norm, DBL_MIN × 100)
```

The absolute floor (`DBL_MIN × 100 ≈ 2.2e-306`) prevents underflow when the reference norm is itself tiny, while the relative branch scales correctly for matrices at any magnitude.

### Tolerance categories

Tolerance sites fall into three categories:

**Category A — Singularity detection** (LU, Cholesky, ILU, ILUT, CSR-LU):
Uses `sparse_rel_tol(factor_norm, DROP_TOL)` where `factor_norm` is `||A||_inf` cached at factorization time. For Cholesky, the reference is `sqrt(||A||_inf)` since L entries scale as the square root of A entries.

**Category B — Convergence / deflation** (bidiag SVD, tridiag QR, QR solve):
Uses `sparse_rel_tol(local_norm, tol)` where `local_norm` is a locally computed norm (e.g., bidiagonal norm, |R(0,0)|). These checks determine when off-diagonal entries are negligible enough to deflate.

**Category C — Lucky breakdown / normalization guards** (GMRES, Lanczos):
Uses `sparse_rel_tol(0, DROP_TOL)` — the pure absolute floor. These guard against division by zero in iterative contexts where an invariant subspace has been found; the threshold must not depend on matrix scale.

### Why relative?

An absolute threshold fails for two classes of matrices:

1. **Uniformly small entries** — e.g., `A = 1e-40 × I`. Every diagonal is `1e-40`, which any fixed threshold above `1e-40` would falsely reject as singular, even though the matrix is perfectly conditioned (κ = 1).

2. **Very large entries** — e.g., `||A||_inf = 1e15`. A fixed threshold of `1e-14` allows near-zero pivots like `1e-10` to pass, even though they represent catastrophic loss of significance relative to the matrix scale.

The relative threshold `DROP_TOL × ||A||_inf` scales with the matrix:
- A `1e-40 × I` matrix has threshold `1e-14 × 1e-40 = 1e-54`, correctly allowing `1e-40` pivots.
- A matrix with `||A||_inf = 1e15` has threshold `1e-14 × 1e15 = 10`, correctly rejecting pivots smaller than 10.

### Implementation

Each factorization function computes and caches `||A||_inf` before modifying the matrix:

| Solver      | Where cached          | Reference norm for solve        |
|-------------|-----------------------|---------------------------------|
| LU          | `mat->factor_norm`    | `factor_norm`                   |
| Cholesky    | `mat->factor_norm`    | `sqrt(factor_norm)`             |
| ILU / ILUT  | `ilu->factor_norm`    | `factor_norm`                   |
| CSR LU      | `csr->factor_norm`    | `factor_norm`                   |
| QR          | `|R(0,0)|` (computed) | `|R(0,0)|`                      |
| SVD         | bidiag norm (local)   | bidiag norm                     |

### QR R-extraction dropping

During QR factorization, R entries are dropped if |R(i,j)| < DROP_TOL × |R(i,i)|. This is relative to the diagonal of the current row, ensuring tiny-scale matrices retain all structurally significant entries.

## Cholesky Factorization

For symmetric positive-definite (SPD) matrices, Cholesky factorization computes A = L·L^T where L is lower triangular. This exploits symmetry for approximately half the storage and computation of LU.

### Algorithm (left-looking, column-by-column)

```
for k = 0 to n-1:
    L(k,k) = sqrt( A(k,k) - sum_{j<k} L(k,j)^2 )
    if L(k,k) <= 0: return NOT_SPD

    for each i > k with nonzero A(i,k) or fill-in:
        L(i,k) = ( A(i,k) - sum_{j<k} L(i,j)*L(k,j) ) / L(k,k)
```

A dense column accumulator is used for each column k to efficiently handle fill-in. After accumulating contributions from previous columns, nonzero entries are written back to the sparse structure.

### Advantages over LU for SPD matrices

- **No pivoting needed** — SPD guarantees all pivots are positive
- **Half the storage** — only L is stored (not both L and U)
- **Half the computation** — symmetry halves the work
- **Better numerical stability** — no need for pivot search or permutation

### Complexity

| Operation | Complexity |
|-----------|-----------|
| Cholesky factor | O(nnz_L × avg_col_nnz) |
| Cholesky solve | O(nnz_L) forward + O(nnz_L) backward |

### Fill-in comparison (SuiteSparse, no reordering)

| Matrix | LU nnz | Cholesky nnz | Savings |
|--------|-------:|---------:|---------|
| nos4 (100×100) | 1510 | 805 | 47% |
| bcsstk04 (132×132) | 8581 | 3664 | 57% |

## CSC Numeric Backend for Cholesky (Sprint 17)

The CSC Cholesky path (`src/sparse_chol_csc.c`) re-implements Cholesky's
inner loop on contiguous column arrays, mirroring the Sprint 10 CSR LU
working-format strategy for SPD matrices.

### Data layout

`CholCsc` stores the lower triangle of `L` (including the diagonal) as
three arrays:

- `col_ptr[0..n]` — column pointers (monotone; `col_ptr[n] == nnz`)
- `row_idx[0..nnz-1]` — per-column row indices, sorted ascending with
  the diagonal first
- `values[0..nnz-1]` — numeric values

Capacity can exceed `nnz` so fill-in produced during elimination absorbs
into the same packed storage without reallocation, up to the
symbolic-analysis prediction (`sym_L.nnz` — see *Symbolic Analysis*).

### Algorithm: left-looking column sweep

For each column `j = 0 .. n-1`:

1. **Scatter** `A[*, j]`'s stored values into a dense row accumulator
   `dense_col[0..n-1]`, tracking touched rows in `dense_pattern`.
2. **cmod** — apply Schur-complement contributions from every prior
   column `k < j` with `L[j, k] != 0`:  binary-search for `L[j, k]` in
   column `k`'s sorted row indices; when present, subtract
   `L[i, k] · L[j, k]` from `dense_col[i]` for every stored
   `L[i, k]` with `i >= j`.
3. **cdiv** — take the square root of `dense_col[j]` (returning
   `SPARSE_ERR_NOT_SPD` on non-positive diagonals) and scale every
   remaining pattern entry by the inverse.
4. **Gather** — sort the pattern, apply a relative drop tolerance
   (`|v| < SPARSE_DROP_TOL · |L[j,j]|`, diagonal always kept), and
   write surviving entries into column `j`'s CSC slot.  If the slot
   needs to grow or shrink, trailing columns are shifted via
   `memmove` and total capacity grows geometrically.

References: George & Liu 1981 (dense-accumulator left-looking
Cholesky), Davis 2006 §4.4 (CSC storage for sparse direct methods).

### Why CSC (not CSR)

Both Cholesky sweeps — forward `L · y = b` and backward `L^T · x = y`
— iterate columns once each.  The backward sweep uses the same
column slice as the forward sweep because the below-diagonal entries
of column `j` of `L` are exactly row `j` of `L^T` (no materialised
transpose needed).  CSR would require a transpose for one of the
sweeps; CSC fits both without reformatting.

### Performance

Measured on SuiteSparse SPD fixtures (3-repeat average, one-shot
factor with AMD reorder inside the timed region on all paths):

| Matrix        |    n  |    nnz(A) | factor_ll | factor_csc | factor_csc_sn | speedup (scalar / sn) |
|---------------|------:|----------:|----------:|-----------:|--------------:|----------------------:|
| nos4          |   100 |       594 |    0.46 ms |    0.42 ms |       0.38 ms | **1.09× / 1.22×** |
| bcsstk04      |   132 |     3,648 |    3.12 ms |    2.67 ms |       3.09 ms | **1.16× / 1.01×** |
| bcsstk14      | 1,806 |    63,454 |  364.29 ms |  208.82 ms |     152.83 ms | **1.74× / 2.38×** |
| s3rmt3m3      | 5,357 |   207,123 | 4018.41 ms | 1914.53 ms |    1179.41 ms | **2.10× / 3.41×** |
| Kuu           | 7,102 |   340,200 | 3147.78 ms | 4112.76 ms |    1416.64 ms |   0.77× / **2.22×** |
| Pres_Poisson  |14,822 |   715,804 |46003.69 ms|17597.98 ms |   10580.68 ms | **2.61× / 4.35×** |

Residuals `||A·x − b||_∞ / ||b||_∞` match the linked-list path to
within double-precision round-off on every matrix above.

Three takeaways from the scaling corpus (Sprint 18 Day 12):

- **Scalar-CSC speedup scales with n.**  The ratio climbs from 1.09×
  at n = 100 to 2.61× at n = 14 822, matching the Sprint 17
  hypothesis that linked-list pointer-chasing overhead grows faster
  than CSC column traversal.
- **Supernodal beats scalar on every non-trivial matrix.**  The
  batched Days 6-10 kernel pulls ahead by another 1.2–2.9× on top
  of scalar CSC on the four new fixtures.  The only exception is
  bcsstk04 (supernodal 1.01× vs scalar 1.16×) where the matrix is
  small enough that supernode-detection overhead eats the dense-
  block win.
- **Kuu scalar regression is real and localised.**  The scalar
  kernel's repeated `shift_columns_right_of` calls during drop-
  tolerance pruning cost it the round on Kuu (0.77× vs linked-list);
  the supernodal path pre-allocates the full sym_L pattern and
  avoids those shifts, landing 2.22× ahead.

`SPARSE_CSC_THRESHOLD` (default `100` in
`include/sparse_matrix.h`) determines where
`sparse_cholesky_factor_opts` switches the linked-list path over to
the CSC supernodal kernel.  These are one-shot numbers: in the
analyze-once / factor-many workflow (`sparse_analyze` once →
`sparse_factor_numeric` to prime the `sparse_factors_t` →
repeated `sparse_refactor_numeric` for each new value pattern)
the AMD cost amortizes and the CSC
kernel's advantage is dramatically larger — measured 2.4× at n = 132
climbing to 24.3× at n ≈ 15 k on the same corpus (Sprint 19 Day 1-2,
captured by `benchmarks/bench_refactor_csc.c`).  See
[`docs/planning/EPIC_2/SPRINT_17/PERF_NOTES.md`](planning/EPIC_2/SPRINT_17/PERF_NOTES.md)
for the full 12-column CSV, both workflow tables, the hypothesis-
check analysis, and reproduction instructions; raw captures in
[`docs/planning/EPIC_2/SPRINT_18/bench_day12.txt`](planning/EPIC_2/SPRINT_18/bench_day12.txt)
(one-shot) and
[`docs/planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt`](planning/EPIC_2/SPRINT_19/bench_day2_refactor.txt)
(analyze-once).

## Supernodal Detection and Batched Kernel (Sprint 17 + Sprint 18 Days 6-9)

A *fundamental supernode* is a contiguous run of columns
`j, j+1, …, j+s-1` that share the same below-diagonal nonzero pattern.
`chol_csc_detect_supernodes` extracts them directly from the CSC L with
a pairwise check (Liu, Ng, Peyton, SIMAX 1993):

1. Column `j+1`'s first stored sub-diagonal row in column `j` is
   exactly `j+1` (so `j+1` is the immediate etree parent of `j`).
2. Column `j+1` has one fewer entry than column `j`.
3. The remaining row indices of `j+1` match `j`'s rows starting at
   position `col_ptr[j] + 2`, element-for-element.

Three O(nnz(column)) conditions on the sorted CSC columns; no
separate etree materialisation required.

The supernode-aware entry point
`chol_csc_eliminate_supernodal(L, min_size)` runs a fully integrated
batched path on every detected supernode (Sprint 18 Days 6-8):

- `chol_csc_supernode_extract` scatters the supernode's CSC columns
  into a dense column-major buffer, building a `row_map` that records
  each local row's global CSC index.
- `chol_csc_supernode_eliminate_diag` applies left-looking external
  cmod from every prior column `k < s_start` and then calls
  `chol_dense_factor` on the `s_size × s_size` diagonal slab.
- `chol_csc_supernode_eliminate_panel` applies the panel triangular
  solve via `chol_dense_solve_lower` one panel row at a time (in
  column-major storage each row lives at stride `lda_panel`, so the
  helper gathers into a contiguous scratch vector, forward-
  substitutes, and scatters the result back).
- `chol_csc_supernode_writeback` gathers the factored dense buffer
  back into the CSC at the supernode's original row positions.

Columns not inside any detected supernode fall back to the scalar
scatter / cmod / cdiv / gather loop in the same `while (j < n)`
dispatch so mixed-structure matrices factor correctly end-to-end.
The companion dense kernels `chol_dense_factor(A, n, lda, tol)` and
`chol_dense_solve_lower(L, n, lda, b)` operate on column-major
`n × n` blocks and return `SPARSE_ERR_NOT_SPD` on non-positive pivots;
the error propagates out of the batched kernel unchanged.

Day 9's parametrised cross-check verifies scalar == batched
byte-for-byte on nos4 and bcsstk04 (both identity and AMD-permuted)
plus synthetic dense / block-diagonal matrices, across min-size
thresholds of 1, 4, and 16.  A boundary test additionally exercises
the dispatch loop on a matrix whose supernodes are a singleton at
column 0 and a large size-`n-1` block at columns `[1, n)`, covering
both the degenerate `s_size == 1` and the non-degenerate paths in a
single run.

## CSC LDL^T Scaffolding (Sprint 17)

`LdltCsc` combines the Cholesky CSC layout for the unit lower triangular
factor `L` with auxiliary arrays encoding Bunch-Kaufman pivot
information:

- `D[0..n-1]`       — diagonal of `D` (1×1 pivot scalars or 2×2 block
  diagonals)
- `D_offdiag[0..n-1]` — 2×2 pivot off-diagonal entries; zero for 1×1
  pivots
- `pivot_size[0..n-1]` — `1` for 1×1 pivots, `2` for both indices of a
  2×2 pair (matching `sparse_ldlt_t`)
- `perm[0..n-1]` — composed symmetric permutation combining any
  fill-reducing reorder with the Bunch-Kaufman pivot swaps

The Day 8 elimination path expands the lower triangle to a full
symmetric `SparseMatrix`, calls the linked-list `sparse_ldlt_factor`
(which implements the four Bunch-Kaufman criteria with 1×1 / 2×2
pivoting, symmetric row-and-column swaps, and element-growth guards),
then unpacks the result back into the CSC layout with unit diagonals
inserted.  Output is bit-identical to `sparse_ldlt_factor` on the same
input; the Sprint 18 native CSC Bunch-Kaufman kernel
(`ldlt_csc_eliminate_native`) replaces this wrapper as the
production path.

The solve (`ldlt_csc_solve`) runs fully on CSC: apply `P` to `b`,
forward-solve the unit lower triangular `L`, block-diagonal solve `D`
(1×1 division or 2×2 inverse `1/det · [[d22, -d21], [-d21, d11]]`),
backward-solve `L^T`, apply `P^T` to recover `x` in user coordinates.
Singularity detection mirrors `sparse_ldlt_solve`'s relative
tolerance.

## Supernodal LDL^T (Sprint 19 Days 10-13)

`ldlt_csc_eliminate_supernodal(F, min_size)` mirrors the Sprint 18
Cholesky batched path but handles the two LDL^T-specific wrinkles:

1. **2×2 pivot atomicity in supernode boundaries.**
   `ldlt_csc_detect_supernodes` extends the Liu-Ng-Peyton three-condition
   check with a fourth invariant: a 2×2 pivot pair `(k, k+1)` (where
   `pivot_size[k] == pivot_size[k+1] == 2`) cannot straddle a supernode
   boundary.  Boundaries that would land on the first column of a 2×2
   pair are extended (if the pattern check still holds) or retracted
   (column factored by the scalar kernel).

2. **Two-pass refactor model.**  The atomicity check needs `pivot_size`,
   which only exists after a first factor.  The intended workflow is:
   (a) call `ldlt_csc_eliminate_native` once to populate `pivot_size`;
   (b) call `ldlt_csc_detect_supernodes` to identify 2×2-safe supernodes;
   (c) on subsequent refactorisations with the same sparsity pattern,
   use the batched path.  In the batched pass, the dense
   `ldlt_dense_factor` (Sprint 19 Day 11) sees a pre-permuted matrix
   where Bunch-Kaufman chooses the same pivots without further swaps;
   `eliminate_diag` validates this by comparing the dense factor's
   output `pivot_size_block` against cached `F->pivot_size` and returns
   `SPARSE_ERR_BADARG` on divergence so callers can fall back to scalar.

The four batched-path helpers parallel the Cholesky kernel:

- `ldlt_csc_supernode_extract`: scatter the supernode's CSC columns into
  a dense column-major buffer, building `row_map`.
- `ldlt_csc_supernode_eliminate_diag`: apply external cmod from priors
  `[0, s_start)` (handling 1×1 and 2×2 prior cmod with the four-term
  rank-2 outer product expansion, using `D_offdiag != 0` as the
  first-of-pair discriminator robust against adjacent 2×2 pairs), then
  call `ldlt_dense_factor`.
- `ldlt_csc_supernode_eliminate_panel`: per-row two-phase solve:
  forward-substitute against unit `L_diag`, then divide by the block
  diagonal `D` (1×1 division or 2×2 inverse), producing the panel L
  values.
- `ldlt_csc_supernode_writeback`: gather the dense buffer back into
  the CSC, scaling the per-column drop threshold to `|D[k]|` for 1×1
  pivots and `|d11| + |d22| + |d21|` for 2×2 pivots — matching the
  scalar `chol_csc_gather` invocations in `ldlt_csc_eliminate_native`.

End-of-sprint speedups (`docs/planning/EPIC_2/SPRINT_19/bench_day14.txt`):
the batched LDL^T path reaches 6.83× over linked-list on bcsstk14
(SPD, n = 1806) and 3.05× on bcsstk04 (SPD, n = 132).  Indefinite
matrices (KKT-style saddle points) currently require the scalar
`ldlt_csc_eliminate` path because the heuristic CSC fill from
`ldlt_csc_from_sparse` doesn't always cover the supernodal cmod's
fill — a `_with_analysis` mirror that pre-allocates full `sym_L`
(matching the Cholesky side's Sprint 19 Day 6 fix for Kuu) is the
natural Sprint 20 follow-up.

## Row-Adjacency Index for the Scalar LDL^T Kernel (Sprint 19 Days 8-9)

`LdltCsc` carries a per-row adjacency index — `row_adj[r]` lists the
prior columns `kp < r` where `L(r, kp)` was stored during the
factorisation.  Populated incrementally by `ldlt_csc_populate_row_adj`
after each column writeback, the index restores the linked-list
reference's sparse-row scaling for `ldlt_csc_cmod_unified`'s Phase A
(per-column diagonal cmod) and Phase B (2×2 cross-term).  Both phases
iterate `F->row_adj[col]` instead of `[0, step_k)` with a binary
search per `kp`, removing an O(step_k · log nnz) scan per elimination
step.  Bench impact (`bench_day9_row_adj.txt`): bcsstk14 jumped from
~2.5× to 3.5× over linked-list; s3rmt3m3 from ~2.5× to 3.8×.

`ldlt_csc_symmetric_swap` propagates the same swap into `row_adj` (the
two slots being swapped trade their entire adjacency lists, since the
swap permutes rows i and j across every factored column).  Storage
overhead is bounded by `n · avg_fill` ≈ 2× L's capacity — geometric
2× growth keeps amortised append O(1).

## Condition Number Estimation

`sparse_lu_condest()` estimates the 1-norm condition number `κ₁(A) = ‖A‖₁ · ‖A⁻¹‖₁` using the Hager/Higham algorithm (Hager 1984, Higham 2000).

### Algorithm

The key insight is that `‖A⁻¹‖₁` can be estimated without forming the inverse, using only forward and transpose solves with the existing LU factors:

```
1. x = [1/n, ..., 1/n]
2. Solve A·w = x
3. ξ = sign(w)
4. Solve Aᵀ·z = ξ
5. If ‖z‖∞ ≤ zᵀ·w: converged, ‖A⁻¹‖₁ ≈ ‖w‖₁
6. Otherwise: x = eⱼ where j = argmax|zⱼ|, go to 2
7. Max 5 iterations
```

The transpose solve (`Aᵀ·z = b`) is implemented by reversing the order of operations: forward-substitute with Uᵀ (using column lists), backward-substitute with Lᵀ (unit diagonal), with appropriate permutation handling.

### Limitations

- The estimate is a lower bound on the true condition number (may underestimate)
- Typically accurate to within a factor of 3-10
- Requires both the original matrix A (for `‖A‖₁`) and the LU factors

## Fill-Reducing Reordering

Before LU factorization, a symmetric permutation P·A·Pᵀ can dramatically reduce fill-in. The library provides four orderings: RCM, AMD (quotient-graph since Sprint 22), Nested Dissection (Sprint 22), and COLAMD (column ordering for unsymmetric / QR — described in `### Column Reordering` further down).

### Reverse Cuthill-McKee (RCM)

BFS-based bandwidth reduction on the symmetrized adjacency graph (A + Aᵀ).

**Algorithm:**
1. Find a pseudo-peripheral starting node (repeated BFS heuristic)
2. BFS visiting neighbors in order of increasing degree
3. Reverse the resulting ordering

**Characteristics:**
- O(nnz · log d_max) time (includes neighbor sorting during graph construction and BFS)
- Best for banded/structured matrices (e.g., FEM meshes, thermal problems)
- On steam1 (240×240 thermal): bandwidth 146→52, fill-in 23k→15k (33% reduction), 5.5x factorization speedup

### Approximate Minimum Degree (AMD) — quotient-graph implementation (Sprint 22 + Sprint 24 closures)

Greedy minimum-degree elimination ordering on the symmetrized adjacency graph.  Sprint 22 (Days 10-13) replaced the original bitset implementation with a quotient-graph representation following Amestoy/Davis/Duff (2004) "Algorithm 837: AMD" (TOMS) and Davis (2006) §7.

**Production implementation: Sprint 22's simplified quotient-graph baseline.**  The current `sparse_reorder_amd_qg` ships the variable-only quotient-graph form: each pivot rebuilds the affected vertices' adjacency lists via a sorted merge, with exact minimum-degree recompute on each rebuilt list.  No element-side adjacency, no supervariable detection, no approximate-degree formula, no dense-row skip.  Workspace `iw[]` is `5·nnz + 6·n + 1` integer entries (Davis 2006 §7); `qg_compact` reclaims slots when fill-in pushes the watermark past the buffer.  See the `src/sparse_reorder_amd_qg.c` "Day 11 implementation notes" block for the rationale.

**Sprint 23's Davis-mechanism additions (reverted by Sprint 24 Day 2).**  Sprint 23 Days 2-5 layered the four canonical Davis mechanisms on top of Sprint 22's simplified baseline: element absorption (Day 3), supervariable detection (Day 4), the approximate-degree formula (Day 5, opt-in `SPARSE_QG_USE_APPROX_DEG`), and a dense-row skip (Day 5).  Workspace grew to `7·nnz + 8·n + 1` to host an element-side adjacency region.  Sprint 23 Day 12's closing bench surfaced a 62-199× wall-time regression on irregular SuiteSparse SPD fixtures (root cause: Day 3's element absorption enabled an O(adjacency-of-adjacency) walk in `qg_recompute_deg` that Sprint 24 Day 1's profile measured at 95 % of total bcsstk14 wall time; supervariable detection's O(k²) per-pivot compare contributed only 1 %).  Sprint 24 Day 2 reverted the four Days 2-5 commits via `git revert`, restoring Sprint 22's variable-only baseline bit-identically (commit history preserved; the commits exist on master via Sprint 23's PR #31 but their effects are unwound).  See `docs/planning/EPIC_2/SPRINT_24/fix_decision_day1.md` for the full root-cause analysis (the profile evidence pointed away from the three originally-considered fix candidates) and the rationale for choosing (c) revert over (d) `qg_recompute_deg` optimization (risk profile + Sprint 24 budget shape).

**Characteristics:**
- O(deg · avg_deg) per pivot — linear in nnz instead of the bitset's O(n² / 64).  Scales to n ≥ 50 000 without paying quadratic memory.
- Memory bound moved from O(n²/64) to O(nnz).  Sprint 22 Day 13's bench measured ≥ 17× memory reduction at n = 20 000 and analytic ~26× at n = 50 000 (`docs/planning/EPIC_2/SPRINT_22/bench_day13_amd_qg.txt`); Sprint 24 Day 9's bench (`docs/planning/EPIC_2/SPRINT_24/bench_day9_amd_qg.txt`) re-measured Pres_Poisson qg peak at 19.19 MB — exact match to Sprint 22's 19.19 MB after the (c) revert returned the memory profile to the Sprint 22 baseline.
- Fill quality is bit-identical to bitset across the SuiteSparse corpus (verified Sprint 22 Day 13; carries through Sprint 24 — the Days 2-5 revert is fill-neutral by construction).
- **Wall-time profile (Sprint 24 Day 9):** qg-AMD ms is at-or-below Sprint 22 Day 13 on every fixture except bcsstk04 (where the 0.9 ms difference is sub-millisecond noise).  bcsstk14 = 125.8 ms; Pres_Poisson = 8 138.8 ms.  qg wins on banded fixtures by 6.5-9× vs bitset (improvement on Sprint 22's 1.8-7×); loses on irregular SuiteSparse SPD by 1.0-2.0× (Sprint 22 was 0.6-1.8×; Sprint 23 was 6-133×).  Day 2's revert closed the Sprint 23 regression and either matched or improved on Sprint 22's baseline.  See `docs/planning/EPIC_2/SPRINT_24/bench_summary_day9.md` for the full 3-sprint side-by-side.

### Nested Dissection (ND) — multilevel vertex separator (Sprint 22 + Sprint 23 + Sprint 24 + Sprint 25 + Sprint 26 + Sprint 27 + Sprint 28 closures)

Recursive "separator-last" ordering on top of a multilevel partitioner.  ND is best on regular 2D / 3D PDE meshes where the geometric advantage of separating the mesh into halves dominates the minimum-degree heuristic AMD is using.

**Pipeline:**
1. **Coarsen** — heavy-edge matching (Karypis & Kumar 1998) collapses pairs of vertices, building a multilevel hierarchy that bottoms out at MAX(20, n/divisor) vertices.  The divisor is 100 by default; `SPARSE_ND_COARSEN_FLOOR_RATIO` (Sprint 24 Day 5) overrides it in [1, 100000].  Larger divisor → tighter coarsest level → potentially tighter cut at the finest uncoarsening step.  Sprint 24 Day 5's sweep on Pres_Poisson (n = 14 822) found ratio = 200 produces a 1.0pp ND/AMD improvement (0.952× → 0.942×); ratios ≥ 400 regress because the coarsest level pegs at the floor of 20 vertices and the brute-force bisection loses cut quality.  See `docs/planning/EPIC_2/SPRINT_24/nd_coarsen_floor_decision.md`.  Sprint 25 Day 1-3 added `SPARSE_ND_COARSENING={heavy_edge,hcc}` (default `heavy_edge`).  `hcc` selects Heavy Connectivity Coarsening (Karypis & Kumar 1998 §5) — the matching score becomes `edge_weight × min(deg(u), deg(v))` rather than HEM's pure `edge_weight`, biasing matching toward dense-connectivity regions.  Day 3's Pres_Poisson measurement: HCC alone closes ND/AMD 0.952× → 0.937× (-1.5pp).  HCC + ratio=200 composes to **0.922×** (Day 9 setting 13; the Sprint 25 best Pres_Poisson combination).  HCC default flip was attempted Day 10 and reverted — bcsstk14 produces a degenerate `sep = 0` empty separator under HCC, blocking the production-default flip independent of fill quality.  See `docs/planning/EPIC_2/SPRINT_25/coarsening_decision.md` and `hcc_design.md`.  **Sprint 26 Day 3 fixed the sep=0 blocker** via a fall-back path in `sparse_graph_partition` that retries with HEM forced when the multilevel pipeline produces sep=0 (`_Thread_local force_hem_override`); bcsstk14 under HCC now produces sep=97.  Sprint 26 Day 13's flip re-attempt found Kuu HCC-alone regresses by +14.6pp vs Sprint 26 default (a separate blocker Sprint 25 had documented but masked by the bcsstk14 issue), so the HCC default flip remains BLOCKED post-Day-3 — but the env-var path now works correctly on bcsstk14.  See `docs/planning/EPIC_2/SPRINT_26/hcc_sep_zero_diagnosis.md`.
2. **Coarsest bisection** — brute-force enumeration for n ≤ 20 (`2^(n-1)` patterns); GGGP (greedy graph-growing partition, peripheral-vertex BFS) for n > 20.  Sprint 25 Day 6-8 added `SPARSE_ND_COARSEST_BISECTION={spectral,gggp,brute}` (default unset → Sprint 22 routing: `brute` for n ≤ 20, `gggp` for n > 20).  `gggp` forces GGGP at all sizes; `brute` forces brute @ n ≤ 20 and falls through to GGGP at n > 20 (the 2^(n-1) enumeration is intractable beyond n=20; `brute` is mainly a debugging knob for the small-n path).  `spectral` builds the graph Laplacian L = D - A, computes the Fiedler vector via the Sprint 20-21 Lanczos eigensolver (`sparse_eigs_sym(SPARSE_EIGS_SMALLEST, k=2)`), and partitions vertices by the median of v_1 (the eigenvector corresponding to the second-smallest eigenvalue).  Falls back to GGGP if Lanczos fails or produces a 60/40-imbalanced cut.  Day 8's Pres_Poisson measurement: spectral alone barely moves nnz_L (0.953× vs default 0.952×), but reduces ND wall time dramatically (~23× speedup as part of Day 9 setting 15: spectral cuts close to FM optimum, so intermediate / finest FM polishes faster).  Default stays `gggp` because spectral's nnz_L benefit is essentially nil on the headline fixture; spectral ships as advisory for callers prioritising ND wall time.  See `docs/planning/EPIC_2/SPRINT_25/spectral_bisection_decision.md` and `spectral_bisection_design.md`.
3. **Uncoarsen with FM** — project the coarsest partition back through the hierarchy with Fiduccia-Mattheyses refinement at every level (rollback-on-regress on the lowest cut seen).  Sprint 23 Day 10 swapped Sprint 22's O(n) max-gain linear scan for an O(1) gain-bucket structure (`src/sparse_graph_fm_buckets.h`); Sprint 23 Day 11 adopted 3-pass FM at the finest uncoarsening level by default (intermediate levels stay single-pass) — overridable via `SPARSE_FM_FINEST_PASSES` env var.  Sprint 25 Day 4-5 added `SPARSE_FM_INTERMEDIATE_PASSES` (default 1, range [1, 10]; out-of-range / non-numeric / missing → default 1) — controls FM passes at every intermediate uncoarsening level (the finest level continues to use `SPARSE_FM_FINEST_PASSES`).  Day 5's Pres_Poisson sweep across passes ∈ {1, 2, 3}: passes=1 (default) = 0.952×; passes=2 = 0.952× (essentially unchanged, -0.04pp); passes=3 = 0.967× (+1.5pp regression).  Default stays at 1 because the PLAN.md Day-5 flip rule (≥ 1pp Pres_Poisson tightening + no smaller-fixture regression past 5pp) fails on the headline fixture for both candidate values: passes=2 doesn't move Pres_Poisson AND regresses Kuu (+6.6pp), passes=3 regresses Pres_Poisson.  Per-fixture advisories DO exist where the workload prioritises non-Pres_Poisson fill: **Kuu at passes=3 closes -23.2pp** (2.275× → 2.043×; the strongest single per-fixture win Sprint 25 produced via this axis alone, at +3.7% wall cost), and **bcsstk14 at passes=2 closes -2.2pp** (1.129× → 1.107×, +4.7% wall) or passes=3 -2.3pp at -11.4% wall.  s3rmt3m3 regresses +1.7pp at passes ≥ 2; nos4 / bcsstk04 are flat (small fixtures).  See `docs/planning/EPIC_2/SPRINT_25/intermediate_fm_decision.md` for the full per-fixture sweep.  Sprint 26 Day 6-8 added `SPARSE_FM_FINEST_STRATEGY={baseline,fifo,annealing,thick_restart}` (default `baseline`).  `fifo` switches the gain-bucket tie-break from LIFO (most-recently-inserted-or-gain-updated wins; Sprint 23 baseline) to FIFO (first-inserted wins) by adding a `tails[]` array to `fm_bucket_array_t` + a `pop_max_tail` variant.  `annealing` and `thick_restart` are recognized for forward-compatibility but unimplemented in Sprint 26 (rejected at Day 6 design — annealing +20-50% wall; thick-restart 2-3× wall would breach the 1.5× wall-check ceiling); both fall through to baseline.  Day 8's cross-corpus sweep: `fifo` alone REGRESSES Pres_Poisson by +3pp (no flip-rule clearance); ships as advisory only in combination with Sprint 25 setting 15-ish (HCC + ratio=200 + spectral + balanced_boundary), where it contributes -1 to -3pp on smaller fixtures (Kuu / bcsstk14 / nos4).  See `docs/planning/EPIC_2/SPRINT_26/finest_fm_decision.md`.
4. **Vertex-separator extraction** — convert the final 2-way edge separator to a 3-way `{0, 1, 2}` vertex separator.  Default lift strategy is `smaller_weight` (METIS convention — lift the smaller-vertex-weight side's boundary); `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (Sprint 24 Day 6) lifts whichever side has the smaller boundary count regardless of vertex weight, with a 70/30 post-lift weight-balance check that falls back to `smaller_weight` if the chosen lift would skew the recursion too far.  `balanced_boundary` is documented advisory for non-Pres_Poisson workloads: Sprint 24 Day 6's sweep showed 8-38 percentage-point ND/AMD improvements on nos4 / bcsstk14 / Kuu (Kuu's 38pp drop is the largest single nnz win Sprint 24 produced) but is essentially neutral on Pres_Poisson (+0.1pp), so it stays off-by-default.  Sprint 26 Day 10/12 extended the env var with three per-vertex variants: `per_vertex` (Day 10's hybrid score `2 × cross_deg + balance_bonus`), `per_vertex_balance` (balance-priority), `per_vertex_degree` (low-total-degree-priority).  All three score boundary vertices individually + greedily pick top-K with the same 70/30 balance gate.  Day 12's cross-corpus sweep found the three weight schemes converge to bit-identical outputs on 5 of 6 fixtures (the 70/30 balance gate dominates the score formula); per_vertex variants regress Pres_Poisson by +29pp (catastrophic) but win on bcsstk04 (-4.6pp).  Per_vertex ships as advisory for bcsstk04-class small irregular fixtures only.  See `docs/planning/EPIC_2/SPRINT_24/nd_sep_strategy_decision.md` and `docs/planning/EPIC_2/SPRINT_26/per_vertex_sep_decision.md`.
5. **Recursive ordering** — on each interior partition: recurse; for subgraphs with `n ≤ ND_BASE_THRESHOLD` (default **128** — **Sprint 27 Day 3 default-flip from Sprint 26's 96**, itself a flip from Sprint 22's 32), call `sparse_reorder_amd_qg` on the leaf and splice the per-leaf permutation into the global perm[] via `vertex_id_map` (Sprint 23 Day 7).  Append the separator vertices last.  Sprint 26 Day 5's threshold flip from 32 → 96 was driven by Day 4's per-recursion-depth profile finding (88% of ND wall on Pres_Poisson concentrates at depths 6-9 = ~169 small-subgraph multilevel-pipeline calls with 60-200 ms per-call constant overhead floor).  Sprint 27 Day 3's relaxed-flip-rule re-sweep (2pp regression cap, was 1pp Sprint 26) found t=128 the new maximum; t=192 fails by Pres_Poisson +2.0% (right at 2pp); t=256 fails clearly (+3.2%).  Cumulative Pres_Poisson ND wall improvements: 38.1 s (Sprint 25) → 12.2 s (Sprint 26 Day 5; -67.9%) → 8.8 s (Sprint 27 Day 2 HCC default) → 7.1 s (Sprint 27 Day 3 t=128; -42% vs Sprint 26).  See `docs/planning/EPIC_2/SPRINT_26/nd_base_threshold_decision.md` and `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md`.  **Sprint 27 Day 2 also flipped `SPARSE_ND_COARSENING` default `heavy_edge` → `hcc`** via the Kuu-safe matching variant (option (a.1) — degree-CV-detection-and-HEM-fall-through; default threshold 0.30 routes Kuu's CV=0.425 to HEM, Pres_Poisson's CV=0.108 stays HCC).  Both Sprint 25 Day 10 default-flip blockers now closed (bcsstk14 sep=0 by Sprint 26 Day 3; Kuu +14.6pp regress by Sprint 27 Day 2).  Day 2 corpus sweep delta vs HEM: Pres_Poisson -3.4 % (0.950× → 0.918×); Kuu -12.3 %; bcsstk14 + s3rmt3m3 within ±0.7 % (under 5pp budget); nos4 / bcsstk04 bit-stable.  See `docs/planning/EPIC_2/SPRINT_27/hcc_kuu_diagnosis.md`.

**Sprint 27 algorithmic-axis closures (advisory, no default flip):**
- **`SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k` + `SPARSE_ND_SEP_LIFT_WEIGHT={hybrid (default), balance, degree}`** (Sprint 27 Day 4) — fixed-K termination instead of the dynamic-K + 70/30 balance gate that Sprint 26 Day 12 found dominates the score formula.  Confirms Sprint 26 Day 12's hypothesis: under fixed-K, the three weight schemes produce a 6× spread in Kuu nnz_L (vs <1pp under dynamic-K).  **Headline win: Kuu under `per_vertex_fixed_k × hybrid` lands -34.7 % nnz_L** (1.229× of AMD; was 1.882× under Sprint 27 Day 2 default).  Pres_Poisson regresses +52 % under hybrid, so this is bimodal-class-specific (Kuu) not a default flip.  See `docs/planning/EPIC_2/SPRINT_27/per_vertex_fixed_k_decision.md`.
- **`SPARSE_FM_FINEST_STRATEGY=annealing` + `SPARSE_FM_ANNEALING_SCHEDULE={linear, exponential (default), cosine}`** (Sprint 27 Days 5-7) — Sprint 26 Day 6's parser-stubbed `annealing` value gets the actual acceptance overlay (`P = exp(g/T)` per Kirkpatrick-1983 §3) plus three temperature schedules.  All three schedules regress Pres_Poisson 2.2-3.1 % (the rollback-to-best-cut floor combined with stochastic acceptance produces trajectories that miss baseline's saved-best path on the regular FE-mesh structure).  bcsstk14-class fixtures see slight wins (-0.7 %); ships as advisory for that class.  See `docs/planning/EPIC_2/SPRINT_27/annealing_fm_decision.md`.
- **`SPARSE_ND_ROOT_BISECT={multilevel (default), spectral}` + `SPARSE_ND_ROOT_BISECT_MAX_N=N` (default 50000)** (Sprint 27 Days 7-9) — extends Sprint 25 Day 7's coarsest-level spectral path (`graph_bisect_coarsest_spectral`, promoted to internal-API on Day 8) to the ROOT level via Lanczos + Fiedler on the full graph Laplacian.  Day-7 hypothesis ("Fiedler at the root captures geometric structure the multilevel pipeline loses") is empirically wrong — multilevel's iterative FM refinement reaches near-optimal cuts the median-bisect-on-Fiedler doesn't beat.  Pres_Poisson +2.3 % regress; Kuu +16.7 %; combination with `SPARSE_FM_FINEST_STRATEGY=annealing` lands 0.947× = +2.4pp regress.  **bcsstk04 is the lone win: -1.3 % nnz_L + 23× wall speedup** (the multilevel pipeline's coarsening overhead dominated this small-n fixture; spectral skips it cleanly).  Ships as advisory for small irregular fixtures (bcsstk04-class, n ≤ ~200).  See `docs/planning/EPIC_2/SPRINT_27/root_spectral_decision.md`.

**Sprint 28 algorithmic-axis closures (advisory, no default flip):**
- **`SPARSE_FM_THICK_RESTART_PERTURB=gain_noise_formal`** (Sprint 28 Days 2-3) — Sprint 27 Day 11 simplified the `gauss_noise` thick-restart variant to "random-flip with k drawn proportional to a half-Gaussian"; the formal variant adds Gaussian noise to the gain-bucket pick step (`noisy_gain = gain + sigma * |max_gain| * randn()` with linear or exponential `sigma` schedules).  Day 3's corpus sweep: catastrophic Pres_Poisson regress of +24pp (1.166× ratio) under both linear and exponential schedules; **bcsstk04 is the lone win** (-1.7 % nnz_L under the linear schedule); bcsstk14, Kuu, s3rmt3m3 all regress (+12 to +67 %).  Ships as advisory for bcsstk04-class small irregular SPDs; default stays `random_flip`.  See `docs/planning/EPIC_2/SPRINT_28/gain_noise_decision.md`.
- **`SPARSE_FM_FINEST_STRATEGY=ensemble` + `SPARSE_FM_ENSEMBLE_STRATEGIES={baseline,fifo,annealing} (default)`** (Sprint 28 Days 4-5) — runs all listed FM strategies in parallel per finest-level call, picks the lowest-cut result.  Default selector `baseline,fifo,annealing` explores 3× the FM landscape at 2-3× wall cost.  Day 5's corpus sweep: Pres_Poisson regresses +1.5pp (0.937× ratio) under all 4 selector list variants (default, drop-FIFO, drop-baseline, 4-way with thick_restart).  Wall cost rules out the default flip even on the smaller-fixture wins.  Ships as advisory; default stays `baseline`.  See `docs/planning/EPIC_2/SPRINT_28/ensemble_fm_decision.md`.
- **`SPARSE_ND_SUPERNODAL_POSTORDER={off (default), on}`** (Sprint 28 Days 6-10) — Item 4 "non-pipeline-level pivot" per Day-1 `pivot_decision_day1.md` (chosen over METIS-style multi-matching coarsening and geometric domain decomposition).  Composes the elimination-tree postorder into `analysis->perm` (Liu 1990 / Davis 2006 §6.5), rebuilds B + recomputes etree+postorder so colcount + symbolic Cholesky run on the final ordering.  Default-off path bit-identical to Sprint 27; env-on path adds ~6-15 % to `sparse_analyze` wall.  Day 9's 24-cell sweep ({AMD, ND} × {off, on} × 6 fixtures): nnz_L invariant on every cell (symmetric permutation preserves symbolic Cholesky fill by construction); supernode-count delta trivial (±1-3 supernodes; total_grouped same or ±3 columns).  Day 12's 24-setting cross-corpus matrix: Item-4 + Sprint 27 default tied for Pres_Poisson best at 0.9226× (bit-equal); Item-4 + Sprint 27 Kuu opt-in (t=256) tied for corpus-wide best at geomean 1.1156.  **Ships infrastructure for future supernodal-numeric-factor kernels** (the existing chol_csc path doesn't currently exploit supernodal kernels; per Day-1 dossier candidate-(c), the post-pass becomes the natural input ordering for such kernels when a future sprint wires them).  Default stays `off`.  See `docs/planning/EPIC_2/SPRINT_28/non_pipeline_decision.md` + `pivot_decision_day1.md`.

**Characteristics:**
- Fill quality on regular meshes (default-path, **Sprint 27 + Sprint 28 production** — Sprint 28 inherited Sprint 27's HCC + Kuu-safe + t=128 defaults; no Sprint 28 default flips): nos4 1.000× of AMD (n < threshold; entirely AMD); bcsstk04 1.184×; Kuu 1.882×; bcsstk14 1.124×; s3rmt3m3 1.028×; **Pres_Poisson 0.9226× of AMD** (Sprint 22 Day 7 was 1.063×; the canonical 2D-PDE benchmark — ND beats AMD).  Sprint 28 Day 12's 24-setting × 6-fixture cross-corpus matrix (`bench_day12_combinations.csv`) confirms the production default is at the empirical floor on Pres_Poisson; no Sprint 28 axis or combination beats it on the headline fixture; Sprint 28's Item-4 (supernodal-etree post-pass) bit-equals the default by symmetric-permutation invariance.  Full corpus capture: `docs/planning/EPIC_2/SPRINT_23/bench_day12.txt` (default-path baseline through Sprint 23), `docs/planning/EPIC_2/SPRINT_25/bench_day9_combinations.csv` (Sprint 25 16-setting matrix), `docs/planning/EPIC_2/SPRINT_27/bench_day13_combinations.csv` (Sprint 27 24-setting matrix), `docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.csv` (Sprint 28 24-setting matrix; current).
- Per-fixture advisory env-var settings (each documented in the cited decision docs).  **Sprint 28 Day 12's 24-setting × 6-fixture sweep** (`docs/planning/EPIC_2/SPRINT_28/bench_day12_combinations.{csv,txt}` + `headline_summary.md`) is the current cross-corpus reference (supersedes Sprint 27 Day 13's matrix while preserving its conclusions; Sprint 28 added Items 1+2+4 axis variants + stack combinations).  The verified-best recipes per fixture-class:
    - **Pres_Poisson (headline)** — Sprint 27 production default (no env vars; HCC + t=128) at **0.9226×** is itself the best across all 24 settings of Sprint 28's Day-12 matrix (tied with Sprint 28 setting 3 `SPARSE_ND_SUPERNODAL_POSTORDER=on` by symmetric-permutation invariance).  No Sprint 27 or Sprint 28 advisory combination beats it; the closest contenders cluster at 0.927-0.944×.  Stacking advisory axes (especially `fixed_k×degree`) can REGRESS Pres_Poisson catastrophically (Sprint 27 settings 5, 21, 23; Sprint 28 settings 4, 5, 10, 17, 22, 23, 24 land 1.17-1.65× — worse than AMD on some).  See `docs/planning/EPIC_2/SPRINT_28/headline_summary.md`.
    - **Kuu (largest single-fixture win Sprint 27 produced)** — `SPARSE_ND_SEP_LIFT_STRATEGY=per_vertex_fixed_k SPARSE_ND_SEP_LIFT_WEIGHT=hybrid build/bench_reorder --nd-threshold 256` (Day 13 setting 18) drops Kuu ND/AMD 1.882× → **1.217×** (−35.3 % nnz_L).  Combines Day-3's t=256 advisory with Day-4's fixed-K hybrid weighting — neither stand-alone beats this; together they compose constructively.  Best Kuu single-axis: setting 17 (`--nd-threshold 256` alone) at 1.772× (−5.6pp).
    - **bcsstk04 (small irregular SPD)** — `SPARSE_ND_ROOT_BISECT=spectral` (Day 9) drops bcsstk04 nnz_L −1.3 % AND wall 23× faster (105 ms → 4.5 ms; the multilevel pipeline's coarsening overhead dominated this small-n fixture; spectral skips it cleanly).  See `root_spectral_decision.md`.
    - **bcsstk14 (mid-irregular)** — `SPARSE_FM_FINEST_STRATEGY=annealing` (Day 7) closes bcsstk14 nnz_L −0.7 % under the exponential or cosine schedule (1.124× → 1.116×).  See `annealing_fm_decision.md`.
    - **s3rmt3m3 (mid-irregular)** — `SPARSE_FM_FINEST_STRATEGY=thick_restart SPARSE_FM_THICK_RESTART_PERTURB=random_flip` (Day 12) closes s3rmt3m3 nnz_L −1.0 % (1.028× → 1.018×).  See `thick_restart_decision.md`.
    - **Corpus-wide best (geomean of 6 ratios)** — setting 17 (`build/bench_reorder --nd-threshold 256`) at geomean 1.116 vs Sprint 27 default's 1.155, driven primarily by Kuu's huge improvement.  Loses Pres_Poisson 2.5pp; ships as advisory for non-FE-mesh-dominated workloads.

    Cross-axis caution: stacking advisory env vars often makes Pres_Poisson WORSE (settings 21, 23, 24 in the Day 13 matrix all regress past 1.0× of AMD).  The `fixed_k×degree` weight scheme particularly destroys Pres_Poisson — opt in only on workloads that don't include FE meshes.

    Historical Sprint 25 advisories — `SPARSE_ND_COARSEN_FLOOR_RATIO=200` (Sprint 24 Day 5), `SPARSE_FM_INTERMEDIATE_PASSES={2,3}` (Sprint 25 Day 4), `SPARSE_ND_COARSEST_BISECTION=spectral` (Sprint 25 Day 6-8), `SPARSE_ND_SEP_LIFT_STRATEGY=balanced_boundary` (Sprint 24 Day 6), and the legacy `per_vertex` / `per_vertex_balance` / `per_vertex_degree` strategies (Sprint 26 Day 10/12) — remain available but are subsumed by Sprint 27's verified recipes for the corresponding fixture classes.  See `SPRINT_24/`, `SPRINT_25/`, `SPRINT_26/` decision docs.
- The Pres_Poisson default-path **0.923× ratio** is the cumulative effect of Sprint 23 Days 7 (leaf-AMD splice), 9-10 (gain-bucket FM lifting per-pass cost from O(n²) to O(|E|)), 11 (multi-pass FM at the finest level), Sprint 26 Day 5 (`nd_base_threshold 32 → 96`), Sprint 27 Day 2 (`SPARSE_ND_COARSENING heavy_edge → hcc` with Kuu-safe degree-CV-fall-through), and Sprint 27 Day 3 (`nd_base_threshold 96 → 128`).  PLAN.md's literal **≤ 0.85× target across Sprints 24-28 remained UNMET** (6 consecutive sprints; current default 0.923× = 7.3pp from target; per-sprint trajectory: 0.952× → 0.942× → 0.922× → 0.9217× → 0.9226× → 0.9226×).  Sprint 27's structural-pipeline-level interventions all REGRESSED Pres_Poisson on the headline fixture (Item 4 annealing FM +2.2-3.1pp, Item 5 root-spectral +2.3pp, Item 6 thick-restart FM +4.7-11.5pp).  Sprint 28's non-pipeline-level pivot (`SPARSE_ND_SUPERNODAL_POSTORDER=on` — Liu 1990 post-permutation acting on the elimination tree AFTER the multilevel pipeline) is bit-equivalent to the default on Pres_Poisson nnz_L by symmetric-permutation invariance — the only intervention that can act AFTER the pipeline produces a 0pp delta on the metric.  **The literal 0.85× target is formally RETIRED with Sprint 28's empirical evidence** (Day 12 cross-corpus matrix; `headline_summary.md` "Sprint 28 Verdict on the Literal 0.85× Pres_Poisson Target — RETIRED").  Sprint 29+ may revisit ONLY with fundamentally different machinery: METIS C library interop (defer to production METIS rather than the in-house multilevel pipeline), geometric mesh-aware ordering with first-class coordinate API (rejected Sprint 27 Day 9; Laplacian-spectral coordinates regress +2.3pp), or hybrid AMD-then-ND-on-separators.  None budgeted for Sprint 29.
- **Wall time on Pres_Poisson is ~10 s** under Sprint 27 production defaults (Sprint 25 baseline was ~38 s; cumulative -73.5 % across Sprints 26-27).  Reductions: Sprint 26 Day 5 t=96 flip cut to ~12 s (-67.9 % vs Sprint 25); Sprint 27 Day 2 HCC default flip to ~8.8 s (-77 %); Sprint 27 Day 3 t=128 flip held wall at the new floor.  See `docs/planning/EPIC_2/SPRINT_27/nd_base_threshold_decision.md` and `hcc_kuu_diagnosis.md` for per-flip wall measurements.

### Performance regression gates

Sprint 24 Day 1 added a `make wall-check` target driven by
`scripts/wall_check.sh` and a per-fixture baseline file at
`docs/planning/EPIC_2/SPRINT_24/wall_check_baseline.txt`.  Sprint
25 Day 12 extended the gate with a third single-fixture
measurement.  The gate now runs:

- `bench_amd_qg --only bcsstk14` — captures qg-AMD's `reorder_ms`
  on the bcsstk14 SuiteSparse fixture (n = 1 806); compared against
  `bcsstk14_qg_amd_ms` baseline.
- `bench_reorder --only Pres_Poisson --skip-factor` — captures
  both AMD's and ND's `reorder_ms` on Pres_Poisson (n = 14 822)
  from a single bench invocation; compared against the
  `pres_poisson_amd_ms` and `pres_poisson_nd_ms` baselines
  respectively.

Each measurement is compared against its baseline in
`wall_check_baseline.txt` using a per-key threshold:

- **`bcsstk14_qg_amd_ms` → 2× threshold** (Sprint 24 Day 1).
  Catches single-day algorithmic regressions (Sprint 23 Days 2-5
  each introduced ~10-50× drift, so 2× is generous-but-not-
  toothless) without flagging on routine host-load noise (typical
  run-to-run drift on this fixture is within ±25 %).
- **`pres_poisson_amd_ms` → 2× threshold** (Sprint 24 Day 1).
  Same calibration as bcsstk14 — both AMD baselines are tight
  gates on the qg-AMD path, which Sprint 23 introduced + Sprint
  24 reverted a 30-200× regression that escaped notice for an
  entire sprint.
- **`pres_poisson_nd_ms` → 1.5× threshold** (Sprint 25 Day 12).
  Wider than the AMD gates because Sprint 25 Day 11 profiling
  (`docs/planning/EPIC_2/SPRINT_25/profile_day11_pres_poisson_nd.txt`)
  measured 16 % within-run variance on this fixture (5 consecutive
  runs spanned 44 321 - 51 562 ms; 99.5 % of wall time is in the
  partition phase, which is sensitive to macOS arm64 thermal
  management + sustained-load variance).  1.5× absorbs the
  variance without going so wide that real algorithmic regressions
  slip through; if Sprint 26 lands a real cost tightening on the
  ND default path, the gate can drop to 1.25×.

The Sprint-24-internal motivation: Sprint 23's qg-AMD wall-time
regression (62-199× vs Sprint 22 baseline; documented in
`SPRINT_23/bench_summary_day12.md "(b)"`) accumulated across four
day-by-day commits with no intermediate signal — the closing-day
bench was the first time the regression was measured end-to-end.
The wall-check gate runs in seconds (one fixture each side, no
factor), so it's cheap to invoke at every day-by-day commit, and
catches the single-day step-change that the corpus-scale closing-
bench would otherwise only surface days later.

The baseline file commits its values in a `KEY=VALUE_MS`
key-value format with `#`-prefixed comment blocks documenting
which day landed each baseline and what the previous values were.
Sprint 24 Day 4 bumps the AMD baselines down to the post-fix
measurements once item 2's wall-time fix lands; Sprint 25 Day 12
adds the `pres_poisson_nd_ms = 47 055 ms` baseline (median of 5
consecutive Day 11 measurements per
`docs/planning/EPIC_2/SPRINT_25/nd_wall_time_decision.md`).
Future sprints that touch the AMD or ND default code paths should
expect to update both the baselines and the comment block.

### Integration with Factorization

The `sparse_lu_factor_opts()` function provides a unified interface:

```c
sparse_lu_opts_t opts = { SPARSE_PIVOT_PARTIAL, SPARSE_REORDER_AMD, 1e-12 };
sparse_lu_factor_opts(LU, &opts);
sparse_lu_solve(LU, b, x);  // reorder/unpermute handled automatically
```

The reordering permutation is stored in the matrix and automatically applied during solve — callers do not need to manually permute/unpermute vectors.

Similarly for Cholesky:

```c
sparse_cholesky_opts_t opts = { SPARSE_REORDER_AMD };
sparse_cholesky_factor_opts(L, &opts);
sparse_cholesky_solve(L, b, x);  // reorder/unpermute handled automatically
```

`SPARSE_REORDER_ND` (Sprint 22) routes through the same dispatch as
`AMD` / `RCM` — `sparse_analyze`, the per-factorization
`*_factor_opts`, and the QR `factor_opts`'s A^TA path all dispatch
ND alongside the others.  COLAMD on QR is recommended for
unsymmetric inputs; ND is best on 2D / 3D PDE meshes; AMD is the
default catch-all for SPD fixtures and where it lives today.

## Sparse Matrix-Matrix Multiply (SpMM)

`sparse_matmul(A, B, &C)` computes C = A*B using Gustavson's row-wise algorithm.

### Algorithm

```
for each row i of A:
    initialize dense accumulator acc[0..n-1] = 0
    for each nonzero A(i,j):
        acc += A(i,j) * row_j(B)
    flush nonzeros from acc into row i of C
```

The dense accumulator avoids hash-based or sort-based merging. Entries with |value| < 1e-15 are dropped during flush to avoid storing numerical zeros from cancellation.

### Complexity

- Time: O(nnz_A × avg_nnz_per_row_B + nnz_C) where the second term is the total flush cost over all rows
- Space: O(n) for the dense accumulator and touched-index list (reused per row)
- The flush step iterates only columns touched during accumulation (via a compact index list), not all n columns.

## CSR/CSC Compressed Formats

The library provides bidirectional conversion between the orthogonal linked-list format and standard compressed sparse formats.

### CSR (Compressed Sparse Row)

```
row_ptr[i] .. row_ptr[i+1]-1  →  indices into col_idx/values for row i
```

- `sparse_to_csr()`: walks each row's linked list (already sorted by column) — O(nnz)
- `sparse_from_csr()`: validates structure then inserts entries — O(nnz × avg_row_nnz) due to per-insert row-list scan

### CSC (Compressed Sparse Column)

```
col_ptr[j] .. col_ptr[j+1]-1  →  indices into row_idx/values for column j
```

- `sparse_to_csc()`: walks each column's linked list (already sorted by row) — O(nnz)
- `sparse_from_csc()`: validates structure then inserts entries — O(nnz × avg_row_nnz) due to per-insert row-list scan

### Transpose Relationship

CSR of A has the same structure as CSC of A^T. This means `sparse_to_csr(A)` produces arrays identical to `sparse_to_csc(A^T)`.

## Thread Safety

The library is safe for concurrent use under these conditions:

**Safe operations (no synchronization needed):**
- Concurrent solves (`sparse_lu_solve`, `sparse_cholesky_solve`) on the same factored matrix with different b/x vectors — solve is read-only on the matrix
- Concurrent factorization of different matrices — each has its own pool allocator
- Concurrent read-only access (nnz, get, matvec) to any matrix
- `sparse_errno()` — uses `_Thread_local` storage

**Unsafe operations (require external synchronization):**
- Concurrent mutation (insert/remove/factor) of the same matrix

**Optional mutex support:** Compile with `-DSPARSE_MUTEX` and `-pthread` to add per-matrix mutex locking on `sparse_insert()` and `sparse_remove()`. This serializes concurrent insert/remove calls on the same matrix. Factorization functions are NOT mutex-protected. Zero overhead when compiled without the flag.

## Conjugate Gradient (CG) Solver

`sparse_solve_cg()` implements the Preconditioned Conjugate Gradient method for symmetric positive-definite (SPD) systems.

### Algorithm

```
Given SPD matrix A, right-hand side b, initial guess x_0:
    r_0 = b - A*x_0
    z_0 = M^{-1}*r_0          (or z_0 = r_0 if no preconditioner)
    p_0 = z_0
    for k = 0, 1, ..., max_iter:
        if ||r_k|| / ||b|| <= tol: converged
        alpha_k = (r_k^T * z_k) / (p_k^T * A*p_k)
        x_{k+1} = x_k + alpha_k * p_k
        r_{k+1} = r_k - alpha_k * A*p_k
        z_{k+1} = M^{-1}*r_{k+1}
        beta_k  = (r_{k+1}^T * z_{k+1}) / (r_k^T * z_k)
        p_{k+1} = z_{k+1} + beta_k * p_k
```

### Convergence Properties

- CG converges in at most n iterations in exact arithmetic (finite termination)
- The convergence rate depends on the condition number: `||e_k||_A ≤ 2 * ((sqrt(κ)-1)/(sqrt(κ)+1))^k * ||e_0||_A`
- Preconditioning reduces the effective condition number, accelerating convergence
- CG is only applicable to SPD matrices; use GMRES for general systems

### Workspace

4 vectors of length n (r, z, p, Ap), allocated as a single block.

## GMRES Solver

`sparse_solve_gmres()` implements the Restarted GMRES(k) method for general (possibly unsymmetric) square systems, with optional left preconditioning.

### Arnoldi Process

GMRES builds an orthonormal basis V = [v_1, ..., v_k] for the Krylov subspace K_k(A, r_0) = span{r_0, Ar_0, ..., A^{k-1}r_0} using Modified Gram-Schmidt orthogonalization:

```
for j = 0 to k-1:
    w = A*v_j                          (or w = M^{-1}*A*v_j with left preconditioning)
    for i = 0 to j:
        H(i,j) = w^T * v_i            (Hessenberg matrix entry)
        w = w - H(i,j) * v_i          (orthogonalize)
    H(j+1,j) = ||w||
    v_{j+1} = w / H(j+1,j)
```

### Givens Rotations

Instead of solving the Hessenberg least-squares problem from scratch each iteration, Givens rotations are applied incrementally to triangularize H and transform the residual vector:

```
for each new column j of H:
    apply all previous rotations (i = 0..j-1)
    compute new rotation: cs[j], sn[j] from H(j,j), H(j+1,j)
    apply to H and residual vector g
    residual norm = |g[j+1]| / ||b||
```

This gives the residual norm cheaply without forming the solution each iteration.

### Restart

After k Arnoldi steps, the solution is formed: x = x_0 + V_k * y_k where y_k solves the upper triangular system from the Givens-transformed H. If not converged, x becomes the new initial guess and the process restarts. The restart parameter k controls memory usage (k+1 vectors of length n) versus convergence speed.

### Lucky Breakdown

If H(j+1,j) = 0, the Krylov subspace is invariant — the exact solution lies in the current subspace. This is detected before the Givens rotation zeroes H(j+1,j).

### Workspace

(k+1)×n for Arnoldi vectors, (k+1)×k for Hessenberg matrix, plus O(k) for Givens data and O(n) for temporaries.

## MINRES Solver

`sparse_solve_minres()` solves symmetric (possibly indefinite) linear systems Ax = b using the Minimum Residual method (Paige & Saunders, 1975).

### Algorithm

MINRES uses a three-term Lanczos recurrence to build an orthonormal basis for the Krylov subspace K_k(A, r_0), then minimizes ||b - Ax|| over this subspace via an implicit QR factorization of the resulting tridiagonal matrix.

```
r_0 = b - A*x_0;  beta_1 = ||r_0||;  v_1 = r_0 / beta_1
For k = 1, 2, ...:
    1. Lanczos step:
       w = A*v_k - alpha_k*v_k - beta_k*v_{k-1}
       where alpha_k = v_k^T * A * v_k
       beta_{k+1} = ||w||;  v_{k+1} = w / beta_{k+1}

    2. QR update (Givens rotations on tridiagonal column k):
       Apply G_{k-2}, G_{k-1} to [beta_k; alpha_k; beta_{k+1}]
       Compute G_k to zero out beta_{k+1} → gives gamma_k

    3. Solution update (short recurrence):
       d_k = (v_k - eps_k*d_{k-2} - delta_k*d_{k-1}) / gamma_k
       x_k = x_{k-1} + phi_k * d_k

    4. Residual: ||r_k|| = |phi_bar_{k+1}| (available cheaply)
       Check convergence: |phi_bar_{k+1}| / ||b|| < tol
```

### Convergence Properties

- **Monotonic residual decrease:** ||r_k|| decreases at every iteration (unlike CG or restarted GMRES).
- **Short recurrences:** O(n) storage per iteration — only 3 direction vectors and 2 Lanczos vectors needed.
- **Symmetric indefinite:** Works on any symmetric matrix, including indefinite (KKT, saddle-point).
- **Exact in n steps:** For an n×n matrix, MINRES produces the exact solution after at most n iterations (in exact arithmetic).

### Preconditioning

Preconditioned MINRES uses the M-inner product where M is an SPD preconditioner. The preconditioner must be SPD even when A is indefinite. IC(0) is a natural choice for SPD systems; for indefinite systems, use a Jacobi (diagonal) preconditioner with |A(i,i)|.

### Workspace

6 vectors of length n (unpreconditioned) or 8 vectors (preconditioned), plus O(1) scalar state for Givens rotations.

## BiCGSTAB Solver

`sparse_solve_bicgstab()` solves general nonsymmetric linear systems Ax = b using the Bi-Conjugate Gradient Stabilized method (Van der Vorst, 1992).

### Algorithm

BiCGSTAB combines the BiCG two-sided Lanczos approach with a polynomial stabilization step. Each iteration performs two matrix-vector products and produces smoother convergence than CGS without requiring A^T.

```
r_0 = b - A*x_0;  r_hat = r_0;  p_0 = r_0
rho_0 = r_hat^T * r_0

For k = 0, 1, 2, ...:
    1. First half-step (BiCG direction):
       v = A * M^{-1} * p_k          (preconditioned)
       alpha = rho_k / (r_hat^T * v)
       s = r_k - alpha * v

    2. Early termination check:
       If ||s|| / ||b|| < tol: accept x += alpha * M^{-1} * p, done

    3. Second half-step (stabilization):
       t = A * M^{-1} * s
       omega = (t^T * s) / (t^T * t)

    4. Solution and residual update:
       x_{k+1} = x_k + alpha * M^{-1} * p_k + omega * M^{-1} * s
       r_{k+1} = s - omega * t

    5. Convergence check:
       If ||r_{k+1}|| / ||b|| < tol: converged

    6. Direction update:
       rho_{k+1} = r_hat^T * r_{k+1}
       beta = (rho_{k+1} / rho_k) * (alpha / omega)
       p_{k+1} = r_{k+1} + beta * (p_k - omega * v)
```

### When to Use BiCGSTAB vs GMRES

- **BiCGSTAB** uses O(n) storage and has no restart parameter. Good when memory is limited or when restarted GMRES stalls due to information loss at restarts.
- **GMRES(k)** uses O(n*k) storage and is generally more robust. Better when the restart parameter k can be set large enough to cover the convergence horizon.
- For symmetric positive-definite systems, **CG** is preferred. For symmetric indefinite, **MINRES** is preferred.

### Breakdown Conditions

- **rho = 0:** r_hat^T * r = 0. The BiCG component has broken down. The solver returns the current best solution.
- **omega = 0:** The stabilization polynomial has failed. The half-step x += alpha * p_hat may still be useful.
- **r_hat^T * v = 0:** Breakdown in the BiCG direction computation.

### Workspace

6 vectors of length n (unpreconditioned) or 8 vectors (preconditioned): r, r_hat, p, v, s, t, plus optionally p_hat and s_hat for preconditioned variants.

## Stagnation Detection

All iterative solvers (CG, GMRES, MINRES, BiCGSTAB) support optional stagnation detection via the `stagnation_window` parameter in their options structs.

### Mechanism

When `stagnation_window > 0`, the solver maintains a ring buffer of the last N relative residual norms. After the buffer fills, it checks whether the ratio of the maximum to minimum residual in the window is less than 1.01 (i.e., less than 1% variation). If so, the residual has effectively plateaued, and the solver exits early with `stagnated = 1` in the result struct.

### When Stagnation Occurs

- **CG/MINRES:** Stagnation typically occurs when the tolerance is set below the achievable accuracy for the given condition number and floating-point precision. For a system with condition number κ, CG/MINRES can typically reduce the residual to O(κ·ε_mach).
- **GMRES:** Stagnation often manifests as restarts failing to improve the true residual. Small restart parameters (restart << n) exacerbate this.
- **BiCGSTAB:** Can exhibit erratic convergence; stagnation detection smooths over the noise via the sliding window.

### Usage

```c
sparse_iter_opts_t opts = {
    .max_iter = 5000,
    .tol = 1e-14,
    .stagnation_window = 15   /* detect stagnation over 15 iterations */
};
sparse_iter_result_t result;
sparse_solve_cg(A, b, x, &opts, NULL, NULL, &result);
if (result.stagnated)
    printf("Solver stagnated at residual %.3e after %d iterations\n",
           result.residual_norm, (int)result.iterations);
```

## Convergence Diagnostics

All iterative solvers support optional per-iteration residual history recording and a user-supplied verbose callback.

### Residual History

Set `residual_history` to a caller-allocated array and `residual_history_len` to its capacity. The solver fills `residual_history[i]` with the relative residual norm at iteration `i`. The actual number of entries written is returned in `result.residual_history_count`.

### Verbose Callback

Set `callback` to a user function of type `sparse_iter_callback_fn` to receive per-iteration progress reports. The callback receives a `sparse_iter_progress_t` struct with the current iteration number, residual norm, and solver name. When a callback is provided, the default `fprintf(stderr, ...)` verbose output is suppressed.

## ILU(0) Preconditioner

`sparse_ilu_factor()` computes an Incomplete LU factorization that preserves the sparsity pattern of the original matrix.

### Algorithm (IKJ variant)

```
W = copy(A)
for i = 1 to n-1:
    for each k < i where W(i,k) != 0:
        W(i,k) = W(i,k) / W(k,k)              (multiplier → L)
        for each j > k where W(k,j) != 0:
            if W(i,j) exists in sparsity pattern:
                W(i,j) -= W(i,k) * W(k,j)      (update)
            else:
                drop                             (ILU(0) no-fill rule)
Extract L (unit lower, below diagonal), U (upper, with diagonal)
```

### Key Properties

- **No fill-in:** L+U has nnz ≤ nnz(A). Only positions present in A are modified.
- **Approximate:** L*U ≈ A, not exact. The quality depends on how much fill would have been dropped.
- **Preconditioner application:** Solve L*U*z = r via forward substitution (L*y = r) then backward substitution (U*z = y).
- **Limitation:** Requires nonzero diagonal entries. Matrices with structurally zero diagonal (e.g., west0067) cause `SPARSE_ERR_SINGULAR`.

### Effectiveness

ILU(0) preconditioning typically reduces CG iteration count by 3-16× and can make GMRES converge where unpreconditioned GMRES stalls (e.g., steam1: 2000 iterations → 2 iterations).

## ILUT Preconditioner

`sparse_ilut_factor()` computes an Incomplete LU factorization with Threshold dropping, allowing controlled fill-in beyond the original sparsity pattern.

### Algorithm

ILUT uses a dense row accumulator and dual drop rules:

```
For each row i = 0..n-1:
    w = row i of A (scattered into dense workspace)
    row_norm = ||w||_2
    For each k < i where w[k] != 0:
        mult = w[k] / U(k,k)
        if |mult| < tol * row_norm: drop (set w[k] = 0)
        else: w[k] = mult; w[j] -= mult * U(k,j) for j > k
    Diagonal modification: if |w[i]| < 1e-30:
        eps = (row_norm > 0) ? tol * row_norm : 1e-10
        w[i] = sign(w[i]) * eps
    Apply dual dropping:
        L entries: keep at most max_fill largest |entries| in columns < i
        U entries: keep at most max_fill largest |entries| in columns > i (always keep diagonal)
    Insert surviving entries into L and U
```

### Comparison with ILU(0)

| Property | ILU(0) | ILUT |
|----------|--------|------|
| Fill-in | None beyond A's pattern | Controlled via tol and max_fill |
| Zero diagonal | Fails (SINGULAR) | Diagonal modification |
| Quality | Depends on A's pattern | Tunable via parameters |
| Cost | O(nnz) per row | O(nnz + fill) per row |

### Parameters

- `tol`: entries with |value| < tol * ||row|| are dropped (default: 1e-3)
- `max_fill`: maximum fill entries kept per row in L and U (default: 10)

## IC(0) Preconditioner

`sparse_ic_factor()` computes an Incomplete Cholesky factorization that preserves the sparsity pattern of the lower triangle of A. IC(0) is the symmetric analogue of ILU(0): it produces L such that L*L^T ≈ A.

### Algorithm (left-looking, column-by-column)

```
For k = 0 to n-1:
    Gather column k of lower triangle of A into dense workspace
    For each j < k where L(k,j) != 0:
        For each i >= k where L(i,j) != 0 and A(i,k) != 0:
            val[i] -= L(k,j) * L(i,j)
    L(k,k) = sqrt(val[k])        (diagonal — must be positive)
    L(i,k) = val[i] / L(k,k)     (off-diagonal, only at positions in A's pattern)
Build U = L^T
```

### Key Properties

- **No fill-in:** L has the same sparsity pattern as the lower triangle of A.
- **Symmetric:** L*L^T preserves symmetry, making IC(0) the natural preconditioner for CG and MINRES on SPD systems.
- **SPD required:** The input must be SPD. Indefinite or non-symmetric matrices cause `SPARSE_ERR_NOT_SPD`.
- **Preconditioner application:** Solve L*L^T*z = r via forward substitution (L*y = r) then backward substitution (L^T*z = y).

### Comparison with ILU(0)

| Property | IC(0) | ILU(0) |
|----------|-------|--------|
| Input requirement | SPD only | General square |
| Storage | nnz(L) ≈ nnz(lower(A)) | nnz(L) + nnz(U) ≈ nnz(A) |
| Symmetry preserved | Yes (L*L^T) | No (L*U) |
| CG compatibility | Natural (SPD preconditioner) | Works but not symmetric |
| Iteration count | Comparable to ILU(0) on SPD | Comparable to IC(0) on SPD |

On SuiteSparse bcsstk04 (132×132 SPD): both IC(0)-CG and ILU(0)-CG converge in 39 iterations (vs 653 unpreconditioned).

## Preconditioning

A preconditioner M ≈ A transforms the system to improve the condition number, accelerating iterative solver convergence.

### Left Preconditioning

Instead of solving Ax = b, solve M^{-1}Ax = M^{-1}b. The preconditioned system has condition number κ(M^{-1}A) which is ideally close to 1.

### Right Preconditioning

Instead of solving Ax = b directly, introduce y = Mx and solve A*M^{-1}*y = b for y via GMRES, then recover x = M^{-1}*y.

The key advantage: the GMRES residual norm equals the true residual ||b - Ax|| (no gap between preconditioned and true residual). With left preconditioning, the preconditioned residual may converge while the true residual lags behind.

Set `opts.precond_side = SPARSE_PRECOND_RIGHT` to enable right preconditioning.

### Preconditioner Interface

The library uses a callback-based preconditioner interface:

```c
typedef sparse_err_t (*sparse_precond_fn)(const void *ctx, idx_t n,
                                          const double *r, double *z);
```

The callback solves Mz = r given input r and outputs z. Available preconditioners:

| Preconditioner | Function | Best for | Quality |
|----------------|----------|----------|---------|
| ILU(0) | `sparse_ilu_precond` | General matrices | Good (3-1000× speedup) |
| ILUT | `sparse_ilut_precond` | Matrices with zero diagonal | Better than ILU(0), tunable |
| IC(0) | `sparse_ic_precond` | SPD matrices | Comparable to ILU(0), preserves symmetry |
| Cholesky | Custom wrapper | SPD matrices | Exact (1 iteration) but expensive setup |
| Diagonal (Jacobi) | Custom wrapper | Poorly scaled matrices | Modest improvement |
| Identity | Pass NULL | Baseline | No preconditioning |

### Selection Guide

- For SPD systems: use IC(0)-CG or ILU(0)-CG (best balance of setup cost and iteration reduction)
- For symmetric indefinite: use MINRES with Jacobi preconditioning, or LDL^T direct solve
- For unsymmetric systems: use ILU-GMRES (right preconditioning recommended)
- If ILU(0) fails (zero diagonal): use ILUT with diagonal modification
- If exact solution needed: use direct solver (LU, Cholesky, LDL^T, or QR)

## Sparse QR Factorization

`sparse_qr_factor()` computes the column-pivoted QR factorization A*P = Q*R using Householder reflections.

### Householder Reflections

Given a vector x, compute v and beta such that (I - beta*v*v^T)*x = ||x||*e_1:

```
v = x
v[0] += sign(x[0]) * ||x||
beta = 2 / (v^T * v)
```

Application to a vector y: y = y - beta * v * (v^T * y), requiring only O(len) work.

### Column-Pivoted QR Algorithm

```
Initialize: column norms, identity permutation
For k = 0 to min(m,n)-1:
    Select pivot: column with largest remaining norm
    Swap columns k and pivot
    Compute Householder vector v_k for column k below diagonal
    Check rank: if |R(k,k)| < tol, stop (rank = k)
    Apply H_k to remaining columns k+1..n-1
    Downdate column norms: ||col_j||^2 -= col_j[k]^2
Store R as sparse upper triangular, Q as Householder vectors (v_k, beta_k)
```

### Column Pivoting

Column pivoting ensures |R(k,k)| >= |R(k+1,k+1)|, providing:
- **Rank revelation:** small R diagonals indicate rank deficiency
- **Numerical stability:** largest norm column eliminated first
- **Permutation tracking:** col_perm[k] = original column index

### Least-Squares Solving

For overdetermined systems (m > n), `sparse_qr_solve()` computes min ||Ax - b||_2:

```
c = Q^T * b
Back-substitute: R[0:rank, 0:rank] * x_p = c[0:rank]
Unpermute: x[col_perm[i]] = x_p[i]
Residual: ||c[rank:]||_2
```

### Rank Estimation and Null Space

- `sparse_qr_rank(qr, tol)`: counts R diagonal entries exceeding tol * |R(0,0)|
- `sparse_qr_nullspace(qr, tol, basis, ndim)`: for each null column j, solves R*z = -R[:,j] and forms the null-space vector [z; e_j], unpermuted to original column space

### Column Reordering

Optional AMD reordering on A^T*A sparsity pattern can reduce fill-in in R:

```c
sparse_qr_opts_t opts = { .reorder = SPARSE_REORDER_AMD };
sparse_qr_factor_opts(A, &opts, &qr);
```

## Parallel SpMV (OpenMP)

When compiled with `-DSPARSE_OPENMP`, the sparse matrix-vector product `sparse_matvec()` is parallelized using OpenMP.

### Implementation

```c
#pragma omp parallel for schedule(dynamic, 64)
for (log_i = 0; log_i < nrows; log_i++) {
    // compute y[log_i] = sum over row entries
}
```

### Design

- **Row-wise partitioning:** each thread computes a disjoint subset of output rows
- **No synchronization needed:** each row writes only to its own y[i]
- **Dynamic scheduling** with chunk size 64 handles load imbalance from varying row lengths
- **Compile-time guard:** `#ifdef SPARSE_OPENMP` ensures zero overhead when disabled

### Limitations

- The linked-list row traversal is inherently less cache-friendly than CSR-based SpMV
- Small matrices (n < 200) see no benefit due to thread overhead
- Speedup is best on larger matrices (n > 1000) with balanced row lengths

## Symbolic Analysis and Numeric Refactorization

### Motivation

Many applications solve sequences of linear systems A(k)*x = b where the sparsity pattern of A stays fixed but the numeric values change at each step. Examples include nonlinear Newton solvers (Jacobian has fixed structure), implicit time-steppers, and optimization with changing constraints. Repeating fill-reducing reordering and symbolic analysis at each step is wasteful.

The `sparse_analyze()` / `sparse_factor_numeric()` / `sparse_refactor_numeric()` API separates the symbolic phase (done once) from the numeric phase (done per system).

### Elimination Tree

The elimination tree (etree) of a symmetric n×n matrix A encodes the parent-child relationships during Cholesky elimination. Column j's parent in the etree is the smallest column index k > j such that L(k,j) ≠ 0 in the Cholesky factor.

**Algorithm (Liu's method):** Process columns j = 0..n-1. For each row entry i < j, find the root r of column i's subtree using union-find with path compression. If r ≠ j, set parent[r] = j and merge. Time: O(nnz · α(n)) where α is the inverse Ackermann function.

**Postorder traversal:** A DFS postorder of the etree visits children before parents, which is the natural bottom-up order for symbolic and numeric factorization.

### Column Counts

`sparse_colcount()` computes the exact number of nonzeros per column in the Cholesky factor L. For each column j (processed in postorder), the off-diagonal row indices are the union of:
- Rows from A's lower triangle in column j
- Rows propagated up from children of j in the etree (excluding j itself)

Because the current implementation explicitly unions and propagates row-index sets up the etree, its runtime is not strictly O(nnz(A)). In practice it scales with the size of the propagated symbolic pattern — typically closer to O(nnz(L)) in high-fill cases — while still using the etree and a marker array to avoid duplicate indices within a column.

### Symbolic Cholesky Factorization

`sparse_symbolic_cholesky()` computes the complete row-index structure of L in compressed-column format, using the same bottom-up etree traversal as column counts but storing the actual indices. Row indices are sorted within each column.

For Cholesky on SPD matrices, the symbolic structure is exact — every predicted position will have a nonzero value. For matrices with numeric cancellation (e.g., when the numeric Cholesky drops tiny fill-in entries below DROP_TOL), the symbolic structure is a superset.

### Symbolic LU Factorization

For unsymmetric LU with partial pivoting, the exact fill depends on the pivot sequence, which isn't known until numeric factorization. `sparse_symbolic_lu()` computes an upper bound by:

1. Building the sparse structure of A^T * A (the column interaction graph) as an explicit sparse pattern
2. Computing symbolic Cholesky of this symmetrized pattern
3. The resulting L structure bounds the actual L; its transpose bounds U

This is the standard approach from Gilbert and Peierls (1988).

### Analyze → Factor → Refactor Workflow

```c
// 1. Analyze once
sparse_analysis_opts_t opts = { SPARSE_FACTOR_CHOLESKY, SPARSE_REORDER_AMD };
sparse_analysis_t analysis = {0};
sparse_analyze(A, &opts, &analysis);

// 2. Factor
sparse_factors_t factors = {0};
sparse_factor_numeric(A, &analysis, &factors);
sparse_factor_solve(&factors, &analysis, b, x);

// 3. Refactor with new values (same pattern)
sparse_refactor_numeric(A_new, &analysis, &factors);
sparse_factor_solve(&factors, &analysis, b2, x2);

// 4. Cleanup
sparse_factor_free(&factors);
sparse_analysis_free(&analysis);
```

The analysis object stores: fill-reducing permutation, elimination tree, postorder, and symbolic column structure. The factors object stores the numeric L (and U for LU, D for LDL^T).

## COLAMD Column Ordering

### Algorithm

COLAMD (Column Approximate Minimum Degree) computes a fill-reducing column permutation for unsymmetric matrices. Unlike AMD, which operates on the symmetrized graph A+A^T, COLAMD works on the column adjacency graph: columns i and j are adjacent if they share a nonzero row in A.

**Column adjacency graph construction:** For each column j, walk j's column entries to find rows containing j, then walk each row to find other columns. A per-column marker array prevents duplicate counting. Dense rows (nnz > 10·√n) are skipped to control cost.

**Minimum degree elimination:** The column adjacency graph is converted to bitset format for O(1) neighbor queries. At each step, the column with minimum degree is eliminated: its adjacency is merged into all remaining neighbors (modeling fill-in), and degrees are updated via popcount. This produces a column permutation that tends to reduce fill-in during QR or LU factorization.

### When to Use Each Ordering

| Ordering | Best for | Handles rectangular | Cost |
|----------|----------|-------------------|------|
| **RCM** | Bandwidth reduction, banded systems | No (square only) | O(nnz) |
| **AMD** | Symmetric fill reduction (Cholesky, LDL^T) | No (square only) | O(n² + nnz) |
| **COLAMD** | Unsymmetric column fill reduction (QR, LU) | Yes | O(n² + nnz) |

For QR factorization, COLAMD is recommended because it directly considers column structure without the overhead of forming A^T*A. For symmetric problems (Cholesky, LDL^T), AMD is preferred.

## QR Minimum-Norm Least Squares

### Problem

For an underdetermined system A*x = b where m < n (more unknowns than equations), there are infinitely many solutions. The minimum-norm solution x* has the smallest ||x||_2 among all solutions.

### Algorithm

1. **Transpose:** Build A^T (an n×m matrix with n > m)
2. **Factor:** Compute QR of A^T: A^T·P = Q·R, where R is m×m upper triangular
3. **Permute:** bp = P^T · b
4. **Forward substitute:** Solve R^T · y = bp (R^T is lower triangular)
5. **Apply Q:** x = Q · y

The result x has minimum 2-norm because the transformation preserves norms and the forward substitution produces the unique solution in the range of A^T.

### Iterative Refinement

`sparse_qr_refine_minnorm()` improves an initial minimum-norm solution by repeatedly:
1. Computing residual r = b - A·x
2. Solving for a minimum-norm correction dx via `sparse_qr_solve_minnorm(A, r, dx)`
3. Updating x += dx

Stops when the residual stops decreasing or max iterations are reached.

## Symmetric Eigensolvers (Sprint 20)

`sparse_eigs_sym(A, k, opts, result)` computes `k` extreme or near-sigma eigenpairs of a symmetric sparse matrix A via a **Lanczos-based Krylov method with full MGS reorthogonalisation**.  The outer solve grows the Krylov subspace dimension and retries as needed rather than performing a true thick restart (a Wu/Simon thick-restart backend that preserves a compact Ritz space across restarts is planned for Sprint 21).  An optional **shift-invert** mode for interior eigenvalues composes with the LDL^T dispatch in `sparse_ldlt.h`.

### The 3-term Lanczos recurrence

Given a symmetric A and a starting vector `v_0` (unit norm), the Lanczos recurrence builds an orthonormal Krylov basis `V = [v_0, v_1, ..., v_{m-1}]` and a symmetric tridiagonal `T = V^T A V` by:

```
for k = 0, 1, 2, ..., m-1:
    w        = A · v_k − beta_{k-1} · v_{k-1}      (beta_{-1} := 0)
    alpha_k  = <w, v_k>
    w        = w − alpha_k · v_k
    beta_k   = ||w||
    if beta_k ≈ 0: invariant subspace — stop
    v_{k+1}  = w / beta_k
```

`alpha[0..m-1]` populates T's main diagonal; `beta[0..m-2]` populates the sub/super-diagonal.  In exact arithmetic `V` is orthonormal and T's eigenvalues — the **Ritz values** — approximate A's eigenvalues, with the approximation sharpest at the extremes of the spectrum (that's why Lanczos converges quickly to `LARGEST` / `SMALLEST` and slowly to interior points — hence shift-invert for interior queries).

### Full reorthogonalisation (why, not just how)

Paige (1972) showed that without reorthogonalisation V^T V drifts from I as k grows, and **ghost Ritz values** — spurious copies of already-converged eigenvalues — emerge in T's spectrum around `k ≈ cond(A)` iterations.  The library's default is full **modified Gram-Schmidt (MGS)** reorthogonalisation: after the standard 3-term step computes a tentative w, the helper subtracts the projection of w onto every prior Lanczos vector:

```
for j = 0, 1, ..., k-1:
    dot = <w, v_j>
    w  -= dot · v_j
```

MGS has the same O(k·n) asymptotic cost per step as classical Gram-Schmidt but is numerically more stable under cancellation because each subtraction uses the currently-orthogonalised w rather than a cached dot-product of the original.  The resulting orthogonality drift stays at 1e-12 or better up to moderate k on well-conditioned Krylov bases; classical GS would bottom out around 1e-6 on wide-spectrum matrices.

The `opts.reorthogonalize = 0` escape hatch disables reorth for the occasional cheap-smoke-test where ghost values don't matter, but every production call runs with reorth on.

### Parallel reorthogonalisation: MGS stays serial in j

Sprint 21 Day 5-6 parallelises MGS under `-DSPARSE_OPENMP`, but only in the inner axis.  The outer `j` loop — sweeping the prior Lanczos vectors — **stays serial**.  Each iteration of the `j` loop reads the `w` that the previous iteration just modified, so the iterations have a read-after-write dependency that parallelism cannot break.  Only the inner dot-product `<w, v_j>` and the inner daxpy `w -= dot · v_j` are data-parallel in the length-`n` axis, and those get `#pragma omp parallel for reduction(+:dot)` and `#pragma omp parallel for` respectively.

Why not classical Gram-Schmidt?  CGS computes every `dot_j = <w_original, v_j>` in one parallel pass, then subtracts them all in a second pass: the `j` loop parallelises trivially.  But each subtraction uses the *original* `w`, not the partially-orthogonalised one, so cancellation errors compound and the orthogonality drift bottoms out around 1e-6 on wide-spectrum matrices — an order of magnitude worse than MGS's 1e-12.  On the eigensolver's convergence criterion (`|beta_m · y_{m-1,j}| / |theta_j|`) that difference determines whether the residual gate fires at 1e-10 or plateaus at 1e-8 because of ghost Ritz pairs.  We pay the serial-j cost to keep the stability; Sprint 21's parallel speedup comes from the `i`-axis work instead.  (There are compromises — iterated CGS, block MGS, TSQR — but the library's size/complexity budget doesn't justify them for the measured 2× at 4 threads we get from the simple pattern.)

A compile-time threshold `SPARSE_EIGS_OMP_REORTH_MIN_N` (default 500) gates both pragmas via an OpenMP `if (n >= threshold)` clause.  Below the threshold the `parallel for` runs on a single-thread team — zero fork/join overhead, serial performance.  The fork/join overhead on macOS Homebrew libomp is 5-20 μs per parallel region, which exceeds the per-reorth work when `n < ~500`; see [`docs/planning/EPIC_2/SPRINT_21/bench_day6_omp_scaling.txt`](planning/EPIC_2/SPRINT_21/bench_day6_omp_scaling.txt) for the scaling sweep that motivates the threshold.

### Ritz extraction

Once Lanczos builds `(V, T)`, the Ritz pairs `(theta_j, V · y_j)` — where `(theta_j, y_j)` are the eigenpairs of T — approximate A's eigenpairs.  The library extracts both eigenvalues and eigenvectors of T via `tridiag_qr_eigenpairs`, an implicit-QR-with-Wilkinson-shift variant that accumulates the Givens rotations into an orthogonal Y matrix so the full-problem Ritz vectors come out as `V · Y[:, j]` via one gemv per requested column.

### Outer loop: growing m on retry

The practical convergence question is *how big m should be*.  The Day 13 implementation runs a single growing-m Lanczos sequence:

```
m = m_init = 3k + 30
loop:
    run lanczos(op, v0, m) → V, alpha, beta
    compute Ritz pairs (theta, Y) from (alpha, beta) via tridiag QR
    check Wu/Simon residual |beta_m · y_{m-1, j}| / |theta_j| ≤ tol for every top-k pair
    if converged: emit and return
    if m == m_cap: emit partial NOT_CONVERGED
    m += k + 20
```

Because `v_0` is deterministic and Lanczos is deterministic under fixed v_0, the first `m_prev` steps of each retry are bit-for-bit identical to the previous retry's basis; the extra `m_new − m_prev` steps are where the convergence tightens.  This **strictly extends** the Krylov basis on every pass — an earlier design that restarted with `v_0 := v_last` on non-convergence saturated at residual ~7e-3 on nos4 after 2000 iterations because the warm-start lost the prior convergence.  The growing-m variant lands at residual 4e-14 in 70 total iterations on the same fixture.

### Thick-restart Lanczos: bounded memory via arrowhead state

The grow-m outer loop's peak memory is `O(m_cap · n)` because the full Lanczos basis `V` lives across retries.  On bcsstk14 (n = 1806) at `m_cap = 500` that's ~7 MB; pushing `m_cap = n` for harder fixtures balloons it to ~26 MB.  Sprint 21's thick-restart backend (`SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`) replaces the grow-m strategy with the Wu/Simon (2000) — Stathopoulos & Saad (2007) restart mechanism that bounds peak memory at `O((k + m_restart) · n)` regardless of total iteration count.

**The arrowhead state.**  After a Lanczos phase of length `m_restart` and Ritz extraction, the locked top-`k_locked` Ritz pairs are kept; the rest of `V` is discarded.  The locked block + the trailing residual seed the next phase, but T's structure is no longer plain tridiagonal — it becomes an *arrowhead* matrix:

```
       [ θ_0   0    0    β_0   0    0    0   ]
       [  0   θ_1   0    β_1   0    0    0   ]
       [  0    0   θ_2   β_2   0    0    0   ]
T  =   [ β_0  β_1  β_2   α_3  β_3   0    0   ]
       [  0    0    0    β_3  α_4  β_4   0   ]
       [  0    0    0    0    β_4  α_5  β_5  ]
       [  0    0    0    0    0    β_5  α_6  ]
```

The top-left `k_locked × k_locked` block is diagonal (the locked Ritz values `θ_j`), the spoke `β_j = β_m · y_{m-1, j}` couples each locked pair to the new Lanczos extension at row/column `k_locked`, and the trailing rows are the standard 3-term Lanczos α/β values from the new phase.  This shape is the "arrow" — three diagonals meeting at a point.

**Reduction to tridiagonal.**  The existing Sprint 20 `tridiag_qr_eigenpairs` consumes a symmetric tridiagonal, not an arrowhead.  The library's `s21_arrowhead_to_tridiag` materialises the arrowhead as a dense K × K symmetric matrix (K = `k_locked + m_ext`) and runs classical Householder reflections (Golub & Van Loan §8.3.1) to chase the spoke entries into a tridiagonal form with the same spectrum — total work O(K^3) per restart, but K stays small (typically `k_locked + m_restart ≤ 100`), so each reduction is microsecond-scale.

**Locked-pair preservation.**  Wu/Simon's claim is that each restart **strictly extends** the converged subspace: the locked Ritz pairs sit in the diagonal block of the new T, so the next Ritz extraction picks up exactly those eigenvalues plus refinements from the new Lanczos extension.  Phase `i+1` cannot worsen pair `j`'s residual relative to phase `i` (modulo finite-precision noise) — the bench's `test_thick_restart_locked_progress_monotone` Day 12 test verifies this at the public-API level by running two iteration budgets and asserting `r_long ≤ r_short`.

**Memory bound.**  Peak `V` columns equals `m_restart + k_locked_cap` plus a small transient (`V_locked_tmp` during the pick-locked step), reported in `result.peak_basis_size`.  For bcsstk14 at `k = 5`, `m_restart = 30`, `k_locked_cap = 5`: peak `V ≈ 40 cols × 1806 × 8 B ≈ 565 KB` — ~15× smaller than the grow-m path's 7 MB.  AUTO routes to thick-restart when `n ≥ SPARSE_EIGS_THICK_RESTART_THRESHOLD` (default 500); explicitly opt in via `opts->backend = SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART` below the threshold for memory profiling.

### Wu/Simon convergence criterion

For a Ritz pair `(theta_j, V · y_j)` the true eigen-equation residual satisfies `||A · V · y_j − theta_j · V · y_j|| = |beta_m · y_{m-1, j}|` (Paige 1972; Bai et al. 2000), where `beta_m` is the final Lanczos `beta` value (which `lanczos_iterate` stores in `beta[m-1]` as the residual norm of the last unaccepted w vector).  Since `Y` is orthogonal, `||V · y_j|| = 1`, so this is already the absolute residual; dividing by `|theta_j|` gives the relative residual that the library reports in `result.residual_norm`.  Wu/Simon is cheap (one row of Y per pair — no extra matvecs), directly bounds the eigen-equation error, and matches what callers care about.

### Shift-invert mode (`SPARSE_EIGS_NEAREST_SIGMA`)

Interior eigenvalues converge painfully slowly in direct Lanczos because they're neither end of the spectrum.  Shift-invert transforms the problem:

```
op = (A − sigma·I)^{-1}
eigenvalues of op = 1 / (lambda − sigma)
```

Values of `lambda` near `sigma` become **largest in magnitude** on `op`'s spectrum — exactly where Lanczos converges fastest.  The library:

1. Copies A and subtracts `sigma` from each diagonal entry to form `A − sigma·I`.
2. Factors via `sparse_ldlt_factor_opts(SPARSE_LDLT_BACKEND_AUTO)` — the Sprint 20 Days 4-6 dispatch routes to the CSC supernodal path on `n ≥ SPARSE_CSC_THRESHOLD`, so big indefinite shifts benefit from the Day 3 batched LDL^T.
3. Drives Lanczos with `sparse_ldlt_solve` as the operator (one triangular solve per iteration against the pre-computed factor).
4. Post-processes Ritz values: `lambda_j = sigma + 1 / theta_j`.

Singular-shift case: if `sigma` coincides with an eigenvalue of A, `A − sigma·I` is singular and LDL^T returns `SPARSE_ERR_SINGULAR`; `sparse_eigs_sym` propagates it, and the caller perturbs sigma slightly and retries.  The inner factor's backend choice surfaces in `result.used_csc_path_ldlt` for observability.

### Convergence heuristics in practice

- **Well-separated extremes.** `LARGEST` / `SMALLEST` converge in O(sqrt(cond(A))) × k iterations on well-separated spectra.  nos4 k=5 LARGEST converges in 70 steps (~35 matvecs/requested pair).
- **Clustered spectra.** Bottom-cluster SPD matrices need close to the full Krylov basis (m ≈ n).  bcsstk04 k=3 SMALLEST hits m_cap = n = 132 and converges cleanly (shift-invert with `sigma ~ 1e-3` would be faster).
- **Shift-invert break-even.** When the target eigenvalues are ≥ 10% of the spectrum distance from the extremes, shift-invert beats direct even accounting for the one-time LDL^T factor cost.  The KKT n=150 run takes 39 Lanczos steps at sigma=0 vs 62 for direct SMALLEST — 34% faster wall time.

See `docs/planning/EPIC_2/SPRINT_20/bench_day13_lanczos.txt` for the measured numbers across SuiteSparse fixtures.  Sprint 21 Day 14's full sweep across all three backends and `which` modes lands at `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt` (and the 3-backend × 3-precond pivot at `docs/planning/EPIC_2/SPRINT_21/bench_day14_compare.txt`).

### LOBPCG: preconditioned block Rayleigh-Ritz

Sprint 21 Days 7-10 add Knyazev's (2001) Locally Optimal Block Preconditioned Conjugate Gradient as `SPARSE_EIGS_BACKEND_LOBPCG`.  Two regimes motivate a third backend:

1. **Ill-conditioned SPD.** When `cond(A)` reaches 1e6+, Lanczos's spectral-gap convergence rate slows to `1 − O(1/sqrt(cond))` per step.  A cheap preconditioner `M ≈ A` (IC(0) from `sparse_ic_factor`, LDL^T from `sparse_ldlt_factor`) accelerates LOBPCG to a rate determined by `cond(M^{-1}·A)` — often four or five orders of magnitude faster on the same fixture.  Lanczos has no inner preconditioning hook (shift-invert is the closest analogue, but it requires a near-eigenvalue `σ` to work).
2. **Block convergence.** When the requested eigenvalues are clustered, Lanczos converges them sequentially while LOBPCG converges them in parallel via the `block_size > k` mechanism.

**The three-block subspace.**  Each iteration maintains three n × `block_size` matrices stored column-major:

- `X` — current eigenvector approximations (init: deterministic golden-ratio per-column starting vectors).
- `W` — preconditioned residual (`M^{-1} · (AX − X·diag(theta))` when `opts->precond` is non-NULL; the raw residual `R` itself when NULL).
- `P` — previous search direction (init: 0; updated each step).

The block Rayleigh-Ritz step concatenates these into an n × (3·`block_size`) basis Q, orthonormalises it (per-column MGS with scale-aware breakdown ejection — the `s21_lobpcg_orthonormalize_block` helper reuses the Lanczos MGS kernel), forms the dense symmetric Gram matrix `G = Q^T · A · Q`, and diagonalises it via the same dense Jacobi rotation eigensolver used for the Day 2 thick-restart arrowhead reduction.  The selection step picks `block_size` Ritz pairs by `which` (LARGEST / SMALLEST / NEAREST_SIGMA via the same shift-invert wiring the Lanczos backends use); the new X / P come from the corresponding eigenvectors of G.

**P-update formulation.**  Knyazev's eq. 2.11 expresses `P_new` as the W and P contributions to the new X (the "search direction" component, excluding the X-block).  In exact arithmetic this matches the orthogonal-projection form `P_new = X_new − X · (X^T · X_new)`, which is what the library uses — when X stays orthonormal across iterations (which it does, because each X_new is built from an orthonormal Q via an orthogonal Y), the two formulas agree.  A **BLOPEX conditioning guard** (Stathopoulos 2007) inspects Jacobi's eigenvalue spread on G; when the smallest |theta_full| collapses below `scale · 1e-12`, treat the iteration as Gram-singular and reset `P_new = 0` (restarts the conjugate-gradient direction track on the next outer iteration).

**Soft-locking.**  Per `opts->lobpcg_soft_lock` (default ON): once a Ritz pair's residual passes `tol`, that column's W and P entries are zeroed before the next Rayleigh-Ritz step.  The orthonormaliser ejects the zero columns, shrinking the active subspace from `(bs + bs + bs)` to `(bs + bs_active_W + bs_active_P)`.  The locked X[:, j] stays in Q, so its Ritz pair is preserved by the RR step (X is in the basis, A·X[:, j] ≈ θ_j·X[:, j], and Y maps that column back to itself).

**Convergence.**  Per-column Wu/Simon residual `||R[:, j]|| / max(|θ_j|, scale)` matches the Lanczos backends' `result.residual_norm` semantics, so the tolerance has problem-physical meaning regardless of preconditioner choice.

**Preconditioning regime.**  LOBPCG's preconditioning naturally targets the SMALLEST end of the spectrum: `M^{-1}` amplifies the small-eigenvalue components of the residual.  For LARGEST modes, the preconditioner doesn't help directly (and can hurt — see the Day 14 `bench_day14_compare.txt` row for nos4 LARGEST + IC0).  The standard LARGEST-with-precond approach is op-negation (apply LOBPCG to `-A`'s SMALLEST), which the library doesn't currently wire — a candidate for a future sprint when the workload demands it.

**Memory.**  Peak `O((4·block_size + scratch) · n)` where the outer loop holds X, R, W, P (each n × `block_size`) plus the RR step's transient n × (3·block_size) Q and AQ scratch.  For `block_size ≤ 30` this is ~5 MB on bcsstk14 — comparable to thick-restart's ~500 KB but with much better convergence on ill-conditioned fixtures.

**AUTO routing.**  AUTO picks LOBPCG when `opts->precond != NULL`, `n ≥ SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` (default 1000), and the effective block size is at least 4.  Without a preconditioner LOBPCG generally underperforms thick-restart Lanczos on the well-conditioned corpus, so AUTO declines to pick it.  Override with explicit `opts->backend = SPARSE_EIGS_BACKEND_LOBPCG` to force the choice — useful for the precond-comparison rows in `bench_eigs --compare`.

### API consistency notes

The API surface mirrors the iterative-solver convention in `sparse_iterative.h`: `sparse_eigs_opts_t` carries all tuning knobs (`which`, `sigma`, `max_iterations`, `tol`, `reorthogonalize`, `compute_vectors`, `backend`, `block_size`, `precond`, `precond_ctx`, `lobpcg_soft_lock`), and `sparse_eigs_t` uses caller-owned buffers for `eigenvalues` / `eigenvectors` plus library-written scalar output fields (`n_requested`, `n_converged`, `iterations`, `residual_norm`, `used_csc_path_ldlt`, `peak_basis_size`, `backend_used`).  All Sprint 21 additions to either struct are trailing fields, so designated-initialiser callers from Sprint 20 compile unchanged with library-default behaviour for the new knobs.  No library-side allocation means no `sparse_eigs_free` helper — callers free their own buffers.
