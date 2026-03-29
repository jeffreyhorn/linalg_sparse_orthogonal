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

## Singularity Detection: Relative Tolerance

Backward substitution checks each diagonal element U(i,i) for near-zero values before dividing. This check uses a **relative** threshold:

```
threshold = DROP_TOL × ||A||_inf
```

where `||A||_inf` is the infinity norm (maximum absolute row sum) of the original matrix, computed and cached during `sparse_lu_factor()`.

### Why Relative?

An absolute threshold (the previous approach, using bare `DROP_TOL = 1e-14`) fails for two classes of matrices:

1. **Matrices with uniformly small entries** — e.g., `A = 1e-16 × I`. Every diagonal entry is `1e-16`, which is smaller than `1e-14`, causing a false singular detection even though the matrix is perfectly well-conditioned (condition number = 1).

2. **Matrices with very large entries** — e.g., diagonal entries of `1e15`. With an absolute threshold, even a near-zero diagonal entry like `1e-10` would pass, even though it represents catastrophic loss of significance relative to the matrix scale.

The relative threshold `DROP_TOL × ||A||_inf` scales with the matrix, so:
- A `1e-16 × I` matrix has threshold `1e-14 × 1e-16 = 1e-30`, correctly allowing `1e-16` pivots.
- A matrix with `||A||_inf = 1e15` has threshold `1e-14 × 1e15 = 10`, correctly rejecting pivots smaller than 10.

### Fallback

If `factor_norm` is not set (e.g., backward substitution is called on a matrix that was not factored via `sparse_lu_factor`), the absolute `DROP_TOL` is used as a fallback for backward compatibility.

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

Before LU factorization, a symmetric permutation P·A·Pᵀ can dramatically reduce fill-in. Two algorithms are provided:

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

### Approximate Minimum Degree (AMD)

Greedy elimination ordering using bitset adjacency for efficient set operations.

**Algorithm:**
1. Build bitset adjacency matrix from the symmetrized sparsity pattern
2. At each step: eliminate the uneliminated node with smallest degree
3. Merge eliminated node's neighbors' adjacency sets (models fill-in)
4. Update degrees

**Characteristics:**
- O(n² · n/64) time with bitset operations
- Generally better fill-in reduction for unstructured matrices
- On orsirr_1 (1030×1030 oil reservoir): fill-in 78k→55k (29% reduction)
- On nos4 (100×100 structural): fill-in 1510→1174 (22% reduction)

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
| Cholesky | Custom wrapper | SPD matrices | Exact (1 iteration) but expensive setup |
| Diagonal (Jacobi) | Custom wrapper | Poorly scaled matrices | Modest improvement |
| Identity | Pass NULL | Baseline | No preconditioning |

### Selection Guide

- For SPD systems: use ILU-CG (best balance of setup cost and iteration reduction)
- For unsymmetric systems: use ILU-GMRES (right preconditioning recommended)
- If ILU(0) fails (zero diagonal): use ILUT with diagonal modification
- If exact solution needed: use direct solver (LU, Cholesky, or QR)

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
