#ifndef SPARSE_MATRIX_INTERNAL_H
#define SPARSE_MATRIX_INTERNAL_H

/*
 * Private header: internal struct definitions for sparse_matrix.c and
 * sparse_lu.c. NOT part of the public API.
 */

#include "sparse_matrix.h" /* picks up SPARSE_NODES_PER_SLAB, SPARSE_DROP_TOL */
#include <float.h>
#include <math.h>
#include <stdatomic.h>

/*
 * Thread safety invariants:
 *
 * - cached_norm (_Atomic double): safe for concurrent reads/writes.
 *   Uses relaxed ordering; the computation is idempotent (all threads
 *   compute the same result from immutable linked-list structure).
 *
 * - factor_norm (plain double): written once during factorization,
 *   read during solve.  No race because factorization completes before
 *   any solve call begins (the caller must ensure this).
 *
 * - Pool, row_headers, col_headers, nnz: mutated by insert/remove and
 *   factorization.  NOT safe for concurrent access.  Protected by the
 *   optional SPARSE_MUTEX (insert/remove only) or external sync.
 *
 * - Permutation arrays: mutated only during factorization (single-threaded).
 *
 * Optional mutex support: compile with -DSPARSE_MUTEX to enable.
 * When enabled, sparse_insert and sparse_remove lock the matrix mutex
 * to serialize concurrent insert/remove calls on the same matrix.
 * Factorization functions are NOT mutex-protected.
 * Default: disabled (zero overhead).
 */
#ifdef SPARSE_MUTEX
#include <pthread.h>
#define SPARSE_LOCK(mat) pthread_mutex_lock(&(mat)->mtx)
#define SPARSE_UNLOCK(mat) pthread_mutex_unlock(&(mat)->mtx)
#else
#define SPARSE_LOCK(mat) ((void)0)
#define SPARSE_UNLOCK(mat) ((void)0)
#endif

#define NODES_PER_SLAB SPARSE_NODES_PER_SLAB
#define DROP_TOL SPARSE_DROP_TOL

/* ═══════════════════════════════════════════════════════════════════════════
 * Tolerance strategy (Sprint 11)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Problem: 21 hardcoded tolerance sites across the library use absolute
 * constants (1e-30 or 1e-300) that do not scale with matrix magnitude.
 * A well-conditioned 1e+20-scale matrix may be flagged singular, while a
 * truly singular 1e-10-scale matrix may slip through.
 *
 * Reference pattern: sparse_lu.c already uses relative tolerance:
 *   sing_tol = (factor_norm > 0) ? DROP_TOL * factor_norm : DROP_TOL
 *
 * Strategy: replace absolute magic numbers with sparse_rel_tol(), which
 * computes  max(user_tol * reference_norm, DBL_MIN * 100).  The absolute
 * floor (DBL_MIN * 100 ≈ 2.2e-306) prevents underflow when the reference
 * norm is itself tiny, while the relative branch scales correctly for
 * large-magnitude matrices.
 *
 * Site catalog (21 sites, 3 categories):
 *
 * Category A — Singularity detection (11 sites)
 *   Should use relative tolerance scaled by matrix/factor norm.
 *   sparse_cholesky.c:281   L(i,i) forward-sub singularity check
 *   sparse_cholesky.c:304   L(i,i) backward-sub singularity check
 *   sparse_ilu.c:94         ILU(0) diagonal during factorization
 *   sparse_ilu.c:124        ILU(0) post-factor diagonal verification
 *   sparse_ilu.c:237        ILU precond solve: U diagonal check
 *   sparse_ilu.c:481        ILUT factorization: U(k,k) pivot check
 *   sparse_ilu.c:514        ILUT diagonal stabilization threshold
 *   sparse_ilu.c:568        ILUT final diagonal check
 *   sparse_lu_csr.c:267     Dense LU backward-sub diagonal check
 *   sparse_lu_csr.c:1137    CSR LU solve: U diagonal check
 *   sparse_lu_csr.c:1220    CSR LU multi-RHS solve: U diagonal check
 *
 * Category B — Convergence / deflation (6 sites)
 *   Absolute floor is acceptable but should derive from the bidiagonal or
 *   tridiagonal norm rather than a fixed constant.
 *   sparse_svd.c:313        Bidiag SVD: absolute tolerance floor
 *   sparse_svd.c:430        2×2 SVD: Jacobi rotation near-zero guard
 *   sparse_dense.c:283      Tridiag QR: off-diagonal deflation floor
 *   sparse_dense.c:294      Tridiag QR: inner-block deflation floor
 *   sparse_qr.c:947         QR solve: R diagonal (rank-deficient → 0)
 *   sparse_qr.c:1142        QR nullspace: R diagonal check
 *
 * Category C — Lucky breakdown / normalization guards (4 sites)
 *   These guard against division by zero in iterative/Lanczos contexts
 *   where an invariant subspace has been found.  Absolute floor is fine.
 *   sparse_iterative.c:590  GMRES: lucky breakdown (H(j+1,j) ≈ 0)
 *   sparse_iterative.c:651  GMRES: Hessenberg diagonal back-sub
 *   sparse_svd.c:836        Lanczos: p-vector normalization guard
 *   sparse_svd.c:875        Lanczos: q-vector normalization guard
 *
 * Application plan (Days 3-5):
 *   Category A: replace with sparse_rel_tol(factor_norm, DROP_TOL)
 *   Category B: replace with sparse_rel_tol(local_norm, DROP_TOL)
 *   Category C: replace with sparse_rel_tol(0, DROP_TOL) — uses floor only
 * ═══════════════════════════════════════════════════════════════════════════
 */

/**
 * sparse_rel_tol — compute a relative tolerance with absolute floor.
 *
 * Returns  max(user_tol * reference_norm, DBL_MIN * 100).
 *
 * @param reference_norm  A representative norm (e.g. ||A||_inf at factor
 *                        time, bidiagonal norm, or 0 for pure floor).
 * @param user_tol        Relative tolerance (e.g. DROP_TOL = 1e-14).
 * @return                Threshold below which a value is "zero".
 */
static inline double sparse_rel_tol(double reference_norm, double user_tol) {
    double rel = user_tol * fabs(reference_norm);
    double floor = DBL_MIN * 100.0;
    return rel > floor ? rel : floor;
}

typedef struct Node {
    idx_t row;
    idx_t col;
    double value;
    struct Node *right; /* next in same row (sorted by col)  */
    struct Node *down;  /* next in same col (sorted by row)  */
} Node;

typedef struct NodeSlab {
    Node nodes[NODES_PER_SLAB];
    struct NodeSlab *next;
    idx_t used;
} NodeSlab;

typedef struct NodePool {
    NodeSlab *head;
    NodeSlab *current;
    Node *free_list; /* singly-linked free list via ->right */
    idx_t num_slabs;
} NodePool;

typedef struct SparseMatrix {
    idx_t rows;
    idx_t cols;
    Node **row_headers;
    Node **col_headers;
    idx_t *row_perm;     /* logical -> physical row */
    idx_t *inv_row_perm; /* physical -> logical row */
    idx_t *col_perm;     /* logical -> physical col */
    idx_t *inv_col_perm; /* physical -> logical col */
    NodePool pool;
    idx_t nnz;
    _Atomic double cached_norm; /* cached ||A||_inf, -1.0 = invalid.
                                 * Atomic to allow concurrent norminf() calls
                                 * on the same matrix from solve threads. */
    double factor_norm;         /* ||A||_inf at factorization time, for relative tol */
    int factored;               /* 1 after factorization, 0 otherwise.
                                 * Solve functions check this to catch
                                 * solve-before-factor bugs at runtime. */
    idx_t *reorder_perm;        /* fill-reducing reorder: perm[new] = old, or NULL */
#ifdef SPARSE_MUTEX
    pthread_mutex_t mtx; /* optional mutex for concurrent mutation */
#endif
} SparseMatrix;

/*
 * Internal errno capture (defined in sparse_types.c)
 */
void sparse_set_errno_(int errnum);

/*
 * Internal pool operations (used by sparse_matrix.c and sparse_lu.c)
 */
Node *pool_alloc(NodePool *pool);
void pool_release(NodePool *pool, Node *node);
void pool_free_all(NodePool *pool);

/*
 * Build CSR adjacency graph of A+A^T (symmetrized, no self-loops).
 * adj_ptr[i]..adj_ptr[i+1]-1 index into adj_list for neighbors of i.
 * Caller must free both adj_ptr and adj_list.
 */
sparse_err_t sparse_build_adj(const SparseMatrix *A, idx_t **adj_ptr, idx_t **adj_list);

#endif /* SPARSE_MATRIX_INTERNAL_H */
