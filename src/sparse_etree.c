#include "sparse_alloc_internal.h"
#include "sparse_analysis_internal.h"
#include <stdint.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════
 * Union-find helpers for etree computation
 * ═══════════════════════════════════════════════════════════════════════ */

/* Find representative with path compression.
 * Precondition: 0 <= i < n where n is the ancestor array length. */
static idx_t uf_find(idx_t *ancestor, idx_t n, idx_t i) {
    if (i < 0 || i >= n)
        return i;
    while (ancestor[i] != i) {
        idx_t gi = ancestor[i];
        if (gi < 0 || gi >= n)
            break;
        ancestor[i] = ancestor[gi]; /* path halving */
        i = ancestor[i];
    }
    return i;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Elimination tree computation — Liu's algorithm
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_etree_compute(const SparseMatrix *A, idx_t *parent) {
    if (!A || !parent)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;

    /* ancestor[i] is the union-find representative for column i */
    idx_t *ancestor = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&ancestor) != SPARSE_OK)
        return SPARSE_ERR_ALLOC;

    /* Initialize: each column is its own ancestor, no parents yet */
    for (idx_t i = 0; i < n; i++) {
        parent[i] = -1;  // NOLINT(clang-analyzer-security.ArrayBound)
        ancestor[i] = i; // NOLINT(clang-analyzer-security.ArrayBound)
    }

    /* Liu's algorithm: process columns j = 0..n-1.
     * For each row j, walk the entries with column i < j (lower-triangular
     * entries A(j,i) in row j, equivalently upper-triangular A(i,j) by symmetry).
     * For each such entry, find the root of column i's etree subtree.
     * If that root r != j, set parent[r] = j and merge r into j. */
    for (idx_t j = 0; j < n; j++) {
        /* Walk row j: columns i < j correspond to lower-triangle entries
         * A(j,i); by symmetry these are the upper-triangle entries A(i,j). */
        for (Node *nd = A->row_headers[j]; nd; nd = nd->right) {
            idx_t i = nd->col;
            if (i >= j)
                break; /* row is sorted by column; stop at diagonal */

            /* Find the root of column i's current subtree */
            idx_t r = uf_find(ancestor, n, i);
            if (r >= 0 && r < n && r != j) {
                parent[r] = j;
                ancestor[r] = j;
            }
        }
    }

    free(ancestor);
    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Column counts of Cholesky factor L
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_colcount(const SparseMatrix *A, const idx_t *parent, const idx_t *postorder,
                             idx_t *colcount) {
    if (!A || !parent || !postorder || !colcount)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    if (n == 0)
        return SPARSE_OK;
    /* Build child lists from parent pointers */
    idx_t *child_head = NULL;
    idx_t *child_next = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_head) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_next) != SPARSE_OK) {
        free(child_head);
        free(child_next);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        child_head[i] = -1;
        child_next[i] = -1;
    }
    for (idx_t i = 0; i < n; i++) {
        idx_t p = parent[i]; // NOLINT(clang-analyzer-security.ArrayBound)
        if (p >= 0 && p < n) {
            child_next[i] = child_head[p];
            child_head[p] = i;
        }
    }

    /* Per-column row index sets (propagated up the etree) */
    idx_t **col_rows = NULL;
    idx_t *col_nrows = NULL;
    idx_t *marker = NULL;
    idx_t *tmp = NULL;
    if (sparse_calloc_idx_array(n, sizeof(idx_t *), (void **)&col_rows) != SPARSE_OK ||
        sparse_calloc_idx_array(n, sizeof(idx_t), (void **)&col_nrows) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&marker) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&tmp) != SPARSE_OK) {
        free(col_rows);
        free(col_nrows);
        free(marker);
        free(tmp);
        free(child_head);
        free(child_next);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++)
        marker[i] = -1;

    /* Process columns in postorder (children before parents).
     * For column j, the off-diagonal rows in L(:,j) are:
     *   (a) rows i > j from A's lower triangle, plus
     *   (b) rows inherited from children (excluding row j). */
    for (idx_t k = 0; k < n; k++) {
        idx_t j = postorder[k]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
        idx_t count = 0;

        /* (a) Original lower-triangle entries in column j */
        for (Node *nd = A->col_headers[j]; nd; nd = nd->down) {
            idx_t i = nd->row;
            if (i > j && marker[i] != j) { // NOLINT(clang-analyzer-security.ArrayBound)
                marker[i] = j;
                tmp[count++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
        }

        /* (b) Rows from children of j in the etree */
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        for (idx_t c = child_head[j]; c >= 0; c = child_next[c]) {
            for (idx_t m = 0; m < col_nrows[c]; m++) {
                idx_t i = col_rows[c][m]; // NOLINT(clang-analyzer-core.NullDereference)
                if (i != j && marker[i] != j) {
                    marker[i] = j;
                    tmp[count++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
                }
            }
            free(col_rows[c]);
            col_rows[c] = NULL;
        }

        colcount[j] = count + 1; /* +1 for the diagonal */

        /* Store this column's row set for its parent to consume */
        if (count > 0) {
            if (sparse_malloc_array((size_t)count, sizeof(idx_t), (void **)&col_rows[j]) !=
                SPARSE_OK) {
                for (idx_t i = 0; i < n; i++)
                    free(col_rows[i]);
                free(col_rows);
                free(col_nrows);
                free(marker);
                free(tmp);
                free(child_head);
                free(child_next);
                return SPARSE_ERR_ALLOC;
            }
            memcpy(col_rows[j], tmp, (size_t)count * sizeof(idx_t));
            col_nrows[j] = count;
        }
    }

    for (idx_t i = 0; i < n; i++)
        free(col_rows[i]);
    free(col_rows);
    free(col_nrows);
    free(marker);
    free(tmp);
    free(child_head);
    free(child_next);

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic Cholesky factorization
 * ═══════════════════════════════════════════════════════════════════════ */

/* qsort comparator for idx_t */
static int cmp_idx(const void *pa, const void *pb) {
    idx_t a = *(const idx_t *)pa;
    idx_t b = *(const idx_t *)pb;
    return (a > b) - (a < b);
}

/* Sort an array of idx_t. Uses insertion sort for small arrays,
 * qsort for larger ones to avoid quadratic behavior. */
static void isort(idx_t *a, idx_t len) {
    if (len <= 32) {
        for (idx_t i = 1; i < len; i++) {
            idx_t key = a[i];
            idx_t j = i - 1;
            while (j >= 0 && a[j] > key) {
                a[j + 1] = a[j];
                j--;
            }
            a[j + 1] = key;
        }
    } else {
        qsort(a, (size_t)len, sizeof(idx_t), cmp_idx);
    }
}

void sparse_symbolic_free(sparse_symbolic_t *sym) {
    if (!sym)
        return;
    free(sym->col_ptr);
    free(sym->row_idx);
    sym->col_ptr = NULL;
    sym->row_idx = NULL;
    sym->n = 0;
    sym->nnz = 0;
}

sparse_err_t sparse_symbolic_cholesky(const SparseMatrix *A, const idx_t *parent,
                                      const idx_t *postorder, const idx_t *colcount,
                                      sparse_symbolic_t *sym) {
    if (!A || !parent || !postorder || !colcount || !sym)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;
    /* sym must be zeroed or previously freed before calling.
     * We zero it here; callers reusing a sym should call
     * sparse_symbolic_free() first. */
    memset(sym, 0, sizeof(*sym));

    if (n == 0) {
        if (sparse_calloc_array(1, sizeof(idx_t), (void **)&sym->col_ptr) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
        return SPARSE_OK;
    }

    {
        size_t col_ptr_len = 0;
        if (sparse_size_add_overflow((size_t)n, 1, &col_ptr_len))
            return SPARSE_ERR_ALLOC;

        /* Compute total nnz and build col_ptr from column counts */
        if (sparse_malloc_array(col_ptr_len, sizeof(idx_t), (void **)&sym->col_ptr) != SPARSE_OK)
            return SPARSE_ERR_ALLOC;
        sym->n = n;
    }

    /* Build col_ptr with overflow-safe accumulation */
    sym->col_ptr[0] = 0;
    size_t total_nnz = 0;
    for (idx_t j = 0; j < n; j++) {
        size_t cj = (size_t)colcount[j]; // NOLINT(clang-analyzer-security.ArrayBound)
        if (sparse_size_add_overflow(total_nnz, cj, &total_nnz) ||
            sparse_size_to_idx_checked(total_nnz, &sym->col_ptr[j + 1])) {
            sparse_symbolic_free(sym);
            return SPARSE_ERR_ALLOC;
        }
        /* Check idx_t representability and monotonicity */
        if (sym->col_ptr[j + 1] < sym->col_ptr[j]) {
            sparse_symbolic_free(sym);
            return SPARSE_ERR_ALLOC;
        }
    }
    sym->nnz = sym->col_ptr[n]; // NOLINT(clang-analyzer-security.ArrayBound)

    {
        size_t row_idx_bytes = 0;
        if (sparse_idx_count_bytes_overflow(sym->nnz, sizeof(idx_t), &row_idx_bytes)) {
            sparse_symbolic_free(sym);
            return SPARSE_ERR_ALLOC;
        }
    }
    if (sparse_malloc_idx_array(sym->nnz, sizeof(idx_t), (void **)&sym->row_idx) != SPARSE_OK) {
        sparse_symbolic_free(sym);
        return SPARSE_ERR_ALLOC;
    }

    /* Build child lists */
    idx_t *child_head = NULL;
    idx_t *child_next = NULL;
    idx_t *marker = NULL;
    idx_t *tmp = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_head) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_next) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&marker) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&tmp) != SPARSE_OK) {
        free(child_head);
        free(child_next);
        free(marker);
        free(tmp);
        sparse_symbolic_free(sym);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        child_head[i] = -1;
        child_next[i] = -1;
        marker[i] = -1;
    }
    for (idx_t i = 0; i < n; i++) {
        idx_t p = parent[i];
        if (p >= 0 && p < n) {
            child_next[i] = child_head[p];
            child_head[p] = i;
        }
    }

    /* Per-column row sets propagated up the etree */
    idx_t **col_rows = NULL;
    idx_t *col_nrows = NULL;
    if (sparse_calloc_idx_array(n, sizeof(idx_t *), (void **)&col_rows) != SPARSE_OK ||
        sparse_calloc_idx_array(n, sizeof(idx_t), (void **)&col_nrows) != SPARSE_OK) {
        free(col_rows);
        free(col_nrows);
        free(child_head);
        free(child_next);
        free(marker);
        free(tmp);
        sparse_symbolic_free(sym);
        return SPARSE_ERR_ALLOC;
    }

    /* Process columns in postorder */
    for (idx_t k = 0; k < n; k++) {
        idx_t j = postorder[k]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
        idx_t count = 0;

        /* (a) Original lower-triangle entries */
        for (Node *nd = A->col_headers[j]; nd; nd = nd->down) {
            idx_t i = nd->row;
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
            if (i > j && marker[i] != j) {
                marker[i] = j;
                tmp[count++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
            }
        }

        /* (b) Rows from children */
        // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound)
        for (idx_t c = child_head[j]; c >= 0; c = child_next[c]) {
            for (idx_t m = 0; m < col_nrows[c]; m++) {
                idx_t i = col_rows[c][m]; // NOLINT(clang-analyzer-core.NullDereference)
                if (i != j && marker[i] != j) {
                    marker[i] = j;
                    tmp[count++] = i; // NOLINT(clang-analyzer-security.ArrayBound)
                }
            }
            free(col_rows[c]);
            col_rows[c] = NULL;
        }

        /* Sort row indices and write into sym->row_idx */
        isort(tmp, count);
        idx_t base = sym->col_ptr[j];
        sym->row_idx[base] = j; /* diagonal first */
        memcpy(&sym->row_idx[base + 1], tmp, (size_t)count * sizeof(idx_t));

        /* Store row set for parent */
        if (count > 0) {
            if (sparse_malloc_array((size_t)count, sizeof(idx_t), (void **)&col_rows[j]) !=
                SPARSE_OK) {
                for (idx_t i = 0; i < n; i++)
                    free(col_rows[i]);
                free(col_rows);
                free(col_nrows);
                free(child_head);
                free(child_next);
                free(marker);
                free(tmp);
                sparse_symbolic_free(sym);
                return SPARSE_ERR_ALLOC;
            }
            memcpy(col_rows[j], tmp, (size_t)count * sizeof(idx_t));
            col_nrows[j] = count;
        }
    }

    for (idx_t i = 0; i < n; i++)
        free(col_rows[i]);
    free(col_rows);
    free(col_nrows);
    free(child_head);
    free(child_next);
    free(marker);
    free(tmp);

    return SPARSE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Symbolic LU factorization (upper bound via column etree of A)
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_symbolic_lu(const SparseMatrix *A, const idx_t *perm, sparse_symbolic_t *sym_L,
                                sparse_symbolic_t *sym_U) {
    if (!A)
        return SPARSE_ERR_NULL;
    if (!sym_L && !sym_U)
        return SPARSE_ERR_NULL;
    if (A->rows != A->cols)
        return SPARSE_ERR_SHAPE;

    idx_t n = A->rows;

    /* Build B = structure of A^T * A (with optional permutation).
     * B(i,j) is nonzero iff columns i and j of A share a nonzero row.
     * This gives the correct column interaction graph for LU fill bounds.
     * If perm is given, columns are permuted: B uses permuted indices. */
    SparseMatrix *B = sparse_create(n, n);
    if (!B)
        return SPARSE_ERR_ALLOC;

    /* Build inverse permutation: inv_perm[old] = new.
     * Reorder routines produce perm[new] = old, so we invert. */
    idx_t *inv_perm = NULL;
    if (perm) {
        unsigned char *seen = NULL;
        size_t seen_bytes = 0;
        size_t inv_perm_bytes = 0;
        if (sparse_idx_count_bytes_overflow(n, sizeof(unsigned char), &seen_bytes) ||
            sparse_idx_count_bytes_overflow(n, sizeof(idx_t), &inv_perm_bytes)) {
            sparse_free(B);
            return SPARSE_ERR_ALLOC;
        }
        if (seen_bytes != 0)
            seen = calloc(1, seen_bytes);
        if (inv_perm_bytes != 0)
            inv_perm = malloc(inv_perm_bytes);
        if ((seen_bytes != 0 && !seen) || (inv_perm_bytes != 0 && !inv_perm)) {
            free(seen);
            free(inv_perm);
            sparse_free(B);
            return SPARSE_ERR_ALLOC;
        }
        for (idx_t i = 0; i < n; i++) {
            idx_t p = perm[i];
            if (p < 0 || p >= n || seen[p]) {
                free(seen);
                free(inv_perm);
                sparse_free(B);
                return SPARSE_ERR_BADARG;
            }
            seen[p] = 1;
            inv_perm[p] = i;
        }
        free(seen);
    }

    /* Workspace: collect permuted column indices per row */
    idx_t *row_cols = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&row_cols) != SPARSE_OK) {
        free(inv_perm);
        sparse_free(B);
        return SPARSE_ERR_ALLOC;
    }

    sparse_err_t ins_err = SPARSE_OK;
    for (idx_t k = 0; k < n && ins_err == SPARSE_OK; k++) {
        /* Collect permuted column indices in row k */
        idx_t cnt = 0;
        for (Node *nd = A->row_headers[k]; nd; nd = nd->right) {
            idx_t pj = inv_perm ? inv_perm[nd->col] : nd->col;
            row_cols[cnt++] = pj; // NOLINT(clang-analyzer-security.ArrayBound)
        }
        /* All pairs (row_cols[a], row_cols[b]) are entries of A^T * A.
         * Skip symmetric duplicate when a == b (diagonal). */
        for (idx_t a = 0; a < cnt && ins_err == SPARSE_OK; a++) {
            for (idx_t b = a; b < cnt && ins_err == SPARSE_OK; b++) {
                ins_err = sparse_insert(B, row_cols[a], row_cols[b], 1.0);
                if (ins_err == SPARSE_OK && a != b) {
                    ins_err = sparse_insert(B, row_cols[b], row_cols[a], 1.0);
                }
            }
        }
    }
    free(row_cols);
    free(inv_perm);

    if (ins_err != SPARSE_OK) {
        sparse_free(B);
        return ins_err;
    }

    /* Ensure diagonal is present (only insert if missing) */
    for (idx_t i = 0; i < n; i++) {
        if (sparse_get(B, i, i) == 0.0) {
            ins_err = sparse_insert(B, i, i, (double)n + 1.0);
            if (ins_err != SPARSE_OK) {
                sparse_free(B);
                return ins_err;
            }
        }
    }

    /* Run the full symbolic Cholesky pipeline on B */
    idx_t *parent = NULL;
    idx_t *postorder = NULL;
    idx_t *cc = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&parent) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&postorder) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&cc) != SPARSE_OK) {
        free(parent);
        free(postorder);
        free(cc);
        sparse_free(B);
        return SPARSE_ERR_ALLOC;
    }

    sparse_err_t err = sparse_etree_compute(B, parent);
    if (err)
        goto cleanup;
    err = sparse_etree_postorder(parent, n, postorder);
    if (err)
        goto cleanup;
    err = sparse_colcount(B, parent, postorder, cc);
    if (err)
        goto cleanup;

    /* Compute symbolic Cholesky of B — this gives the lower triangle L.
     * For LU, L's structure is a superset of the actual L columns, and
     * the transpose of L's structure is a superset of U's columns. */
    sparse_symbolic_t sym_full;
    err = sparse_symbolic_cholesky(B, parent, postorder, cc, &sym_full);
    if (err)
        goto cleanup;

    /* Delay writing to *sym_L until all work succeeds, to avoid
     * leaving a partially-populated output on error. */

    /* Build sym_U (upper triangle) by transposing the symbolic L.
     * U column j contains all rows i such that L column i contains row j,
     * plus the diagonal. */
    if (sym_U) {
        memset(sym_U, 0, sizeof(*sym_U));
        sym_U->n = n;

        /* Count entries per column of U (= entries per row of L) */
        idx_t *u_cnt = NULL;
        if (sparse_calloc_idx_array(n, sizeof(idx_t), (void **)&u_cnt) != SPARSE_OK) {
            sparse_symbolic_free(&sym_full);
            err = SPARSE_ERR_ALLOC;
            goto cleanup;
        }

        for (idx_t j = 0; j < n; j++) {
            // NOLINTNEXTLINE(clang-analyzer-security.ArrayBound,clang-analyzer-core.UndefinedBinaryOperatorResult)
            for (idx_t p = sym_full.col_ptr[j]; p < sym_full.col_ptr[j + 1]; p++) {
                // NOLINTNEXTLINE(clang-analyzer-core.uninitialized.Assign)
                idx_t i = sym_full.row_idx[p];
                u_cnt[i]++; /* row i of L → column i of U has entry in row j */
            }
        }

        /* Build col_ptr for U */
        {
            size_t u_col_ptr_len = 0;
            size_t u_col_ptr_bytes = 0;
            if (sparse_size_add_overflow((size_t)n, 1, &u_col_ptr_len) ||
                sparse_count_bytes_overflow(u_col_ptr_len, sizeof(idx_t), &u_col_ptr_bytes)) {
                free(u_cnt);
                sparse_symbolic_free(&sym_full);
                err = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
            sym_U->col_ptr = malloc(u_col_ptr_bytes);
        }
        if (!sym_U->col_ptr) {
            free(u_cnt);
            sparse_symbolic_free(&sym_full);
            err = SPARSE_ERR_ALLOC;
            goto cleanup;
        }
        sym_U->col_ptr[0] = 0;
        {
            size_t u_total = 0;
            for (idx_t j = 0; j < n; j++) {
                size_t cj = (size_t)u_cnt[j];
                if (sparse_size_add_overflow(u_total, cj, &u_total) ||
                    sparse_size_to_idx_checked(u_total, &sym_U->col_ptr[j + 1])) {
                    free(u_cnt);
                    sparse_symbolic_free(sym_U);
                    sparse_symbolic_free(&sym_full);
                    err = SPARSE_ERR_ALLOC;
                    goto cleanup;
                }
            }
        }
        sym_U->nnz = sym_U->col_ptr[n]; // NOLINT(clang-analyzer-security.ArrayBound)

        idx_t u_row_idx_count = (sym_U->nnz > 0) ? sym_U->nnz : 1;
        {
            size_t u_row_idx_bytes = 0;
            if (sparse_idx_count_bytes_overflow(u_row_idx_count, sizeof(idx_t), &u_row_idx_bytes)) {
                free(u_cnt);
                sparse_symbolic_free(sym_U);
                sparse_symbolic_free(&sym_full);
                err = SPARSE_ERR_ALLOC;
                goto cleanup;
            }
        }
        if (sparse_malloc_idx_array(u_row_idx_count, sizeof(idx_t), (void **)&sym_U->row_idx) !=
            SPARSE_OK) {
            free(u_cnt);
            sparse_symbolic_free(sym_U);
            sparse_symbolic_free(&sym_full);
            err = SPARSE_ERR_ALLOC;
            goto cleanup;
        }

        /* Fill in row indices: for each L entry (i,j), place row j in U column i */
        memset(u_cnt, 0, (size_t)n * sizeof(idx_t));
        for (idx_t j = 0; j < n; j++) {
            for (idx_t p = sym_full.col_ptr[j]; p < sym_full.col_ptr[j + 1]; p++) {
                idx_t i = sym_full.row_idx[p];
                idx_t pos = sym_U->col_ptr[i] + u_cnt[i];
                sym_U->row_idx[pos] = j;
                u_cnt[i]++;
            }
        }

        /* Sort row indices within each U column */
        for (idx_t j = 0; j < n; j++) {
            idx_t len = sym_U->col_ptr[j + 1] - sym_U->col_ptr[j];
            if (len > 1) {
                idx_t base = sym_U->col_ptr[j]; // NOLINT(clang-analyzer-security.ArrayBound)
                isort(&sym_U->row_idx[base],    // NOLINT(clang-analyzer-security.ArrayBound)
                      len);
            }
        }

        free(u_cnt);
    }

    /* All work succeeded — now assign sym_L (deferred to avoid leaks on error) */
    if (sym_L) {
        *sym_L = sym_full;
    } else {
        sparse_symbolic_free(&sym_full);
    }

cleanup:
    free(parent);
    free(postorder);
    free(cc);
    sparse_free(B);
    return err;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Postorder traversal of the elimination tree
 * ═══════════════════════════════════════════════════════════════════════ */

sparse_err_t sparse_etree_postorder(const idx_t *parent, idx_t n, idx_t *postorder) {
    if (n < 0)
        return SPARSE_ERR_BADARG;
    if (n > 0 && (!parent || !postorder))
        return SPARSE_ERR_NULL;
    if (n == 0)
        return SPARSE_OK;
    /* Build child lists from parent pointers.
     * child_head[i] = first child of i, child_next[c] = next sibling of c.
     * Use -1 as sentinel. */
    idx_t *child_head = NULL;
    idx_t *child_next = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_head) != SPARSE_OK ||
        sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&child_next) != SPARSE_OK) {
        free(child_head);
        free(child_next);
        return SPARSE_ERR_ALLOC;
    }

    for (idx_t i = 0; i < n; i++) {
        child_head[i] = -1;
        child_next[i] = -1;
    }

    /* Validate parent values and build linked lists of children */
    for (idx_t i = 0; i < n; i++) {
        idx_t p = parent[i]; // NOLINT(clang-analyzer-core.uninitialized.Assign)
        if (p != -1 && (p < 0 || p >= n)) {
            free(child_head);
            free(child_next);
            return SPARSE_ERR_BADARG;
        }
        if (p >= 0) {
            child_next[i] = child_head[p];
            child_head[p] = i;
        }
    }

    /* Iterative DFS postorder using an explicit stack */
    idx_t *stack = NULL;
    if (sparse_malloc_idx_array(n, sizeof(idx_t), (void **)&stack) != SPARSE_OK) {
        free(child_head);
        free(child_next);
        return SPARSE_ERR_ALLOC;
    }

    idx_t pos = 0; /* next position in postorder[] */

    /* Process all roots (nodes with parent == -1) */
    for (idx_t r = 0; r < n; r++) {
        if (parent[r] != -1)
            continue; /* not a root */

        idx_t top = 0;
        stack[top] = r;

        while (top >= 0) {
            idx_t v = stack[top];

            if (child_head[v] >= 0 && top + 1 < n) {
                /* Push first remaining child and remove it from child list */
                idx_t c = child_head[v];
                child_head[v] = child_next[c];
                stack[++top] = c;
            } else {
                /* All children visited — emit this node */
                postorder[pos++] = v;
                top--;
            }
        }
    }

    free(child_head);
    free(child_next);
    free(stack);

    /* Every node must appear exactly once in the postorder */
    if (pos != n)
        return SPARSE_ERR_BADARG;

    return SPARSE_OK;
}
