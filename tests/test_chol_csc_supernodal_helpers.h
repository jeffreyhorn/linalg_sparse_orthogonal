#ifndef TEST_CHOL_CSC_SUPERNODAL_HELPERS_H
#define TEST_CHOL_CSC_SUPERNODAL_HELPERS_H

/* Family-local helper seam for the supernodal/writeback/dispatch proof group
 * in `test_chol_csc.c`. Keep this narrow and specific instead of widening the
 * shared solver test helper layer with CSC-family details.
 */

static void detect_supernodes_alloc(const CholCsc *L, idx_t min_size, idx_t **starts_out,
                                    idx_t **sizes_out, idx_t *count_out) {
    idx_t n = L->n;
    *starts_out = NULL;
    *sizes_out = NULL;
    *count_out = 0;
    idx_t *starts = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    if (!starts || !sizes) {
        free(starts);
        free(sizes);
        REQUIRE_OK(SPARSE_ERR_ALLOC);
    }
    idx_t count = 0;
    REQUIRE_OK(chol_csc_detect_supernodes(L, min_size, starts, sizes, &count));
    *starts_out = starts;
    *sizes_out = sizes;
    *count_out = count;
}

/* Count the grouped supernode columns for size >= min_size by converting a
 * sparse factored `L` back to `CholCsc` and rerunning supernode detection.
 * Return -1 on failure so callers can skip the derived check cleanly.
 */
static idx_t day8_count_supernodes(const SparseMatrix *L_sparse, idx_t min_size, idx_t *count_out) {
    CholCsc *L = NULL;
    if (chol_csc_from_sparse(L_sparse, NULL, 2.0, &L) != SPARSE_OK)
        return -1;
    idx_t n = L->n;
    idx_t *starts = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t *sizes = malloc((size_t)(n > 0 ? n : 1) * sizeof(idx_t));
    idx_t count = 0;
    if (!starts || !sizes ||
        chol_csc_detect_supernodes(L, min_size, starts, sizes, &count) != SPARSE_OK) {
        free(starts);
        free(sizes);
        chol_csc_free(L);
        return -1;
    }
    idx_t total = 0;
    for (idx_t i = 0; i < count; i++)
        total += sizes[i];
    free(starts);
    free(sizes);
    chol_csc_free(L);
    *count_out = count;
    return total;
}

/* Factor `A` through the scalar and supernodal paths and assert the two
 * factored CSC values still match within `tol`.
 */
static void day9_assert_batched_matches_scalar(const SparseMatrix *A, const idx_t *perm,
                                               idx_t min_size, double tol, const char *label) {
    (void)label; /* reserved for future diagnostic messages */
    CholCsc *Ls = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &Ls));
    REQUIRE_OK(chol_csc_eliminate(Ls));

    CholCsc *Ln = NULL;
    REQUIRE_OK(chol_csc_from_sparse(A, perm, 2.0, &Ln));
    REQUIRE_OK(chol_csc_eliminate_supernodal(Ln, min_size));
    REQUIRE_OK(chol_csc_validate(Ln));

    ASSERT_TRUE(day8_chol_csc_match(Ls, Ln, tol));

    chol_csc_free(Ls);
    chol_csc_free(Ln);
}

/* Build a diagonally dominant SPD matrix with a fixed sparsity pattern so the
 * dispatch residual checks remain deterministic across reruns.
 */
static SparseMatrix *day11_build_spd(idx_t n, double density, unsigned int seed) {
    unsigned int rng = seed;
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++)
        sparse_insert(A, i, i, (double)n);
    for (idx_t i = 1; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            rng = rng * 1664525u + 1013904223u;
            double p = (double)(rng & 0xffffff) / (double)0x1000000;
            if (p < density) {
                rng = rng * 1664525u + 1013904223u;
                double v = ((double)(rng & 0xffff) / (double)0x10000) - 0.5;
                sparse_insert(A, i, j, v);
                sparse_insert(A, j, i, v);
            }
        }
    }
    return A;
}

#endif
