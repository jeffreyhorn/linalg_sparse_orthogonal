#ifndef TEST_GRAPH_FIXTURES_H
#define TEST_GRAPH_FIXTURES_H

#include "sparse_graph_internal.h"
#include "sparse_matrix.h"
#include "sparse_types.h"
#include "test_framework.h"

#include <stdlib.h>

static inline int tf_graph_insert_or_free(SparseMatrix *A, idx_t row, idx_t col, double value) {
    if (sparse_insert(A, row, col, value) != SPARSE_OK) {
        sparse_free(A);
        return 0;
    }
    return 1;
}

static inline SparseMatrix *tf_make_grid_2d(idx_t r, idx_t c) {
    SparseMatrix *A = sparse_create(r * c, r * c);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < r; i++) {
        for (idx_t j = 0; j < c; j++) {
            idx_t v = i * c + j;
            if (!tf_graph_insert_or_free(A, v, v, 1.0))
                return NULL;
            if (j + 1 < c) {
                if (!tf_graph_insert_or_free(A, v, v + 1, 1.0))
                    return NULL;
                if (!tf_graph_insert_or_free(A, v + 1, v, 1.0))
                    return NULL;
            }
            if (i + 1 < r) {
                if (!tf_graph_insert_or_free(A, v, v + c, 1.0))
                    return NULL;
                if (!tf_graph_insert_or_free(A, v + c, v, 1.0))
                    return NULL;
            }
        }
    }
    return A;
}

static inline SparseMatrix *tf_make_path_1d(idx_t n) {
    SparseMatrix *A = sparse_create(n, n);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < n; i++) {
        if (!tf_graph_insert_or_free(A, i, i, 1.0))
            return NULL;
        if (i + 1 < n) {
            if (!tf_graph_insert_or_free(A, i, i + 1, 1.0))
                return NULL;
            if (!tf_graph_insert_or_free(A, i + 1, i, 1.0))
                return NULL;
        }
    }
    return A;
}

static inline SparseMatrix *tf_make_mesh_3d(idx_t d) {
    SparseMatrix *A = sparse_create(d * d * d, d * d * d);
    if (!A)
        return NULL;
    for (idx_t z = 0; z < d; z++) {
        for (idx_t y = 0; y < d; y++) {
            for (idx_t x = 0; x < d; x++) {
                idx_t v = x + y * d + z * d * d;
                if (!tf_graph_insert_or_free(A, v, v, 1.0))
                    return NULL;
                if (x + 1 < d) {
                    if (!tf_graph_insert_or_free(A, v, v + 1, 1.0))
                        return NULL;
                    if (!tf_graph_insert_or_free(A, v + 1, v, 1.0))
                        return NULL;
                }
                if (y + 1 < d) {
                    if (!tf_graph_insert_or_free(A, v, v + d, 1.0))
                        return NULL;
                    if (!tf_graph_insert_or_free(A, v + d, v, 1.0))
                        return NULL;
                }
                if (z + 1 < d) {
                    if (!tf_graph_insert_or_free(A, v, v + d * d, 1.0))
                        return NULL;
                    if (!tf_graph_insert_or_free(A, v + d * d, v, 1.0))
                        return NULL;
                }
            }
        }
    }
    return A;
}

static inline SparseMatrix *tf_make_two_cliques_with_bridge(idx_t k) {
    SparseMatrix *A = sparse_create(2 * k, 2 * k);
    if (!A)
        return NULL;
    for (idx_t i = 0; i < k; i++) {
        if (!tf_graph_insert_or_free(A, i, i, 1.0))
            return NULL;
        for (idx_t j = i + 1; j < k; j++) {
            if (!tf_graph_insert_or_free(A, i, j, 1.0))
                return NULL;
            if (!tf_graph_insert_or_free(A, j, i, 1.0))
                return NULL;
        }
    }
    for (idx_t i = k; i < 2 * k; i++) {
        if (!tf_graph_insert_or_free(A, i, i, 1.0))
            return NULL;
        for (idx_t j = i + 1; j < 2 * k; j++) {
            if (!tf_graph_insert_or_free(A, i, j, 1.0))
                return NULL;
            if (!tf_graph_insert_or_free(A, j, i, 1.0))
                return NULL;
        }
    }
    if (!tf_graph_insert_or_free(A, 0, (idx_t)k, 1.0))
        return NULL;
    if (!tf_graph_insert_or_free(A, (idx_t)k, 0, 1.0))
        return NULL;
    return A;
}

static inline int tf_check_partition_invariant(const sparse_graph_t *G, const idx_t *part) {
    for (idx_t i = 0; i < G->n; i++) {
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            int p_i = (int)part[i];
            int p_j = (int)part[j];
            if ((p_i == 0 && p_j == 1) || (p_i == 1 && p_j == 0))
                return 0;
        }
    }
    return 1;
}

static inline void tf_count_partition_sides(const sparse_graph_t *G, const idx_t *part, idx_t *n0,
                                            idx_t *n1, idx_t *nsep) {
    *n0 = 0;
    *n1 = 0;
    *nsep = 0;
    for (idx_t i = 0; i < G->n; i++) {
        if (part[i] == 0)
            (*n0)++;
        else if (part[i] == 1)
            (*n1)++;
        else if (part[i] == 2)
            (*nsep)++;
    }
}

static inline void tf_count_bipartition_sides(const sparse_graph_t *G, const idx_t *part, idx_t *n0,
                                              idx_t *n1) {
    *n0 = 0;
    *n1 = 0;
    for (idx_t i = 0; i < G->n; i++) {
        ASSERT_TRUE(part[i] == 0 || part[i] == 1);
        if (part[i] == 0)
            (*n0)++;
        else
            (*n1)++;
    }
}

static inline idx_t tf_compute_cut(const sparse_graph_t *G, const idx_t *part) {
    idx_t cut = 0;
    for (idx_t i = 0; i < G->n; i++) {
        for (idx_t k = G->xadj[i]; k < G->xadj[i + 1]; k++) {
            idx_t j = G->adjncy[k];
            if (j <= i)
                continue;
            if (part[i] != part[j])
                cut += G->ewgt ? G->ewgt[k] : 1;
        }
    }
    return cut;
}

static inline void tf_compute_side_weights(const sparse_graph_t *G, const idx_t *part, idx_t *w0,
                                           idx_t *w1) {
    *w0 = 0;
    *w1 = 0;
    for (idx_t i = 0; i < G->n; i++) {
        idx_t w = G->vwgt ? G->vwgt[i] : 1;
        if (part[i] == 0)
            *w0 += w;
        else
            *w1 += w;
    }
}

#endif
