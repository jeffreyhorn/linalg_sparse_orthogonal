#include "sparse_alloc_internal.h"
#include "sparse_errno_internal.h"
#include "sparse_matrix_internal.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

sparse_err_t sparse_save_mm(const SparseMatrix *mat, const char *filename) {
    if (!mat || !filename)
        return SPARSE_ERR_NULL;

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }

    if (sparse_stream_printf_checked(fp, "%%%%MatrixMarket matrix coordinate real general\n") !=
            SPARSE_OK ||
        sparse_stream_printf_checked(fp, "%" SPARSE_PRIDX " %" SPARSE_PRIDX " %" SPARSE_PRIDX "\n",
                                     mat->rows, mat->cols, mat->nnz) != SPARSE_OK) {
        fclose(fp);
        return SPARSE_ERR_IO;
    }

    for (idx_t log_i = 0; log_i < mat->rows; log_i++) {
        idx_t phys_i = mat->row_perm[log_i];
        Node *node = mat->row_headers[phys_i];
        while (node) {
            idx_t log_j = mat->inv_col_perm[node->col];
            if (sparse_stream_printf_checked(fp, "%" SPARSE_PRIDX " %" SPARSE_PRIDX " %.15g\n",
                                             log_i + 1, log_j + 1, node->value) != SPARSE_OK) {
                fclose(fp);
                return SPARSE_ERR_IO;
            }
            node = node->right;
        }
    }

    if (fclose(fp) != 0) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }
    sparse_set_errno_(0);
    return SPARSE_OK;
}

sparse_err_t sparse_load_mm(SparseMatrix **mat_out, const char *filename) {
    SparseBuildEntry *entries = NULL;
    if (!mat_out || !filename)
        return SPARSE_ERR_NULL;
    *mat_out = NULL;

    FILE *fp = fopen(filename, "r");
    if (!fp) {
        sparse_set_errno_(errno);
        return SPARSE_ERR_IO;
    }

    char line[1024];
    if (!fgets(line, (int)sizeof(line), fp)) {
        if (ferror(fp)) {
            sparse_set_errno_(errno);
            fclose(fp);
            return SPARSE_ERR_IO;
        }
        fclose(fp);
        return SPARSE_ERR_PARSE; /* empty file */
    }

    if (strstr(line, "MatrixMarket") == NULL || strstr(line, "coordinate") == NULL) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }

    /* Detect symmetric and pattern-only formats from the header */
    int is_symmetric = (strstr(line, "symmetric") != NULL);
    int is_pattern = (strstr(line, "pattern") != NULL);

    /* Skip comment lines */
    while (fgets(line, (int)sizeof(line), fp)) {
        if (line[0] != '%')
            break;
    }

    idx_t m, n, nnz_file;
    if (sscanf(line, "%" SPARSE_SCNIDX " %" SPARSE_SCNIDX " %" SPARSE_SCNIDX, &m, &n, &nnz_file) !=
        3) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }
    if (m < 0 || n < 0 || nnz_file < 0 || (is_symmetric && m != n)) {
        fclose(fp);
        return SPARSE_ERR_PARSE;
    }

    size_t nnz_file_count = 0;
    size_t triplet_capacity = 0;
    if (sparse_idx_to_size_checked(nnz_file, &nnz_file_count) ||
        sparse_size_mul_overflow(nnz_file_count, is_symmetric ? 2U : 1U, &triplet_capacity)) {
        fclose(fp);
        return SPARSE_ERR_ALLOC;
    }
    if (sparse_malloc_array(triplet_capacity, sizeof(*entries), (void **)&entries) != SPARSE_OK) {
        fclose(fp);
        return SPARSE_ERR_ALLOC;
    }

    size_t entry_count = 0;
    for (idx_t k = 0; k < nnz_file; k++) {
        idx_t i, j;
        sparse_scalar_t v = 1.0; /* default for pattern matrices */
        if (is_pattern) {
            if (fscanf(fp, "%" SPARSE_SCNIDX " %" SPARSE_SCNIDX, &i, &j) != 2) {
                sparse_err_t ioerr =
                    ferror(fp) ? (sparse_set_errno_(errno), SPARSE_ERR_IO) : SPARSE_ERR_PARSE;
                free(entries);
                fclose(fp);
                return ioerr;
            }
        } else {
            if (fscanf(fp, "%" SPARSE_SCNIDX " %" SPARSE_SCNIDX " %lf", &i, &j, &v) != 3) {
                sparse_err_t ioerr =
                    ferror(fp) ? (sparse_set_errno_(errno), SPARSE_ERR_IO) : SPARSE_ERR_PARSE;
                free(entries);
                fclose(fp);
                return ioerr;
            }
        }
        if (i <= 0 || j <= 0) {
            free(entries);
            fclose(fp);
            return SPARSE_ERR_PARSE;
        }
        i--; /* 1-based -> 0-based */
        j--;
        if (i >= m || j >= n) {
            free(entries);
            fclose(fp);
            return SPARSE_ERR_PARSE;
        }
        {
            idx_t order = 0;
            if (entry_count >= triplet_capacity ||
                sparse_size_to_idx_checked(entry_count, &order)) {
                free(entries);
                fclose(fp);
                return SPARSE_ERR_ALLOC;
            }
            entries[entry_count++] = (SparseBuildEntry){
                .row = i,
                .col = j,
                .value = v,
                .order = order,
            };
            /* For symmetric matrices, also insert the mirror entry */
            if (is_symmetric && i != j) {
                if (entry_count >= triplet_capacity ||
                    sparse_size_to_idx_checked(entry_count, &order)) {
                    free(entries);
                    fclose(fp);
                    return SPARSE_ERR_ALLOC;
                }
                entries[entry_count++] = (SparseBuildEntry){
                    .row = j,
                    .col = i,
                    .value = v,
                    .order = order,
                };
            }
        }
    }

    if (fclose(fp) != 0) {
        sparse_set_errno_(errno);
        free(entries);
        return SPARSE_ERR_IO;
    }

    idx_t entry_count_idx = 0;
    if (sparse_size_to_idx_checked(entry_count, &entry_count_idx)) {
        free(entries);
        return SPARSE_ERR_ALLOC;
    }

    SparseMatrix *mat = NULL;
    sparse_err_t build_err = sparse_matrix_build_from_entries(m, n, entries, entry_count_idx,
                                                              /*entries_sorted=*/0, &mat);
    free(entries);
    if (build_err != SPARSE_OK)
        return build_err;

    sparse_set_errno_(0);
    *mat_out = mat;
    return SPARSE_OK;
}
