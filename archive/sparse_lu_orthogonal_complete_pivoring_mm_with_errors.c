// sparse_lu_orthogonal_complete_pivoting_mm_with_errors.c
// Orthogonal linked-list sparse matrix with complete pivoting LU
// Features: node pool, auto nnz, memory estimator, Matrix Market I/O, full solver, error checking
// Compile: gcc -O2 -Wall -o sparse_lu sparse_lu_orthogonal_complete_pivoting_mm_with_errors.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>  // for PRId32
#include <errno.h>

#define DROP_TOL          1e-14
#define NODES_PER_SLAB    4096

typedef int32_t idx_t;

// ──────────────────────────────────────────────────────────────────────────────
// Error codes
// ──────────────────────────────────────────────────────────────────────────────

#define ERR_OK            0
#define ERR_NULL_PTR      1
#define ERR_MALLOC_FAIL   2
#define ERR_INVALID_DIM   3
#define ERR_SINGULAR      4
#define ERR_FILE_OPEN     5
#define ERR_FILE_READ     6
#define ERR_FILE_WRITE    7
#define ERR_PARSE_FAIL    8

// ──────────────────────────────────────────────────────────────────────────────
// Node pool
// ──────────────────────────────────────────────────────────────────────────────

typedef struct Node {
    idx_t row;
    idx_t col;
    double value;
    struct Node* right;
    struct Node* down;
} Node;

typedef struct NodeSlab {
    Node nodes[NODES_PER_SLAB];
    struct NodeSlab* next;
    idx_t used;
} NodeSlab;

typedef struct NodePool {
    NodeSlab* head;
    NodeSlab* current;
    idx_t num_slabs;
} NodePool;

static Node* pool_alloc(NodePool* pool) {
    if (!pool) return NULL;
    if (!pool->current || pool->current->used >= NODES_PER_SLAB) {
        NodeSlab* slab = malloc(sizeof(NodeSlab));
        if (!slab) return NULL;
        slab->used = 0;
        slab->next = NULL;
        if (pool->current) pool->current->next = slab;
        else pool->head = slab;
        pool->current = slab;
        pool->num_slabs++;
    }
    return &pool->current->nodes[pool->current->used++];
}

static void pool_free_all(NodePool* pool) {
    if (!pool) return;
    NodeSlab* slab = pool->head;
    while (slab) {
        NodeSlab* next = slab->next;
        free(slab);
        slab = next;
    }
    pool->head = pool->current = NULL;
    pool->num_slabs = 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Sparse matrix
// ──────────────────────────────────────────────────────────────────────────────

typedef struct SparseMatrix {
    idx_t rows;
    idx_t cols;
    Node** rowHeaders;
    Node** colHeaders;
    idx_t* row_perm;
    idx_t* inv_row_perm;
    idx_t* col_perm;
    idx_t* inv_col_perm;
    NodePool node_pool;
    idx_t nnz;
} SparseMatrix;

// ──────────────────────────────────────────────────────────────────────────────
// Creation / Destruction
// ──────────────────────────────────────────────────────────────────────────────

SparseMatrix* createSparseMatrix(idx_t n) {
    if (n <= 0) return NULL;
    SparseMatrix* mat = malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;
    mat->rows = mat->cols = n;
    mat->rowHeaders = calloc(n, sizeof(Node*));
    mat->colHeaders = calloc(n, sizeof(Node*));
    mat->row_perm = malloc(n * sizeof(idx_t));
    mat->inv_row_perm = malloc(n * sizeof(idx_t));
    mat->col_perm = malloc(n * sizeof(idx_t));
    mat->inv_col_perm = malloc(n * sizeof(idx_t));
    if (!mat->rowHeaders || !mat->colHeaders ||
        !mat->row_perm || !mat->inv_row_perm ||
        !mat->col_perm || !mat->inv_col_perm) {
        free(mat->rowHeaders);
        free(mat->colHeaders);
        free(mat->row_perm);
        free(mat->inv_row_perm);
        free(mat->col_perm);
        free(mat->inv_col_perm);
        free(mat);
        return NULL;
    }
    for (idx_t i = 0; i < n; i++) {
        mat->row_perm[i] = mat->inv_row_perm[i] =
        mat->col_perm[i] = mat->inv_col_perm[i] = i;
    }
    mat->node_pool.head = mat->node_pool.current = NULL;
    mat->node_pool.num_slabs = 0;
    mat->nnz = 0;
    return mat;
}

void freeSparseMatrix(SparseMatrix* mat) {
    if (!mat) return;
    pool_free_all(&mat->node_pool);
    free(mat->rowHeaders);
    free(mat->colHeaders);
    free(mat->row_perm);
    free(mat->inv_row_perm);
    free(mat->col_perm);
    free(mat->inv_col_perm);
    free(mat);
}

// ──────────────────────────────────────────────────────────────────────────────
// Insert / Remove / Get / Set
// ──────────────────────────────────────────────────────────────────────────────

static Node* createNodeFromPool(SparseMatrix* mat, idx_t r, idx_t c, double v) {
    Node* node = pool_alloc(&mat->node_pool);
    if (!node) return NULL;
    node->row = r; node->col = c; node->value = v;
    node->right = node->down = NULL;
    return node;
}

int removeNode(SparseMatrix* mat, idx_t pr, idx_t pc) {
    if (!mat) return ERR_NULL_PTR;
    if (pr >= mat->rows || pc >= mat->cols) return ERR_INVALID_DIM;

    Node *p = NULL, *c = mat->rowHeaders[pr];
    while (c && c->col != pc) { p = c; c = c->right; }
    if (!c) return ERR_OK;  // Not found, no error
    if (p) p->right = c->right; else mat->rowHeaders[pr] = c->right;

    p = NULL; c = mat->colHeaders[pc];
    while (c && c->row != pr) { p = c; c = c->down; }
    if (p) p->down = c->down; else mat->colHeaders[pc] = c->down;

    mat->nnz--;
    return ERR_OK;
}

int insert(SparseMatrix* mat, idx_t pr, idx_t pc, double val) {
    if (!mat) return ERR_NULL_PTR;
    if (pr >= mat->rows || pc >= mat->cols) return ERR_INVALID_DIM;

    if (val == 0.0) return removeNode(mat, pr, pc);

    Node *prv = NULL, *cur = mat->rowHeaders[pr];
    while (cur && cur->col < pc) { prv = cur; cur = cur->right; }
    if (cur && cur->col == pc) { cur->value = val; return ERR_OK; }

    Node* node = createNodeFromPool(mat, pr, pc, val);
    if (!node) return ERR_MALLOC_FAIL;
    mat->nnz++;

    node->right = cur;
    if (prv) prv->right = node; else mat->rowHeaders[pr] = node;

    prv = NULL; cur = mat->colHeaders[pc];
    while (cur && cur->row < pr) { prv = cur; cur = cur->down; }
    node->down = cur;
    if (prv) prv->down = node; else mat->colHeaders[pc] = node;

    return ERR_OK;
}

double getPhysValue(SparseMatrix* mat, idx_t pr, idx_t pc) {
    if (!mat || pr >= mat->rows || pc >= mat->cols) return 0.0;
    Node* cur = mat->rowHeaders[pr];
    while (cur && cur->col < pc) cur = cur->right;
    return (cur && cur->col == pc) ? cur->value : 0.0;
}

double getValue(SparseMatrix* mat, idx_t lr, idx_t lc) {
    if (!mat || lr >= mat->rows || lc >= mat->cols) return 0.0;
    return getPhysValue(mat, mat->row_perm[lr], mat->col_perm[lc]);
}

int setValue(SparseMatrix* mat, idx_t lr, idx_t lc, double v) {
    if (!mat || lr >= mat->rows || lc >= mat->cols) return ERR_INVALID_DIM;
    return insert(mat, mat->row_perm[lr], mat->col_perm[lc], v);
}

// ──────────────────────────────────────────────────────────────────────────────
// NNZ & Memory
// ──────────────────────────────────────────────────────────────────────────────

idx_t countNNZ(SparseMatrix* mat) {
    return mat ? mat->nnz : 0;
}

size_t estimateMemoryUsage(SparseMatrix* mat) {
    if (!mat) return 0;
    return sizeof(SparseMatrix) +
           2 * mat->rows * sizeof(Node*) +
           4 * mat->rows * sizeof(idx_t) +
           mat->node_pool.num_slabs * sizeof(NodeSlab);
}

// ──────────────────────────────────────────────────────────────────────────────
// Matrix Market I/O
// ──────────────────────────────────────────────────────────────────────────────

int saveToMatrixMarket(SparseMatrix* mat, const char* filename) {
    if (!mat || !filename) return ERR_NULL_PTR;
    FILE* fp = fopen(filename, "w");
    if (!fp) return ERR_FILE_OPEN;

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%" PRId32 " %" PRId32 " %" PRId32 "\n", mat->rows, mat->cols, mat->nnz);

    for (idx_t i = 0; i < mat->rows; i++) {
        idx_t pr = mat->row_perm[i];
        Node* node = mat->rowHeaders[pr];
        while (node) {
            idx_t lc = mat->inv_col_perm[node->col];
            fprintf(fp, "%" PRId32 " %" PRId32 " %.15g\n", i+1, lc+1, node->value);
            node = node->right;
        }
    }

    if (fclose(fp) != 0) return ERR_FILE_WRITE;
    return ERR_OK;
}

int loadFromMatrixMarket(SparseMatrix** mat_ptr, const char* filename) {
    if (!mat_ptr || !filename) return ERR_NULL_PTR;
    FILE* fp = fopen(filename, "r");
    if (!fp) return ERR_FILE_OPEN;

    char line[1024];
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return ERR_FILE_READ; }

    if (strstr(line, "MatrixMarket") == NULL || strstr(line, "coordinate") == NULL) {
        fclose(fp); return ERR_PARSE_FAIL;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '%') break;
    }

    idx_t m, n, nnz_file;
    if (sscanf(line, "%" PRId32 " %" PRId32 " %" PRId32, &m, &n, &nnz_file) != 3) {
        fclose(fp); return ERR_PARSE_FAIL;
    }

    SparseMatrix* mat = createSparseMatrix(m);
    if (!mat) { fclose(fp); return ERR_MALLOC_FAIL; }

    for (idx_t k = 0; k < nnz_file; k++) {
        idx_t i, j;
        double v;
        if (fscanf(fp, "%" PRId32 " %" PRId32 " %lf", &i, &j, &v) != 3) {
            freeSparseMatrix(mat);
            fclose(fp);
            return ERR_FILE_READ;
        }
        i--; j--;
        if (i < 0 || i >= m || j < 0 || j >= n) continue;
        if (insert(mat, i, j, v) != ERR_OK) {
            freeSparseMatrix(mat);
            fclose(fp);
            return ERR_MALLOC_FAIL;
        }
    }

    fclose(fp);
    *mat_ptr = mat;
    return ERR_OK;
}

// ──────────────────────────────────────────────────────────────────────────────
// Complete pivoting LU
// ──────────────────────────────────────────────────────────────────────────────

int computeLU(SparseMatrix* mat, double tol) {
    if (!mat) return ERR_NULL_PTR;
    idx_t n = mat->rows;

    for (idx_t k = 0; k < n; k++) {
        double max_val = 0.0;
        idx_t pivot_r = k, pivot_c = k;

        for (idx_t lj = k; lj < n; lj++) {
            idx_t pj = mat->col_perm[lj];
            Node* col_node = mat->colHeaders[pj];
            while (col_node) {
                idx_t pi = col_node->row;
                idx_t li = mat->inv_row_perm[pi];
                if (li >= k) {
                    double av = fabs(col_node->value);
                    if (av > max_val) {
                        max_val = av;
                        pivot_r = li;
                        pivot_c = lj;
                    }
                }
                col_node = col_node->down;
            }
        }

        if (max_val < tol) return ERR_SINGULAR;

        if (pivot_r != k) {
            idx_t t = mat->row_perm[k];
            mat->row_perm[k] = mat->row_perm[pivot_r];
            mat->row_perm[pivot_r] = t;
            mat->inv_row_perm[mat->row_perm[k]] = k;
            mat->inv_row_perm[mat->row_perm[pivot_r]] = pivot_r;
        }

        if (pivot_c != k) {
            idx_t t = mat->col_perm[k];
            mat->col_perm[k] = mat->col_perm[pivot_c];
            mat->col_perm[pivot_c] = t;
            mat->inv_col_perm[mat->col_perm[k]] = k;
            mat->inv_col_perm[mat->col_perm[pivot_c]] = pivot_c;
        }

        idx_t pk = mat->col_perm[k];
        Node* elim = mat->colHeaders[pk];
        while (elim) {
            idx_t pi = elim->row;
            idx_t li = mat->inv_row_perm[pi];
            if (li > k) {
                double pivot = getValue(mat, k, k);
                if (fabs(pivot) < tol) return ERR_SINGULAR;
                double mult = getValue(mat, li, k) / pivot;
                if (setValue(mat, li, k, mult) != ERR_OK) return ERR_MALLOC_FAIL;

                idx_t prk = mat->row_perm[k];
                Node* uj = mat->rowHeaders[prk];
                while (uj) {
                    idx_t pj = uj->col;
                    idx_t lj = mat->inv_col_perm[pj];
                    if (lj > k) {
                        double newv = getValue(mat, li, lj) - mult * uj->value;
                        if (fabs(newv) < DROP_TOL * max_val) {
                            if (setValue(mat, li, lj, 0.0) != ERR_OK) return ERR_MALLOC_FAIL;
                        } else {
                            if (setValue(mat, li, lj, newv) != ERR_OK) return ERR_MALLOC_FAIL;
                        }
                    }
                    uj = uj->right;
                }
            }
            elim = elim->down;
        }
    }
    return ERR_OK;
}

// ──────────────────────────────────────────────────────────────────────────────
// Solvers
// ──────────────────────────────────────────────────────────────────────────────

int applyRowPerm(SparseMatrix* mat, const double* b, double* pb) {
    if (!mat || !b || !pb) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        pb[i] = b[mat->row_perm[i]];
    }
    return ERR_OK;
}

int applyInvColPerm(SparseMatrix* mat, const double* z, double* x) {
    if (!mat || !z || !x) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        x[i] = z[mat->col_perm[i]];
    }
    return ERR_OK;
}

int forwardSubstitution(SparseMatrix* mat, const double* pb, double* y) {
    if (!mat || !pb || !y) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        idx_t pi = mat->row_perm[i];
        Node* node = mat->rowHeaders[pi];
        while (node) {
            idx_t j = mat->inv_col_perm[node->col];
            if (j < i) sum += node->value * y[j];
            else break;
            node = node->right;
        }
        y[i] = pb[i] - sum;
    }
    return ERR_OK;
}

int backwardSubstitution(SparseMatrix* mat, const double* y, double* x) {
    if (!mat || !y || !x) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    for (idx_t i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        double u_ii = 0.0;
        idx_t pi = mat->row_perm[i];
        Node* node = mat->rowHeaders[pi];
        while (node) {
            idx_t j = mat->inv_col_perm[node->col];
            if (j == i) u_ii = node->value;
            else if (j > i) sum += node->value * x[j];
            node = node->right;
        }
        if (fabs(u_ii) < DROP_TOL) return ERR_SINGULAR;
        x[i] = (y[i] - sum) / u_ii;
    }
    return ERR_OK;
}

int solveLU(SparseMatrix* mat, const double* b, double* x) {
    if (!mat || !b || !x) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    double* pb = malloc(n * sizeof(double));
    double* y = malloc(n * sizeof(double));
    double* z = malloc(n * sizeof(double));
    if (!pb || !y || !z) {
        free(pb); free(y); free(z);
        return ERR_MALLOC_FAIL;
    }

    int err = applyRowPerm(mat, b, pb);
    if (err != ERR_OK) goto cleanup;

    err = forwardSubstitution(mat, pb, y);
    if (err != ERR_OK) goto cleanup;

    err = backwardSubstitution(mat, y, z);
    if (err != ERR_OK) goto cleanup;

    err = applyInvColPerm(mat, z, x);

cleanup:
    free(pb); free(y); free(z);
    return err;
}

// ──────────────────────────────────────────────────────────────────────────────
// Display
// ──────────────────────────────────────────────────────────────────────────────

int displayLogical(SparseMatrix* mat) {
    if (!mat) return ERR_NULL_PTR;
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            printf("%8.4f ", getValue(mat, i, j));
        }
        printf("\n");
    }
    return ERR_OK;
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

int main(void) {
    SparseMatrix* mat = createSparseMatrix(3);
    if (!mat) {
        fprintf(stderr, "Failed to create matrix\n");
        return 1;
    }

    if (insert(mat, 0, 0, 1.0) != ERR_OK ||
        insert(mat, 0, 2, 3.0) != ERR_OK ||
        insert(mat, 1, 1, 5.0) != ERR_OK ||
        insert(mat, 2, 0, 7.0) != ERR_OK ||
        insert(mat, 2, 2, 9.0) != ERR_OK) {
        fprintf(stderr, "Insertion failed\n");
        freeSparseMatrix(mat);
        return 1;
    }

    printf("Original matrix (logical view):\n");
    displayLogical(mat);
    printf("NNZ = %" PRId32 "   Memory ≈ %zu bytes\n\n", mat->nnz, estimateMemoryUsage(mat));

    int err = computeLU(mat, 1e-10);
    if (err != ERR_OK) {
        fprintf(stderr, "LU decomposition failed with error %d\n", err);
        freeSparseMatrix(mat);
        return 1;
    }

    printf("\nAfter LU (logical view):\n");
    displayLogical(mat);
    printf("NNZ after LU = %" PRId32 "\n\n", mat->nnz);

    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0};

    err = solveLU(mat, b, x);
    if (err != ERR_OK) {
        fprintf(stderr, "Solve failed with error %d\n", err);
        freeSparseMatrix(mat);
        return 1;
    }

    printf("Solution x:\n");
    for (idx_t i = 0; i < 3; i++) printf("%.6f ", x[i]);
    printf("\n");

    if (saveToMatrixMarket(mat, "lu.mtx") != ERR_OK) {
        fprintf(stderr, "Save failed\n");
    }

    SparseMatrix* mat2 = NULL;
    if (loadFromMatrixMarket(&mat2, "lu.mtx") != ERR_OK) {
        fprintf(stderr, "Load failed\n");
    } else {
        printf("\nLoaded matrix NNZ = %" PRId32 "\n", mat2->nnz);
        freeSparseMatrix(mat2);
    }

    freeSparseMatrix(mat);
    return 0;
}
