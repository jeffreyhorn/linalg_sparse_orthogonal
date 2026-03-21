// sparse_lu_orthogonal_complete_pivoting_mm.c
// Orthogonal linked-list sparse matrix with complete pivoting LU
// Features: node pool, auto nnz, memory estimator, Matrix Market I/O, full solver
// Compile: gcc -O2 -Wall -o sparse_lu sparse_lu_orthogonal_complete_pivoting_mm.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>  // for PRId32

#define DROP_TOL          1e-14
#define NODES_PER_SLAB    4096

typedef int32_t idx_t;

// ──────────────────────────────────────────────────────────────────────────────
// Node pool (slab allocator)
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
    SparseMatrix* mat = malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;
    mat->rows = mat->cols = n;
    mat->rowHeaders = calloc(n, sizeof(Node*));
    mat->colHeaders = calloc(n, sizeof(Node*));
    mat->row_perm     = malloc(n * sizeof(idx_t));
    mat->inv_row_perm = malloc(n * sizeof(idx_t));
    mat->col_perm     = malloc(n * sizeof(idx_t));
    mat->inv_col_perm = malloc(n * sizeof(idx_t));
    if (!mat->rowHeaders || !mat->colHeaders ||
        !mat->row_perm || !mat->inv_row_perm ||
        !mat->col_perm || !mat->inv_col_perm) {
        free(mat->rowHeaders); free(mat->colHeaders);
        free(mat->row_perm); free(mat->inv_row_perm);
        free(mat->col_perm); free(mat->inv_col_perm);
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

void removeNode(SparseMatrix* mat, idx_t pr, idx_t pc) {
    Node *p = NULL, *c = mat->rowHeaders[pr];
    while (c && c->col != pc) { p = c; c = c->right; }
    if (!c) return;
    if (p) p->right = c->right; else mat->rowHeaders[pr] = c->right;

    p = NULL; c = mat->colHeaders[pc];
    while (c && c->row != pr) { p = c; c = c->down; }
    if (p) p->down = c->down; else mat->colHeaders[pc] = c->down;

    mat->nnz--;
}

void insert(SparseMatrix* mat, idx_t pr, idx_t pc, double val) {
    if (val == 0.0) { removeNode(mat, pr, pc); return; }

    Node *prv = NULL, *cur = mat->rowHeaders[pr];
    while (cur && cur->col < pc) { prv = cur; cur = cur->right; }
    if (cur && cur->col == pc) { cur->value = val; return; }

    Node* node = createNodeFromPool(mat, pr, pc, val);
    if (!node) return;
    mat->nnz++;

    node->right = cur;
    if (prv) prv->right = node; else mat->rowHeaders[pr] = node;

    prv = NULL; cur = mat->colHeaders[pc];
    while (cur && cur->row < pr) { prv = cur; cur = cur->down; }
    node->down = cur;
    if (prv) prv->down = node; else mat->colHeaders[pc] = node;
}

double getPhysValue(SparseMatrix* mat, idx_t pr, idx_t pc) {
    Node* cur = mat->rowHeaders[pr];
    while (cur && cur->col < pc) cur = cur->right;
    return (cur && cur->col == pc) ? cur->value : 0.0;
}

double getValue(SparseMatrix* mat, idx_t lr, idx_t lc) {
    return getPhysValue(mat, mat->row_perm[lr], mat->col_perm[lc]);
}

void setValue(SparseMatrix* mat, idx_t lr, idx_t lc, double v) {
    insert(mat, mat->row_perm[lr], mat->col_perm[lc], v);
}

// ──────────────────────────────────────────────────────────────────────────────
// NNZ & Memory estimator
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
// Matrix Market I/O (coordinate format)
// ──────────────────────────────────────────────────────────────────────────────

int saveToMatrixMarket(SparseMatrix* mat, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return -1;

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
    fclose(fp);
    return 0;
}

int loadFromMatrixMarket(SparseMatrix** mat_ptr, const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[1024];
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return -1; }

    if (strstr(line, "MatrixMarket") == NULL ||
        strstr(line, "coordinate") == NULL) {
        printf("Only coordinate format supported\n");
        fclose(fp);
        return -1;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '%') break;
    }

    idx_t m, n, nnz_file;
    if (sscanf(line, "%" PRId32 " %" PRId32 " %" PRId32, &m, &n, &nnz_file) != 3) {
        fclose(fp);
        return -1;
    }

    SparseMatrix* mat = createSparseMatrix(m > n ? m : n);
    if (!mat) { fclose(fp); return -1; }

    for (idx_t k = 0; k < nnz_file; k++) {
        idx_t i, j;
        double v = 1.0;
        if (fscanf(fp, "%" PRId32 " %" PRId32 " %lf", &i, &j, &v) != 3) break;
        i--; j--;  // 1-based → 0-based
        if (i < mat->rows && j < mat->cols) {
            insert(mat, i, j, v);
        }
    }

    fclose(fp);
    *mat_ptr = mat;
    return 0;
}

// ──────────────────────────────────────────────────────────────────────────────
// Complete pivoting LU decomposition
// ──────────────────────────────────────────────────────────────────────────────

void computeLU(SparseMatrix* mat, double tol) {
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

        if (max_val < tol) {
            printf("Near-singular matrix at step %d (pivot < %g)\n", (int)k, tol);
            return;
        }

        // Swap rows
        if (pivot_r != k) {
            idx_t t = mat->row_perm[k];
            mat->row_perm[k] = mat->row_perm[pivot_r];
            mat->row_perm[pivot_r] = t;
            mat->inv_row_perm[mat->row_perm[k]] = k;
            mat->inv_row_perm[mat->row_perm[pivot_r]] = pivot_r;
        }

        // Swap columns
        if (pivot_c != k) {
            idx_t t = mat->col_perm[k];
            mat->col_perm[k] = mat->col_perm[pivot_c];
            mat->col_perm[pivot_c] = t;
            mat->inv_col_perm[mat->col_perm[k]] = k;
            mat->inv_col_perm[mat->col_perm[pivot_c]] = pivot_c;
        }

        // Elimination
        idx_t pk = mat->col_perm[k];
        Node* elim = mat->colHeaders[pk];
        while (elim) {
            idx_t pi = elim->row;
            idx_t li = mat->inv_row_perm[pi];
            if (li > k) {
                double mult = getValue(mat, li, k) / getValue(mat, k, k);
                setValue(mat, li, k, mult);

                idx_t prk = mat->row_perm[k];
                Node* uj = mat->rowHeaders[prk];
                while (uj) {
                    idx_t pj = uj->col;
                    idx_t lj = mat->inv_col_perm[pj];
                    if (lj > k) {
                        double newv = getValue(mat, li, lj) - mult * uj->value;
                        if (fabs(newv) < DROP_TOL * max_val) {
                            setValue(mat, li, lj, 0.0);
                        } else {
                            setValue(mat, li, lj, newv);
                        }
                    }
                    uj = uj->right;
                }
            }
            elim = elim->down;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Forward / Backward substitution & full solver
// ──────────────────────────────────────────────────────────────────────────────

void applyRowPerm(SparseMatrix* mat, const double* b, double* pb) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) pb[i] = b[mat->row_perm[i]];
}

void applyInvColPerm(SparseMatrix* mat, const double* z, double* x) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) x[i] = z[mat->col_perm[i]];
}

void forwardSubstitution(SparseMatrix* mat, const double* pb, double* y) {
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
        y[i] = pb[i] - sum;  // L_ii = 1
    }
}

void backwardSubstitution(SparseMatrix* mat, const double* y, double* x) {
    idx_t n = mat->rows;
    for (idx_t i = n - 1; i != (idx_t)-1; i--) {
        double sum = 0.0;
        double u_ii = 0.0;
        idx_t pi = mat->row_perm[i];
        Node* node = mat->rowHeaders[pi];
        while (node) {
            idx_t j = mat->inv_col_perm[node->col];
            if (j == i)      u_ii = node->value;
            else if (j > i)  sum += node->value * x[j];
            node = node->right;
        }
        if (fabs(u_ii) < 1e-14) {
            printf("Zero/small pivot in U at row %d\n", (int)i);
            return;
        }
        x[i] = (y[i] - sum) / u_ii;
    }
}

void solveLU(SparseMatrix* mat, const double* b, double* x) {
    idx_t n = mat->rows;
    double *pb = malloc(n * sizeof(double));
    double *y  = malloc(n * sizeof(double));
    double *z  = malloc(n * sizeof(double));
    if (!pb || !y || !z) {
        free(pb); free(y); free(z);
        return;
    }

    applyRowPerm(mat, b, pb);           // Pb
    forwardSubstitution(mat, pb, y);    // L y = Pb
    backwardSubstitution(mat, y, z);    // U z = y
    applyInvColPerm(mat, z, x);         // x = Q⁻¹ z

    free(pb);
    free(y);
    free(z);
}

// ──────────────────────────────────────────────────────────────────────────────
// Display (logical view)
// ──────────────────────────────────────────────────────────────────────────────

void displayLogical(SparseMatrix* mat) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            printf("%8.4f ", getValue(mat, i, j));
        }
        printf("\n");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Example / test
// ──────────────────────────────────────────────────────────────────────────────

int main(void) {
    SparseMatrix* mat = createSparseMatrix(3);
    if (!mat) return 1;

    // Small test matrix (same as before)
    insert(mat, 0, 0, 1.0);
    insert(mat, 0, 2, 3.0);
    insert(mat, 1, 1, 5.0);
    insert(mat, 2, 0, 7.0);
    insert(mat, 2, 2, 9.0);

    printf("Original matrix (logical view):\n");
    displayLogical(mat);
    printf("NNZ = %" PRId32 "   Memory ≈ %zu bytes\n\n", mat->nnz, estimateMemoryUsage(mat));

    printf("Computing LU with complete pivoting...\n");
    computeLU(mat, 1e-10);

    printf("\nAfter LU (L below diag, U on/above - logical view):\n");
    displayLogical(mat);
    printf("NNZ after factorization = %" PRId32 "\n\n", mat->nnz);

    // Solve A x = b
    double b[3] = {1.0, 2.0, 3.0};
    double x[3] = {0};

    solveLU(mat, b, x);

    printf("Solution to A x = [1, 2, 3]^T:\n");
    printf("x ≈ [%.6f, %.6f, %.6f]\n\n", x[0], x[1], x[2]);

    // Quick residual check
    printf("Residuals (A x - b):\n");
    for (idx_t i = 0; i < 3; i++) {
        double ax = 0.0;
        for (idx_t j = 0; j < 3; j++) {
            ax += getValue(mat, i, j) * x[j];
        }
        printf("row %d: %12.4e\n", (int)i, ax - b[i]);
    }

    saveToMatrixMarket(mat, "lu_result.mtx");
    printf("\nSaved factored matrix to lu_result.mtx (Matrix Market format)\n");

    freeSparseMatrix(mat);
    return 0;
}
