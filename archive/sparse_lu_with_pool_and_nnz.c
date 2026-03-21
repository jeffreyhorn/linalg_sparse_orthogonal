// sparse_lu_orthogonal_with_pool_and_nnz.c
// Orthogonal (cross-linked) sparse matrix with complete pivoting LU
// Features: node pool allocator, drop tolerance, nnz counting, solvers
// Compile: gcc -O2 -Wall -o sparse_lu sparse_lu_orthogonal_with_pool_and_nnz.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define DROP_TOL          1e-14
#define NODES_PER_SLAB    4096     // adjust based on expected matrix size

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
} NodePool;

static Node* pool_alloc(NodePool* pool) {
    if (!pool->current || pool->current->used >= NODES_PER_SLAB) {
        NodeSlab* new_slab = malloc(sizeof(NodeSlab));
        if (!new_slab) return NULL;
        new_slab->used = 0;
        new_slab->next = NULL;

        if (pool->current) {
            pool->current->next = new_slab;
        } else {
            pool->head = new_slab;
        }
        pool->current = new_slab;
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
} SparseMatrix;

// ──────────────────────────────────────────────────────────────────────────────
// Creation / Destruction
// ──────────────────────────────────────────────────────────────────────────────

SparseMatrix* createSparseMatrix(idx_t n) {
    SparseMatrix* mat = malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;

    mat->rows = n;
    mat->cols = n;

    mat->rowHeaders    = calloc(n, sizeof(Node*));
    mat->colHeaders    = calloc(n, sizeof(Node*));
    mat->row_perm      = malloc(n * sizeof(idx_t));
    mat->inv_row_perm  = malloc(n * sizeof(idx_t));
    mat->col_perm      = malloc(n * sizeof(idx_t));
    mat->inv_col_perm  = malloc(n * sizeof(idx_t));

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
        mat->row_perm[i]     = i;
        mat->inv_row_perm[i] = i;
        mat->col_perm[i]     = i;
        mat->inv_col_perm[i] = i;
    }

    mat->node_pool.head = mat->node_pool.current = NULL;

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

static Node* createNodeFromPool(SparseMatrix* mat, idx_t row, idx_t col, double value) {
    Node* node = pool_alloc(&mat->node_pool);
    if (!node) return NULL;
    node->row   = row;
    node->col   = col;
    node->value = value;
    node->right = NULL;
    node->down  = NULL;
    return node;
}

void removeNode(SparseMatrix* mat, idx_t phys_row, idx_t phys_col) {
    Node *prev = NULL, *curr = mat->rowHeaders[phys_row];
    while (curr && curr->col != phys_col) {
        prev = curr;
        curr = curr->right;
    }
    if (!curr) return;
    if (prev) prev->right = curr->right;
    else      mat->rowHeaders[phys_row] = curr->right;

    prev = NULL; curr = mat->colHeaders[phys_col];
    while (curr && curr->row != phys_row) {
        prev = curr;
        curr = curr->down;
    }
    if (prev) prev->down = curr->down;
    else      mat->colHeaders[phys_col] = curr->down;

    // Nodes are not individually freed — handled by pool
}

void insert(SparseMatrix* mat, idx_t phys_row, idx_t phys_col, double val) {
    if (val == 0.0) {
        removeNode(mat, phys_row, phys_col);
        return;
    }

    Node *prev_r = NULL, *curr_r = mat->rowHeaders[phys_row];
    while (curr_r && curr_r->col < phys_col) {
        prev_r = curr_r;
        curr_r = curr_r->right;
    }
    if (curr_r && curr_r->col == phys_col) {
        curr_r->value = val;
        return;
    }

    Node* node = createNodeFromPool(mat, phys_row, phys_col, val);
    if (!node) return;

    node->right = curr_r;
    if (prev_r) prev_r->right = node;
    else        mat->rowHeaders[phys_row] = node;

    Node *prev_c = NULL, *curr_c = mat->colHeaders[phys_col];
    while (curr_c && curr_c->row < phys_row) {
        prev_c = curr_c;
        curr_c = curr_c->down;
    }
    node->down = curr_c;
    if (prev_c) prev_c->down = node;
    else        mat->colHeaders[phys_col] = node;
}

double getPhysValue(SparseMatrix* mat, idx_t phys_row, idx_t phys_col) {
    Node* curr = mat->rowHeaders[phys_row];
    while (curr && curr->col < phys_col) curr = curr->right;
    return (curr && curr->col == phys_col) ? curr->value : 0.0;
}

double getValue(SparseMatrix* mat, idx_t log_row, idx_t log_col) {
    return getPhysValue(mat, mat->row_perm[log_row], mat->col_perm[log_col]);
}

void setValue(SparseMatrix* mat, idx_t log_row, idx_t log_col, double val) {
    insert(mat, mat->row_perm[log_row], mat->col_perm[log_col], val);
}

// ──────────────────────────────────────────────────────────────────────────────
// NNZ counting
// ──────────────────────────────────────────────────────────────────────────────

idx_t countNNZ(SparseMatrix* mat) {
    if (!mat) return 0;
    idx_t nnz = 0;
    for (idx_t i = 0; i < mat->rows; i++) {
        Node* node = mat->rowHeaders[i];
        while (node) {
            nnz++;
            node = node->right;
        }
    }
    return nnz;
}

idx_t countNNZLogical(SparseMatrix* mat) {
    if (!mat) return 0;
    idx_t nnz = 0;
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            if (fabs(getValue(mat, i, j)) > 1e-20) {
                nnz++;
            }
        }
    }
    return nnz;
}

// ──────────────────────────────────────────────────────────────────────────────
// Complete pivoting LU (P A Q = L U)
// ──────────────────────────────────────────────────────────────────────────────

void computeLU(SparseMatrix* mat, double tol) {
    idx_t n = mat->rows;

    for (idx_t k = 0; k < n; k++) {
        double maxv = 0.0;
        idx_t pivot_r = k, pivot_c = k;

        for (idx_t lj = k; lj < n; lj++) {
            idx_t pj = mat->col_perm[lj];
            Node* col = mat->colHeaders[pj];
            while (col) {
                idx_t pi = col->row;
                idx_t li = mat->inv_row_perm[pi];
                if (li >= k) {
                    double av = fabs(col->value);
                    if (av > maxv) {
                        maxv    = av;
                        pivot_r = li;
                        pivot_c = lj;
                    }
                }
                col = col->down;
            }
        }

        if (maxv < tol) {
            printf("Near-singular at step %d (pivot < %g)\n", (int)k, tol);
            return;
        }

        if (pivot_r != k) {
            idx_t t = mat->row_perm[k]; mat->row_perm[k] = mat->row_perm[pivot_r];
            mat->row_perm[pivot_r] = t;
            mat->inv_row_perm[mat->row_perm[k]] = k;
            mat->inv_row_perm[mat->row_perm[pivot_r]] = pivot_r;
        }

        if (pivot_c != k) {
            idx_t t = mat->col_perm[k]; mat->col_perm[k] = mat->col_perm[pivot_c];
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
                double mult = getValue(mat, li, k) / getValue(mat, k, k);
                setValue(mat, li, k, mult);

                idx_t prk = mat->row_perm[k];
                Node* uj = mat->rowHeaders[prk];
                while (uj) {
                    idx_t pj = uj->col;
                    idx_t lj = mat->inv_col_perm[pj];
                    if (lj > k) {
                        double newv = getValue(mat, li, lj) - mult * uj->value;
                        if (fabs(newv) < DROP_TOL * maxv) {
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
// Solvers
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
        while (node && mat->inv_col_perm[node->col] < i) {
            idx_t j = mat->inv_col_perm[node->col];
            sum += node->value * y[j];
            node = node->right;
        }
        y[i] = pb[i] - sum;
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
            printf("Zero pivot in U at %d\n", (int)i);
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
    if (!pb || !y || !z) { free(pb); free(y); free(z); return; }

    applyRowPerm(mat, b, pb);
    forwardSubstitution(mat, pb, y);
    backwardSubstitution(mat, y, z);
    applyInvColPerm(mat, z, x);

    free(pb); free(y); free(z);
}

// ──────────────────────────────────────────────────────────────────────────────
// Debug / Display
// ──────────────────────────────────────────────────────────────────────────────

void displayLogical(SparseMatrix* mat) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < n; j++) {
            printf("%8.3f ", getValue(mat, i, j));
        }
        printf("\n");
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main example
// ──────────────────────────────────────────────────────────────────────────────

int main(void) {
    idx_t n = 3;
    SparseMatrix* mat = createSparseMatrix(n);
    if (!mat) return 1;

    insert(mat, 0, 0, 1.0);
    insert(mat, 0, 2, 3.0);
    insert(mat, 1, 1, 5.0);
    insert(mat, 2, 0, 7.0);
    insert(mat, 2, 2, 9.0);

    printf("Original matrix (logical view):\n");
    displayLogical(mat);
    printf("Non-zeros (storage) : %d\n", (int)countNNZ(mat));
    printf("Non-zeros (logical) : %d\n\n", (int)countNNZLogical(mat));

    printf("Computing LU ...\n");
    computeLU(mat, 1e-10);

    printf("\nAfter LU (L below diag, U on/above diag - logical view):\n");
    displayLogical(mat);
    printf("Non-zeros after factorization: %d\n\n", (int)countNNZ(mat));

    double b[3] = {1, 2, 3};
    double x[3] = {0};
    solveLU(mat, b, x);

    printf("Solution to A x = [1, 2, 3]ᵀ:\n");
    printf("x ≈ [%.6f, %.6f, %.6f]\n\n", x[0], x[1], x[2]);

    printf("Residual check (A x - b):\n");
    for (idx_t i = 0; i < n; i++) {
        double ax = 0.0;
        for (idx_t j = 0; j < n; j++) {
            ax += getValue(mat, i, j) * x[j];
        }
        printf("row %d: %12.3e\n", (int)i, ax - b[i]);
    }

    freeSparseMatrix(mat);
    return 0;
}
