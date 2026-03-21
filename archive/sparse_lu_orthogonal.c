// sparse_lu_orthogonal.c
// Orthogonal (cross-linked) sparse matrix with complete pivoting LU
// and forward/backward substitution
// Compile: gcc -O2 -Wall -o sparse_lu sparse_lu_orthogonal.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>     // for int32_t
#include <math.h>

#define DROP_TOL 1e-14

typedef int32_t idx_t;

// Node for each non-zero entry
typedef struct Node {
    idx_t row;          // physical row index
    idx_t col;          // physical column index
    double value;
    struct Node* right; // next in same row (sorted by col)
    struct Node* down;  // next in same column (sorted by row)
} Node;

typedef struct SparseMatrix {
    idx_t rows;
    idx_t cols;                 // currently assumes square matrix
    Node** rowHeaders;
    Node** colHeaders;
    idx_t* row_perm;            // logical → physical row
    idx_t* inv_row_perm;        // physical → logical row
    idx_t* col_perm;            // logical → physical column
    idx_t* inv_col_perm;        // physical → logical column
} SparseMatrix;

// ──────────────────────────────────────────────────────────────────────────────
// Creation / Destruction
// ──────────────────────────────────────────────────────────────────────────────

Node* createNode(idx_t row, idx_t col, double value) {
    Node* node = (Node*)malloc(sizeof(Node));
    if (!node) return NULL;
    node->row   = row;
    node->col   = col;
    node->value = value;
    node->right = NULL;
    node->down  = NULL;
    return node;
}

SparseMatrix* createSparseMatrix(idx_t n) {
    SparseMatrix* mat = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;

    mat->rows = n;
    mat->cols = n;

    mat->rowHeaders    = (Node**)calloc(n, sizeof(Node*));
    mat->colHeaders    = (Node**)calloc(n, sizeof(Node*));
    mat->row_perm      = (idx_t*)malloc(n * sizeof(idx_t));
    mat->inv_row_perm  = (idx_t*)malloc(n * sizeof(idx_t));
    mat->col_perm      = (idx_t*)malloc(n * sizeof(idx_t));
    mat->inv_col_perm  = (idx_t*)malloc(n * sizeof(idx_t));

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

    return mat;
}

void freeSparseMatrix(SparseMatrix* mat) {
    if (!mat) return;
    for (idx_t i = 0; i < mat->rows; i++) {
        Node* curr = mat->rowHeaders[i];
        while (curr) {
            Node* tmp = curr;
            curr = curr->right;
            free(tmp);
        }
    }
    free(mat->rowHeaders);
    free(mat->colHeaders);
    free(mat->row_perm);
    free(mat->inv_row_perm);
    free(mat->col_perm);
    free(mat->inv_col_perm);
    free(mat);
}

// ──────────────────────────────────────────────────────────────────────────────
// Core operations (physical indices)
// ──────────────────────────────────────────────────────────────────────────────

void removeNode(SparseMatrix* mat, idx_t phys_row, idx_t phys_col) {
    // Remove from row list
    Node *prev = NULL, *curr = mat->rowHeaders[phys_row];
    while (curr && curr->col != phys_col) {
        prev = curr;
        curr = curr->right;
    }
    if (!curr) return;
    if (prev) prev->right = curr->right;
    else      mat->rowHeaders[phys_row] = curr->right;

    // Remove from column list
    prev = NULL; curr = mat->colHeaders[phys_col];
    while (curr && curr->row != phys_row) {
        prev = curr;
        curr = curr->down;
    }
    if (prev) prev->down = curr->down;
    else      mat->colHeaders[phys_col] = curr->down;

    free(curr);
}

void insert(SparseMatrix* mat, idx_t phys_row, idx_t phys_col, double val) {
    if (val == 0.0) {
        removeNode(mat, phys_row, phys_col);
        return;
    }

    // Check if exists (in row list – sorted)
    Node *prev_r = NULL, *curr_r = mat->rowHeaders[phys_row];
    while (curr_r && curr_r->col < phys_col) {
        prev_r = curr_r;
        curr_r = curr_r->right;
    }
    if (curr_r && curr_r->col == phys_col) {
        curr_r->value = val;
        return;
    }

    Node* node = createNode(phys_row, phys_col, val);
    if (!node) return;

    node->right = curr_r;
    if (prev_r) prev_r->right = node;
    else        mat->rowHeaders[phys_row] = node;

    // Insert into column list (sorted by row)
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
    idx_t pr = mat->row_perm[log_row];
    idx_t pc = mat->col_perm[log_col];
    return getPhysValue(mat, pr, pc);
}

void setValue(SparseMatrix* mat, idx_t log_row, idx_t log_col, double val) {
    idx_t pr = mat->row_perm[log_row];
    idx_t pc = mat->col_perm[log_col];
    insert(mat, pr, pc, val);
}

// ──────────────────────────────────────────────────────────────────────────────
// Complete pivoting LU (P A Q = L U)
// L = unit lower triangular (1s on diag, not stored)
// U = upper triangular (diag + upper part stored)
// ──────────────────────────────────────────────────────────────────────────────

void computeLU(SparseMatrix* mat, double tol) {
    idx_t n = mat->rows;

    for (idx_t k = 0; k < n; k++) {
        // Find pivot: max |A[i,j]| for i≥k, j≥k (logical)
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
                        maxv     = av;
                        pivot_r  = li;
                        pivot_c  = lj;
                    }
                }
                col = col->down;
            }
        }

        if (maxv < tol) {
            printf("Near-singular at step %d (pivot < %g)\n", (int)k, tol);
            return;
        }

        // Swap rows
        if (pivot_r != k) {
            idx_t t = mat->row_perm[k];
            mat->row_perm[k] = mat->row_perm[pivot_r];
            mat->row_perm[pivot_r] = t;
            mat->inv_row_perm[mat->row_perm[k]]     = k;
            mat->inv_row_perm[mat->row_perm[pivot_r]] = pivot_r;
        }

        // Swap columns
        if (pivot_c != k) {
            idx_t t = mat->col_perm[k];
            mat->col_perm[k] = mat->col_perm[pivot_c];
            mat->col_perm[pivot_c] = t;
            mat->inv_col_perm[mat->col_perm[k]]     = k;
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
                setValue(mat, li, k, mult);  // L part

                // row_i -= mult * row_k   (only j > k)
                idx_t prk = mat->row_perm[k];
                Node* uj = mat->rowHeaders[prk];
                while (uj) {
                    idx_t pj = uj->col;
                    idx_t lj = mat->inv_col_perm[pj];
                    if (lj > k) {
                        double u_kj = uj->value;
                        double a_ij = getValue(mat, li, lj);
                        double newv = a_ij - mult * u_kj;
                        // Apply drop tolerance
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
    for (idx_t i = 0; i < n; i++)
        pb[i] = b[mat->row_perm[i]];
}

void applyInvColPerm(SparseMatrix* mat, const double* z, double* x) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++)
        x[i] = z[mat->col_perm[i]];
}

void forwardSubstitution(SparseMatrix* mat, const double* pb, double* y) {
    idx_t n = mat->rows;
    for (idx_t i = 0; i < n; i++) {
        double sum = 0.0;
        idx_t pi = mat->row_perm[i];
        Node* node = mat->rowHeaders[pi];
        while (node) {
            idx_t pj = node->col;
            idx_t j  = mat->inv_col_perm[pj];
            if (j < i) {
                sum += node->value * y[j];
            } else if (j >= i) {
                break;  // lists sorted → no more L entries
            }
            node = node->right;
        }
        y[i] = pb[i] - sum;  // L_{i,i} = 1
    }
}

void backwardSubstitution(SparseMatrix* mat, const double* y, double* x) {
    idx_t n = mat->rows;
    for (idx_t i = n-1; i != (idx_t)-1; i--) {
        double sum = 0.0;
        idx_t pi = mat->row_perm[i];
        Node* node = mat->rowHeaders[pi];
        double u_ii = 0.0;
        idx_t found_diag = 0;

        while (node) {
            idx_t pj = node->col;
            idx_t j  = mat->inv_col_perm[pj];
            if (j == i) {
                u_ii = node->value;
                found_diag = 1;
            } else if (j > i) {
                sum += node->value * x[j];
            }
            node = node->right;
        }

        if (!found_diag || fabs(u_ii) < 1e-14) {
            printf("Zero/small pivot in U at row %d\n", (int)i);
            return;
        }
        x[i] = (y[i] - sum) / u_ii;
    }
}

void solveLU(SparseMatrix* mat, const double* b, double* x) {
    idx_t n = mat->rows;
    double* pb = malloc(n * sizeof(double));
    double* y  = malloc(n * sizeof(double));
    double* z  = malloc(n * sizeof(double));

    if (!pb || !y || !z) {
        free(pb); free(y); free(z);
        return;
    }

    applyRowPerm(mat, b, pb);
    forwardSubstitution(mat, pb, y);
    backwardSubstitution(mat, y, z);
    applyInvColPerm(mat, z, x);

    free(pb);
    free(y);
    free(z);
}

// ──────────────────────────────────────────────────────────────────────────────
// Utility / Debug
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
// Example
// ──────────────────────────────────────────────────────────────────────────────

int main(void) {
    idx_t n = 3;
    SparseMatrix* mat = createSparseMatrix(n);
    if (!mat) return 1;

    // Example matrix (same as before)
    insert(mat, 0, 0, 1.0);
    insert(mat, 0, 2, 3.0);
    insert(mat, 1, 1, 5.0);
    insert(mat, 2, 0, 7.0);
    insert(mat, 2, 2, 9.0);

    printf("Original matrix (logical view):\n");
    displayLogical(mat);

    computeLU(mat, 1e-10);

    printf("\nAfter LU (L below diag + U on/above, logical view):\n");
    displayLogical(mat);

    printf("\nRow perm  (logical→phys): ");
    for (idx_t i = 0; i < n; i++) printf("%d ", mat->row_perm[i]);
    printf("\nCol perm  (logical→phys): ");
    for (idx_t i = 0; i < n; i++) printf("%d ", mat->col_perm[i]);
    printf("\n");

    // Solve A x = b
    double b[3] = {1, 2, 3};
    double x[3] = {0};

    solveLU(mat, b, x);

    printf("\nSolution to A x = [1,2,3]ᵀ:\n");
    printf("x = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);

    // Quick residual check
    printf("\nResidual check (A x - b):\n");
    for (idx_t i = 0; i < n; i++) {
        double ax = 0.0;
        for (idx_t j = 0; j < n; j++)
            ax += getValue(mat, i, j) * x[j];
        printf("%12.3e\n", ax - b[i]);
    }

    freeSparseMatrix(mat);
    return 0;
}
