// sparse_lu_orthogonal_with_pool_nnz_mem_io.c
// Orthogonal sparse matrix + complete pivoting LU + solvers
// Features: node pool, drop tol, auto nnz counter, mem estimator, file I/O
// Compile: gcc -O2 -Wall -o sparse_lu sparse_lu_orthogonal_with_pool_nnz_mem_io.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>  // For PRId32

#define DROP_TOL          1e-14
#define NODES_PER_SLAB    4096     // Tune as needed

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
    idx_t num_slabs;  // For memory estimation
} NodePool;

static Node* pool_alloc(NodePool* pool) {
    if (!pool->current || pool->current->used >= NODES_PER_SLAB) {
        NodeSlab* new_slab = (NodeSlab*)malloc(sizeof(NodeSlab));
        if (!new_slab) return NULL;
        new_slab->used = 0;
        new_slab->next = NULL;

        if (pool->current) {
            pool->current->next = new_slab;
        } else {
            pool->head = new_slab;
        }
        pool->current = new_slab;
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
// Sparse matrix structure
// ──────────────────────────────────────────────────────────────────────────────

typedef struct SparseMatrix {
    idx_t rows;
    idx_t cols;                 // currently assumes square for LU
    Node** rowHeaders;
    Node** colHeaders;
    idx_t* row_perm;
    idx_t* inv_row_perm;
    idx_t* col_perm;
    idx_t* inv_col_perm;
    NodePool node_pool;
    idx_t nnz;                  // automatic non-zero count
} SparseMatrix;

// ──────────────────────────────────────────────────────────────────────────────
// Creation / Destruction
// ──────────────────────────────────────────────────────────────────────────────

SparseMatrix* createSparseMatrix(idx_t n) {
    SparseMatrix* mat = (SparseMatrix*)malloc(sizeof(SparseMatrix));
    if (!mat) return NULL;

    mat->rows = n;
    mat->cols = n;

    mat->rowHeaders = (Node**)calloc(n, sizeof(Node*));
    mat->colHeaders = (Node**)calloc(n, sizeof(Node*));
    mat->row_perm = (idx_t*)malloc(n * sizeof(idx_t));
    mat->inv_row_perm = (idx_t*)malloc(n * sizeof(idx_t));
    mat->col_perm = (idx_t*)malloc(n * sizeof(idx_t));
    mat->inv_col_perm = (idx_t*)malloc(n * sizeof(idx_t));

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
        mat->row_perm[i] = i;
        mat->inv_row_perm[i] = i;
        mat->col_perm[i] = i;
        mat->inv_col_perm[i] = i;
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
// Core operations (physical indices)
// ──────────────────────────────────────────────────────────────────────────────

static Node* createNodeFromPool(SparseMatrix* mat, idx_t row, idx_t col, double value) {
    Node* node = pool_alloc(&mat->node_pool);
    if (!node) return NULL;
    node->row = row;
    node->col = col;
    node->value = value;
    node->right = NULL;
    node->down = NULL;
    return node;
}

void removeNode(SparseMatrix* mat, idx_t phys_row, idx_t phys_col) {
    // Remove from row list
    Node *prev = NULL, *curr = mat->rowHeaders[phys_row];
    while (curr && curr->col != phys_col) {
        prev = curr;
        curr = curr->right;
    }
    if (!curr) return;
    if (prev) {
        prev->right = curr->right;
    } else {
        mat->rowHeaders[phys_row] = curr->right;
    }

    // Remove from column list
    prev = NULL;
    curr = mat->colHeaders[phys_col];
    while (curr && curr->row != phys_row) {
        prev = curr;
        curr = curr->down;
    }
    if (prev) {
        prev->down = curr->down;
    } else {
        mat->colHeaders[phys_col] = curr->down;
    }

    mat->nnz--;  // Decrement counter
    // Note: no free(curr) - pool handles it
}

void insert(SparseMatrix* mat, idx_t phys_row, idx_t phys_col, double value) {
    if (value == 0.0) {
        removeNode(mat, phys_row, phys_col);
        return;
    }

    // Check if exists
    Node *prevRow = NULL;
    Node *currRow = mat->rowHeaders[phys_row];
    while (currRow && currRow->col < phys_col) {
        prevRow = currRow;
        currRow = currRow->right;
    }
    if (currRow && currRow->col == phys_col) {
        currRow->value = value;  // Overwrite
        return;
    }

    // Create new
    Node* newNode = createNodeFromPool(mat, phys_row, phys_col, value);
    if (!newNode) return;
    mat->nnz++;  // Increment counter

    // Insert into row list
    newNode->right = currRow;
    if (prevRow == NULL) {
        mat->rowHeaders[phys_row] = newNode;
    } else {
        prevRow->right = newNode;
    }

    // Insert into column list
    Node *prevCol = NULL;
    Node *currCol = mat->colHeaders[phys_col];
    while (currCol && currCol->row < phys_row) {
        prevCol = currCol;
        currCol = currCol->down;
    }
    newNode->down = currCol;
    if (prevCol == NULL) {
        mat->colHeaders[phys_col] = newNode;
    } else {
        prevCol->down = newNode;
    }
}

double getPhysValue(SparseMatrix* mat, idx_t phys_row, idx_t phys_col) {
    Node* curr = mat->rowHeaders[phys_row];
    while (curr && curr->col < phys_col) {
        curr = curr->right;
    }
    if (curr && curr->col == phys_col) {
        return curr->value;
    }
    return 0.0;
}

double getValue(SparseMatrix* mat, idx_t log_row, idx_t log_col) {
    idx_t phys_row = mat->row_perm[log_row];
    idx_t phys_col = mat->col_perm[log_col];
    return getPhysValue(mat
