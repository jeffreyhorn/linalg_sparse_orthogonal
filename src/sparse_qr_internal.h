#ifndef SPARSE_QR_INTERNAL_H
#define SPARSE_QR_INTERNAL_H

#include "sparse_matrix.h"
#include "sparse_qr.h"

double s29_qr_now_s(void);

double sparse_qr_householder_compute(const double *x, double *v, idx_t len);
void sparse_qr_householder_apply(const double *v, double beta, double *y, idx_t len);

void sparse_qr_extract_column(const SparseMatrix *A, idx_t col, double *dense);
void sparse_qr_householder_apply_to_column(const double *v, double beta, double *dense, idx_t start,
                                           idx_t m);

#endif
