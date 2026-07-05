#ifndef SPARSE_LU_CSR_INTERNAL_H
#define SPARSE_LU_CSR_INTERNAL_H

#include "sparse_lu_csr.h"

sparse_err_t lu_csr_grow(LuCsr *csr, idx_t needed);
sparse_err_t lu_csr_validate(const LuCsr *csr);

#endif
