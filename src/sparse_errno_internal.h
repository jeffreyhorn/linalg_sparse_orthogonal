#ifndef SPARSE_ERRNO_INTERNAL_H
#define SPARSE_ERRNO_INTERNAL_H

/*
 * Private header: internal errno capture helper shared between
 * sparse_types.c and internal library call sites.
 */

void sparse_set_errno_(int errnum);

#endif /* SPARSE_ERRNO_INTERNAL_H */
