#include "sparse_types.h"

static _Thread_local int last_errno = 0;

void sparse_set_errno_(int errnum)
{
    last_errno = errnum;
}

int sparse_errno(void)
{
    return last_errno;
}

const char *sparse_strerror(sparse_err_t err)
{
    switch (err) {
    case SPARSE_OK:           return "success";
    case SPARSE_ERR_NULL:     return "null pointer argument";
    case SPARSE_ERR_ALLOC:    return "memory allocation failure";
    case SPARSE_ERR_BOUNDS:   return "index out of bounds";
    case SPARSE_ERR_SINGULAR: return "singular or nearly singular matrix";
    case SPARSE_ERR_FOPEN:    return "cannot open file";
    case SPARSE_ERR_FREAD:    return "file read error";
    case SPARSE_ERR_FWRITE:   return "file write error";
    case SPARSE_ERR_PARSE:    return "file format parse error";
    case SPARSE_ERR_SHAPE:    return "matrix shape mismatch";
    case SPARSE_ERR_IO:       return "I/O error (check sparse_errno())";
    }
    return "unknown error";
}
