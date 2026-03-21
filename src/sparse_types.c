#include "sparse_types.h"

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
    }
    return "unknown error";
}
