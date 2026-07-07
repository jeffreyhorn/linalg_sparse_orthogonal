# Matrix Market Format Support

## Overview

The library supports reading and writing sparse matrices in the
[Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) coordinate
format. This is a simple text-based format widely used for exchanging sparse
matrices, including the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

The public API surface is the pair of functions declared in
[`sparse_matrix.h`](../include/sparse_matrix.h):

- `sparse_load_mm(...)`
- `sparse_save_mm(...)`

This is not a separate public Matrix I/O module and does not expose a public
matrix builder API. Loaded matrices are normal caller-owned `SparseMatrix`
objects and are freed with `sparse_free(...)`.

## Supported Features

### Writing (`sparse_save_mm`)

Matrices are always written as:

```
%%MatrixMarket matrix coordinate real general
```

- Values are written with full double-precision: `%.15g`
- Only stored non-zeros are written (physical ordering)
- Indices are 1-based in the file (converted from internal 0-based)
- On success, `sparse_errno()` is reset to 0
- On file open/write/close failure, the function returns `SPARSE_ERR_IO` and
  captures the system errno for `sparse_errno()`

### Reading (`sparse_load_mm`)

The loader supports the following Matrix Market header combinations:

| Object | Format | Value type | Symmetry |
|--------|--------|-----------|----------|
| matrix | coordinate | real | general |
| matrix | coordinate | real | symmetric |
| matrix | coordinate | pattern | general |
| matrix | coordinate | pattern | symmetric |
| matrix | coordinate | integer | general |
| matrix | coordinate | integer | symmetric |

#### Symmetry handling

For **symmetric** matrices, the file typically contains only one triangle. The
loader mirrors each off-diagonal entry `(i, j, v)` by also inserting
`(j, i, v)`, so the loaded matrix is fully populated in the returned
`SparseMatrix`. Symmetric Matrix Market inputs must be square.

#### Pattern matrices

**Pattern** matrices have no value field; they specify only the sparsity
structure. The loader assigns value `1.0` to every pattern entry.

#### Integer matrices

**Integer** values are read and stored as `double`.

#### Duplicate entries and zero values

If a Matrix Market file contains duplicate coordinates, the last entry for
that coordinate in file order wins. If the final value for a coordinate is
`0.0`, that coordinate is omitted from the returned sparse matrix. This matches
the normal sparse storage rule: explicit zero and absent entry are both read
through `sparse_get(...)` as `0.0`.

For symmetric inputs, mirrored off-diagonal entries participate in the same
duplicate-resolution rule after expansion.

#### Ownership and error handling

On success, `sparse_load_mm(...)` stores a new caller-owned matrix in
`*mat_out`; free it with `sparse_free(...)`. On error, `*mat_out` is set to
`NULL`.

Return codes:

| Condition | Return code |
|-----------|-------------|
| Success | `SPARSE_OK` |
| NULL output pointer or filename | `SPARSE_ERR_NULL` |
| File open/read/close failure | `SPARSE_ERR_IO` |
| Unsupported header, malformed dimensions, malformed data, out-of-range coordinates, zero coordinates, or rectangular symmetric input | `SPARSE_ERR_PARSE` |
| Allocation failure or supported-size overflow | `SPARSE_ERR_ALLOC` |

For `SPARSE_ERR_IO`, call `sparse_errno()` to inspect the captured system
errno. Successful load/save calls reset `sparse_errno()` to `0`. Parse errors
do not represent system I/O failures and should be handled through the
`SPARSE_ERR_PARSE` return code.

## Unsupported Features

The following Matrix Market features are **not** supported:

| Feature | Status |
|---------|--------|
| `array` (dense) format | Not supported — use coordinate format |
| `complex` value type | Not supported — only real values |
| `skew-symmetric` symmetry | Not supported |
| `Hermitian` symmetry | Not supported (complex not supported) |

Attempting to load an unsupported format returns `SPARSE_ERR_PARSE`.

Comment lines beginning with `%` between the header and size line are skipped.
Inputs still need a parseable size line and exactly `nnz` parseable data lines
in the expected positions.

## File Format Reference

A Matrix Market file consists of:

1. **Header line** (required):
   ```
   %%MatrixMarket matrix coordinate real general
   ```

2. **Comment lines** (optional, any number):
   ```
   % This is a comment
   % Author: ...
   ```

3. **Size line**:
   ```
   rows cols nnz
   ```
   Where rows and cols are dimensions and nnz is the number of entries that
   follow. Symmetric inputs must have `rows == cols`.

4. **Data lines** (exactly nnz lines):
   ```
   row col value
   ```
   For `real` and `integer` formats. Indices are 1-based.

   For `pattern` format:
   ```
   row col
   ```
   (No value field.)

## Example Files

The `tests/data/` directory contains reference matrices:

| File | Description | Format |
|------|-------------|--------|
| `identity_5.mtx` | 5×5 identity | real general |
| `diagonal_10.mtx` | 10×10 diagonal (d[i] = i+1) | real general |
| `tridiagonal_20.mtx` | 20×20 Poisson-1D tridiagonal | real general |
| `symmetric_4.mtx` | 4×4 symmetric matrix | real symmetric |
| `pattern_3.mtx` | 3×3 pattern-only matrix | pattern general |
| `bcsstk01.mtx` | 6×6 SPD structural matrix | real symmetric |
| `unsymm_5.mtx` | 5×5 unsymmetric diag-dominant | real general |
| `bad_header.mtx` | Invalid header (for error testing) | — |

## Using SuiteSparse Matrices

You can download matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) in Matrix Market
format and load them directly:

```c
SparseMatrix *A = NULL;
sparse_err_t err = sparse_load_mm(&A, "downloaded_matrix.mtx");
if (err != SPARSE_OK) {
    fprintf(stderr, "Load failed: %s\n", sparse_strerror(err));
    if (err == SPARSE_ERR_IO) {
        fprintf(stderr, "system errno: %d\n", sparse_errno());
    }
    return;
}

/* Use A with the normal public matrix and solver APIs. */
sparse_free(A);
```

For a complete load/use example, see
[`examples/example_matrix_market.c`](../examples/example_matrix_market.c).

Note that very large matrices may require significant memory due to the
per-node overhead of the orthogonal linked-list representation.
