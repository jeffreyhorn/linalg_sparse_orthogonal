# Matrix Market Format Support

## Overview

The library supports reading and writing sparse matrices in the [Matrix Market](https://math.nist.gov/MatrixMarket/formats.html) coordinate format. This is a simple text-based format widely used for exchanging sparse matrices, including the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).

## Supported Features

### Writing (`sparse_save_mm`)

Matrices are always written as:

```
%%MatrixMarket matrix coordinate real general
```

- Values are written with full double-precision: `%.15g`
- Only stored non-zeros are written (physical ordering)
- Indices are 1-based in the file (converted from internal 0-based)

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

For **symmetric** matrices, the file typically contains only the lower triangle. The loader mirrors each off-diagonal entry (i, j, v) to also insert (j, i, v), so the loaded matrix is fully populated.

#### Pattern matrices

**Pattern** matrices have no value field — they specify only the sparsity structure. The loader assigns value 1.0 to all entries.

#### Integer matrices

**Integer** values are read and stored as `double`.

## Unsupported Features

The following Matrix Market features are **not** supported:

| Feature | Status |
|---------|--------|
| `array` (dense) format | Not supported — use coordinate format |
| `complex` value type | Not supported — only real values |
| `skew-symmetric` symmetry | Not supported |
| `Hermitian` symmetry | Not supported (complex not supported) |
| Comment lines (lines starting with `%`) | Skipped correctly during loading |
| Blank lines between header and data | Handled correctly |

Attempting to load an unsupported format returns `SPARSE_ERR_PARSE`.

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
   Where rows and cols are dimensions and nnz is the number of entries that follow.

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

You can download matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) in Matrix Market format and load them directly:

```c
SparseMatrix *A = NULL;
sparse_err_t err = sparse_load_mm(&A, "downloaded_matrix.mtx");
if (err != SPARSE_OK) {
    fprintf(stderr, "Load failed: %s\n", sparse_strerror(err));
}
```

Note that very large matrices may require significant memory due to the per-node overhead of the orthogonal linked-list representation (~32 bytes per non-zero).
