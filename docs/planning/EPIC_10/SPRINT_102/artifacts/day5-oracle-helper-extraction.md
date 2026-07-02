# Sprint 102 Day 5 Oracle Helper Extraction

## Purpose

Day 5 implements the bounded helper extraction selected on Day 4. The change
consolidates external dense-reference subprocess/vector parsing while keeping
direct-solver fixture construction, solver execution, tolerances, residuals,
and assertions family-local.

## Implemented Helper

Added an opt-in test-only helper to `tests/test_solver_helpers.h`:

```c
typedef enum {
    TF_EXTERNAL_REFERENCE_ERROR = -1,
    TF_EXTERNAL_REFERENCE_SKIP = 0,
    TF_EXTERNAL_REFERENCE_OK = 1
} tf_external_reference_status_t;

tf_external_reference_status_t tf_read_external_reference_vector(
    const char *cmd,
    const char *label,
    double *x_out,
    idx_t n,
    char *reason,
    size_t reason_cap);
```

The helper is gated behind:

```c
#define TF_ENABLE_EXTERNAL_REFERENCE_HELPER
```

This keeps the subprocess reader out of unrelated tests that include
`tests/test_solver_helpers.h` only for residual helpers.

## Behavior Consolidated

| behavior | new owner |
|---|---|
| `popen` / `_popen` command execution | `tf_read_external_reference_vector(...)` |
| `OK n` header parsing | `tf_read_external_reference_vector(...)` |
| `SKIP reason` handling | `tf_read_external_reference_vector(...)` |
| `ERROR reason` handling | `tf_read_external_reference_vector(...)` |
| vector line parsing | `tf_read_external_reference_vector(...)` |
| dimension mismatch failure | `tf_read_external_reference_vector(...)` |
| truncated output failure | `tf_read_external_reference_vector(...)` |
| parse failure | `tf_read_external_reference_vector(...)` |
| non-zero helper exit failure | `tf_read_external_reference_vector(...)` |
| newline trimming for helper reasons | `tf_external_reference_copy_reason(...)` |

## Family-Local Behavior Preserved

| behavior | owner remains |
|---|---|
| Cholesky Matrix Market fixture loading | `tests/test_chol_csc.c` |
| Cholesky factor/solve path | `tests/test_chol_csc.c` |
| Cholesky tolerances and residual checks | `tests/test_chol_csc.c` |
| LDLT KKT fixture construction | `tests/test_ldlt_csc.c` |
| LDLT permutation handling | `tests/test_ldlt_csc.c` |
| LDLT factor/solve path | `tests/test_ldlt_csc.c` |
| LDLT tolerances and residual checks | `tests/test_ldlt_csc.c` |
| Python dense-reference math | `tests/chol_external_dense_reference.py`; `tests/ldlt_external_dense_reference.py` |

## Touched Files

| file | change |
|---|---|
| `tests/test_solver_helpers.h` | added opt-in external reference status enum and vector reader |
| `tests/test_chol_csc.c` | replaced duplicated external-reference parser with helper call |
| `tests/test_ldlt_csc.c` | replaced duplicated external-reference parser with helper call |

No public headers, library sources, CMake files, Makefile targets, or public
documentation were changed.

## Focused Validation Results

| command | result |
|---|---|
| `make format` | passed |
| `make build/test_chol_csc` | passed |
| `./build/test_chol_csc` | passed; 92 tests, 0 failures, 0 skips, 20844 assertions |
| `make build/test_ldlt_csc` | passed |
| `./build/test_ldlt_csc` | passed; 98 tests, 0 failures, 0 skips, 2288 assertions |

External dense-reference behavior preserved:

| lane | preserved evidence |
|---|---|
| Cholesky CSC `nos4` | passed; solver output compared with Python dense reference |
| Cholesky CSC `bcsstk04` with AMD | passed; solver output compared with Python dense reference |
| LDLT CSC `kkt5` | passed; `max\|x-x_ref\| = 0.000e+00`, zero residual |
| LDLT CSC `kkt10` | passed; `max\|x-x_ref\| = 3.553e-15`, residual `2.292e-16` |

## Full Validation Results

Because Day 5 changed `.c` and `.h` test files, the required full quality
chain was run:

| command | result |
|---|---|
| `make format` | passed |
| `make lint` | passed |
| `make test` | passed |

## Non-Claims Preserved

Day 5 earns only this maintainability claim:

> External dense-reference vector parsing is shared by direct-solver tests.

Day 5 does not claim:

- new LU, QR, SVD, Cholesky, or LDLT oracle coverage;
- direct CSR/CSC solver APIs;
- external oracle coverage for every direct solver;
- public API behavior changes;
- portable performance superiority;
- broad state-of-the-art solver parity.

## Day 5 Conclusion

The first oracle helper extraction is complete. Direct-solver external
dense-reference parsing now has one opt-in test-support owner, while Cholesky
and LDLT keep their mathematical behavior and claim boundaries local to their
family tests.
