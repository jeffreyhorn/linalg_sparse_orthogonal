# Sprint 129 Day 13 - Bidiagonal/Golub-Kahan Helper Decision

## Decision

Day 13 extracted one bounded Bidiagonal-owned helper and left Golub-Kahan,
SVD, QR-iteration, and partial-SVD helpers in their current owners.

The accepted movement was:

- from `tests/test_bidiag.c`
- to `tests/test_bidiag_helpers.h`
- renamed from `bidiag_reconstruction_error`
- to `tf_bidiag_reconstruction_max_error`

This keeps the helper test-only, Bidiagonal-owned, and explicitly scoped to
implicit Householder bidiagonal reconstruction.

## Source And Build Ownership

| Area | Day 13 action | Build impact |
| --- | --- | --- |
| Bidiagonal reconstruction helper | Moved to `tests/test_bidiag_helpers.h` and included from `tests/test_bidiag.c`. | Header-only helper; no `Makefile` or `CMakeLists.txt` source-list update required. |
| Bidiagonal call sites | Updated to call `tf_bidiag_reconstruction_max_error`. | Existing `test_bidiag` target continues to own validation. |
| Golub-Kahan reconstruction | No movement. | Existing `tests/test_svd.c` owner unchanged. |
| SVD helper headers | No movement. | Existing SVD helper ownership unchanged. |
| Partial-SVD helpers | No movement. | Existing partial-SVD helper ownership unchanged. |

## Preserved Semantics

The extracted helper preserves the Day 12 checklist:

- transposed `sparse_bidiag_t` recursion still transposes the input matrix and
  checks a non-transposed view;
- `sparse_get_phys` remains the comparison path;
- implicit left and right Householder replay order is unchanged;
- allocation failure still reports `HUGE_VAL`;
- dense reconstruction layout and cleanup behavior are unchanged;
- call-site matrices, tolerances, assertions, and diagnostics remain visible
  in `tests/test_bidiag.c`.

## Non-Claims

Day 13 did not create a general SVD helper owner. In particular:

- `gk_reconstruction_error` remains in `tests/test_svd.c` because it validates
  explicit Golub-Kahan `U`/`V`, `diag`, and `superdiag` products.
- Wide Golub-Kahan skip/scoping behavior remains owner-local to SVD tests and
  does not inherit Bidiagonal transpose recursion.
- Full SVD reconstruction and orthogonality helpers remain in
  `tests/test_svd_helpers.h`.
- Partial-SVD residual helpers remain in `tests/test_svd_partial_helpers.h`.
- No public API, production source, Matrix Market fixture, or maintainer-guide
  ownership claim changed.

## Rollback Boundary

The movement can be rolled back by moving
`tf_bidiag_reconstruction_max_error` back into `tests/test_bidiag.c` and
reverting the call-site rename. No build source lists, public API declarations,
or production code depend on the new helper header.

## Validation Plan

Because Day 13 changed `.c` and `.h` files, the required validation package is:

- focused Bidiagonal validation:
  `make build/test_bidiag && ./build/test_bidiag`
- focused SVD/GK validation:
  `make build/test_svd && ./build/test_svd`
- full quality gate:
  `make format && make lint && make test`

## Validation Results

Day 13 validation passed:

- `make build/test_bidiag && ./build/test_bidiag`
- `make build/test_svd && ./build/test_svd`
- `make format && make lint && make test`

## Result

Bidiagonal/Golub-Kahan helper ownership is clearer after Day 13. The only
extracted helper is Bidiagonal-owned and preserves the implicit Householder
reconstruction semantics from the original test file. Golub-Kahan and SVD
helpers remain explicitly out of scope for this movement.
