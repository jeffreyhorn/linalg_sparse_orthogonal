# Sprint 113 Day 10: SVD Proof-Owner Cleanup

## Purpose

Implement the bounded SVD cleanup selected on Day 9 while preserving the
partial-SVD vector proof values at the test call sites.

## Selected Owner

Day 10 cleaned up the duplicated partial-SVD `A*v ~= sigma*u` residual loops in
`tests/test_svd_partial_helpers.h`.

Primary tests touched:

- `test_partial_svd_vectors_Av`;
- `test_partial_svd_vectors_wide`.

## Before/After Metrics

- Before: `tests/test_svd_partial_helpers.h` had 915 lines.
- After: `tests/test_svd_partial_helpers.h` has 907 lines.
- Net change: -8 lines after centralizing the mechanical residual loop.

`tests/test_svd.c` remained at 2893 lines.

## Code Changes

- Added local helper `partial_svd_max_av_residual`.
- Centralized only the mechanical residual computation:
  - temporary `Av` allocation;
  - temporary `v` allocation;
  - `Vt` row extraction into `v`;
  - `sparse_matvec(A, v, Av)`;
  - `||A*v_s - sigma_s*u_s||_2` computation;
  - maximum residual tracking across retained singular triplets.
- Updated `test_partial_svd_vectors_Av` to call the helper.
- Updated `test_partial_svd_vectors_wide` to call the helper.

## Proof Visibility Preserved

The cleanup intentionally left the behavior-sensitive proof values at the test
sites:

- fixture shapes:
  - 8x8 tridiagonal-like partial-vector fixture;
  - 4x8 wide diagonal partial-vector fixture;
- inserted matrix values;
- selected partial ranks:
  - `k = 3`;
  - `k = 2`;
- SVD options:
  - `compute_uv = 1`;
  - `economy = 1`;
  - `max_iter = 0`;
  - `tol = 0.0`;
- expected singular-value tolerances for the wide fixture;
- residual diagnostics:
  - `partial SVD A*v ~= sigma*u`;
  - `wide 4x8 partial vectors`;
- residual threshold `1e-6`;
- `sparse_svd_partial` call and result checks;
- `sparse_svd_free` and fixture cleanup ownership.

## Validation

Focused validation passed:

```sh
make build/test_svd && build/test_svd
```

Result:

- `test_svd`: 98 tests run;
- 0 failed;
- 0 skipped;
- 1562 assertions.

Full required quality chain passed because a header file changed:

```sh
make format && make lint && make test
```

Result:

- formatting completed;
- strict warning build completed;
- `clang-tidy` completed;
- `cppcheck` completed, including `tests/test_svd.c` and the included partial
  helper header;
- full test suite passed.

## Drift Assessment

No scope drift was introduced:

- no public API changes;
- no install-header changes;
- no helper-target changes;
- no Makefile or CMake source-list changes;
- no reviewed CTest registration changes;
- no broad SVD reconstruction abstraction;
- no shared U/Vt orthogonality abstraction;
- no Moore-Penrose product helper.

## Remaining SVD Cleanup Queue

The following SVD cleanup candidates remain deferred:

- reconstruction helper movement;
- U/Vt orthogonality helper movement;
- Moore-Penrose product helper extraction;
- dense low-rank proof-loop cleanup;
- sparse low-rank proof-loop cleanup;
- condition-number proof cleanup.
