# Sprint 113 Day 9: SVD Proof Boundary Refresh

## Purpose

Refresh the remaining SVD proof-owner cleanup candidates, select one bounded
Day 10 cleanup target, and define the proof values that must stay visible after
cleanup.

Day 9 is intentionally a boundary and selection day. It does not modify SVD
code.

## Current SVD Test Surface

Primary files reviewed:

- `tests/test_svd.c`: 2893 lines.
- `tests/test_svd_partial_helpers.h`: 915 lines.

Focused baseline validation passed:

```sh
make build/test_svd && build/test_svd
```

Result:

- `test_svd`: 98 tests run;
- 0 failed;
- 0 skipped;
- 1562 assertions.

## Candidate Comparison

| Candidate | Current owner | Duplication pressure | Behavior risk | Proof clarity risk | Day 9 disposition |
|---|---|---:|---:|---:|---|
| Reconstruction helper movement | `tests/test_svd.c` GK and full/economy reconstruction checks | Medium | Medium | High | Defer. Reconstruction formulas differ by storage contract and should not be merged without a broader design pass. |
| U/Vt orthogonality helper movement | `tests/test_svd.c` and partial helper vector tests | Medium | Medium | Medium | Defer. There are multiple leading-dimension conventions across economy/full and `U`/`Vt` checks. |
| Moore-Penrose helper extraction | `tests/test_svd.c` pseudoinverse tests | Medium | Medium | High | Defer. Dense product dimensions and expected inverse entries must remain close to the pseudoinverse assertions. |
| Dense low-rank proof-loop cleanup | `tests/test_svd.c` low-rank dense residual checks | Low | Medium | Medium | Defer. The retained singular-value error-bound proof is already readable. |
| Sparse low-rank proof-loop cleanup | `tests/test_svd.c` sparse-vs-dense and corpus low-rank checks | Medium | Medium | High | Defer. Fixture names, drop tolerances, and env gates are part of the proof contract. |
| Partial-SVD vector/residual cleanup | `tests/test_svd_partial_helpers.h` partial vector residual tests | High | Low | Low | Select for Day 10. Repeated `A*v ~= sigma*u` loops can share a local residual helper while preserving visible matrices, ranks, tolerances, and assertions. |
| Condition-number proof cleanup | `tests/test_svd.c` condition-number tests | Low | Low | Medium | Defer. Expected finite/infinite condition values are already compact and should remain local. |

## Selected Day 10 Target

Day 10 should clean up the duplicated partial-SVD vector residual loop in
`tests/test_svd_partial_helpers.h`.

Primary target tests:

- `test_partial_svd_vectors_Av`;
- `test_partial_svd_vectors_wide`.

Both tests currently:

- compute a singular vector `v` from `svd.Vt`;
- run `sparse_matvec(A, v, Av)`;
- compare `Av` against `sigma[s] * U[:, s]`;
- accumulate the squared residual;
- track a maximum residual over the retained partial singular triplets.

## Required Proof Visibility

The Day 10 cleanup must keep the following proof values visible at or near each
test call site:

- fixture shape and inserted diagonal/off-diagonal values;
- selected partial rank `k`;
- `compute_uv`, `economy`, `max_iter`, and `tol` options;
- expected singular-value tolerances;
- residual threshold `1e-6`;
- printed diagnostic label for the residual lane;
- `sparse_svd_partial` call and result checks;
- `svd.U`, `svd.Vt`, `svd.sigma`, `svd.m`, and `svd.n` ownership expectations.

The helper may centralize only the mechanical residual computation:

- temporary `Av` allocation;
- temporary `v` allocation;
- extraction of each `Vt` row into `v`;
- `sparse_matvec`;
- `||A*v_s - sigma_s*u_s||_2` computation;
- maximum residual across `s = 0..k-1`.

## Non-Claims

Day 10 must not introduce:

- a broad SVD reconstruction abstraction;
- a shared U/Vt orthogonality abstraction;
- a Moore-Penrose dense product helper;
- new public API;
- new install-header declarations;
- Makefile or CMake source-list changes;
- new CTest registrations.

## Validation Requirements

If Day 10 changes `tests/test_svd_partial_helpers.h` or any other C/header
file, run:

```sh
make build/test_svd && build/test_svd
make format && make lint && make test
git diff --check
```

The focused SVD output should still include the existing diagnostics:

- `partial SVD A*v ~= sigma*u: max_resid=...`;
- `wide 4x8 partial vectors: max_resid=...`.

## Remaining SVD Queue

Unselected SVD cleanup candidates remain deferred:

- reconstruction helper movement;
- U/Vt orthogonality helper movement;
- Moore-Penrose product helper extraction;
- dense low-rank proof-loop cleanup;
- sparse low-rank proof-loop cleanup;
- condition-number proof cleanup.
