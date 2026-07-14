# Sprint 124 Day 7 QR Q-Basis/Economy Decision Package

## Purpose

Day 7 decides the Sprint 124 Q-basis/economy external evidence lane after the
Day 6 sign, orientation, projection, subspace, and economy-shape semantics.
The decision must add only basis-invariant proof and must not claim raw Q
column equality, sign orientation, rank-deficient subspace parity, sparse-mode
parity, or broad external QR parity.

## Inputs Reviewed

| Input | Decision Use |
| --- | --- |
| Sprint 124 Plan Day 7 | Requires accepted/deferred Q-basis/economy decision, metric protocol, future-owner handoff, validation plan, and non-claim register. |
| Day 6 Q-basis/economy semantics artifact | Defines acceptable metrics, shape expectations, owner boundaries, and raw-basis rejection rules. |
| `include/sparse_qr.h` | Defines economy `Q` shape and `sparse_qr_form_q` layout. |
| `tests/test_qr.c` | Primary Q/economy owner and target for the accepted external evidence. |
| `tests/qr_external_dense_reference.py` | Existing bounded Python helper; extended with a separate projector protocol for one named fixture. |
| `docs/maintainer_guide.md` | QR evidence and non-claim table that must remain fixture-scoped. |

## Decision Summary

Accepted one bounded economy-Q projector evidence lane:

`qr_economy_projector_5x3`

The lane compares the product's economy-Q projector `Q Q^T` to a
standard-library Python dense reference projector `A (A^T A)^{-1} A^T` for the
existing full-column-rank 5x3 compatible QR matrix. It also checks economy
shape and thin-Q orthogonality in the existing `tests/test_qr.c` owner.

This explicitly avoids raw Q-column equality. A valid QR factor can flip signs
or choose a different but equivalent basis in ambiguous cases, so Day 7 proves
projection-space behavior for one named, non-degenerate economy fixture rather
than basis orientation.

## Accepted Fixture Protocol

| Field | Value |
| --- | --- |
| Fixture key | `qr_economy_projector_5x3` |
| Matrix | Existing bounded 5x3 compatible QR matrix from `qr_overdetermined_compatible_5x3`. |
| Product path | `tests/test_qr.c` factors the matrix with `SPARSE_REORDER_NONE` and `economy = 1`, forms thin Q, and computes `Q Q^T`. |
| Reference path | `tests/qr_external_dense_reference.py` computes `A (A^T A)^{-1} A^T` using Python standard-library dense elimination. |
| Helper output | `OK 29`, followed by `q_rows`, `q_cols`, `r_rows`, `r_cols`, and 25 row-major projector entries. |
| Shape expectations | `q_rows = 5`, `q_cols = 3`, `r_rows = 3`, `r_cols = 3`. |
| Projector tolerance | `max |Q Q^T - P_ref| < 1e-8`. |
| Orthogonality tolerance | `max |Q^T Q - I| < 1e-10`. |
| Windows behavior | Preserve existing external QR helper skip behavior on Windows. |
| Build membership impact | None. The test is registered inside existing `test_qr`; no new executable, Makefile entry, CMake entry, or CTest member is added. |

## Implemented Changes

| Surface | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Added `qr_economy_projector_5x3` and a standard-library economy-projector reference path that emits shape and projector entries. |
| `tests/test_qr.c` | Added external-reference reader, matching sparse 5x3 fixture, economy factorization, thin-Q projector comparison, orthogonality check, and existing-suite registration. |
| `docs/maintainer_guide.md` | Updated QR evidence wording to include the bounded economy projector fixture while preserving broad Q-basis/economy non-claims. |

## Deferred Q-Basis/Economy Work

| Deferred Work | Future Owner | Promotion Gate |
| --- | --- | --- |
| Raw Q-column external comparison | Future QR basis owner | Define sign normalization, orientation, ordering, uniqueness, and failure diagnostics for a non-degenerate fixture. |
| Rank-deficient Q/nullspace subspace external evidence | Future QR subspace owner | Add projector or principal-angle helper semantics tied to a pinned rank threshold. |
| Wide economy external evidence | Future QR economy shape owner | Define wide-case `m x m` Q shape, R shape, projection metric, and fixture-specific skip/tolerance policy. |
| Sparse-mode Q/economy external evidence | Future QR sparse-mode owner | Compare product metrics only and preserve no performance/backend parity claim. |
| SuiteSparse Q/economy corpus evidence | Future corpus/platform owner | Define corpus availability, platform skip policy, time budget, and support-tier wording. |

## Failure Diagnostics

The accepted test identifies:

- fixture key;
- reference helper status;
- expected Q and R dimensions;
- product QR factorization status;
- projector max difference;
- orthogonality max difference;
- whether a failure is helper protocol, shape contract, QR product projector,
  thin-Q orthogonality, unsupported platform, or optional-helper availability.

## Validation Checklist

Day 7 touched `.c` and Python helper files, so the required code gate is:

1. `python3 tests/qr_external_dense_reference.py qr_economy_projector_5x3`
2. `make build/test_qr && ./build/test_qr`
3. `make format`
4. `make lint`
5. `make test`
6. `git diff --check`
7. Focused trailing-whitespace scan over Sprint 124 files and touched QR files

The helper must emit `OK 29`; values 1-4 must be `5`, `3`, `3`, `3`. A
different output count or shape is a fixture protocol failure.

## Non-Claim Register

Day 7 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR factorization parity;
- raw Q-basis equality, Q-sign, Q-orientation, or unique basis parity;
- rank-deficient, repeated, clustered, near-dependent, or nullspace subspace
  parity;
- broad economy-mode external oracle parity beyond the named 5x3 projector
  fixture;
- sparse-mode, reorder, backend, corpus, or performance parity;
- solve residual, minimum-norm, rank-deficient, nullspace, or pseudoinverse
  coverage beyond previously named fixtures;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- scalability, memory behavior, or state-of-the-art behavior.

## Validation Notes

Focused validation passed before full quality:

1. `python3 tests/qr_external_dense_reference.py qr_economy_projector_5x3`
   emitted `OK 29` with shape values `5`, `3`, `3`, `3`.
2. `make build/test_qr && ./build/test_qr` passed with 66 tests, 0 failures,
   0 skips, and 628 assertions.

Full required quality validation passed:

1. `make format`
2. `make lint`
3. `make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 3 is complete or explicitly deferred. | Complete | One bounded projector/economy external lane is implemented; broader Q-basis/economy work is explicitly deferred. |
| Sign and subspace semantics are represented in the decision. | Complete | The accepted lane uses projector comparison instead of raw Q columns and defers subspace/basis work. |
| Accepted work cannot create unsupported vector-orientation claims. | Complete | Fixture compares `Q Q^T` against a reference projector and preserves raw-basis non-claims. |
