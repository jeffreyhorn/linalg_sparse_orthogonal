# Sprint 128 Day 7 Remaining Threshold-Family Evidence

## Purpose

Day 7 implements the bounded threshold-family evidence selected by the Day 6
policy and explicitly defers the remaining candidates that still need stricter
metadata, support-tier, or proof-boundary work.

## Inputs

| Input | Use |
| --- | --- |
| Day 6 remaining threshold-family policy | Provides candidate order, acceptance gate, expected-rank rules, diagnostics, and non-claims. |
| `tests/qr_external_dense_reference.py` | Owns standard-library reference values for bounded QR threshold fixtures. |
| `tests/test_qr.c` | Owns product QR rank and rank-info evidence. |
| `docs/maintainer_guide.md` | Records fixture ownership and claim boundaries. |

## Implemented Fixture

| Field | Value |
| --- | --- |
| Fixture key | `qr_rank_threshold_dependent_row_4x3_perturbed_family` |
| Matrix family | Existing dependent-row `4 x 3` QR fixture with the formerly dependent entry `(2, 2)` changed from `3.0` to `3.0 + perturbation`. |
| Perturbation | `1e-6` |
| Thresholds | `1e-10`, `1e-8`, `1e-6` |
| Expected ranks | `3`, `3`, `2` |
| Primary claim | Product QR rank and rank-info diagnostics agree with fixture-local expected ranks under explicit relative thresholds. |
| Support tier | Checked-in deterministic unit fixture. |
| Skip behavior | Windows keeps the existing external-helper skip path. |

## Diagnostics Captured

The focused QR run reported:

| Relative threshold | Absolute threshold | Expected rank | Product rank | Rank-info rank | Pivot ratio | R diagonal magnitudes |
| --- | --- | --- | --- | --- | --- | --- |
| `1e-10` | `3.742e-10` | `3` | `3` | `3` | `7.939e-08` | `[3.742e+00, 2.204e+00, 2.970e-07]` |
| `1e-8` | `3.742e-08` | `3` | `3` | `3` | `7.939e-08` | `[3.742e+00, 2.204e+00, 2.970e-07]` |
| `1e-6` | `3.742e-06` | `2` | `2` | `2` | `7.939e-08` | `[3.742e+00, 2.204e+00, 2.970e-07]` |

The fixture therefore exercises the strict threshold rule without relying on
residual, nullspace, subspace, solve, or minimum-norm metrics.

## Files Updated

| File | Change |
| --- | --- |
| `tests/qr_external_dense_reference.py` | Added helper values and command routing for `qr_rank_threshold_dependent_row_4x3_perturbed_family`. |
| `tests/test_qr.c` | Added fixture-key allow-list entry, product QR threshold test, rank-info diagnostic checks, and test registration. |
| `docs/maintainer_guide.md` | Updated the QR ownership row to include both perturbed threshold fixtures. |
| `docs/planning/EPIC_11/SPRINT_128/WORKING_NOTES.md` | Recorded Day 7 implementation, validation, deferrals, and non-claims. |

## Explicit Deferrals

| Candidate | Day 7 Decision | Reason |
| --- | --- | --- |
| Wide threshold family | Deferred | Needs rank/nullity semantics and underdetermined non-claims before threshold evidence can be separated from solution-selection and subspace behavior. |
| Default-threshold diagnostic | Deferred | Needs product-local `tol <= 0` wording and stability proof so it is not promoted into a global rank policy. |
| SuiteSparse threshold candidate | Deferred | Belongs to Days 8-9 corpus gates with support-tier, missing-data, runtime, and expected-rank metadata. |
| Near-threshold nullspace/subspace family | Deferred | Requires completed threshold expected-rank metadata plus accepted projection metrics before promotion. |

## Non-Claims

This Day 7 evidence does not claim:

- a global QR rank-threshold, default-threshold, or numerical-rank policy;
- residual, compatible-solve, nullspace, subspace, or minimum-norm behavior;
- wide, sparse-mode, economy-mode, reorder, SuiteSparse, optional-data,
  platform, backend, performance, or external-library parity;
- that the perturbed dependent-row fixture replaces the completed diagonal,
  scaled diagonal, or perturbed duplicate-column threshold fixtures.

## Validation

Focused validation:

- `python3 -m py_compile tests/qr_external_dense_reference.py`
- `python3 tests/qr_external_dense_reference.py qr_rank_threshold_dependent_row_4x3_perturbed_family`
- `make build/test_qr && ./build/test_qr`

Required full validation is needed before Day 7 closeout because `.c` and
Python helper files changed:

- `make format && make lint && make test`

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Added threshold batch or explicit deferral is justified against Day 6 gates. | Complete | Implemented bounded dependent-row threshold fixture and deferred remaining lanes. |
| Expected-rank metadata and diagnostics are recorded. | Complete | See implemented fixture and diagnostics tables. |
| Threshold non-claims are updated. | Complete | See maintainer guide and non-claims section. |
| Focused QR checks pass. | Complete | `make build/test_qr && ./build/test_qr` passed 74 QR tests. |
