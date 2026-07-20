# Sprint 128 Day 13 Cross-Check, Helper, And Integrated Validation

## Scope

Day 13 closes Sprint 128 Item 7 by applying the Day 12 exact
minimum-norm, QR-vs-SVD cross-check, and helper movement gate.

No new exact minimum-norm fixture, QR-vs-SVD cross-check, or helper movement
is accepted for implementation on Day 13. The available candidates either
duplicate Sprint 125-127 evidence or lack the non-duplicate fixture key,
closed-form expected values, tolerance policy, diagnostics, and owner-local
placement required before code edits.

Day 13 therefore records explicit deferral, preserves the completed evidence,
and validates the Sprint 128 touched QR lanes.

## Gate Inputs

| Source | Role |
| --- | --- |
| Sprint 128 Day 12 artifact | Defines the acceptance gates for exact lanes, QR-vs-SVD cross-checks, and helper movement. |
| Sprint 125 `test_minnorm_vs_pinv` | Existing bounded QR-vs-SVD minimum-norm cross-check. |
| Sprint 126 `qr_minnorm_5x10_exact_values` | Existing exact minimum-norm fixture with closed-form values and norm. |
| Sprint 127 `qr_minnorm_3x6_exact_values` | Existing exact minimum-norm fixture with closed-form values and norm. |
| Sprint 128 Day 5 wide nullspace evidence | Sprint 128 accepted code evidence requiring focused QR validation. |
| Sprint 128 Day 7 threshold-family evidence | Sprint 128 accepted code evidence requiring focused QR validation. |

## Candidate Disposition

| Candidate | Day 13 decision | Rationale |
| --- | --- | --- |
| New `3 x 6` QR-vs-SVD cross-check | Deferred | The `3 x 6` fixture already has exact values from Sprint 127; adding an SVD comparison would duplicate coverage and risk overstating pseudoinverse oracle scope. |
| New `5 x 10` QR-vs-SVD cross-check | Deferred | The `5 x 10` fixture already has exact values from Sprint 126; no additional non-oracle metadata improves the current claim. |
| New `2 x 4` exact or QR-vs-SVD lane | Deferred | Sprint 125 already owns the bounded `2 x 4` cross-check and QR solve exact evidence. |
| Larger synthetic exact underdetermined fixture | Deferred | No non-duplicate fixture key, closed-form expected vector, exact norm, tolerance policy, and owner-local diagnostics are pinned. |
| SuiteSparse-derived exact or QR-vs-SVD lane | Deferred | Days 10-11 kept SuiteSparse minimum-norm evidence limited to the existing `west0067` smoke because independent rank/nullity and oracle metadata are not pinned. |
| Generic minimum-norm helper consolidation | Deferred | The Day 12 helper gate requires behavior-specific ownership, visible call-site tolerances, and no generic public helper movement. |

## Implementation Decision

Day 13 makes no C, header, Python helper, Matrix Market, build, maintainer,
or public API edits.

This is intentional. The completed exact and cross-check evidence already
covers the bounded claims accepted in prior sprints, while every new candidate
available on Day 13 would either duplicate existing evidence or weaken the
owner boundary by promoting SVD-pseudoinverse or generic helper wording beyond
the tested behavior.

## Validation Results

Because Day 13 itself is documentation-only, the required Day 13 quality gate
is markdown and diff hygiene. Integrated validation also re-ran focused checks
for Sprint 128 code touched earlier in the sprint:

| Command | Result |
| --- | --- |
| `python3 -m py_compile tests/qr_external_dense_reference.py` | Passed. |
| `python3 tests/qr_external_dense_reference.py qr_rankdef_wide_3x5_nullspace_subspace` | Passed; helper emitted `OK 29`. |
| `python3 tests/qr_external_dense_reference.py qr_rank_threshold_dependent_row_4x3_perturbed_family` | Passed; helper emitted `OK 9`. |
| `make build/test_qr && ./build/test_qr` | Passed: 74 tests, 0 failures, 0 skips, 885 assertions. |
| `make build/test_qr_solve && ./build/test_qr_solve` | Passed: 19 tests, 0 failures, 0 skips, 1104 assertions. |
| `make build/test_colamd && ./build/test_colamd` | Passed: 70 tests, 0 failures, 0 skips, 317 assertions. |

Full `make format && make lint && make test` was already required and passed
for the Sprint 128 Day 5 and Day 7 code changes. Day 13 does not add new code,
so the full quality gate is not required again by the sprint rules.

Final Day 13 hygiene checks:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_128
```

## Non-Claims Preserved

- No new exact minimum-norm fixture is accepted on Day 13.
- No new QR-vs-SVD cross-check is accepted on Day 13.
- No SVD-pseudoinverse global oracle claim.
- No generic minimum-norm, pseudoinverse, cross-solver, or corpus helper.
- No SuiteSparse, optional-large, performance, platform, rank-deficient corpus,
  Q-basis, economy, public API, CMake, CTest, CI, backend, package, ABI,
  ecosystem, LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, or
  state-of-the-art parity claim.

## Future Promotion Gate

A future sprint may promote a new exact or cross-check lane only when it
provides all of the following before code edits:

1. A non-duplicate fixture key and owner-local test name.
2. Closed-form expected values and exact norm, or a bounded QR-vs-SVD role
   that is not described as an oracle.
3. Residual, value, norm, and rank/nullity tolerances recorded before
   implementation.
4. Diagnostics that print the fixture key, max residual, solution norm, and
   expected norm when exact values are asserted.
5. Focused validation for touched helpers/tests and full quality validation
   when `.c`, `.h`, or helper script files change.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Project-plan Item 7 is complete or explicitly deferred. | Complete | Day 13 explicitly defers every candidate that fails the Day 12 gate. |
| Touched code or scripts have appropriate validation evidence. | Complete | QR helper, QR, QR solve, and COLAMD focused checks passed. |
| Helper movement does not blur ownership. | Complete | No helpers are moved; future movement remains behavior-specific and owner-local. |
