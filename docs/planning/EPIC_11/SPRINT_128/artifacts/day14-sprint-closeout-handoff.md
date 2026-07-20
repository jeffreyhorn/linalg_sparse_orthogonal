# Sprint 128 Day 14 Closeout And Handoff

## Scope

Day 14 closes Sprint 128 by reconciling the project-plan items, day-level
artifacts, validation evidence, explicit deferrals, non-claims, and Sprint 129
handoff boundaries.

Sprint 128 accepted two bounded code evidence lanes:

- `qr_rankdef_wide_3x5_nullspace_subspace`
- `qr_rank_threshold_dependent_row_4x3_perturbed_family`

All other residual, corpus, optional-large, exact minimum-norm, QR-vs-SVD, and
helper-movement candidates were explicitly deferred because they lacked
distinct trust value, independent metadata, or behavior-specific ownership
needed before implementation.

## Project-Plan Reconciliation

| Item | Disposition | Evidence |
| --- | --- | --- |
| 1. Sprint 127 Residual Dedupe and Dependency Map | Complete | Day 1 mapped Sprint 127 carry-forward work against Sprint 121-127 QR residual, nullspace/subspace, threshold, SuiteSparse, optional-large, minimum-norm, and helper evidence. |
| 2. Compatible and Wide Residual Evidence Gate | Complete by explicit deferral | Days 2-3 preserved existing compatible and wide residual evidence and deferred new compatible zero-residual and wide residual-only fixtures until distinct trust value and underdetermined output semantics are pinned. |
| 3. Wide and Near-Threshold Nullspace/Subspace Gate | Complete with one accepted fixture plus deferrals | Days 4-5 accepted `qr_rankdef_wide_3x5_nullspace_subspace`; near-threshold and SuiteSparse subspace evidence remain deferred behind rank/nullity, projector metrics, support tier, and output-semantics gates. |
| 4. Remaining Threshold-Family Gate | Complete with one accepted fixture plus deferrals | Days 6-7 accepted `qr_rank_threshold_dependent_row_4x3_perturbed_family`; wide, default-threshold, SuiteSparse threshold, and additional near-threshold families remain deferred behind expected-rank and diagnostic metadata. |
| 5. SuiteSparse Rank-Deficient QR Corpus Gate | Complete by explicit deferral | Days 8-9 rejected SuiteSparse rank-deficient QR corpus promotion because checked-in corpus candidates lack independent expected-rank metadata, threshold semantics, support tier, skip behavior, and runtime gates. |
| 6. SuiteSparse and Optional-Large Minimum-Norm Gate | Complete by explicit deferral | Days 10-11 preserved the existing `west0067` first-30-row minimum-norm smoke and deferred additional SuiteSparse/optional-large lanes until extraction, shape, nnz, RHS, rank/nullity, residual, norm, support-tier, skip, runtime, and validation metadata are pinned. |
| 7. Exact Minimum-Norm and Helper Movement Gate | Complete by explicit deferral | Days 12-13 preserved Sprint 125-127 exact and QR-vs-SVD baselines and deferred duplicate exact lanes, additional QR-vs-SVD checks, SuiteSparse-derived exact lanes, and generic helper consolidation. |

## Evidence Index

| Day | Artifact | Outcome |
| --- | --- | --- |
| 1 | `day1-residual-dedupe-baseline.md` | Sprint 128 scope, duplicate fences, item owners, and validation requirements established. |
| 2 | `day2-compatible-wide-residual-semantics-policy.md` | Compatible zero-residual and wide residual-only proof-value policy pinned. |
| 3 | `day3-compatible-wide-residual-evidence-decision.md` | Compatible zero-residual and wide residual-only evidence explicitly deferred. |
| 4 | `day4-wide-near-threshold-subspace-policy.md` | Wide, near-threshold, and SuiteSparse subspace acceptance gates established. |
| 5 | `day5-wide-near-threshold-subspace-evidence.md` | `qr_rankdef_wide_3x5_nullspace_subspace` implemented and validated. |
| 6 | `day6-remaining-threshold-family-policy.md` | Remaining threshold-family acceptance gates established. |
| 7 | `day7-remaining-threshold-family-evidence.md` | `qr_rank_threshold_dependent_row_4x3_perturbed_family` implemented and validated. |
| 8 | `day8-suitesparse-rankdef-qr-corpus-gate.md` | SuiteSparse rank-deficient QR corpus promotion gate established. |
| 9 | `day9-suitesparse-rankdef-qr-evidence-decision.md` | SuiteSparse rank-deficient QR corpus evidence explicitly deferred. |
| 10 | `day10-suitesparse-optional-large-minnorm-policy.md` | SuiteSparse and optional-large minimum-norm metadata gate established. |
| 11 | `day11-suitesparse-optional-large-minnorm-evidence-decision.md` | Additional SuiteSparse and optional-large minimum-norm evidence explicitly deferred. |
| 12 | `day12-exact-minnorm-crosscheck-helper-gate.md` | Exact minimum-norm, QR-vs-SVD, and helper movement gate established. |
| 13 | `day13-crosscheck-helper-integrated-validation.md` | Additional exact/cross-check/helper work explicitly deferred and focused validation rerun. |
| 14 | `day14-sprint-closeout-handoff.md` | Final evidence, deferral, validation, non-claim, and Sprint 129 handoff index published. |

## Validation Package

Sprint 128 code changes were validated when introduced:

| Scope | Validation |
| --- | --- |
| Day 5 wide nullspace subspace fixture | `python3 -m py_compile tests/qr_external_dense_reference.py`; helper invocation for `qr_rankdef_wide_3x5_nullspace_subspace`; `make build/test_qr && ./build/test_qr`; full `make format && make lint && make test`. |
| Day 7 threshold-family fixture | `python3 -m py_compile tests/qr_external_dense_reference.py`; helper invocation for `qr_rank_threshold_dependent_row_4x3_perturbed_family`; `make build/test_qr && ./build/test_qr`; full `make format && make lint && make test`. |
| Day 11 minimum-norm owner preservation | `make build/test_colamd && ./build/test_colamd` passed with the accepted `west0067` smoke diagnostic preserved. |
| Day 13 integrated focused validation | QR helper compile and both Sprint 128 helper keys passed; `make build/test_qr && ./build/test_qr` passed 74 tests; `make build/test_qr_solve && ./build/test_qr_solve` passed 19 tests; `make build/test_colamd && ./build/test_colamd` passed 70 tests. |

Day 14 changes documentation only. Required Day 14 hygiene:

```text
git diff --check
rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_128
```

## Deferred Debt And Future Owners

| Deferred item | Blocker | Future owner |
| --- | --- | --- |
| Compatible zero-residual rank-deficient QR residual fixture | No non-duplicate proof value beyond existing residual/rank coverage. | Future QR residual semantics sprint. |
| Wide residual-only QR fixture | Underdetermined output semantics and non-minimum-norm wording need a distinct fixture-specific claim. | Future QR residual semantics sprint. |
| Near-threshold nullspace/subspace fixture | Rank/nullity, projector metric, and failure interpretation are not pinned for a non-duplicate fixture. | Future QR nullspace/subspace sprint. |
| SuiteSparse QR nullspace/subspace fixture | Independent rank/nullity metadata and support-tier gates are not pinned. | Future SuiteSparse QR corpus sprint. |
| Wide/default/SuiteSparse threshold families | Expected ranks, threshold semantics, support tier, skip behavior, and diagnostics are not pinned. | Future QR threshold-family sprint. |
| SuiteSparse rank-deficient QR corpus evidence | Checked-in candidates lack independent expected-rank metadata and threshold interpretation. | Future SuiteSparse corpus metadata sprint. |
| Additional SuiteSparse or optional-large minimum-norm evidence | Extraction, shape, nnz, RHS, rank/nullity, residual/norm metric, support tier, skip, runtime, and validation metadata are incomplete. | Future minimum-norm corpus sprint. |
| Additional exact underdetermined fixtures | Non-duplicate fixture keys, closed-form expected vectors, exact norms, and owner-local diagnostics are not pinned. | Future minimum-norm exact-values sprint. |
| Additional QR-vs-SVD cross-checks | Existing `2 x 4`, `3 x 6`, and `5 x 10` evidence already covers bounded claims; new checks risk oracle overclaim without a narrower role. | Future QR-vs-SVD claim-gate sprint. |
| Generic helper consolidation | Behavior-specific ownership, visible call-site tolerances, and helper naming are not pinned. | Sprint 129 helper ownership work. |

## Non-Claim Register

Sprint 128 does not claim:

- compatible zero-residual or wide residual-only behavior beyond existing
  bounded fixtures
- wide nullspace unique-basis behavior
- near-threshold or SuiteSparse nullspace/subspace parity
- broad threshold robustness across all matrix families
- independent SuiteSparse rank-deficient QR corpus coverage
- optional-large corpus coverage
- broad QR minimum-norm optimality
- SVD-pseudoinverse as a global QR oracle
- generic helper API ownership
- Q-basis, economy, sparse-mode, reorder, refinement, package, ABI, public API,
  CI, CMake, CTest, performance, memory, platform, backend, ecosystem,
  LAPACK, NumPy, SciPy, BLAS, PETSc, Trilinos, Eigen, or state-of-the-art
  parity

## Sprint 129 Handoff

Sprint 129 may begin from these settled boundaries:

1. Sprint 128 QR residual, subspace, threshold, SuiteSparse, minimum-norm,
   QR-vs-SVD, and helper claim gates are closed unless a future sprint supplies
   the missing metadata listed above.
2. Sprint 129 Q-basis and economy work should not reopen Sprint 128 residual or
   minimum-norm decisions unless a Q/economy fixture has a distinct
   behavior-specific claim.
3. Helper ownership work should use behavior-specific helper names, keep
   fixture keys and tolerances visible at owner call sites, and avoid generic
   public helper movement without an explicit review gate.
4. Any code changes in Sprint 129 that touch `.c`, `.h`, or Python helper files
   should run focused executable checks plus `make format && make lint &&
   make test`.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| All Sprint 128 project-plan deliverables are present or explicitly deferred. | Complete | Project-plan reconciliation table maps every item to evidence or deferral. |
| No unresolved item lacks a future owner and dependency statement. | Complete | Deferred debt table names blockers and future owners. |
| Sprint 129 can begin without re-litigating Sprint 128 claim boundaries. | Complete | Handoff and non-claim register define what Sprint 129 may build on. |

