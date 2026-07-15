# Sprint 124 Day 14 Closeout and Handoff

## Purpose

Close Sprint 124 from the Day 13 validation baseline, consolidate accepted
oracle evidence, preserve explicit deferrals and non-claims, and hand a stable
oracle-truth package to Sprint 125 corpus and report-index work.

## Sprint 124 Closeout Summary

Sprint 124 completed all seven project-plan items either through bounded
implementation plus validation or explicit deferral with named future owners
and promotion gates.

| Project-plan item | Final state | Evidence |
| --- | --- | --- |
| 1. Rank-Deficient QR Oracle Design | Complete with one bounded implementation and explicit deferrals | Day 2 policy plus Day 3 `qr_rankdef_duplicate_5x4_rank_only` rank-only fixture. |
| 2. QR Minimum-Norm Oracle Design | Complete with one bounded implementation and explicit deferrals | Day 4 behavior contract plus Day 5 `qr_underdetermined_minnorm_2x4` exact minimum-norm fixture. |
| 3. QR Q-Basis and Economy Oracle Design | Complete with one bounded implementation and explicit deferrals | Day 6 semantics plus Day 7 `qr_economy_projector_5x3` economy-projector fixture. |
| 4. Partial-SVD Vector and Subspace Semantics | Complete with one bounded implementation and explicit deferrals | Day 8 semantics plus Day 9 `partial_svd_vector_residual_diag6_k2` vector-residual fixture. |
| 5. Partial-SVD Residual Semantics Batch | Complete by explicit deferral package | Day 10 scenario matrix plus Day 11 residual deferral package. |
| 6. Helper Ownership Follow-Through | Complete by explicit deferral package | Day 12 minimum-norm and Bidiagonal/Golub-Kahan helper ownership decision. |
| 7. Validation, Docs, and Claim Gate | Complete | Day 13 validation and claim-gate artifact. |

## Accepted Implementation Package

| Accepted lane | Fixture or behavior | Owner surfaces | Claim boundary |
| --- | --- | --- | --- |
| QR rank-only rank-deficient evidence | `qr_rankdef_duplicate_5x4_rank_only` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One exact 5x4 duplicate-column rank check at threshold `0.0`; no solve, nullspace, minimum-norm, basis, economy, SuiteSparse, or broad QR parity claim. |
| QR exact minimum-norm evidence | `qr_underdetermined_minnorm_2x4` | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One exact 2x4 underdetermined solution/residual/norm check; no broad minimum-norm, COLAMD, fallback, refinement, rank-deficient, SuiteSparse, or SVD-pseudoinverse parity claim. |
| QR economy projector evidence | `qr_economy_projector_5x3` | `tests/test_qr.c`, `tests/qr_external_dense_reference.py`, `docs/maintainer_guide.md` | One full-column-rank 5x3 economy-Q projector and orthogonality check; no raw Q-column, sign/orientation, rank-deficient subspace, sparse-mode, reorder, corpus, or performance claim. |
| Partial-SVD vector-residual evidence | `partial_svd_vector_residual_diag6_k2` | `tests/test_svd_partial_helpers.h`, `tests/test_svd.c`, `tests/svd_external_dense_reference.py`, `docs/maintainer_guide.md` | One exact square diagonal `k=2` vector-residual and orthogonality check anchored by existing singular-value helper output; no raw vector equality, broad vector/subspace, repeated/clustered, rank-deficient, convergence, corpus, or low-rank optimality claim. |

## Validation Baseline

Day 13 is the authoritative Sprint 124 validation baseline.

Focused helper validation passed:

1. `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_rank_only`
2. `python3 tests/qr_external_dense_reference.py qr_underdetermined_minnorm_2x4`
3. `python3 tests/qr_external_dense_reference.py qr_economy_projector_5x3`
4. `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2`

Focused executable validation passed:

| Command | Result |
| --- | --- |
| `make build/test_qr_solve && ./build/test_qr_solve` | 17 tests, 0 failures, 0 skips, 1069 assertions |
| `make build/test_qr && ./build/test_qr` | 66 tests, 0 failures, 0 skips, 628 assertions |
| `make build/test_svd && ./build/test_svd` | 109 tests, 0 failures, 0 skips, 1803 assertions |

Full required quality validation passed:

1. `make format`
2. `make lint`
3. `make test`

The final test phase ended with `All tests passed.`

Day 14 changes planning documentation only. No source, header, helper script,
build metadata, package metadata, public API, or public solver-selection
wording changed on Day 14.

## Consolidated Residual and Future-Owner Queue

| Residual | Future owner | Promotion gate |
| --- | --- | --- |
| Rank-deficient QR residual-only evidence | Future QR solve oracle owner | Prove residual evidence adds trust beyond deterministic tests without implying nullspace, minimum-norm, or broad QR parity. |
| Rank-deficient QR nullspace evidence | Future QR basis/subspace owner | Define sign, ordering, nullity, projection/subspace metric, null residual semantics, and fixture-specific tolerance. |
| Near-rank-deficient QR threshold evidence | Future numerical-rank owner | Define threshold family, expected ranks, stability policy, and non-global interpretation. |
| SuiteSparse rank-deficient QR evidence | Future corpus/platform owner | Define optional corpus availability, platform skip behavior, support tier, diagnostics, and claim boundary. |
| COLAMD/reordered QR minimum-norm evidence | Future QR minimum-norm/COLAMD owner | Define ordering options, expected residual/norm behavior, skip behavior, and non-superiority wording. |
| QR fallback minimum-norm evidence | Future QR fallback owner | Define whether the fixture proves ordinary solve fallback or underdetermined minimum-norm behavior. |
| Rank-deficient QR minimum-norm evidence | Future rank-deficient/minimum-norm owner | Combine rank policy with solution norm, residual, nullspace boundaries, and fixture-local tolerances. |
| QR refinement minimum-norm evidence | Future QR refinement owner | Define before/after residual expectations, iteration-budget semantics, and failure interpretation. |
| QR-vs-SVD-pseudoinverse evidence | Future QR/SVD cross-solver owner | Define whether SVD pseudoinverse is an oracle, cross-check, or independent behavior owner. |
| SuiteSparse QR minimum-norm evidence | Future corpus/platform owner | Define corpus availability, skip behavior, support tier, and platform implications. |
| Raw QR Q-column comparison | Future QR basis owner | Define sign normalization, orientation, ordering, uniqueness, and diagnostics for a non-degenerate fixture. |
| Rank-deficient Q/nullspace subspace evidence | Future QR subspace owner | Add projector or principal-angle helper semantics tied to a pinned rank threshold. |
| Wide QR economy evidence | Future QR economy owner | Define wide-case Q/R shapes, projection metric, skip policy, and fixture-local tolerances. |
| QR sparse-mode Q/economy evidence | Future QR sparse-mode owner | Compare product metrics only while preserving no performance/backend parity claim. |
| SuiteSparse QR Q/economy evidence | Future corpus/platform owner | Define corpus availability, platform skip policy, time budget, and support-tier wording. |
| Partial-SVD rectangular vector residual | Future partial-SVD vector owner | Pick one bounded shape lane first, define matrix, `k`, dimensions, tolerances, and claim wording before editing tests. |
| Partial-SVD repeated-spectrum subspace | Future subspace owner | Add projector or principal-angle protocol for left and right subspaces; forbid raw vector equality. |
| Partial-SVD clustered-spectrum subspace/convergence | Future convergence/subspace owner | Define spectral gap, ordered versus set-based value policy, projector tolerance, iteration budget, and failure meaning. |
| Partial-SVD rank-deficient subspace | Future rank/subspace owner | Define rank threshold, zero singular-value tolerance, and range/null-space projector ownership. |
| Partial-SVD SuiteSparse vector residual | Future corpus owner | Define file availability rules, conditioning notes, per-fixture residual windows, and non-external-oracle wording. |
| Partial-SVD low-rank optimality | Future low-rank owner | Choose Frobenius or spectral norm evidence, dense versus sparse-output semantics, and sparse drop-tolerance handling. |
| Partial-SVD convergence budget | Future convergence owner | Add deterministic options, iteration cap, tolerance, and budget-failure classification. |
| Partial-SVD nonsymmetric rectangular value residual | Future external value owner | Add a non-diagonal dense-reference fixture without extending vector/subspace claims. |
| Minimum-norm helper movement | Future helper owner | Use behavior-specific helper names and run focused QR solve, COLAMD, SVD, and full quality validation. |
| Bidiagonal/Golub-Kahan helper extraction | Future Bidiagonal/GK helper owner | Use a dedicated owner preserving transpose, reconstruction, explicit `U`/`V`, wide skip, and QR-iteration semantics; run focused bidiag/SVD and full quality validation. |

## Final Non-Claim Register

Sprint 124 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, ecosystem, dense-library, or broad external package parity;
- broad QR factorization, QR solve, rank-deficient solve, nullspace,
  minimum-norm, Q-basis, economy, sparse-mode, reorder, backend, corpus, or
  performance parity;
- raw QR Q-basis equality, Q sign/orientation, unique basis parity, or broad
  projection/subspace parity;
- broad SVD or partial-SVD parity, raw singular-vector parity, repeated or
  clustered spectrum parity, rank-deficient subspace parity, low-rank global
  optimality, convergence-budget guarantees, corpus parity, or performance
  parity;
- QR/SVD pseudoinverse oracle parity beyond explicitly named bounded checks;
- helper API expansion, generic helper consolidation, CMake/CTest membership
  expansion, package behavior, ABI behavior, public API behavior, platform
  support, scalability, memory behavior, or state-of-the-art behavior.

## Sprint 125 Corpus and Report-Index Handoff

Sprint 125 can treat Sprint 124's oracle-truth input package as stable:

| Sprint 125 need | Sprint 124 input |
| --- | --- |
| Corpus taxonomy inputs | Accepted and deferred fixture classes are listed in Day 3, Day 5, Day 7, Day 9, Day 10, Day 11, and this closeout. |
| External-reference script inventory | `tests/qr_external_dense_reference.py` now owns the Sprint 124 QR rank, minimum-norm, and projector protocols; `tests/svd_external_dense_reference.py` remains singular-value only for the Day 9 vector-residual lane. |
| Expected-failure and skip interpretation | Each accepted lane preserves existing missing-`python3` and Windows external-helper skip behavior; helper `ERROR` remains a protocol failure. |
| Report-index evidence fields | Use the four accepted lanes, their owners, Day 13 validation commands, and the maintainer-guide trust-boundary rows. |
| Claim-boundary inputs | Use the final non-claim register above and the Day 13 solver-selection no-update rationale. |
| Future-owner queue | Use the consolidated residual table above; do not promote any deferred item without its named promotion gate. |

## Day 14 Validation

Day 14 is documentation-only. Required validation:

1. `git diff --check`
2. Focused trailing-whitespace scan over Sprint 124 planning files and touched
   maintainer/test/helper surfaces

No full C quality gate is required for Day 14 because Day 14 does not change
`.c`, `.h`, helper script, build, package, or public API files. The full code
quality baseline remains the Day 13 `make format && make lint && make test`
pass.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| All seven Sprint 124 project-plan items are complete or explicitly deferred. | Complete | See closeout summary table. |
| Every deferred item has owner, dependency, and promotion gate. | Complete | See consolidated residual and future-owner queue. |
| Sprint 125 has a stable oracle-truth input package. | Complete | See Sprint 125 corpus and report-index handoff. |
