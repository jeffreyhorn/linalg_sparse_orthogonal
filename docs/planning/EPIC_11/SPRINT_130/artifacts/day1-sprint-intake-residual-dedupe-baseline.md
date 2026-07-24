# Sprint 130 Day 1 - Sprint Intake And Residual Dedupe Baseline

## Purpose

Day 1 establishes Sprint 130 scope, duplicate fences, day-level owners,
validation expectations, and the completed-versus-deferred partial-SVD
residual evidence map.

Sprint 130 is a partial-SVD residual expansion and solver-selection
claim-gate sprint. It starts from Sprint 124's single accepted bounded
partial-SVD vector-residual lane and the Sprint 124 residual deferral package.
It does not start by adding tests; every deferred lane must first pass a
metric, oracle, tolerance, diagnostics, support-tier, skip-behavior, and
failure-interpretation gate.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 project-plan section | Defines seven items and the explicit goal to expand or defer partial-SVD residual evidence before refreshing solver-selection wording. |
| Sprint 130 plan | Defines the 14-day sequence and 166-hour budget. |
| Sprint 124 partial-SVD vector/subspace semantics | Defines sign-invariant residual policy, projection/subspace metrics, tolerance, skip, and failure interpretation. |
| Sprint 124 partial-SVD vector/subspace decision | Accepts `partial_svd_vector_residual_diag6_k2` and defers the remaining residual/subspace lanes. |
| Sprint 124 residual scenario matrix | Names repeated, clustered, rank-deficient, rectangular, corpus, low-rank, convergence, and nonsymmetric lanes with required diagnostics. |
| Sprint 124 residual deferral package | Carries future owners and promotion gates for the deferred partial-SVD lanes. |
| Sprint 129 closeout and handoff | Keeps QR Q/economy/helper boundaries closed while Sprint 130 starts partial-SVD work. |
| `tests/test_svd.c` | Main SVD and partial-SVD test registration owner. |
| `tests/test_svd_partial_helpers.h` | Primary partial-SVD helper and bounded evidence owner. |
| `tests/svd_external_dense_reference.py` | External dense-reference singular-value helper; any vector/subspace protocol expansion requires a later gate. |
| `src/sparse_svd_partial.c` | Partial-SVD implementation owner; not touched by Day 1. |
| `include/sparse_svd.h` | Public partial-SVD API wording owner; not touched by Day 1. |
| `docs/maintainer_guide.md` | Evidence table and solver-selection wording owner. |

## Project-Plan Owner Map

| Item | Sprint 130 owner days | Likely touched files | Required validation |
| --- | --- | --- | --- |
| 1. Partial-SVD Dedupe and Metric Map | Days 1-2 | Sprint 130 artifacts, working notes, maybe `docs/maintainer_guide.md` if claim wording boundaries are clarified | Documentation hygiene; claim-boundary scan if maintainer wording changes. |
| 2. Rectangular and Nonsymmetric Evidence | Days 3-6 | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py`, Sprint 130 artifacts, maybe `docs/maintainer_guide.md` | Focused helper invocation if Python changes; `make build/test_svd && ./build/test_svd`; full quality gate if `.c` or `.h` changes. |
| 3. Repeated and Clustered Spectrum Evidence | Days 7-8 | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, possible dense-reference helper expansion, Sprint 130 artifacts, maybe maintainer evidence | Projector/principal-angle helper checks if added; focused SVD validation; full quality gate if `.c` or `.h` changes. |
| 4. Rank-Deficient Subspace Evidence | Days 9-10 | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py`, Sprint 130 artifacts, maybe maintainer evidence | Focused rank/subspace validation; helper invocation if reference protocol changes; full quality gate if `.c` or `.h` changes. |
| 5. SuiteSparse and Low-Rank Optimality Evidence | Days 11-12 | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, SuiteSparse data references, `src/sparse_svd.c` only if low-rank behavior changes, Sprint 130 artifacts, maybe maintainer evidence | Focused SuiteSparse and low-rank diagnostics, optional-data skip proof, support-tier/runtime note, full quality gate if `.c` or `.h` changes. |
| 6. Convergence-Budget Evidence | Day 13 | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h`, `src/sparse_svd_partial.c` only if options or semantics change, Sprint 130 artifacts | Focused convergence-budget test, explicit partial-result/failure semantics, full quality gate if `.c` or `.h` changes. |
| 7. Solver-Selection Wording Gate | Day 14 | `docs/maintainer_guide.md`, README/tutorial/public docs only if evidence earns wording, Sprint 130 artifacts | Evidence-to-claim traceability, non-claim scan, docs hygiene; no code validation unless code changed. |

## Completed Evidence Fence

The following completed partial-SVD work is preserved as baseline evidence and
must not be repackaged as new Sprint 130 proof without a distinct metric or
claim boundary:

| Completed scope | Baseline to preserve |
| --- | --- |
| Bounded external square top-k values | `partial_svd_diag6_k2` proves only ordered leading singular values for one square diagonal fixture. |
| Bounded external tall top-k values | `partial_svd_tall_diag_8x5_k3` proves only ordered leading singular values for one tall diagonal fixture. |
| Bounded square vector residual | `partial_svd_vector_residual_diag6_k2` proves sign-invariant residuals and U/V orthogonality for one exact square diagonal fixture. |
| Internal vector availability and orthogonality | Existing vector tests prove local output presence and orthogonality only. |
| Internal rectangular behavior | Existing tall, wide, and rectangular reconstruction tests are implementation regression coverage, not external residual/subspace evidence. |
| Internal rank-deficient behavior | Existing rank-deficient partial-SVD tests exercise value behavior but do not define rank-deficient subspace or null-space parity. |
| SuiteSparse partial-SVD smoke | Existing `nos4` and `west0067` coverage is internal/corpus smoke, not external corpus residual parity. |
| Low-rank approximation tests | Existing low-rank tests are fixture-specific reconstruction checks, not global optimality evidence. |
| Timing smoke | Existing timing tests are not convergence-budget evidence. |

## Deferred Evidence Map

| Deferred lane | Required Day 2+ gate | Default Day 1 posture |
| --- | --- | --- |
| Rectangular vector residual | Shape-specific dimensions, residual metrics for both `A v_i` and `A^T u_i`, orthogonality, oracle, and tolerance policy. | Deferred to Days 3-4. |
| Nonsymmetric rectangular residual | Non-diagonal dense-reference or analytic fixture, value/residual boundary, left/right vector semantics, and no subspace overclaim. | Deferred to Days 5-6. |
| Repeated-spectrum subspace | Projector or principal-angle metric, basis-dimension checks, unordered or set-based value policy, and tie interpretation. | Deferred to Days 7-8. |
| Clustered-spectrum subspace | Spectral gap diagnostics, ordering/set policy, projector tolerance, convergence-budget interaction, and near-tie failure meaning. | Deferred to Days 7-8. |
| Rank-deficient subspace | Numerical-rank threshold, zero singular-value tolerance, range/null-space split, projector metrics, and rank/nullity diagnostics. | Deferred to Days 9-10. |
| SuiteSparse corpus residual | Optional-data skip behavior, support tier, runtime, conditioning notes, fixture-specific residual windows, and non-external-oracle wording. | Deferred to Days 11-12. |
| Low-rank optimality | Frobenius or spectral norm target, reconstruction metric, dense versus sparse-output semantics, and sparse drop-tolerance policy. | Deferred to Day 12. |
| Convergence budget | Options surface, iteration cap, tolerance, deterministic start or partial-result policy, and budget-failure classification. | Deferred to Day 13. |
| Solver-selection wording | Evidence-to-public-wording traceability and bounded non-claim language. | Deferred to Day 14; default is no update unless evidence earns it. |

## Duplicate Fence

Sprint 130 may promote a deferred partial-SVD lane only when all of the
following are true before implementation:

1. The candidate has a non-duplicate fixture key, helper name, or corpus slice.
2. The evidence class is explicit: value, vector residual, subspace,
   rank/nullity, corpus residual, low-rank optimality, convergence budget, or
   solver-selection wording.
3. The metric, expected shape, expected rank/nullity when relevant, tolerance,
   oracle, diagnostics, support tier, skip behavior, runtime expectation, and
   failure interpretation are pinned.
4. The artifact states why the candidate adds trust beyond
   `partial_svd_vector_residual_diag6_k2` and the existing internal SVD tests.
5. The validation plan includes focused owner tests and the full quality gate
   when `.c` or `.h` files change.
6. Public or maintainer wording changes have a direct evidence-to-claim trace
   and preserve non-claims for unimplemented lanes.

If any condition is missing, the candidate remains a deferred Sprint 130 item
with blocker, dependency, future owner, and promotion gate recorded.

## Validation Boundary

| Change class | Day 1 rule |
| --- | --- |
| Documentation-only Sprint 130 artifacts | `git diff --check` and trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_130`. |
| Partial-SVD helper or test edits | Focused `make build/test_svd && ./build/test_svd`, plus `make format && make lint && make test` if `.c` or `.h` files change. |
| Python external-reference helper changes | `python3 -m py_compile tests/svd_external_dense_reference.py`, focused helper invocation, affected executable, and diff hygiene. |
| SuiteSparse or optional-data evidence | Present/missing data behavior, skip-path proof, diagnostics, runtime/support-tier note, and required focused/full validation. |
| Low-rank behavior changes | Focused low-rank SVD checks and explicit optimality metric validation; full quality gate if code changes. |
| Convergence-budget behavior changes | Focused budget test with iteration/tolerance diagnostics and explicit partial-result failure semantics; full quality gate if code changes. |
| Maintainer/public solver-selection wording | Evidence-to-claim traceability, non-claim scan, path/link hygiene, and docs hygiene. |

## Non-Claims Preserved

Day 1 does not claim:

- broad partial-SVD external parity;
- singular-vector parity beyond the single bounded square diagonal
  vector-residual fixture;
- repeated-spectrum, clustered-spectrum, or rank-deficient subspace parity;
- SuiteSparse corpus residual parity or optional-data platform coverage;
- low-rank global optimality;
- convergence-budget guarantees or performance parity;
- raw vector orientation, sign, ordering, or unique-basis stability;
- solver-selection wording readiness beyond current workflow guidance;
- LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity;
- package, ABI, public API, install-header, CMake, Makefile, CI, CTest,
  scalability, memory, or state-of-the-art parity.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every Sprint 130 project-plan item has a day-level owner. | Complete | Owner map ties Items 1-7 to Days 1-14 and likely touched files. |
| Completed Sprint 124 evidence is not duplicated silently. | Complete | Completed evidence fence identifies `partial_svd_diag6_k2`, `partial_svd_tall_diag_8x5_k3`, `partial_svd_vector_residual_diag6_k2`, and internal-only coverage boundaries. |
| Deferred rectangular, spectral, subspace, corpus, optimality, and convergence lanes are visible before new evidence is accepted. | Complete | Deferred evidence map records required gates and default posture for each lane. |

## Day 2 Handoff

Day 2 should turn this baseline into a metric map. It should classify each
deferred lane by shape, spectrum, rank behavior, corpus status, optimality
claim, convergence claim, solver-selection impact, metric owner, tolerance,
oracle, diagnostics, and failure interpretation before any test or public
wording changes are accepted.
