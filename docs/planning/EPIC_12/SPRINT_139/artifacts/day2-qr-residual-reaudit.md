# Sprint 139 Day 2: QR Residual Reaudit

## Purpose

Day 2 re-ranks QR residual candidates before implementation starts. The goal is
to choose one residual that can be fully closed inside Sprint 139 with the
Sprint 138 corpus lane, focused QR proof ownership, oracle evidence, and
bounded documentation wording.

This is a planning and evidence-selection artifact. It does not change QR
source, tests, corpus rows, oracle commands, public documentation, or support
claims.

## Reaudit Inputs

| Input | Reaudit use |
| --- | --- |
| `docs/planning/EPIC_12/PROJECT_PLAN.md` Sprint 139 | Defines required QR residual families to consider: rank-deficient, rectangular, least-squares, minimum-norm, nullspace, COLAMD, and SuiteSparse/corpus behavior. |
| `docs/planning/EPIC_12/SPRINT_137/RETROSPECTIVE.md` | Selects QR rank-deficient nullspace/subspace closure as the Sprint 139 target. |
| `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md` | Freezes broad QR wording until the selected residual closes. |
| `docs/planning/EPIC_12/SPRINT_138/RETROSPECTIVE.md` | Confirms Sprint 138 created the first QR corpus lane but did not close solver-backed QR behavior. |
| `docs/planning/EPIC_12/SPRINT_138/artifacts/day8-deterministic-fixture-lane-design.md` | Defines `qr_rank_deficient_6x4_nullspace_v1`, expected rank/nullity, null vector, and tolerance. |
| `tests/corpus/README.md` | Requires Sprint 139 to use the first QR lane and avoid raw-basis equality. |
| `tests/test_qr.c` | Current owner for QR factorization, rank, nullspace, projector, and basis-safe assertions. |
| `tests/test_qr_solve.c` | Current owner for QR solve, least-squares residual, rank-deficient residual, and minimum-norm evidence. |
| `tests/qr_external_dense_reference.py` | Existing bounded dense-reference helper for QR residual, rank, projector, and minimum-norm checks. |
| `docs/maintainer_guide.md` | Current maintainer-facing QR evidence and non-claim table. |

## Candidate Ranking Rubric

Scores use `1` for low/weak and `5` for high/strong.

| Score | User-facing risk | Feasibility | Validation cost | Complete-closure fit |
| --- | --- | --- | --- | --- |
| 5 | Important API behavior that affects solver selection or trust. | Existing fixture, oracle, and test surfaces make closure direct. | Focused local validation is enough unless code changes. | Can close one precise claim inside Sprint 139. |
| 3 | Useful behavior but already partly bounded or less visible. | Requires moderate fixture/test/doc work. | Needs several focused lanes or build updates. | Can close a subclaim but may leave adjacent ambiguity. |
| 1 | Mostly broad positioning or future expansion. | Depends on external data, hosted review, or unrelated sprint work. | High or unavailable locally. | Cannot be closed fully this sprint without overclaiming. |

## QR Residual Candidate Ranking

| Rank | Candidate residual | Current evidence | Gap | Risk | Feasibility | Validation cost | Closure fit | Decision |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Corpus-backed rank-deficient nullspace residual on `qr_rank_deficient_6x4_nullspace_v1` | Sprint 138 fixture, generator, expected rank/nullity/residual rows, and QR nullspace test patterns exist. | No solver-backed proof owns this exact maintained corpus fixture yet. | 5 | 5 | 4 | 5 | Select as priority closure. |
| 2 | Existing duplicate-column rank-deficient nullspace projector lane | `tests/test_qr.c` has bounded projector checks such as `qr_rankdef_duplicate_5x4_nullspace_projector`. | It is not the Sprint 138 maintained corpus fixture and does not close the new corpus-backed residual. | 4 | 4 | 4 | 3 | Keep as backup/pattern. |
| 3 | Rank-threshold policy for near-dependent QR fixtures | Existing threshold fixtures and dense-reference helper cover bounded cases. | Global rank-threshold policy remains explicitly non-claimed. | 4 | 3 | 3 | 2 | Defer; too policy-wide for this sprint's selected corpus lane. |
| 4 | Rectangular least-squares residual behavior | `tests/test_qr_solve.c` has overdetermined compatible/incompatible fixtures and dense-reference comparisons. | Existing evidence is solve-focused and not tied to the Sprint 138 nullspace fixture. | 4 | 3 | 3 | 2 | Defer unless selected lane exposes solve-side defect. |
| 5 | Minimum-norm underdetermined behavior | `tests/test_qr_solve.c` has bounded minimum-norm fixtures and cross-checks. | Broad minimum-norm behavior remains a non-claim and would require solve-specific closure. | 4 | 3 | 3 | 2 | Defer; not the selected nullspace residual. |
| 6 | COLAMD/reordered QR behavior | QR option and COLAMD evidence exists through tests and docs. | Reorder behavior adds permutation/fill concerns outside the first corpus lane. | 3 | 2 | 2 | 1 | Defer; not needed for fixture-local nullspace closure. |
| 7 | SuiteSparse rank-deficient QR subset | Optional-data row `suitesparse_rank_deficient_qr_subset_v1` exists but is disabled by default. | Requires external data availability, licensing/review, hosted evidence, and support-tier policy. | 5 | 1 | 1 | 1 | Out of scope for Sprint 139 local closure. |
| 8 | Broad QR external-library parity | Maintainer guide lists bounded external helper evidence and explicit non-claims. | Would require broad LAPACK/NumPy/SciPy/SuiteSparse parity design. | 5 | 1 | 1 | 1 | Non-goal; violates fixture-local closure boundary. |

## Selected Priority Residual

Sprint 139 selects:

`qr_rank_deficient_6x4_nullspace_v1` solver-backed rank/nullity/nullspace
residual closure.

Selection rationale:

- It is the only QR residual with a maintained Sprint 138 corpus fixture,
  generator row, expected-result rows, oracle/report command, and explicit
  handoff.
- It directly closes the Sprint 137 selected target:
  QR rank-deficient nullspace/subspace residual closure backed by the Sprint
  138 corpus lane.
- It can be closed without raw QR basis parity by comparing rank, nullity, and
  normalized nullspace residual.
- It keeps SuiteSparse, optional external data, broad QR correctness,
  minimum-norm, least-squares, reorder, platform, performance, and
  state-of-the-art claims fenced.

Required Day 3 design focus:

- decide whether this closure uses a focused proof extracted from
  `tests/test_qr.c` or a dedicated QR corpus proof test
- define a C-side fixture builder that exactly matches the maintained generator
  entries or define a reusable bridge from corpus metadata
- define solver-backed observed rows for rank, nullity, and normalized residual
- define whether the residual row compares the solver-produced unit null vector
  directly via `||A*v|| / ||v||` or via a two-way subspace/projector metric
- preserve support tier as `local_only` unless later reviewed evidence promotes
  it

## Backup Candidate

Backup candidate:

`qr_rankdef_duplicate_5x4_nullspace_projector`

Backup rationale:

- It already has projector-style QR evidence and dense-reference helper support.
- It exercises rank-deficient nullspace behavior and can inform the proof-owner
  shape.
- It is not the maintained Sprint 138 corpus lane, so it should be used only if
  Day 3 finds that `qr_rank_deficient_6x4_nullspace_v1` has a blocking fixture,
  schema, or implementation contradiction.

If the backup is needed, Sprint 139 must explicitly document why the Sprint
138 lane cannot be used and must not claim the Sprint 138 handoff residual is
closed.

## Deferred Residuals and Out-of-Scope Boundaries

| Deferred residual | Defer reason | Future gate |
| --- | --- | --- |
| Broad rank-threshold policy | Current bounded threshold fixtures do not define a global numerical rank policy. | Later sprint with explicit tolerance policy and public wording review. |
| Rectangular least-squares behavior | Existing QR solve evidence is bounded and not the selected corpus lane. | Dedicated solve fixture/oracle rows if a future residual selects solve behavior. |
| Minimum-norm behavior | Existing lanes remain bounded; broad minimum-norm remains a non-claim. | Separate minimum-norm claim gate with fixture, oracle, and docs. |
| COLAMD/reordered QR behavior | Reorder behavior adds permutation and fill considerations beyond nullspace closure. | Reorder-specific fixture and proof owner. |
| SuiteSparse rank-deficient QR subset | Optional external data is disabled by default and lacks reviewed pass evidence. | License/data review plus hosted support-tier proof. |
| Broad LAPACK/NumPy/SciPy/SuiteSparse parity | Violates fixture-local closure and needs an external parity strategy. | Separate external parity product decision. |
| Partial-SVD rank-deficient subspace behavior | Owned by Sprint 140 partial-SVD residual closure. | Sprint 140 selected residual and comparison semantics. |

## Evidence and Gap Map

| Evidence family | Present today | Remaining gap for selected closure |
| --- | --- | --- |
| Corpus fixture metadata | `tests/corpus/manifests/fixtures.tsv` defines `qr_rank_deficient_6x4_nullspace_v1`. | Day 3 must decide whether to reuse metadata directly or mirror it in C helper form. |
| Generator metadata | `tests/corpus/manifests/generators.tsv` records deterministic generation and hashes. | Solver-backed proof needs a matching matrix builder or reader. |
| Expected rows | Rank, nullity, and residual rows are `ready_for_oracle`. | Observed solver-backed QR rows do not exist yet. |
| Existing QR nullspace tests | `tests/test_qr.c` verifies several nullspace/projector lanes. | No focused owner for this exact corpus fixture. |
| Existing dense helper | `tests/qr_external_dense_reference.py` supports bounded QR references. | It does not currently expose the Sprint 138 corpus fixture key. |
| Documentation | Maintainer and corpus docs preserve boundaries. | Earned public wording waits until proof and validation land. |

## Day 2 Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected residual can be closed without broadening unsupported claims. | Complete | `qr_rank_deficient_6x4_nullspace_v1` uses fixture-local rank, nullity, and normalized residual evidence. |
| Lower-priority residuals have explicit defer reasons. | Complete | Deferred residual table separates rank-threshold, least-squares, minimum-norm, COLAMD, SuiteSparse, parity, and partial-SVD work. |
| Fixture and oracle design can proceed from a single bounded QR behavior. | Complete | Day 3 design focus is limited to the selected corpus-backed rank-deficient nullspace closure. |
