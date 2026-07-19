# Sprint 127 Working Notes

## Sprint Goal

Convert Sprint 126's remaining rank-deficient QR residual,
nullspace/subspace, threshold-family, SuiteSparse corpus, optional-large, and
minimum-norm deferred debt into bounded evidence or explicit future-owner
decisions before Q/economy, partial-SVD, corpus-index, performance, package,
and adoption work consume those truth boundaries.

## Starting Constraints

- Treat Sprint 126's residual deferred debt and non-claim register as the
  source of truth for Sprint 127 scope.
- Do not reopen completed Sprint 126 intake, dependent-row residual,
  rank-1/nullity-2 projector, scaled threshold-family, SuiteSparse QR corpus
  deferral, SuiteSparse minimum-norm deferral, 5 x 10 exact minimum-norm,
  validation, maintainer evidence, or closeout work.
- Keep compatible zero-residual, wide residual-only, nullspace/subspace,
  threshold-family, SuiteSparse rank-deficient QR corpus, SuiteSparse and
  optional-large minimum-norm, exact underdetermined, QR-vs-SVD cross-check,
  and helper-movement work separate because each lane has different proof
  boundaries.
- Preserve non-claims around broad LAPACK, NumPy, SciPy, BLAS, SuiteSparse,
  PETSc, Trilinos, Eigen, ARPACK, vendor-backend, dense-library, QR,
  compatible solve, wide solve, rank-deficient solve, nullspace, minimum-norm,
  Q-basis, economy, sparse-mode, reorder, backend, corpus, optional-data,
  performance, platform, package, public API, and state-of-the-art parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and focused
  markdown whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 127 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 127 | Defines seven Sprint 127 items for deferred dedupe, compatible/wide residual semantics, nullspace/subspace expansion, threshold-family follow-through, SuiteSparse rank-deficient QR corpus evidence, SuiteSparse and optional-large minimum-norm gating, and minimum-norm exact/cross-check/helper claim gates. |
| `docs/planning/EPIC_11/SPRINT_127/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_11/SPRINT_126/RETROSPECTIVE.md` | Defines the carry-forward residual deferred debt and the explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_126/WORKING_NOTES.md` | Captures completed Sprint 126 evidence, validation rules, duplicate fences, and future-owner handoffs. |
| Sprint 126 Day 1 artifact | Source for residual dedupe and duplicate fences against completed Sprint 121-126 work. |
| Sprint 126 Day 2-3 artifacts | Source for compatible/dependent-row/wide residual semantics and completed `qr_rankdef_dependent_row_4x3_residual_only` evidence. |
| Sprint 126 Day 4-5 artifacts | Source for nullspace/subspace policy and completed `qr_rank1_4x3_nullspace_projector` evidence. |
| Sprint 126 Day 6-7 artifacts | Source for threshold-family policy and completed `qr_rank_threshold_diag4_scaled_family` evidence. |
| Sprint 126 Day 8-9 artifacts | Source for SuiteSparse rank-deficient QR corpus gate and explicit deferral boundaries. |
| Sprint 126 Day 10-11 artifacts | Source for SuiteSparse minimum-norm corpus gate and explicit deferral boundaries. |
| Sprint 126 Day 12-13 artifacts | Source for larger underdetermined exact-value and QR-vs-SVD gate plus completed `qr_minnorm_5x10_exact_values` evidence. |
| Sprint 126 Day 14 artifact | Source for validation, maintainer evidence, non-claim register, and Sprint 127 handoff requirements. |
| Sprint 121-125 artifacts | Source for earlier QR/SVD/rank taxonomy, external-reference lanes, nullspace policies, threshold policies, minimum-norm owner maps, bounded QR-vs-SVD decisions, and helper ownership decisions. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, deferred dedupe baseline, duplicate fence, validation boundary | Items 1-7 |
| 2 | Compatible zero-residual and wide residual-only semantics policy | Item 2 |
| 3 | Compatible/wide residual evidence decision or explicit deferral | Item 2 |
| 4 | Nullspace/subspace expansion policy | Item 3 |
| 5 | Nullspace/subspace evidence batch or explicit deferral | Item 3 |
| 6 | Threshold-family follow-through policy | Item 4 |
| 7 | Threshold-family evidence batch or explicit deferral | Item 4 |
| 8 | SuiteSparse rank-deficient QR corpus evidence gate | Item 5 |
| 9 | SuiteSparse rank-deficient QR evidence decision | Item 5 |
| 10 | SuiteSparse and optional-large minimum-norm gate | Item 6 |
| 11 | SuiteSparse and optional-large minimum-norm evidence decision | Item 6 |
| 12 | Exact underdetermined and QR-vs-SVD cross-check helper claim gate | Items 6-7 |
| 13 | Exact minimum-norm and QR-vs-SVD cross-check evidence decision | Item 7 |
| 14 | Validation, claim gate, residual publication, and Sprint 128 handoff | Items 1-7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused markdown whitespace scan over Sprint 127 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Python external-reference helper edits | `python3 -m py_compile` for the helper, focused helper invocation, affected test executable, and `git diff --check`. |
| Fixture or test registration edits | Focused executable proof plus Make/CMake/CTest impact check if membership changes. |
| SuiteSparse optional-corpus edits | Focused optional-data present/missing behavior, skip-path proof, diagnostics check, runtime note, and support-tier note. |
| Maintainer or public wording edits | Evidence-to-claim traceability, claim-boundary scan, link/path hygiene, and explicit non-claim update. |

## Scope Boundaries

- Sprint 127 may add bounded evidence only after the relevant trust,
  metadata, tolerance, skip, metric, diagnostics, support-tier, runtime, and
  failure interpretation are explicit.
- Sprint 127 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 127 must not relabel completed Sprint 126 fixtures as broader
  compatible, wide, multi-dimensional, threshold, SuiteSparse, optional-large,
  underdetermined, QR-vs-SVD, raw-basis, helper, or parity proof.
- Sprint 127 must not absorb Sprint 128 Q-basis/economy/helper ownership work
  except as prerequisite handoff data.
- Sprint 127 must not update public solver-selection wording unless Day 14
  proves evidence supports bounded user-facing wording beyond current guidance.

## Day 1 Notes

- Created the Sprint 127 working-notes baseline.
- Created the Day 1 artifact directory entry.
- Mapped every Sprint 127 project-plan item to a day-level owner.
- Recorded duplicate fences for completed Sprint 121-126 QR residual,
  nullspace/subspace, threshold, minimum-norm, SuiteSparse, SVD-pseudoinverse,
  Q/economy, and helper evidence.
- Established validation expectations for documentation, C code, Python
  helper, fixture/test registration, SuiteSparse optional-corpus, maintainer,
  and public wording changes.

## Day 2 Notes

- Reviewed current compatible, dependent-row, duplicate-column, wide,
  underdetermined, minimum-norm, and sparse/economy QR evidence across
  `tests/test_qr.c`, `tests/test_qr_solve.c`, `tests/test_colamd.c`,
  `tests/test_qr_helpers.h`, and `tests/qr_external_dense_reference.py`.
- Kept Sprint 125's `qr_rankdef_duplicate_5x4_residual_only` and Sprint
  126's `qr_rankdef_dependent_row_4x3_residual_only` as completed
  residual-only baselines, not candidates to repeat.
- Defined compatible zero-residual evidence as acceptable only when the
  fixture has explicit rank-deficient structure, a named compatible RHS, a
  residual-only diagnostic that adds trust beyond existing compatible solves,
  and no solution-selection, nullspace, or minimum-norm implication.
- Defined wide residual-only evidence as deferred by default until
  underdetermined output semantics, solution-selection boundaries,
  Q/economy/sparse-mode boundaries, and residual-only proof value are pinned.
- Selected Day 3's only conditionally acceptable implementation path as a
  policy-backed compatible zero-residual decision if a non-duplicate
  diagnostic can be proven before code changes; otherwise Day 3 should
  explicitly defer Item 2.
- Recorded candidate tables, output-semantics rules, tolerance and diagnostic
  policy, promotion gates, and explicit non-claims in the Day 2 artifact.

## Day 3 Notes

- Applied the Day 2 compatible zero-residual and wide residual-only semantics
  gate to all Item 2 candidates.
- Explicitly deferred `qr_rankdef_duplicate_5x4_compatible_zero_residual`
  because it does not yet add a distinct diagnostic beyond existing
  full-rank compatible solves, duplicate-column rank/residual evidence, and
  rank-deficient solve smoke.
- Explicitly deferred `qr_rankdef_dependent_row_4x3_compatible_zero_residual`
  because Sprint 126 already accepted the incompatible dependent-row
  residual-only lane and existing deterministic dependent-row checks cover the
  structural fixture.
- Explicitly deferred wide residual-only candidates because
  underdetermined output semantics, solution-selection boundaries,
  Q/economy/sparse-mode boundaries, and residual-only proof value are not
  pinned tightly enough for code registration.
- Did not change C, header, Python helper, build, maintainer, or public
  wording files.
- Recorded future-owner promotion gates, validation expectations, and
  residual non-claims in the Day 3 artifact.

## Day 4 Notes

- Refreshed the Sprint 125-126 nullspace/subspace policy for Sprint 127's
  remaining wide-shape, dependent-row, near-threshold, and SuiteSparse
  candidate set.
- Kept `qr_rankdef_duplicate_5x4_nullspace_projector` and
  `qr_rank1_4x3_nullspace_projector` as completed projector baselines, not
  candidates to repeat.
- Required every future nullspace/subspace fixture to pin fixture key, shape,
  expected rank, expected nullity, threshold semantics, metric, tolerance,
  diagnostics, skip behavior, support tier when applicable, and proof
  boundary before implementation.
- Chose full projectors for tiny low-nullity fixtures and two-way projection
  residuals for wide, multi-dimensional, near-threshold, and SuiteSparse
  candidates where full projectors become noisy or too large.
- Deferred SuiteSparse subspace evidence by policy until expected rank/nullity
  metadata, support tier, optional-data behavior, runtime budget, and
  diagnostics are available.
- Recorded Day 5 candidate ordering, acceptance gates, deferral gates,
  validation requirements, and raw-basis/minimum-norm/Q-economy non-claims in
  the Day 4 artifact.

## Day 5 Notes

- Accepted `qr_rankdef_dependent_row_4x3_nullspace_projector` as the bounded
  dependent-row nullspace/subspace evidence batch.
- Added a Python standard-library exact projector reference for the existing
  dependent-row 4 x 3 fixture, using normalized nullspace vector
  `[-1, -2, 1] / sqrt(6)`, expected rank `2`, nullity `1`, and threshold
  `0.0`.
- Added a focused QR test that reuses `tf_qr_make_dependent_row_4x3()`,
  obtains the product nullspace basis, normalizes it, compares its projector
  against the external reference, and reports projector diff, null residual,
  and norm error.
- Preserved raw basis equality, basis orientation, Q/economy, sparse-mode,
  minimum-norm, pseudoinverse, SuiteSparse, optional-data, platform, and broad
  QR non-claims.
- Deferred wide-shape, near-threshold, SuiteSparse, sparse-mode,
  principal-angle, and raw-basis lanes to their policy owners and promotion
  gates.
- Focused validation passed: helper emitted `OK 13`; `make build/test_qr &&
  ./build/test_qr` passed 71 tests and printed projector diff `5.551e-17`,
  null residual `2.544e-16`, and norm error `4.441e-16`.
- Because Day 5 changed `.c` and Python helper code, full `make format &&
  make lint && make test` validation was required before closeout and passed.

## Day 6 Notes

- Reviewed completed Sprint 125 `qr_rank_threshold_diag4_family` and Sprint
  126 `qr_rank_threshold_diag4_scaled_family` threshold evidence as baselines
  that must not be repeated as Sprint 127 evidence.
- Reframed Sprint 127 Item 4 around the remaining perturbed
  duplicate-column, dependent-row, wide, default-threshold, and SuiteSparse
  threshold-family candidates.
- Chose `qr_rank_threshold_duplicate_5x4_perturbed_family` as the preferred
  Day 7 implementation candidate only if perturbation values, expected ranks,
  strict threshold comparisons, and R-diagonal diagnostics can be pinned
  before code edits.
- Kept dependent-row threshold evidence as a secondary Day 7 candidate because
  it must remain rank-threshold evidence and not reuse residual or projector
  metrics as proof.
- Deferred wide threshold, default-threshold policy, near-threshold subspace,
  and SuiteSparse threshold lanes unless their future owners can satisfy
  rank/nullity, support-tier, optional-data, and non-claim gates.
- Did not change C, header, Python helper, build, maintainer, or public
  wording files for Day 6.
- Recorded the threshold-family follow-through policy, diagnostics,
  perturbation separation rules, Day 7 acceptance gate, deferred promotion
  gates, and non-claim register in the Day 6 artifact.

## Day 7 Notes

- Accepted `qr_rank_threshold_duplicate_5x4_perturbed_family` as the bounded
  threshold-family evidence batch for Sprint 127 Item 4.
- Reused the existing duplicate-column 5 x 4 fixture and inserted a single
  perturbation `6e-8` at row `0`, column `3`, turning the duplicate-column
  structural fixture into a near-threshold rank fixture.
- Added Python helper triples for perturbation, threshold, and expected rank:
  rank `4` at `1e-10` and rank `3` at `1e-6`.
- Added a focused QR test that checks helper metadata, factors the perturbed
  fixture, compares `sparse_qr_rank()` and `sparse_qr_rank_info()` against the
  expected ranks, and prints absolute thresholds, pivot ratio, and R diagonal
  magnitudes.
- Preserved global rank-threshold, default-threshold, dense-library parity,
  residual, nullspace/subspace, minimum-norm, Q/economy, sparse-mode,
  SuiteSparse, optional-data, platform, performance, public API, and package
  non-claims.
- Deferred dependent-row threshold, wide threshold, default-threshold,
  SuiteSparse threshold, and near-threshold subspace lanes to their future
  owner gates.
- Focused validation passed: helper emitted `OK 6`; `make build/test_qr &&
  ./build/test_qr` passed 72 tests and printed product/rank-info agreement for
  both accepted thresholds.
- Because Day 7 changed `.c` and Python helper code, full `make format &&
  make lint && make test` validation was required before closeout and passed.

## Day 8 Notes

- Reviewed Sprint 125 and Sprint 126 SuiteSparse rank-deficient QR corpus
  policies and explicit deferrals.
- Inventoried the checked-in SuiteSparse Matrix Market corpus with current
  shape and nnz metadata.
- Preserved `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` as existing
  full-rank QR controls, not rank-deficient QR corpus evidence.
- Kept `steam1.mtx` deferred because no independent expected-rank or
  threshold/rank metadata is available for a QR rank-deficient claim.
- Kept `fs_541_1.mtx` and `orsirr_1.mtx` behind optional-large support-tier,
  runtime, expected-rank, and skip-behavior gates.
- Kept `bcsstk14.mtx`, `s3rmt3m3.mtx`, `Kuu.mtx`, `bloweybq.mtx`,
  `Pres_Poisson.mtx`, and `tuma1.mtx` report-only for this lane.
- Did not change C, header, Python helper, build, maintainer, Matrix Market,
  optional-data, or public wording files for Day 8.
- Recorded the Day 9 implementation-or-deferral checklist, expected-rank
  metadata policy, support-tier policy, runtime budget, skip behavior, and
  bounded corpus non-claims in the Day 8 artifact.

## Day 9 Notes

- Applied the Day 8 SuiteSparse rank-deficient QR corpus gate to the checked-in
  corpus.
- Explicitly deferred SuiteSparse rank-deficient QR corpus evidence because no
  candidate has independent expected-rank/nullity metadata or threshold/rank
  pairs.
- Rejected `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` as rank-deficient
  evidence because focused QR diagnostics report full-rank behavior.
- Kept `steam1.mtx` deferred until independent expected-rank metadata and
  focused QR diagnostics are available.
- Kept `fs_541_1.mtx` and `orsirr_1.mtx` behind optional-large support-tier,
  runtime, expected-rank, and skip-proof gates.
- Kept `bcsstk14.mtx`, `s3rmt3m3.mtx`, `Kuu.mtx`, `bloweybq.mtx`,
  `Pres_Poisson.mtx`, and `tuma1.mtx` report-only for this lane.
- Focused validation passed: `make build/test_qr_solve && ./build/test_qr_solve`
  passed 19 tests and reconfirmed ranks `100`, `132`, and `67` for `nos4`,
  `bcsstk04`, and `west0067`.
- Did not change C, header, Python helper, build, maintainer, Matrix Market,
  optional-data, or public wording files for Day 9.

## Day 10 Notes

- Reviewed the Sprint 125 and Sprint 126 minimum-norm owner maps, SuiteSparse
  corpus gates, explicit deferrals, and current owner-local evidence in
  `tests/test_colamd.c`, `tests/test_qr_solve.c`, and `tests/test_svd.c`.
- Preserved the Sprint 125 `west0067.mtx` first-30-row 30 x 67 smoke as the
  only accepted default checked-in SuiteSparse minimum-norm corpus baseline.
- Rejected a repeat of the same `west0067` extraction as duplicate unless a
  future owner adds a distinct pinned metric beyond the existing residual and
  feasible-vector norm-bound assertions.
- Kept alternate `west0067` row windows, `steam1` submatrices, rank-deficient
  SuiteSparse candidates, and SuiteSparse QR-vs-SVD corpus cross-checks gated
  behind extraction, shape, nnz, rank/nullity, residual, norm, diagnostics, and
  runtime metadata.
- Kept `fs_541_1.mtx` and `orsirr_1.mtx` behind optional-large
  `SPARSE_TEST_LARGE=1` support-tier, present/missing skip proof, runtime
  budget, and candidate-specific residual/norm rules.
- Kept `bcsstk14.mtx`, `s3rmt3m3.mtx`, `Kuu.mtx`, `bloweybq.mtx`,
  `Pres_Poisson.mtx`, and `tuma1.mtx` report-only for this lane until a future
  sprint promotes support tier and pins rank/nullity plus residual/norm
  metadata.
- Did not change C, header, Python helper, build, maintainer, Matrix Market,
  optional-data, or public wording files for Day 10.
- Recorded the Day 11 candidate decision rules, metadata protocol,
  diagnostics/tolerance policy, optional-large skip behavior, runtime
  expectations, duplicate fence, and non-claim register in the Day 10 artifact.

## Day 11 Notes

- Applied the Day 10 SuiteSparse and optional-large minimum-norm gate to the
  candidate set.
- Explicitly deferred additional SuiteSparse and optional-large minimum-norm
  corpus evidence because no non-duplicate candidate has independent
  extraction, rank/nullity, residual, norm, support-tier, skip, and runtime
  metadata pinned.
- Preserved the Sprint 125 `west0067.mtx` first-30-row 30 x 67 smoke as the
  only accepted SuiteSparse minimum-norm corpus baseline.
- Rejected the same `west0067` first-30-row extraction as duplicate for Day 11
  because it would not add trust beyond the existing residual and
  feasible-vector norm-bound assertions.
- Kept alternate `west0067` windows, `steam1` submatrices, rank-deficient
  SuiteSparse candidates, and SuiteSparse QR-vs-SVD corpus cross-checks
  deferred until fixture-local metadata and non-oracle wording are available.
- Kept `fs_541_1.mtx` and `orsirr_1.mtx` behind optional-large
  `SPARSE_TEST_LARGE=1`, present/missing skip proof, runtime budget, and
  candidate-specific residual/norm targets.
- Focused validation passed: `make build/test_colamd && ./build/test_colamd`
  passed 70 tests, 0 failures, 0 skips, 310 assertions, and reconfirmed
  `west0067` 30 x 67 minimum-norm smoke diagnostics.
- Did not change C, header, Python helper, build, maintainer, Matrix Market,
  optional-data, or public wording files for Day 11.

## Day 12 Notes

- Reviewed the existing exact underdetermined and QR-vs-SVD minimum-norm
  evidence in `tests/test_colamd.c`, `tests/test_qr_solve.c`,
  `tests/qr_external_dense_reference.py`, `tests/test_svd.c`, and the Sprint
  125-126 minimum-norm artifacts.
- Preserved `qr_underdetermined_minnorm_2x4` and
  `qr_minnorm_5x10_exact_values` as completed exact minimum-norm baselines
  and rejected repeats of those lanes.
- Accepted `qr_minnorm_3x6_exact_values` as the Day 13 implementation
  candidate because the existing 3 x 6 fixture has independent two-variable
  row constraints with closed-form values `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]`
  and exact norm `sqrt(8.4)`.
- Deferred additional QR-vs-SVD minimum-norm cross-checks so SVD
  pseudoinverse remains a bounded named cross-check, not a global QR oracle.
- Deferred generic minimum-norm helper movement unless a future owner uses
  behavior-specific helper names, keeps tolerances and fixture keys at the
  call site, and validates every touched owner executable.
- Kept SuiteSparse and optional-large exact/cross-check candidates governed by
  the Day 10-11 corpus deferrals rather than relabeling synthetic fixtures as
  corpus evidence.
- Did not change C, header, Python helper, build, maintainer, Matrix Market,
  optional-data, or public wording files for Day 12.
- Recorded the Day 13 implementation checklist, exact-value tolerance policy,
  helper movement boundary, QR-vs-SVD non-oracle wording, and non-claim
  register in the Day 12 artifact.

## Day 13 Notes

- Implemented `qr_minnorm_3x6_exact_values` in the owner-local
  `test_minnorm_3x6` fixture.
- Added exact expected values `[1.2, 1.2, 1.0, 0.6, 0.4, 2.0]`, max residual
  calculation, per-entry value assertions, and exact norm assertion
  `sqrt(8.4)`.
- Preserved the existing residual assertions and kept the diagnostic focused
  on max residual and solution norm.
- Did not add a QR-vs-SVD comparison; Sprint 125's bounded 2 x 4
  `test_minnorm_vs_pinv` lane remains the only accepted QR-vs-SVD
  minimum-norm cross-check.
- Did not move helper ownership or add generic minimum-norm helpers.
- Did not change header, Python helper, build, maintainer, Matrix Market,
  optional-data, SuiteSparse corpus, or public wording files for Day 13.
- Focused validation passed: `make build/test_colamd && ./build/test_colamd`
  passed 70 tests, 0 failures, 0 skips, and 317 assertions.
- Full validation passed: `make format && make lint && make test`.

## Day 14 Notes

- Inventoried all Sprint 127 code, helper, documentation, and artifact
  changes.
- Preserved the accepted implementation package as three bounded lanes:
  `qr_rankdef_dependent_row_4x3_nullspace_projector`,
  `qr_rank_threshold_duplicate_5x4_perturbed_family`, and
  `qr_minnorm_3x6_exact_values`.
- Updated `docs/maintainer_guide.md` only to add
  `qr_minnorm_3x6_exact_values` to the bounded exact minimum-norm fixture list.
- Kept public solver-selection, README, public headers, package metadata,
  build metadata, CMake/CTest membership, optional-data, and public API wording
  unchanged.
- Reconfirmed that Sprint 127 did not add broad QR, nullspace, minimum-norm,
  SuiteSparse, optional-large, helper API, platform, performance, or parity
  claims.
- Published the residual deferred debt queue for compatible/wide residuals,
  near-threshold and SuiteSparse subspace/threshold work, SuiteSparse
  rank-deficient QR, additional SuiteSparse/optional-large minimum-norm,
  additional QR-vs-SVD cross-checks, larger exact underdetermined lanes,
  generic helper movement, and Q/economy follow-through.
- Handed Sprint 128 the Q-basis/economy/helper prerequisites: named
  output-shape semantics, projection metrics, corpus support-tier and skip
  rules, behavior-specific helper names, call-site tolerances, and claim
  boundaries.
- Prepared the retrospective input package from the Sprint 127 plan, working
  notes, day artifacts, touched test/helper surfaces, maintainer evidence row,
  and Day 13 full quality-gate result.
