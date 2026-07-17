# Sprint 126 Working Notes

## Sprint Goal

Convert Sprint 125's remaining rank-deficient QR, nullspace, threshold,
SuiteSparse, and minimum-norm residual debt into bounded evidence or explicit
future-owner decisions before Q/economy, corpus-index, and adoption work consume
those truth boundaries.

## Starting Constraints

- Treat Sprint 125's residual deferred debt and non-claim register as the
  source of truth for Sprint 126 scope.
- Do not reopen completed Sprint 125 intake, residual-only rank-deficient QR,
  nullspace projector, threshold-family, minimum-norm owner-local evidence,
  bounded QR-vs-SVD decision, SuiteSparse submatrix smoke, validation, or
  closeout work.
- Keep compatible/wide residual fixtures, nullspace/subspace evidence,
  threshold families, SuiteSparse rank-deficient QR corpus evidence,
  SuiteSparse minimum-norm evidence, larger underdetermined exact-value lanes,
  and QR-vs-SVD minimum-norm cross-checks separate because each lane has
  different proof boundaries.
- Preserve non-claims around broad LAPACK, NumPy, SciPy, BLAS, SuiteSparse,
  PETSc, Trilinos, Eigen, ARPACK, vendor-backend, dense-library, QR,
  nullspace, minimum-norm, Q-basis, economy, sparse-mode, reorder, backend,
  corpus, performance, platform, package, public API, and state-of-the-art
  parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and focused
  markdown whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 126 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 126 | Defines seven Sprint 126 items for residual dedupe, compatible/wide residual fixtures, nullspace/subspace expansion, threshold family expansion, SuiteSparse rank-deficient QR corpus gating, SuiteSparse and underdetermined minimum-norm evidence, and QR-vs-SVD cross-check gating. |
| `docs/planning/EPIC_11/SPRINT_126/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| `docs/planning/EPIC_11/SPRINT_125/RETROSPECTIVE.md` | Defines the carry-forward residual deferred debt and the explicit non-claim register. |
| `docs/planning/EPIC_11/SPRINT_125/WORKING_NOTES.md` | Captures completed Sprint 125 fixtures, validation rules, duplicate fences, and future-owner handoffs. |
| Sprint 125 Day 1-3 artifacts | Source for deferred QR/minimum-norm dedupe and completed residual-only rank-deficient QR evidence. |
| Sprint 125 Day 4-5 artifacts | Source for nullspace/subspace policy and completed duplicate-column 5x4 nullspace projector evidence. |
| Sprint 125 Day 6-7 artifacts | Source for threshold-family policy and completed diagonal threshold family evidence. |
| Sprint 125 Day 8-9 artifacts | Source for SuiteSparse rank-deficient QR corpus policy and explicit deferral boundaries. |
| Sprint 125 Day 10-12 artifacts | Source for minimum-norm behavior owner map, owner-local evidence, QR-vs-SVD bounded cross-check decision, and `west0067` minimum-norm submatrix smoke. |
| Sprint 125 Day 13-14 artifacts | Source for validation, claim-gate, closeout, and Sprint 126 handoff requirements. |
| Sprint 121-124 artifacts | Source for earlier QR/SVD/rank taxonomy, bounded external-reference lanes, Q/economy policies, and helper ownership decisions. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, residual dedupe baseline, duplicate fence, validation boundary | Items 1-7 |
| 2 | Compatible, dependent-row, and wide residual fixture trust policy | Item 2 |
| 3 | Compatible/wide residual evidence batch or explicit deferral | Item 2 |
| 4 | Expanded nullspace/subspace policy refresh | Item 3 |
| 5 | Nullspace/subspace evidence batch or explicit deferral | Item 3 |
| 6 | Threshold family expansion policy | Item 4 |
| 7 | Threshold family evidence batch or explicit deferral | Item 4 |
| 8 | SuiteSparse rank-deficient QR corpus gate | Item 5 |
| 9 | SuiteSparse rank-deficient QR evidence decision | Item 5 |
| 10 | SuiteSparse minimum-norm evidence gate | Item 6 |
| 11 | SuiteSparse minimum-norm evidence batch or explicit deferral | Item 6 |
| 12 | Larger underdetermined exact-value and QR-vs-SVD cross-check gate | Items 6-7 |
| 13 | Underdetermined and QR-vs-SVD evidence batch or explicit deferral | Items 6-7 |
| 14 | Full validation, claim gate, residual publication, and Sprint 127 handoff | Items 1-7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused markdown whitespace scan over Sprint 126 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Python external-reference helper edits | `python3 -m py_compile` for the helper, focused helper invocation, affected test executable, and `git diff --check`. |
| Fixture or test registration edits | Focused executable proof plus Make/CMake/CTest impact check if membership changes. |
| SuiteSparse optional-corpus edits | Focused optional-data present/missing behavior, skip-path proof, diagnostics check, and support-tier note. |
| Maintainer or public wording edits | Evidence-to-claim traceability, claim-boundary scan, link/path hygiene, and explicit non-claim update. |

## Scope Boundaries

- Sprint 126 may add bounded evidence only after the relevant trust,
  metadata, tolerance, skip, metric, diagnostics, and failure interpretation are
  explicit.
- Sprint 126 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 126 must not relabel completed Sprint 125 fixtures as broader
  compatible, wide, multi-dimensional, threshold, SuiteSparse, underdetermined,
  QR-vs-SVD, raw-basis, or parity proof.
- Sprint 126 must not absorb Sprint 127 Q-basis/economy/helper movement work
  except as handoff prerequisites.
- Sprint 126 must not update public solver-selection wording unless Day 14
  proves evidence supports bounded user-facing wording beyond current guidance.

## Day 1 Notes

- Created the Sprint 126 working-notes baseline.
- Created the Day 1 artifact directory entry.
- Mapped every Sprint 126 project-plan item to a day-level owner.
- Recorded duplicate fences for completed Sprint 121-125 QR, nullspace,
  threshold, minimum-norm, SuiteSparse, SVD-pseudoinverse, Q/economy, and
  helper evidence.
- Established validation expectations for documentation, C code, Python
  helper, fixture/test registration, SuiteSparse optional-corpus, maintainer,
  and public wording changes.

## Day 2 Notes

- Inventoried current residual-only, compatible solve, dependent-row, wide,
  and minimum-norm evidence across `tests/test_qr.c`,
  `tests/test_qr_solve.c`, `tests/test_colamd.c`,
  `tests/test_qr_helpers.h`, and `tests/qr_external_dense_reference.py`.
- Kept the completed Sprint 125
  `qr_rankdef_duplicate_5x4_residual_only` lane as the baseline residual-only
  evidence and fenced it from compatible zero-residual, dependent-row, wide,
  SuiteSparse, nullspace, and minimum-norm follow-through.
- Defined Day 3's highest-value accepted candidate as a dependent-row
  residual-only policy decision, with implementation allowed only if a standard
  library external reference proves distinct residual trust without solution,
  rank, nullspace, or minimum-norm assertions.
- Deferred compatible zero-residual evidence by default because it mostly
  repeats existing compatible solve behavior and has high minimum-norm claim
  confusion risk unless Day 3 proves a distinct diagnostic value.
- Deferred wide residual evidence by default to minimum-norm or
  nullspace/subspace owners unless Day 3 can define a residual-only shape
  contract that does not imply underdetermined solution selection.
- Recorded fixture-local residual tolerances, diagnostics, skip behavior,
  proof boundaries, and explicit non-claims in the Day 2 artifact.

## Day 3 Notes

- Accepted the Day 2 preferred
  `qr_rankdef_dependent_row_4x3_residual_only` candidate as the only Day 3
  implementation batch.
- Added a Python standard-library residual reference for the existing
  dependent-row 4x3 fixture with RHS `[1.0, -2.0, 5.0, 0.0]`.
- Added a focused `test_qr_solve` residual-only check that compares the QR
  solve returned residual against the external reference value
  `4.2840332837724997`.
- Kept the test assertion surface intentionally narrow: non-zero expected
  residual and absolute residual agreement only.
- Deferred compatible zero-residual, wide residual-only, wide sparse-mode, and
  SuiteSparse residual fixtures to future owners because each still depends on
  zero-residual diagnostic value, solution-selection policy, Q/economy
  boundaries, or corpus support-tier metadata.
- Recorded helper output, focused test output, validation requirements, and
  non-claims in the Day 3 artifact.

## Day 4 Notes

- Refreshed Sprint 125's nullspace/subspace policy for Sprint 126's expanded
  candidate set.
- Identified five candidate families: multi-dimensional nullspace,
  wide-shape nullspace, dependent-row projector, near-threshold nullspace, and
  SuiteSparse nullspace/subspace evidence.
- Kept `qr_rankdef_duplicate_5x4_nullspace_projector` as completed baseline
  evidence, not a candidate to repeat.
- Required every future nullspace/subspace fixture to pin expected rank,
  nullity, threshold, matrix shape, metric, tolerance, diagnostics, and
  failure interpretation before implementation.
- Selected full projectors as the default metric for small dense references
  and two-way projection residuals as the preferred scalable metric for
  multi-dimensional, wide, or SuiteSparse candidates.
- Rejected raw basis equality, basis ordering, unique orientation, principal
  angle implementation, minimum-norm, pseudoinverse, Q/economy, sparse-mode,
  backend, and corpus parity claims for Day 4.
- Recorded the Day 5 candidate order, acceptance gates, deferral gates,
  validation requirements, and non-claims in the Day 4 artifact.

## Day 5 Notes

- Accepted `qr_rank1_4x3_nullspace_projector` as the first expanded
  multi-dimensional nullspace/subspace fixture.
- Added a Python standard-library exact projector reference for the rank-1
  4x3 fixture, with expected rank 1, nullity 2, threshold 0.0, and projector
  `I - 11^T / 3`.
- Added a focused QR test that obtains the product nullspace basis,
  orthonormalizes it locally, computes the product projector, and compares it
  against the external reference without raw basis equality.
- Recorded focused validation: helper output `OK 13`; focused `test_qr`
  passed with projector diff `2.220e-16`, null residual `5.088e-16`, and
  orthogonality error `2.220e-16`.
- Deferred wide-shape, dependent-row projector, near-threshold, SuiteSparse,
  sparse-mode, principal-angle, and raw-basis lanes to their policy owners and
  promotion gates.
- Because Day 5 changed `.c` and Python helper code, full `make format &&
  make lint && make test` validation is required before closeout.

## Day 6 Notes

- Refreshed Sprint 125's threshold-family policy for Sprint 126's expanded
  scaled, perturbed, dependent-row, wide, and SuiteSparse candidate set.
- Kept `qr_rank_threshold_diag4_family` as completed baseline evidence, not a
  Sprint 126 candidate to repeat.
- Identified the scaled diagonal ladder as the preferred Day 7 implementation
  candidate because it tests relative-threshold scale invariance without
  changing the existing rank ladder.
- Required every accepted threshold-family fixture to pin fixture key, matrix
  construction, scale or perturbation, threshold list, expected ranks,
  absolute-threshold diagnostics, stability rules, and non-global
  interpretation before implementation.
- Deferred perturbed duplicate-column, dependent-row, wide, and SuiteSparse
  threshold families unless Day 7 records their promotion gates or proves
  stable fixture-local metadata.
- Recorded diagnostics, tolerance, support-tier, skip behavior, validation,
  and no-global-rank-policy non-claims in the Day 6 artifact.

## Day 7 Notes

- Accepted `qr_rank_threshold_diag4_scaled_family` as the bounded expanded
  QR threshold-family evidence batch.
- Added helper output for three scale values: `1e-6`, `1`, and `1e6`, each
  checked at thresholds `1e-14`, `1e-10`, and `1e-6` with expected ranks
  `3`, `2`, and `1`.
- Added a focused QR test that builds each scaled diagonal fixture, compares
  `sparse_qr_rank()` and `sparse_qr_rank_info()` against the external
  expected ranks, and prints scale, relative threshold, absolute threshold,
  expected/product/info ranks, and R diagonal magnitudes.
- Preserved the completed `qr_rank_threshold_diag4_family` baseline without
  relabeling it as new Sprint 126 work.
- Deferred perturbed duplicate-column, dependent-row, wide, SuiteSparse, and
  default-threshold evidence to future owners and promotion gates.
- Focused validation passed: helper emitted `OK 27`; `make build/test_qr &&
  ./build/test_qr` passed 70 tests, 0 failures, 0 skips.
- Because Day 7 changed `.c` and Python helper code, full `make format &&
  make lint && make test` validation is required before closeout.

## Day 8 Notes

- Re-ran the SuiteSparse rank-deficient QR corpus gate against the current
  Sprint 126 scope, Sprint 125 Day 8-9 decisions, checked-in Matrix Market
  inventory, and current QR/SuiteSparse test owners.
- Kept `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` classified as existing
  full-rank QR controls, not rank-deficient candidates.
- Classified `steam1.mtx` as a possible default checked-in Day 9 investigation
  target only if independent expected-rank metadata is available before test
  registration.
- Kept `fs_541_1.mtx` and `orsirr_1.mtx` behind the existing
  `SPARSE_TEST_LARGE=1` optional-large convention unless Day 9 defines and
  validates a narrower QR-specific gate.
- Kept `bcsstk14.mtx`, `s3rmt3m3.mtx`, `Kuu.mtx`, `bloweybq.mtx`,
  `Pres_Poisson.mtx`, and `tuma1.mtx` report-only for QR rank-deficient
  evidence because they lack pinned expected-rank metadata and are too large
  or too unsupported for default evidence.
- Defined the Day 9 metadata protocol: matrix path, support tier, claim type,
  expected rank/nullity or threshold/rank pairs, independent metadata source,
  threshold semantics, diagnostics, skip behavior, and validation commands.
- Day 9 may either promote one narrowly named fixture-local candidate after
  satisfying the Day 8 protocol or explicitly defer Project Plan Item 5.
- Day 8 changed documentation only, so required validation is `git diff
  --check` plus a focused trailing-whitespace scan over Sprint 126 docs.

## Day 9 Notes

- Explicitly deferred SuiteSparse rank-deficient QR corpus evidence under the
  Day 8 metadata protocol.
- Rejected `west0067.mtx`, `nos4.mtx`, and `bcsstk04.mtx` as
  rank-deficient evidence because the focused QR solve executable reports them
  as full-rank controls: ranks `67`, `100`, and `132`, respectively.
- Deferred `steam1.mtx` because it is checked in and default-tier but still
  lacks independent expected-rank/nullity or threshold/rank metadata for a QR
  rank-deficient claim.
- Deferred `fs_541_1.mtx` and `orsirr_1.mtx` because they remain
  optional-large under `SPARSE_TEST_LARGE=1` and lack independent expected-rank
  metadata plus QR-specific runtime/skip proof.
- Deferred report-only matrices (`bcsstk14.mtx`, `s3rmt3m3.mtx`, `Kuu.mtx`,
  `bloweybq.mtx`, `Pres_Poisson.mtx`, and `tuma1.mtx`) until a future owner
  promotes support tier and pins expected-rank metadata.
- Focused validation passed: `make build/test_qr_solve && ./build/test_qr_solve`
  ran 19 tests, 0 failures, 0 skips, and 1104 assertions.
- Day 9 changed documentation only, so full C quality gates are not required;
  documentation validation remains `git diff --check` plus Sprint 126
  trailing-whitespace scan.

## Day 10 Notes

- Created the SuiteSparse minimum-norm corpus gate for Sprint 126 Project Plan
  Item 6.
- Preserved Sprint 125's accepted `west0067` first-30-row 30 x 67
  minimum-norm smoke as the baseline default checked-in corpus evidence and
  explicitly fenced it from optional-large, rank-deficient SuiteSparse,
  platform, and performance claims.
- Identified Day 11 candidates: non-duplicate `west0067` extractions,
  `steam1` submatrices, optional-large `fs_541_1` or `orsirr_1` submatrices,
  report-only matrix submatrices, and rank-deficient SuiteSparse
  minimum-norm corpus paths.
- Required every accepted SuiteSparse minimum-norm candidate to pin matrix
  path, extraction rule, shape, nnz, support tier, RHS, expected rank/nullity
  when claimed, residual metric/tolerance, solution-norm target, diagnostics,
  skip behavior, and validation commands before implementation.
- Required optional-large candidates to prove `SPARSE_TEST_LARGE=1` or a
  narrower gate, missing-data skip behavior, runtime budget, and numerical
  failure behavior before registration.
- Set Day 11's default decision to explicit deferral unless a non-duplicate
  candidate has independent extraction, rank/nullity, residual, and norm
  metadata.
- Day 10 changed documentation only, so required validation is `git diff
  --check` plus a focused trailing-whitespace scan over Sprint 126 docs.

## Day 11 Notes

- Explicitly deferred additional SuiteSparse minimum-norm evidence under the
  Day 10 metadata protocol.
- Preserved Sprint 125's `west0067` first-30-row 30 x 67 minimum-norm
  submatrix smoke as the only accepted SuiteSparse minimum-norm corpus
  evidence.
- Rejected a repeat of the `west0067` first-30-row smoke as duplicate Sprint
  125 evidence.
- Deferred new `west0067` row-window, `steam1`, optional-large `fs_541_1` and
  `orsirr_1`, report-only matrix, rank-deficient SuiteSparse, and
  SuiteSparse QR-vs-SVD corpus candidates because they lack pinned extraction,
  rank/nullity, residual tolerance, norm target, support-tier promotion,
  optional skip proof, or bounded cross-check wording.
- Focused owner-local validation passed: `make build/test_colamd &&
  ./build/test_colamd` ran 70 tests, 0 failures, 0 skips, and 299 assertions;
  the existing `west0067` smoke printed max residual `1.78e-15` and
  `||x||=4.30 <= ||1||=8.19`.
- Day 11 changed documentation only, so full C quality gates are not required;
  documentation validation remains `git diff --check` plus Sprint 126
  trailing-whitespace scan.

## Day 12 Notes

- Created the underdetermined minimum-norm and QR-vs-SVD cross-check gate for
  Sprint 126 Project Plan Items 6 and 7.
- Accepted one Day 13 implementation candidate:
  `qr_minnorm_5x10_exact_values` in `tests/test_colamd.c::test_minnorm_5x10`.
- Pinned the 5 x 10 fixture's independent closed-form derivation: each row is
  `2a + b = rhs_i`, so the minimum-norm pair is
  `[a, b] = [2*rhs_i/5, rhs_i/5]`.
- Pinned expected 5 x 10 solution values
  `[0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, expected norm
  `sqrt(11)`, residual tolerance `1e-10`, value tolerance `1e-10`, and norm
  tolerance `1e-10`.
- Deferred new QR-vs-SVD minimum-norm cross-checks because the existing Sprint
  125 2 x 4 cross-check remains the bounded baseline and broader additions
  risk implying SVD-pseudoinverse-as-global-oracle semantics.
- Deferred SuiteSparse QR-vs-SVD corpus cross-checks under the Day 11
  SuiteSparse minimum-norm deferral.
- Day 12 changed documentation only, so required validation is `git diff
  --check` plus a focused trailing-whitespace scan over Sprint 126 docs.

## Day 13 Notes

- Accepted and implemented `qr_minnorm_5x10_exact_values` in
  `tests/test_colamd.c::test_minnorm_5x10`.
- Added exact expected solution values
  `[0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- Preserved the existing residual assertions and added max residual
  diagnostics.
- Added the exact norm assertion `||x|| = sqrt(11)`.
- Deferred additional QR-vs-SVD minimum-norm cross-checks; the Sprint 125 2 x
  4 bounded cross-check remains the only accepted QR-vs-SVD minimum-norm
  comparison.
- Focused validation passed: `make build/test_colamd && ./build/test_colamd`
  ran 70 tests, 0 failures, 0 skips, and 310 assertions; the 5 x 10 fixture
  printed max residual `8.88e-16` and `||x||=3.3166`.
- Because Day 13 changed `.c` code, full `make format && make lint && make
  test` validation is required before closeout.

## Day 14 Notes

- Inventoried all Sprint 126 code, helper, maintainer-doc, planning-doc, and
  artifact changes.
- Refreshed the QR maintained evidence row in `docs/maintainer_guide.md` with
  Sprint 126's accepted bounded fixtures:
  `qr_rankdef_dependent_row_4x3_residual_only`,
  `qr_rank1_4x3_nullspace_projector`,
  `qr_rank_threshold_diag4_scaled_family`, and
  `qr_minnorm_5x10_exact_values`.
- Left public solver-selection, README, headers, package metadata, and API
  wording unchanged because Sprint 126 evidence remains fixture-scoped or
  owner-local.
- Published the Day 14 validation, claim-gate, residual queue, non-claim
  register, and Sprint 127 Q/economy/helper handoff artifact.
- Day 14 changed documentation only after the Sprint 126 code package, but
  the same-day full `make format && make lint && make test` gate passed for
  the sprint-level touched C/helper surfaces.
- `git diff --check`, a focused trailing-whitespace scan, and Python cache
  scan passed after closeout edits.
