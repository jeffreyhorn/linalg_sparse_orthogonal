# Sprint 125 Working Notes

## Sprint Goal

Convert Sprint 124's rank-deficient QR and minimum-norm deferred debt into
behavior-specific evidence or explicit deferrals before broader corpus and
adoption work depend on these claims.

## Starting Constraints

- Treat Sprint 124's residual deferred debt and non-claim register as the
  source of truth for Sprint 125 scope.
- Do not reopen completed Sprint 124 intake, policy, fixture, validation, or
  closeout work.
- Do not duplicate completed bounded fixtures:
  `qr_rankdef_duplicate_5x4_rank_only`,
  `qr_underdetermined_minnorm_2x4`,
  `qr_economy_projector_5x3`, or
  `partial_svd_vector_residual_diag6_k2`.
- Keep rank-deficient QR residual-only, nullspace/subspace,
  near-rank-deficient threshold, SuiteSparse corpus, and minimum-norm behavior
  evidence separate because each lane has different proof boundaries.
- Preserve QR minimum-norm ownership across QR solve, COLAMD, fallback,
  rank-deficient, refinement, QR-vs-SVD-pseudoinverse, and optional
  SuiteSparse scenarios.
- Preserve non-claims around broad LAPACK, NumPy, SciPy, SuiteSparse, BLAS,
  PETSc, Trilinos, Eigen, ARPACK, vendor-backend, dense-library, QR,
  nullspace, minimum-norm, Q-basis, economy, sparse-mode, reorder, backend,
  corpus, performance, platform, package, public API, and state-of-the-art
  parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and focused
  markdown whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 125 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 125 | Defines seven Sprint 125 items for deferred QR/minimum-norm dedupe, rank-deficient residual evidence, nullspace/subspace policy, near-rank-deficient thresholds, SuiteSparse evidence, minimum-norm behavior evidence, and validation/claim gates. |
| `docs/planning/EPIC_11/SPRINT_125/PLAN.md` | Provides day-level execution order and 164-hour budget. |
| `docs/planning/EPIC_11/SPRINT_124/RETROSPECTIVE.md` | Defines the carry-forward rank-deficient QR and minimum-norm deferred debt plus explicit non-claims. |
| `docs/planning/EPIC_11/SPRINT_124/WORKING_NOTES.md` | Captures completed Sprint 124 fixtures, deferred work, validation expectations, and claim boundaries. |
| Sprint 124 Day 2-3 artifacts | Source for rank-deficient QR rank policy, completed rank-only fixture, and residual/nullspace/threshold/SuiteSparse deferrals. |
| Sprint 124 Day 4-5 artifacts | Source for minimum-norm behavior contract, completed exact 2x4 minimum-norm fixture, and COLAMD/fallback/rank-deficient/refinement/QR-vs-SVD/SuiteSparse deferrals. |
| Sprint 124 Day 13-14 artifacts | Source for validation requirements, maintainer evidence, solver-selection claim gate, and final handoff boundaries. |
| Sprint 121-123 artifacts | Source for earlier QR/SVD/rank taxonomy, external-reference lanes, and duplicate fences. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, deferred QR/minimum-norm dedupe map, duplicate fence, validation boundary | Items 1-7 |
| 2 | Residual-only rank-deficient QR trust gate | Item 2 |
| 3 | Rank-deficient residual evidence batch or explicit deferral | Item 2 |
| 4 | Nullspace and subspace policy design | Item 3 |
| 5 | Nullspace/subspace evidence decision | Item 3 |
| 6 | Near-rank-deficient threshold-family design | Item 4 |
| 7 | Near-rank-deficient threshold evidence decision | Item 4 |
| 8 | SuiteSparse rank-deficient QR corpus policy | Item 5 |
| 9 | SuiteSparse rank-deficient QR evidence decision | Item 5 |
| 10 | Minimum-norm behavior owner map | Item 6 |
| 11 | Minimum-norm core evidence batch | Item 6 |
| 12 | QR-vs-SVD-pseudoinverse and SuiteSparse minimum-norm decision | Item 6 |
| 13 | Validation, maintainer evidence, and claim gate | Item 7 |
| 14 | Sprint closeout and Sprint 126 handoff | Items 1-7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused markdown whitespace scan over Sprint 125 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Python external-reference helper edits | `python3 -m py_compile` for the helper, focused helper invocation, affected test executable, and `git diff --check`. |
| Fixture or test registration edits | Focused executable proof plus Make/CMake/CTest impact check if membership changes. |
| SuiteSparse optional-corpus edits | Focused optional-data path check, skip-path proof, diagnostics check, and support-tier note. |
| Maintainer or public wording edits | Evidence-to-claim traceability, claim-boundary scan, link/path hygiene, and explicit non-claim update. |

## Scope Boundaries

- Sprint 125 may add bounded evidence only after the relevant trust,
  tolerance, skip, metric, diagnostics, and failure interpretation are explicit.
- Sprint 125 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 125 must not relabel completed Sprint 124 rank-only or exact
  minimum-norm fixtures as residual-only, nullspace, threshold, SuiteSparse,
  COLAMD, fallback, rank-deficient minimum-norm, refinement, or QR-vs-SVD proof.
- Sprint 125 must not update public solver-selection wording unless Day 13
  proves evidence supports a user-facing claim beyond current workflow
  guidance.

## Day 1 Notes

- Created the Sprint 125 working-notes baseline.
- Created the Day 1 artifact directory entry.
- Mapped every Sprint 125 project-plan item to a day-level owner.
- Recorded duplicate fences for completed Sprint 121-124 QR, minimum-norm,
  Q/economy, and partial-SVD evidence.
- Established validation expectations for documentation, C code, Python
  helper, fixture/test registration, SuiteSparse optional-corpus, maintainer,
  and public wording changes.

## Day 2 Notes

- Inventoried current rank-deficient QR residual evidence across
  `tests/test_qr.c`, `tests/test_qr_solve.c`,
  `tests/qr_external_dense_reference.py`, and `tests/test_colamd.c`.
- Kept residual-only evidence separate from the completed
  `qr_rankdef_duplicate_5x4_rank_only` rank fixture and the completed
  `qr_underdetermined_minnorm_2x4` minimum-norm fixture.
- Defined residual-only evidence as acceptable only when it proves the product
  QR solve residual agrees with an independently computed bounded residual for
  a named rank-deficient fixture and explicitly does not assert rank,
  nullspace, minimum-norm, or pseudoinverse behavior.
- Identified `qr_rankdef_duplicate_5x4_residual_only` as the strongest Day 3
  candidate if it uses the existing 5x4 duplicate-column matrix with a RHS
  that has a non-zero least-squares residual and a standard-library reference
  output protocol.
- Rejected zero-residual compatible rank-deficient residual evidence for Day 3
  because it adds little trust beyond existing deterministic solve checks and
  risks being misread as minimum-norm evidence.
- Day 2 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files.

## Day 3 Notes

- Accepted and implemented one bounded residual-only rank-deficient QR external
  fixture: `qr_rankdef_duplicate_5x4_residual_only`.
- Reused the existing 5x4 duplicate-column matrix from
  `qr_rankdef_duplicate_5x4_rank_only` and added a deliberately incompatible
  RHS with a non-zero least-squares residual.
- Extended `tests/qr_external_dense_reference.py` with a standard-library
  column-space projection residual helper that emits `OK 1` plus the residual
  norm only.
- Extended `tests/test_qr_solve.c` with a shared 5x4 fixture builder and a
  residual-only test that compares the product QR returned residual against
  the external reference without asserting solution, norm, nullspace, rank, or
  pseudoinverse behavior.
- Updated `docs/maintainer_guide.md` so the QR evidence table names the new
  bounded residual-only fixture while preserving broad QR, nullspace,
  minimum-norm, Q/economy, backend, corpus, and performance non-claims.
- Deferred compatible zero-residual, dependent-row residual, wide
  rank-deficient residual, and SuiteSparse rank-deficient residual lanes to
  later owners with promotion gates.
- Focused validation passed:
  `python3 -m py_compile tests/qr_external_dense_reference.py`,
  `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_residual_only`,
  and `make build/test_qr_solve && ./build/test_qr_solve`.
- Full required quality validation passed: `make format && make lint &&
  make test`.

## Day 4 Notes

- Inventoried current QR rank/nullspace evidence in `tests/test_qr.c`,
  including duplicate-column rank deficiency, rank-1 nullspace, known
  nullspace, 3x5 rectangular nullity, dependent-row null residual, diagonal
  threshold behavior, rank-deficient economy, and sparse-mode rank-deficient
  parity.
- Defined nullity as `n - rank` only when the fixture pins its rank threshold
  and expected rank; Sprint 125 must not introduce a global QR rank threshold.
- Defined raw nullspace vector equality as unsuitable for external evidence
  unless a future fixture proves sign, ordering, and orientation are unique and
  stable.
- Selected projection/subspace metrics as the preferred policy for Day 5:
  compare projectors `P = Z Z^T` or use two-way projection residuals rather
  than basis-column equality.
- Preserved `||A*v||` null residual as a valid per-vector diagnostic, not a
  complete subspace-equivalence proof.
- Day 4 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files and touched
  maintainer/test/helper files.

## Day 5 Notes

- Accepted one bounded nullspace/subspace evidence fixture:
  `qr_rankdef_duplicate_5x4_nullspace_projector`.
- Reused the same 5x4 duplicate-column matrix as the existing Sprint 124/125
  rank-only and residual-only lanes, with expected rank 3, nullity 1, and
  threshold 0.0.
- Extended `tests/qr_external_dense_reference.py` with a standard-library
  reference projector for the null vector `[0, -1/sqrt(2), 0, 1/sqrt(2)]`.
- Extended `tests/test_qr.c` with a projector comparison that normalizes the
  product nullspace vector and compares `P = z z^T` against the external
  reference instead of comparing raw basis vectors.
- Preserved `||A*v||` as a diagnostic and secondary correctness check, not a
  complete subspace-equivalence proof.
- Updated `docs/maintainer_guide.md` to include the bounded nullspace
  projector fixture while keeping broad QR, raw Q-basis, Q-sign/orientation,
  nullspace, minimum-norm, economy, sparse-mode, reorder, backend, corpus, and
  performance non-claims.
- Deferred dependent-row, multi-dimensional, wide-shape, near-rank-threshold,
  and SuiteSparse nullspace/subspace evidence until their metric, threshold,
  skip, and support-tier gates are explicit.
- Focused validation passed:
  `python3 -m py_compile tests/qr_external_dense_reference.py`,
  `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_nullspace_projector`,
  and `make build/test_qr && ./build/test_qr`.
- Full required quality validation passed: `make format && make lint &&
  make test`.

## Day 6 Notes

- Inventoried current QR threshold behavior in `tests/test_qr.c`,
  `include/sparse_qr.h`, and `src/sparse_qr.c`.
- Recorded that `sparse_qr_rank()` and `sparse_qr_rank_info()` use relative
  threshold semantics: explicit `tol > 0` maps to
  `tol * abs(R(0,0))`, while `tol <= 0` uses
  `eps * max(m,n) * abs(R(0,0))`.
- Selected the diagonal bucket ladder as the preferred Day 7 threshold-family
  candidate: diagonal `[1, 1e-8, 1e-12, 0]` with expected ranks 3, 2, and 1
  at tolerances `1e-14`, `1e-10`, and `1e-6`.
- Defined scaled diagonal, perturbed duplicate-column, dependent-row,
  wide-shape, and SuiteSparse near-threshold families as lower-priority or
  deferred candidates with promotion gates.
- Required any accepted threshold evidence to name fixture key, thresholds,
  expected ranks, scale, strict comparison rule, `R` diagonal diagnostics, and
  absolute thresholds.
- Preserved fixture-local interpretation only: Day 6 does not create a global
  QR rank-threshold policy, dense-library threshold parity, residual claim,
  nullspace basis claim, minimum-norm claim, pseudoinverse claim, corpus claim,
  backend claim, performance claim, or public API claim.
- Day 6 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files and touched
  maintainer/test/helper files.

## Day 7 Notes

- Accepted one bounded rank-only threshold fixture:
  `qr_rank_threshold_diag4_family`.
- Added a standard-library external helper path that emits `OK 6` with
  threshold/rank pairs for the diagonal ladder `[1, 1e-8, 1e-12, 0]`.
- Added a QR factorization test that compares `sparse_qr_rank()` and
  `sparse_qr_rank_info()` against expected ranks 3, 2, and 1 at tolerances
  `1e-14`, `1e-10`, and `1e-6`.
- Printed fixture-local diagnostics for each threshold: relative tolerance,
  absolute threshold, expected rank, product rank, rank-info rank, and `R`
  diagonal magnitudes.
- Updated `docs/maintainer_guide.md` with the bounded threshold-rank fixture
  and an explicit no-global-rank-threshold non-claim.
- Deferred scaled diagonal, perturbed duplicate-column, dependent-row,
  wide-shape, and SuiteSparse near-threshold evidence to later owners with
  promotion gates.
- Focused validation passed:
  `python3 -m py_compile tests/qr_external_dense_reference.py`,
  `python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_family`,
  and `make build/test_qr && ./build/test_qr`.
- Full required quality validation passed: `make format && make lint &&
  make test`.

## Day 8 Notes

- Inventoried the checked-in SuiteSparse corpus under
  `tests/data/suitesparse` and separated default checked-in matrices,
  optional-large matrices, and report-only/heavy matrices.
- Confirmed current QR-related SuiteSparse coverage uses `west0067`, `nos4`,
  and `bcsstk04` as solve, reconstruction, refine, economy, sparse-mode, or
  reorder controls, not as documented rank-deficient QR evidence.
- Recorded that `bcsstk04` currently asserts full rank in QR solve coverage,
  so it must remain a full-rank control unless future work explicitly changes
  the expected-rank claim.
- Preserved the existing `SPARSE_TEST_LARGE=1` convention for large
  SuiteSparse paths and required missing-data skips to apply only to
  explicitly optional tests.
- Defined Day 9 acceptance gates for any SuiteSparse rank-deficient QR
  evidence: named matrix, support tier, threshold semantics, expected rank or
  threshold/rank pairs, diagnostics, skip behavior, focused validation, and
  full quality-gate requirements for code changes.
- Required diagnostics to include matrix identity, path, shape, nnz, support
  tier, load/factorization status, rank values, threshold context, `R`
  diagonal summary, and residual or reconstruction metrics when relevant.
- Preserved SuiteSparse QR non-claims for broad corpus correctness, backend or
  dense-library parity, performance, global rank-threshold policy, nullspace,
  minimum-norm, pseudoinverse, platform, package, ABI, and public API support.
- Day 8 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files and touched
  maintainer/test/helper files.

## Day 9 Notes

- Explicitly deferred SuiteSparse rank-deficient QR evidence because no
  checked-in SuiteSparse matrix has a documented, small, rank-deficient QR
  expectation with pinned rank, threshold, nullity, residual semantics, and
  support-tier behavior.
- Ran `./build/test_qr_solve` as the focused diagnostic path and confirmed the
  current default SuiteSparse QR controls report full ranks: `nos4` rank 100,
  `bcsstk04` rank 132, and `west0067` rank 67.
- Rejected reusing `nos4`, `bcsstk04`, or `west0067` as rank-deficient
  evidence because they are current full-rank controls, with `bcsstk04`
  already asserting `qr.rank == n`.
- Deferred `steam1`, `fs_541_1`, `orsirr_1`, and heavier report-only corpus
  matrices until a future owner pins expected-rank metadata, support tier,
  diagnostics, and skip behavior before test registration.
- Preserved existing SuiteSparse QR controls as controls only and kept bounded
  Sprint 125 rank-deficient proof on the synthetic fixture lanes:
  `qr_rankdef_duplicate_5x4_residual_only`,
  `qr_rankdef_duplicate_5x4_nullspace_projector`, and
  `qr_rank_threshold_diag4_family`.
- Preserved SuiteSparse QR non-claims for broad corpus correctness, backend or
  dense-library parity, performance, global rank-threshold policy, broad
  rank-deficient QR behavior, nullspace, minimum-norm, pseudoinverse,
  platform, package, ABI, and public API support.
- Day 9 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files and touched
  maintainer/test/helper files.

## Day 10 Notes

- Inventoried current QR minimum-norm ownership across `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, `tests/test_svd.c`, and
  `tests/qr_external_dense_reference.py`.
- Kept the completed `qr_underdetermined_minnorm_2x4` fixture as the only
  accepted exact external QR minimum-norm lane and marked it complete rather
  than duplicating it.
- Split the remaining minimum-norm work into behavior owner keys for COLAMD,
  fallback, rank-deficient, refinement, zero-row, QR-vs-SVD-pseudoinverse, and
  SuiteSparse submatrix evidence.
- Assigned Day 11 to core QR minimum-norm behavior decisions in
  `tests/test_colamd.c` and `tests/test_qr_solve.c`, with focused validation
  expected through `test_colamd` and `test_qr_solve` before the full quality
  gate if code changes.
- Assigned Day 12 to QR-vs-SVD-pseudoinverse and SuiteSparse minimum-norm
  decisions, with separate SVD cross-check and corpus support-tier gates.
- Preserved helper boundaries: measurement helpers may be considered only with
  behavior-specific names, while generic `assert_minnorm`, `check_minnorm`, and
  `minnorm_oracle` patterns remain rejected.
- Preserved non-claims around broad QR minimum-norm parity, global optimality,
  SVD-pseudoinverse-as-global-oracle, COLAMD/reorder/fallback/refinement/
  rank-deficient/SuiteSparse superiority, backend parity, public API, platform,
  performance, and corpus behavior.
- Day 10 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 125 files and touched
  maintainer/test/helper files.

## Day 11 Notes

- Accepted a bounded core minimum-norm evidence batch in `tests/test_colamd.c`
  without adding a generic minimum-norm helper or changing external-reference
  protocols.
- Strengthened `test_minnorm_with_colamd` with exact COLAMD-path solution
  values `[0.75, 0.75, 1.5, 0.75, 0.75]` and norm `sqrt(4.5)`.
- Strengthened `test_minnorm_fallback_overdetermined` with exact ordinary
  fallback solution values `[1, 2, 3]` in addition to residual.
- Strengthened `test_minnorm_rank_deficient` with expected rank-deficient
  minimum-norm values `[0.5, 0.5, 0.5, 0.5]` and norm `1.0`.
- Strengthened `test_refine_minnorm` with a bounded post-refinement residual
  and solution norm check while preserving residual non-increase semantics.
- Strengthened `test_minnorm_zero_row` with zero-row residual, expected
  solution values `[1, 0, 1, 0]`, and norm `sqrt(2)`.
- Deferred larger-shape, QR-vs-SVD-pseudoinverse, SuiteSparse submatrix, and
  helper-movement lanes to their Day 12 or future owners.
- Focused validation passed:
  `make build/test_colamd && ./build/test_colamd` and
  `make build/test_qr_solve && ./build/test_qr_solve`.
- Full required validation passed: `make format && make lint && make test`,
  followed by `git diff --check` and the focused trailing-whitespace scan.

## Day 12 Notes

- Accepted `qr_minnorm_vs_svd_pinv_crosscheck` as a bounded cross-check, not a
  global SVD oracle for QR minimum-norm behavior.
- Strengthened `test_minnorm_vs_pinv` so QR and SVD-pseudoinverse solutions
  both equal `[0.5, 0.5, 0.5, 0.5]`, both have norm `1.0`, both satisfy
  `A*x=b`, and both match each other.
- Accepted `qr_minnorm_suitesparse_submatrix` as a default checked-in corpus
  smoke on the first 30 rows of `west0067.mtx`, with explicit 30 x 67 shape,
  residual below `1e-8`, positive solution norm, and
  `||x_min|| <= ||ones|| + 1e-8`.
- Kept optional-large SuiteSparse minimum-norm, additional QR-vs-SVD fixtures,
  SuiteSparse rank-deficient minimum-norm corpus, and generic helper movement
  deferred with promotion gates.
- Focused validation passed:
  `make build/test_colamd && ./build/test_colamd` and
  `make build/test_svd && ./build/test_svd`.
- Full required validation passed: `make format && make lint && make test`,
  followed by `git diff --check`, the focused trailing-whitespace scan, and a
  Python-cache scan.

## Day 13 Notes

- Inventoried the Sprint 125 changed surfaces:
  `tests/qr_external_dense_reference.py`, `tests/test_qr.c`,
  `tests/test_qr_solve.c`, `tests/test_colamd.c`,
  `docs/maintainer_guide.md`, and Sprint 125 planning artifacts.
- Refreshed the maintainer QR evidence row so it names
  `tests/test_colamd.c` as the owner for bounded owner-local minimum-norm
  lanes and keeps SVD-pseudoinverse-as-global-oracle and broad SuiteSparse
  corpus claims fenced.
- Audited public/support wording in `docs/solver_selection.md`, `README.md`,
  and public headers and made no public wording changes because the Sprint 125
  evidence remains fixture-scoped or owner-local.
- Focused helper validation passed:
  `python3 -m py_compile tests/qr_external_dense_reference.py`,
  `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_residual_only`,
  `python3 tests/qr_external_dense_reference.py qr_rankdef_duplicate_5x4_nullspace_projector`,
  and
  `python3 tests/qr_external_dense_reference.py qr_rank_threshold_diag4_family`.
- Focused executable validation passed:
  `make build/test_qr && ./build/test_qr`,
  `make build/test_qr_solve && ./build/test_qr_solve`,
  `make build/test_colamd && ./build/test_colamd`, and
  `make build/test_svd && ./build/test_svd`.
- Full required validation passed: `make format && make lint && make test`,
  followed by `git diff --check`, the focused trailing-whitespace scan, and a
  Python-cache scan.

## Day 14 Notes

- Reconciled all seven Sprint 125 project-plan items against Days 1-13
  artifacts and marked each complete through bounded implementation, explicit
  policy, or explicit deferral with promotion gates.
- Published the Day 14 closeout and handoff artifact with the accepted
  evidence package, validation baseline, residual owner queue, final
  non-claim register, and Sprint 126 inputs.
- Published the Sprint 125 retrospective with definition-of-done checklist,
  final metrics, movement and claim outcomes, residual deferred debt, and key
  deliverables.
- Confirmed Day 14 is documentation-only: no source, header, helper script,
  build metadata, package metadata, public API, README, or solver-selection
  wording changed.
- The full code quality baseline remains the Day 13
  `make format && make lint && make test` pass.
- Final Day 14 validation passed: `git diff --check`, focused
  trailing-whitespace scan, and Python-cache scan.
