# Sprint 122 Working Notes

## Sprint Goal

Convert Sprint 121's residual SVD, QR, partial-SVD, helper-ownership, and
solver-selection claim-gate decisions into explicit proof owners before the
Epic 11 corpus, adoption, packaging, and closeout work depends on them.

## Starting Constraints

- Treat Sprint 121 outputs as the source of truth for residual scope.
- Do not reopen completed Sprint 121 audit, taxonomy, helper extraction,
  fixture expansion, or first SVD external-reference pilot work unless a
  Sprint 122 item explicitly identifies a gap.
- Keep SVD external fixture decisions separate from QR external dense-reference
  design and partial-SVD external parity design.
- Define fixture size, tolerance, skip behavior, failure interpretation, trust
  boundary, and non-claim wording before adding any external oracle lane.
- Preserve the existing non-claims around broad LAPACK, SciPy, NumPy,
  SuiteSparse, PETSc, Trilinos, Eigen, package, ABI, performance, scalability,
  public API, and state-of-the-art parity.
- If any `.c` or `.h` file changes, run `make format && make lint && make test`
  before closeout. Documentation-only changes require `git diff --check` and
  focused whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 122 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 122 | Defines residual oracle, helper, documentation, and validation items. |
| `docs/planning/EPIC_11/SPRINT_122/PLAN.md` | Provides day-level execution order and time budget. |
| `docs/planning/EPIC_11/SPRINT_121/RETROSPECTIVE.md` | Defines residual deferred debt and duplicate fences from Sprint 121. |
| `docs/planning/EPIC_11/SPRINT_121/WORKING_NOTES.md` | Captures day-by-day proof-owner findings and non-claim notes. |
| Sprint 121 Day 2-3 artifacts | Source audit for SVD, partial-SVD, QR, rank-deficient, and minimum-norm proof ownership. |
| Sprint 121 Day 4 artifact | Matrix taxonomy and evidence-class source for future oracle fixture keys. |
| Sprint 121 Day 5-7 artifacts | Helper extraction plan and SVD/QR helper boundaries. |
| Sprint 121 Day 8-10 artifacts | Deterministic fixture expansion baseline for rank-deficient, least-squares, pseudoinverse, low-rank, and partial-SVD evidence. |
| Sprint 121 Day 11-12 artifacts | Bounded SVD external-reference pilot and explicit external parity non-claims. |
| Sprint 121 Day 13-14 artifacts | Validation package, deferred validation queue, and closeout residual map. |
| Sprint 120 artifacts | Fixture and oracle patterns to reuse if Sprint 122 selects implementation work. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, residual source map, duplicate fence, validation boundary | Items 1-7 |
| 2 | Residual oracle dedupe and owner map | Item 1 |
| 3 | Additional SVD external fixture decision criteria | Item 2 |
| 4 | Additional SVD external fixture implementation or explicit deferral package | Item 2 |
| 5 | QR external dense-reference lane constraints and fixture decision | Item 3 |
| 6 | QR external dense-reference lane implementation or explicit deferral package | Item 3 |
| 7 | Partial-SVD external parity semantics and oracle design | Item 4 |
| 8 | Partial-SVD external parity implementation or explicit deferral package | Item 4 |
| 9 | Minimum-norm helper ownership decision | Item 5 |
| 10 | Bidiagonal/Golub-Kahan helper boundary decision | Item 5 |
| 11 | Solver-selection claim inventory and evidence gate | Item 6 |
| 12 | Solver-selection wording update or explicit no-update rationale | Item 6 |
| 13 | Validation package, non-claim scan, and residual queue | Item 7 |
| 14 | Sprint closeout, retrospective inputs, and future-sprint handoff | Item 7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused trailing-whitespace scan of Sprint 122 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Script or generated helper edits | Focused syntax check plus affected behavior check. |
| Makefile, CMake, or CTest membership edits | Source-list inspection and relevant CMake/CTest proof, including Windows count impact if test membership changes. |
| External-reference fixture edits | Focused helper invocation, affected test executable, skip-path proof, and failure interpretation note. |
| Public documentation wording edits | Claim scan against current evidence and explicit non-claim list. |

## Scope Boundaries

- Sprint 122 may design, implement, or explicitly defer bounded residual oracle
  lanes, but it must not imply broad dense-library or package parity.
- Additional SVD external work is limited to bounded fixture diversity beyond
  `svd_rect_fullrank_6x4` if the design proves the extra fixture adds evidence.
- QR external-reference work must stay separate from current deterministic QR
  and least-squares fixture expansion.
- Partial-SVD external parity must handle vector, subspace, ordering,
  convergence, and tolerance semantics separately from full-SVD singular-value
  parity.
- Minimum-norm and Bidiagonal/Golub-Kahan helper movement must preserve visible
  scenario ownership and specialized semantics.
- Solver-selection wording may only advance when evidence gates support the
  wording; otherwise the output is an explicit no-update rationale.

## Day 1 Notes

- Created the Sprint 122 artifact directory.
- Established the working-notes baseline.
- Mapped Sprint 121 residual deferred debt to Sprint 122 day-level owners.
- Recorded duplicate fences for completed Sprint 121 work.
- Set validation expectations for documentation, code, build metadata, and
  public wording changes.

## Day 2 Notes

- Converted the Sprint 121 residual deferred debt into six active residual
  owners: SVD external fixtures, QR external dense-reference design,
  partial-SVD external parity design, minimum-norm helper ownership,
  Bidiagonal/Golub-Kahan helper boundaries, and solver-selection claim gates.
- Rejected completed Sprint 121 audit, taxonomy, helper extraction, fixture
  expansion, SVD pilot, validation, and closeout work as duplicates rather than
  unresolved Sprint 122 work.
- Classified broad parity, platform, ABI, performance, public API, and
  state-of-the-art statements as non-claim constraints, not implementation
  items.
- Added dependency order and proof-gate expectations so later days can decide
  whether to implement bounded lanes or defer them with auditable rationale.

## Day 3 Notes

- Reviewed the existing `svd_rect_fullrank_6x4` external-reference pilot and
  confirmed it covers one non-diagonal rectangular full-column-rank
  singular-value comparison only.
- Inventory filtered candidate SVD fixtures against Sprint 121 deterministic
  coverage and rejected exact diagonal, repeated-spectrum, low-rank,
  pseudoinverse, condition-number, SuiteSparse, vector/subspace, and
  performance-shaped candidates as duplicates or out of scope for this item.
- Identified one strongest Day 4 candidate:
  `svd_rankdef_duplicate_5x4_external_sigma`, a small non-diagonal rectangular
  exact-rank-deficient singular-value fixture using the existing pure-Python
  Gram/Jacobi reference path.
- Recorded fixture-size, tolerance, skip behavior, failure interpretation, and
  non-claim criteria for Day 4 implementation or explicit deferral.

## Day 4 Notes

- Accepted and implemented the bounded
  `svd_rankdef_duplicate_5x4` external dense-reference fixture.
- Extended `tests/svd_external_dense_reference.py` with the rank-deficient
  5x4 fixture while retaining Python standard-library-only reference behavior.
- Added one `test_svd` case that compares positive singular values against the
  external reference and separately asserts the zero-tail singular value stays
  below `1e-8`.
- Preserved Makefile, CMake, CTest, public docs, public API, package, ABI,
  platform, performance, and broad external parity non-claims.
- Focused validation passed:
  `python3 tests/svd_external_dense_reference.py svd_rankdef_duplicate_5x4`,
  `make format`, and `make build/test_svd && ./build/test_svd`.
- Full branch validation later passed after Day 6:
  `make lint` and `make test`.

## Day 5 Notes

- Reviewed Sprint 121 QR and least-squares audit and expansion artifacts before
  any QR external implementation decision.
- Classified current QR proof inputs into square solve, overdetermined
  compatible, overdetermined incompatible, underdetermined minimum-norm,
  rank-deficient, reconstruction, orthogonality, refinement, economy/sparse
  mode, and reorder-adjacent surfaces.
- Identified `qr_overdetermined_incompatible_4x2_external_ls` as the strongest
  Day 6 QR external dense-reference candidate because it adds bounded
  independent least-squares residual evidence without reopening deterministic
  Sprint 121 fixture expansion.
- Kept QR external parity as a non-claim pending Day 6 design, implementation,
  or explicit deferral.

## Day 6 Notes

- Accepted and implemented the bounded
  `qr_overdetermined_incompatible_4x2` external least-squares reference lane.
- Added `tests/qr_external_dense_reference.py`, a Python standard-library-only
  helper that emits the least-squares solution and residual norm for one fixed
  4x2 incompatible tall fixture.
- Added one `test_qr_solve` case that compares the QR solve result and reported
  residual against the external reference inside the existing executable.
- Preserved Makefile, CMake, CTest, public docs, public API, package, ABI,
  platform, performance, minimum-norm, rank-deficient, Q-basis, and broad QR
  parity non-claims.
- Focused validation passed:
  `python3 tests/qr_external_dense_reference.py qr_overdetermined_incompatible_4x2`
  and `make format && make build/test_qr_solve && ./build/test_qr_solve`.
- Full branch validation passed: `make lint` and `make test`.

## Day 7 Notes

- Inventoried current partial-SVD evidence and confirmed it remains primarily
  internal-reference proof against this library's full SVD plus deterministic
  vector and reconstruction checks.
- Separated partial-SVD external semantics from full-SVD singular-value parity:
  top-k value agreement, vector residuals, subspace angles, ordering,
  convergence budgets, and degenerate spectra need separate decision gates.
- Identified `partial_svd_diag6_k2_external_sigma` as the lowest-risk Day 8
  candidate because it can validate top-k singular values without claiming
  vector/subspace parity.
- Deferred vector/subspace, SuiteSparse, repeated-spectrum, convergence-budget,
  and low-rank optimality external lanes until their semantics have explicit
  owners.

## Day 8 Notes

- Accepted and implemented the bounded `partial_svd_diag6_k2` top-k external
  singular-value lane.
- Extended `tests/svd_external_dense_reference.py` to emit exactly the top two
  singular values for the selected partial-SVD fixture.
- Added `test_partial_svd_external_dense_reference_diag6_k2` under the existing
  partial-SVD helper owner and registered it inside `test_svd`.
- Kept vector, subspace, convergence-budget, repeated-spectrum, SuiteSparse,
  low-rank optimality, and broad partial-SVD external parity claims explicitly
  unsupported.
- Focused validation passed:
  `python3 tests/svd_external_dense_reference.py partial_svd_diag6_k2` and
  `make format && make build/test_svd && ./build/test_svd`.
- Full branch validation passed: `make lint` and `make test`.

## Day 9 Notes

- Inventoried minimum-norm ownership across `tests/test_qr_solve.c`,
  `tests/test_colamd.c`, `tests/test_svd.c`, `tests/test_qr_helpers.h`, and
  `tests/test_svd_helpers.h`.
- Decided not to migrate minimum-norm helpers in Sprint 122 because the current
  proof owners intentionally encode separate QR solve, COLAMD/reordering,
  SVD pseudoinverse, refinement, rank-deficient, fallback, and SuiteSparse
  semantics.
- Defined future migration boundaries for a QR solve / minimum-norm
  consolidation owner, including behavior-specific helper names and explicit
  tolerance inputs.
- Preserved current test membership, Makefile, CMake, CTest, production
  source, public docs, package, ABI, platform, and support-level non-claims.
- Day 9 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 122 docs.

## Day 10 Notes

- Inventoried Bidiagonal and Golub-Kahan ownership across `tests/test_bidiag.c`,
  `tests/test_svd.c`, `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h`,
  `include/sparse_bidiag.h`, `include/sparse_svd.h`, and `src/sparse_svd.c`.
- Decided not to consolidate Bidiagonal/Golub-Kahan helpers into the general
  SVD helper layer in Sprint 122.
- Preserved specialized wide-matrix transpose handling, implicit Householder
  reconstruction, explicit extracted-`U`/`V` reconstruction, bidiagonal QR
  iteration, and fixture-specific tolerance ownership at the scenario level.
- Defined a future limited extraction path for a Bidiagonal/Golub-Kahan-specific
  helper owner, limited to named measurement helpers and fixture builders.
- Completed Sprint 122 Item 5 helper-boundary work together with Day 9's
  minimum-norm decision.
- Day 10 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 122 docs.

## Day 11 Notes

- Audited public and support wording across `README.md`,
  `docs/solver_selection.md`, `docs/tutorial.md`, `examples/README.md`,
  `benchmarks/README.md`, `INSTALL.md`, and `docs/maintainer_guide.md`.
- Compared current wording with Sprint 121 residuals and Sprint 122 SVD, QR,
  partial-SVD, minimum-norm, and Bidiagonal/Golub-Kahan decisions.
- Recorded an evidence-to-wording matrix for QR external evidence, SVD dense
  reference evidence, partial-SVD external top-k evidence, helper consolidation,
  cross-solver oracle evidence, benchmark claims, and platform/package claims.
- Preserved unsupported external dense-library parity, vector/subspace parity,
  broad cross-solver parity, portable performance, platform, package, ABI,
  public API expansion, and state-of-the-art claims as explicit non-claims.
- Defined Day 12's claim-gate checklist: either publish a no-update rationale
  or make only evidence-linked maintainer/support wording updates.
- Day 11 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 122 docs.

## Day 12 Notes

- Reviewed the Day 11 solver-selection claim inventory and evidence-to-wording
  matrix.
- Decided not to expand public solver-selection wording in Sprint 122 because
  the added SVD, QR, and partial-SVD external lanes are bounded fixture proof,
  not family-wide external parity or support-level promotion.
- Recorded explicit claim gates for SVD external evidence, QR external evidence,
  partial-SVD external evidence, minimum-norm helper ownership,
  Bidiagonal/Golub-Kahan helper ownership, cross-solver parity,
  support-level platform wording, performance/state-of-the-art wording, and
  package/ABI wording.
- Preserved README, solver-selection, tutorial, examples, benchmark, install,
  maintainer, public header, package, CMake, Makefile, and CI wording unchanged.
- Completed Sprint 122 Item 6 claim-gate work together with Day 11's inventory.
- Day 12 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 122 docs.

## Day 13 Notes

- Inventoried all Sprint 122 touched surfaces: planning docs, external-reference
  Python helpers, QR/SVD test owners, and the partial-SVD helper header.
- Re-ran external-helper protocol checks for `svd_rankdef_duplicate_5x4`,
  `partial_svd_diag6_k2`, and `qr_overdetermined_incompatible_4x2`.
- Re-ran focused C test owners:
  `make build/test_qr_solve && ./build/test_qr_solve` and
  `make build/test_svd && ./build/test_svd`.
- Recorded the full C quality gate already passed after the last C/header
  change: `make format`, `make lint`, and `make test`.
- Packaged downstream residual handoffs for future SVD oracle, QR oracle,
  partial-SVD numerical oracle, helper migration, Bidiagonal/Golub-Kahan
  extraction, public wording refresh, and maintainer evidence-table refresh.
- Day 13 changed documentation only after the focused checks; final validation
  used `git diff --check` and a focused trailing-whitespace scan over Sprint
  122 docs.

## Day 14 Notes

- Reviewed all Sprint 122 artifacts and confirmed Days 1-13 are represented in
  the artifact index.
- Confirmed all Sprint 122 project-plan items have completion dispositions:
  Items 1, 6, and 7 completed through planning/validation artifacts; Items 2,
  3, and 4 completed with bounded external-reference lanes; Item 5 completed
  with explicit helper-boundary deferrals.
- Published the final Sprint 122 non-claim register covering broad external
  parity, vector/subspace parity, cross-solver parity, performance,
  platform/package/ABI support, public API expansion, and state-of-the-art
  claims.
- Published dependency-ordered residual deferred debt for future SVD oracle, QR
  oracle, partial-SVD numerical oracle, minimum-norm helper migration,
  Bidiagonal/Golub-Kahan helper extraction, maintainer evidence refresh, and
  public solver-selection wording refresh owners.
- Packaged retrospective-ready evidence: bounded oracle expansion, preserved
  helper ownership semantics, no public wording expansion, and completed
  validation.
- Day 14 changed documentation only; validation used `git diff --check` and a
  focused trailing-whitespace scan over Sprint 122 docs.
