# Sprint 129 Working Notes

## Sprint Goal

Resolve the remaining Sprint 124 QR Q-basis/economy and helper ownership debt
in dependency order, preserving basis and helper semantics before the
corpus/index architecture consumes them.

Sprint 129 intentionally does not continue grinding Sprint 128 residual QR
debt unless a Q/economy/helper item has a distinct behavior-specific claim and
satisfies the Sprint 128 promotion gate.

## Starting Constraints

- Treat Sprint 129 project-plan scope as Q-basis, economy, sparse-mode,
  SuiteSparse Q/economy, minimum-norm helper ownership, and
  Bidiagonal/Golub-Kahan helper ownership.
- Do not reopen Sprint 128 compatible zero-residual, wide residual-only,
  near-threshold subspace, SuiteSparse rank-deficient QR corpus,
  SuiteSparse/optional-large minimum-norm, extra exact underdetermined, or
  extra QR-vs-SVD residual debt unless directly required by a Sprint 129
  Q/economy/helper claim and the promotion gate is satisfied first.
- Preserve raw Q-basis non-claims: no sign, orientation, ordering, unique-basis,
  or raw-basis parity unless a fixture-local rule makes equality meaningful.
- Prefer shape, orthogonality, reconstruction, projection, projector, or
  principal-angle metrics over raw basis comparison.
- Keep QR solve, COLAMD/minimum-norm, SVD-pseudoinverse, Bidiagonal,
  Golub-Kahan, sparse-mode, SuiteSparse, optional-data, and public wording
  owners separate unless a behavior-specific helper movement proves otherwise.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and focused
  markdown whitespace validation.

## Input Artifact Inventory

| Input | Role in Sprint 129 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 129 | Defines seven Sprint 129 items for Q-basis/economy policy, raw Q evidence, rank-deficient Q/nullspace evidence, wide economy/sparse-mode evidence, SuiteSparse Q/economy evidence, minimum-norm helper movement, and Bidiagonal/Golub-Kahan helper extraction. |
| `docs/planning/EPIC_11/SPRINT_129/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| Sprint 124 Day 6 artifact | Source for Q-basis/economy sign, orientation, projection, subspace, economy-shape, and owner policies. |
| Sprint 124 Day 7 artifact | Source for accepted `qr_economy_projector_5x3` evidence and Q/economy decision gates. |
| Sprint 124 Day 12 artifact | Source for minimum-norm and Bidiagonal/Golub-Kahan helper movement deferrals and naming policies. |
| Sprint 125-128 retrospectives and artifacts | Source for rank-deficient QR, nullspace/subspace, threshold, SuiteSparse, optional-large, minimum-norm, QR-vs-SVD, and helper claim gates. |
| Sprint 128 retrospective residual queue | Source for no-reopen boundary and end-of-epic deferred QR residual queue. |
| `tests/test_qr.c` | Primary owner for Q formation, application, orthogonality, economy, sparse-mode, rank, nullspace, reconstruction, and QR mode behavior. |
| `tests/test_qr_solve.c` | Owner for QR solve behavior and solve-oriented external fixtures; not a default Q-basis/economy host. |
| `tests/test_colamd.c` | Owner for behavior-specific minimum-norm, COLAMD, fallback, refinement, zero-row, QR-vs-SVD-pseudoinverse, and SuiteSparse submatrix lanes. |
| `tests/test_svd.c` | Owner for SVD pseudoinverse, Golub-Kahan extraction, bidiagonal QR iteration, and full/partial SVD checks. |
| `tests/test_bidiag.c` | Owner for Bidiagonal reduction, implicit Householder reconstruction, wide-transpose behavior, and Bidiag lifecycle. |
| `tests/qr_external_dense_reference.py` | May grow Q/economy helper protocols only after metric, shape, tolerance, skip, and non-claim rules are explicit. |
| `tests/test_qr_helpers.h`, `tests/test_svd_helpers.h`, `tests/test_svd_partial_helpers.h` | Existing helper surfaces; future movement must preserve behavior-specific names and call-site tolerances. |
| `docs/maintainer_guide.md` | Maintainer evidence table; update only for accepted bounded evidence or helper movement. |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, no-reopen boundary, duplicate fence, owner map, validation boundary | Items 1-7 |
| 2 | Q-basis and economy policy refresh | Item 1 |
| 3 | Raw Q-column evidence decision or explicit deferral | Item 2 |
| 4 | Rank-deficient Q/nullspace policy gate | Item 3 |
| 5 | Rank-deficient Q/nullspace evidence or explicit deferral | Item 3 |
| 6 | Wide economy and sparse-mode policy | Item 4 |
| 7 | Wide economy and sparse-mode evidence or explicit deferral | Item 4 |
| 8 | SuiteSparse Q/economy gate | Item 5 |
| 9 | SuiteSparse Q/economy evidence decision or explicit deferral | Item 5 |
| 10 | Minimum-norm helper ownership gate | Item 6 |
| 11 | Minimum-norm helper movement or explicit deferral | Item 6 |
| 12 | Bidiagonal/Golub-Kahan helper gate | Item 7 |
| 13 | Bidiagonal/Golub-Kahan helper extraction or explicit deferral | Item 7 |
| 14 | Sprint closeout, non-claim update, and Sprint 130 handoff | Items 1-7 |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused markdown whitespace scan over Sprint 129 files. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Python external-reference helper edits | `python3 -m py_compile` for the helper, focused helper invocation, affected test executable, and `git diff --check`. |
| QR Q/economy test edits | Focused `make build/test_qr && ./build/test_qr`, plus full quality gate if `.c` or `.h` files changed. |
| QR solve/minimum-norm helper edits | Focused QR solve, COLAMD, and SVD checks as applicable, plus full quality gate if `.c` or `.h` files changed. |
| Bidiagonal/Golub-Kahan helper edits | Focused Bidiagonal and SVD checks, source-list/build impact checks if ownership moves, plus full quality gate if `.c` or `.h` files changed. |
| SuiteSparse or optional-data evidence edits | Focused present/missing behavior, skip-path proof, diagnostics, runtime/support-tier note, and required quality gates for touched code. |
| Maintainer or public wording edits | Evidence-to-claim traceability, claim-boundary scan, link/path hygiene, and explicit non-claim update. |

## Scope Boundaries

- Sprint 129 may add bounded Q/economy/helper evidence only after metric,
  shape, tolerance, support-tier, diagnostics, and failure interpretation are
  explicit.
- Sprint 129 may explicitly defer work when the future owner, dependency, and
  promotion gate are recorded.
- Sprint 129 must not relabel Sprint 128 residual, subspace, threshold,
  SuiteSparse, optional-large, or minimum-norm deferrals as immediate work.
- Sprint 129 must not update public solver-selection wording unless closeout
  proves evidence supports bounded user-facing wording beyond current guidance.
- Sprint 129 must not create generic helper APIs that blur QR solve, COLAMD,
  SVD-pseudoinverse, Bidiagonal, Golub-Kahan, sparse-mode, SuiteSparse, or
  optional-data ownership.

## Day 1 Notes

- Created the Sprint 129 working-notes baseline.
- Created the Sprint 129 artifact directory and Day 1 artifact entry.
- Mapped every Sprint 129 project-plan item to a day-level owner.
- Recorded duplicate fences for completed Sprint 124-128 Q/economy,
  nullspace/subspace, threshold, SuiteSparse, optional-large, minimum-norm,
  QR-vs-SVD, and helper evidence.
- Established the Sprint 128 residual no-reopen rule: residual QR debt remains
  in the end-of-epic queue unless a Sprint 129 Q/economy/helper candidate has a
  distinct behavior-specific claim and satisfies the promotion gate before
  implementation.
- Established validation expectations for documentation, C code, Python helper,
  QR Q/economy, QR solve/minimum-norm, Bidiagonal/Golub-Kahan, SuiteSparse,
  maintainer, and public wording changes.

## Day 2 Notes

- Refreshed the Sprint 124 Q-basis/economy policy for the Sprint 129 scope.
- Preserved `qr_economy_projector_5x3` as the completed bounded economy
  projector baseline, not a candidate to repeat as raw Q evidence.
- Reconfirmed raw Q-column equality is rejected by default and may proceed only
  for a non-degenerate fixture with explicit sign normalization, column order,
  storage layout, permutation interpretation, tolerance, diagnostics, and
  distinct trust value.
- Defined the preferred metric order for Q/economy evidence: shape,
  orthogonality, reconstruction, projection, projector distance, and
  principal-angle bounds before raw Q values.
- Defined economy and sparse-mode output policy for tall full-rank, tall
  rank-deficient, square, wide, sparse-mode, and SuiteSparse surfaces.
- Kept SuiteSparse Q/economy evidence deferred by default until Day 8-9 can
  pin matrix metadata, support tier, skip behavior, runtime expectations,
  expected shapes, metrics, and diagnostics.
- Preserved the Sprint 128 no-reopen boundary for compatible zero-residual,
  wide residual-only, near-threshold, SuiteSparse rank-deficient corpus,
  optional-large minimum-norm, extra exact, and extra QR-vs-SVD residual debt.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation for
  Day 2.

## Day 3 Notes

- Applied the Day 2 raw Q-column acceptance gate to full-rank tall, economy,
  rank-deficient, wide, sparse-mode, and SuiteSparse raw Q candidates.
- Explicitly deferred full-rank tall raw Q-column evidence because existing
  orthogonality, reconstruction, and Q-application checks already cover the
  durable behavior without pinning implementation-specific sign/orientation.
- Explicitly deferred economy raw Q-column evidence based on
  `qr_economy_projector_5x3` because the accepted projector lane already
  provides basis-invariant trust and raw values would mostly duplicate it.
- Rejected rank-deficient raw Q equality as a metric; projector, projection, or
  principal-angle metrics remain required for rank-deficient subspace claims.
- Deferred wide and sparse-mode raw Q values to the Days 6-7 economy/sparse-mode
  gates, where shape and product metrics must be pinned first.
- Rejected SuiteSparse raw Q evidence for Day 3 because support-tier, skip,
  runtime, and independent expected-basis metadata are not available.
- Preserved the Sprint 128 residual no-reopen boundary; no residual,
  threshold, SuiteSparse corpus, optional-large, extra exact, or extra
  QR-vs-SVD item was pulled back into Sprint 129.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation for
  Day 3.

## Day 4 Notes

- Reviewed the Sprint 125-128 rank-deficient nullspace/subspace projector and
  threshold evidence lanes before defining the Day 4 gate.
- Recorded completed-evidence fences for duplicate-column nullity-1,
  rank-1/nullity-2, dependent-row, wide rank-deficient, and threshold-family
  rank evidence so Day 5 does not repackage them as new work.
- Rejected raw rank-deficient Q or nullspace basis equality because valid
  deficient bases can rotate, reorder, or change sign without changing the
  represented subspace.
- Defined the accepted rank-deficient metric order: rank/nullity metadata,
  null residual, orthonormality, full projector for tiny fixtures, two-way
  projection residual for wider or larger fixtures, and principal-angle bounds
  only when projector/projection metrics are insufficient.
- Marked dependent-row Q-application projection as the only tentative Day 5
  candidate, and only if it proves Q-specific behavior without duplicating the
  existing dependent-row nullspace projector.
- Deferred wide economy, sparse-mode, near-threshold, SuiteSparse corpus, and
  minimum-norm candidates to their later Sprint 129 owners or the end-of-epic
  queue.
- Preserved the Sprint 128 no-reopen boundary for compatible zero-residual,
  wide residual-only, near-threshold, SuiteSparse rank-deficient corpus,
  optional-large, extra exact, and extra QR-vs-SVD debt.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation for
  Day 4.

## Day 5 Notes

- Applied the Day 4 rank-deficient Q/nullspace gate and accepted one
  non-duplicate Q-application evidence lane.
- Added `test_qr_dependent_row_q_transpose_column_space_rhs` to
  `tests/test_qr.c`.
- Reused `tf_qr_make_dependent_row_4x3()` with expected rank `2`, threshold
  `0.0`, and RHS `b = 2*A(:,0) - A(:,1) = [2, -1, 1, 5]^T`.
- Checked that `Q^T b` has negligible residual-tail norm after the product
  rank and that `Q * (Q^T b)` round-trips to `b`.
- Kept the claim Q-specific and solve-adjacent; it is not a new nullspace
  projector, residual-only solve, minimum-norm, raw basis, economy,
  sparse-mode, or SuiteSparse claim.
- Deferred duplicate-column projector, rank-1/nullity-2 projector, raw
  deficient-basis, wide economy, sparse-mode, near-threshold, SuiteSparse
  corpus, and minimum-norm candidates.
- Did not change Python helpers, Matrix Market data, build files, maintainer
  guide text, public API wording, or public documentation for Day 5.

## Day 6 Notes

- Reviewed the current wide Q, economy, sparse-mode, and SuiteSparse-adjacent
  QR coverage before defining the Day 6 policy.
- Recorded completed-evidence fences for wide Q orthogonality, tall economy
  projector evidence, economy solve equivalence, economy shape smokes,
  sparse-mode dense/sparse solve agreement, sparse-mode Q orthogonality,
  sparse-mode reconstruction, and the Sprint 128 wide nullspace projector.
- Defined wide economy candidates and kept raw Q, residual-only, and
  minimum-norm lanes rejected or deferred.
- Defined sparse-mode Q/economy candidates and identified tall sparse-mode
  plus economy behavior as the preferred Day 7 implementation candidate if it
  can satisfy shape, product metric, tolerance, diagnostics, and non-claim
  requirements.
- Pinned output-shape policy for tall economy, tall rank-deficient economy,
  square economy, wide economy, sparse-mode Q/economy, and SuiteSparse
  Q/economy.
- Defined Day 7 acceptance requirements for fixture ownership, matrix shape,
  rank, mode flags, R shape, formed-Q shape, metric choice, tolerance,
  diagnostics, and non-claims.
- Preserved the Sprint 128 no-reopen boundary for compatible zero-residual,
  wide residual-only, near-threshold, SuiteSparse rank-deficient corpus,
  optional-large, extra exact, and extra QR-vs-SVD debt.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation for
  Day 6.

## Day 7 Notes

- Applied the Day 6 wide economy and sparse-mode policy and accepted one
  bounded sparse-mode plus economy evidence lane.
- Added `test_sparse_mode_economy_tall_q_shape` to `tests/test_qr.c`.
- Reused `tf_qr_make_tall_diagonal_dominant(24, 6, 8.0, 0.25, 1)` with dense
  economy and sparse-mode economy QR options.
- Checked rank `6`, 6 x 6 R shape, 24 x 6 sparse-mode economy thin-Q
  orthogonality, and dense economy versus sparse economy solve/residual
  equivalence.
- Deferred wide economy shape/orthogonality, sparse-mode wide economy,
  wide/economy nullspace projection, wide residual-only, wide or sparse-mode
  minimum-norm, raw basis, SuiteSparse Q/economy, and sparse-mode performance
  lanes.
- Kept the claim behavior-specific: this is sparse-mode economy Q/R shape and
  product evidence, not raw basis, residual-only, minimum-norm, SuiteSparse,
  backend, platform, performance, or broad sparse QR parity evidence.
- Did not change Python helpers, Matrix Market data, build files, maintainer
  guide text, public API wording, or public documentation for Day 7.

## Day 8 Notes

- Inventoried checked-in SuiteSparse Matrix Market fixtures relevant to
  Q/economy behavior and recorded dimensions, nnz, existing QR-adjacent
  coverage, and support tiers.
- Classified `west0067`, `nos4`, and `bcsstk04` as checked-in small
  smoke/control candidates; `steam1`, `fs_541_1`, and `orsirr_1` as
  checked-in non-default candidates; and larger checked-in matrices as
  report-only Q/economy candidates unless a separate runtime budget is
  recorded.
- Reconfirmed that product-observed Q, R, rank, residual, solve, fill, and
  timing values are controls, not independent oracle values.
- Marked `nos4` square economy Q orthogonality and `nos4` sparse-mode economy
  Q orthogonality as the only tentatively promotable Day 9 lanes, each still
  requiring pinned shape, metric, tolerance, diagnostics, runtime posture, and
  non-claim wording before code changes.
- Deferred `west0067`, `bcsstk04`, large SuiteSparse Q/economy, raw
  SuiteSparse Q-column, SuiteSparse rank-deficient corpus, and SuiteSparse
  minimum-norm candidates to later owners or end-of-epic queues unless their
  promotion gates are satisfied first.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation
  for Day 8.

## Day 9 Notes

- Applied the Day 8 SuiteSparse Q/economy gate and accepted one checked-in
  small-control corpus lane.
- Added `test_suitesparse_nos4_sparse_mode_economy_q_orthogonality` to
  `tests/test_qr.c`.
- Used checked-in `tests/data/suitesparse/nos4.mtx` metadata to pin 100 x 100
  Q and R shape under `economy = 1` plus `sparse_mode = 1`.
- Checked sparse-mode economy formed-Q orthogonality with `Q^T Q ~= I` as the
  primary basis-invariant metric.
- Kept dense-economy versus sparse-mode-economy rank, solve, and residual
  agreement as controls only, not independent oracle values.
- Deferred `nos4` economy-only Q shape, `west0067`, `bcsstk04`, large
  SuiteSparse Q/economy, raw SuiteSparse Q-column, SuiteSparse rank-deficient
  corpus, and SuiteSparse minimum-norm lanes to later owners or end-of-epic
  queues unless their promotion gates are satisfied first.
- Did not change Python helpers, Matrix Market data, build files, maintainer
  guide text, public API wording, or public documentation for Day 9.

## Day 10 Notes

- Reviewed current minimum-norm owners across QR solve, COLAMD/minimum-norm,
  SVD pseudoinverse, fallback, refinement, zero-row, QR-vs-SVD cross-check,
  and SuiteSparse submatrix smoke lanes.
- Recorded that QR solve owns the external dense-reference
  `qr_underdetermined_minnorm_2x4` fixture, COLAMD owns most behavior-specific
  minimum-norm lanes, and SVD owns pseudoinverse/Moore-Penrose helper logic.
- Identified repeated 2 x 4 split-constraint fixture construction as the only
  tentative Day 11 movement candidate, but only under a behavior-specific
  helper name with expected values, tolerances, and diagnostics kept visible
  at call sites.
- Rejected generic `tf_minnorm_*` fixture or assertion helpers because they
  would blur QR solve, COLAMD, SVD pseudoinverse, fallback, refinement,
  zero-row, and SuiteSparse ownership.
- Deferred SVD pseudoinverse application helpers and SuiteSparse submatrix
  builders unless a future owner-specific promotion gate proves they reduce
  real duplication without hiding layout/support-tier details.
- Defined Day 11 focused validation requirements for QR solve, COLAMD, and
  SVD owner checks, plus the full `make format && make lint && make test`
  gate for any `.c` or `.h` edit.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation
  for Day 10.

## Day 11 Notes

- Applied the Day 10 minimum-norm helper ownership gate to the tentative 2 x 4
  fixture-builder movement candidate.
- Found that the concrete 2 x 4 fixture layouts differ by owner: QR solve and
  SVD pseudoinverse use `row0: x0 + x1`, `row1: x2 + x3`, while the
  COLAMD/minimum-norm exact lane uses `row0: x0 + x2`, `row1: x1 + x3`.
- Explicitly deferred helper movement because a shared helper would require a
  generic topology parameter or QR/SVD/COLAMD-neutral naming that hides owner
  semantics.
- Kept QR solve expected values, SVD pseudoinverse storage-layout comments,
  COLAMD owner-local options, tolerances, and diagnostics visible at call
  sites.
- Deferred QR-solve-local, COLAMD-local, SVD-local, SVD pseudoinverse apply,
  and SuiteSparse submatrix helper movement to future owner-specific gates.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation
  for Day 11.

## Day 12 Notes

- Reviewed Bidiagonal and Golub-Kahan helper ownership across
  `tests/test_bidiag.c`, `tests/test_svd.c`, `tests/test_svd_helpers.h`, and
  `tests/test_svd_partial_helpers.h`.
- Recorded that `bidiag_reconstruction_error` owns implicit Householder replay,
  `sparse_bidiag_t` transpose recursion, and Bidiagonal-reduction
  reconstruction diagnostics.
- Recorded that `gk_reconstruction_error` owns explicit Golub-Kahan `U`/`V`,
  `diag`, and `superdiag` reconstruction inside SVD tests, including
  owner-local wide-matrix scoping.
- Rejected any shared Bidiagonal/GK reconstruction helper because it would hide
  the difference between implicit Householder replay and explicit-vector
  Golub-Kahan products.
- Marked a Bidiagonal-owned extraction of `bidiag_reconstruction_error` as the
  only tentative Day 13 movement candidate, and only if the current transpose,
  cleanup, comparison, and diagnostic semantics remain visible and unchanged.
- Deferred GK helper movement, SVD helper movement, QR-iteration helper
  movement, and partial-SVD helper movement to their owner-specific gates.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation
  for Day 12.

## Day 13 Notes

- Applied the Day 12 Bidiagonal/Golub-Kahan helper gate and moved exactly one
  helper: the Bidiagonal reconstruction helper from `tests/test_bidiag.c` into
  `tests/test_bidiag_helpers.h`.
- Renamed the helper to `tf_bidiag_reconstruction_max_error` so the owner is
  explicit and not confused with Golub-Kahan, full SVD, QR-iteration, or
  partial-SVD helper ownership.
- Preserved the existing implicit Householder replay order, transposed
  `sparse_bidiag_t` recursion, `sparse_get_phys` comparison path, cleanup
  behavior, dense reconstruction layout, and call-site diagnostics.
- Left `gk_reconstruction_error` in `tests/test_svd.c`; explicit Golub-Kahan
  `U`/`V`, `diag`, `superdiag`, and wide-matrix scoping remain SVD-owner
  semantics.
- Did not move SVD helper headers, partial-SVD helper headers, QR-iteration
  helpers, Matrix Market fixtures, production code, public API declarations,
  maintainer guide text, or build source lists.
- Recorded the source-list no-change rationale: the new helper is header-only
  and included by the existing `test_bidiag` target, so no `Makefile` or
  `CMakeLists.txt` registration is required.
- Passed focused validation with `make build/test_bidiag && ./build/test_bidiag`
  and `make build/test_svd && ./build/test_svd`.
- Passed the required full quality gate with
  `make format && make lint && make test`.

## Day 14 Notes

- Reconciled all seven Sprint 129 project-plan items against the daily
  artifacts and recorded each item as implemented or explicitly deferred.
- Published the final evidence index for the dependent-row QR `Q^T b` lane,
  tall sparse-mode economy Q-shape lane, `nos4` sparse-mode economy Q
  orthogonality lane, and Bidiagonal reconstruction helper movement.
- Published the final deferral index for raw Q-column equality, extra
  projector lanes, wide economy/nullspace interaction, near-threshold
  Q/nullspace, SuiteSparse Q/economy, SuiteSparse rank-deficient QR corpus,
  SuiteSparse and optional-large minimum-norm evidence, owner-specific helper
  movement, Golub-Kahan helper movement, and partial-SVD helper movement.
- Refreshed non-claims so Sprint 129 does not imply raw basis equality,
  Householder orientation stability, dense/sparse backend parity, wide
  minimum-norm behavior, SuiteSparse rank-deficient QR support, generic
  QR/SVD/minimum-norm helper ownership, Golub-Kahan ownership from the
  Bidiagonal helper extraction, or public solver-selection readiness.
- Recorded the Sprint 130 handoff: begin with partial-SVD residual expansion
  and solver-selection claim gates, using Sprint 129 Q/economy/helper evidence
  as closed context rather than a reopened implementation queue.
- Did not change C tests, headers, Python helpers, Matrix Market data, build
  files, maintainer guide text, public API wording, or public documentation
  for Day 14.
