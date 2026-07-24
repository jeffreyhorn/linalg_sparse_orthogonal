# Sprint 130 Working Notes

## Sprint Goal

Expand or explicitly defer Sprint 124 partial-SVD residual evidence under
dedicated metric policies, then refresh public solver-selection wording only
where earned.

Sprint 130 is a partial-SVD residual, subspace, corpus, optimality,
convergence-budget, and solver-selection claim-gate sprint. It should not
reopen Sprint 129 QR Q-basis, economy, sparse-mode, SuiteSparse Q/economy,
minimum-norm helper, or Bidiagonal/Golub-Kahan helper boundaries unless a
Sprint 130 partial-SVD item directly depends on one of those decisions.

## Starting Constraints

- Treat `partial_svd_vector_residual_diag6_k2` as the only completed bounded
  external partial-SVD vector-residual baseline from Sprint 124.
- Treat `partial_svd_diag6_k2` and `partial_svd_tall_diag_8x5_k3` as bounded
  external singular-value evidence, not vector/subspace evidence.
- Do not duplicate Sprint 124 internal vector, reconstruction, rectangular,
  rank-deficient, SuiteSparse smoke, timing, or low-rank tests as new external
  residual claims without a distinct metric and claim boundary.
- Repeated, clustered, and rank-deficient cases require subspace, projection,
  rank/nullity, and threshold policies before implementation.
- SuiteSparse and optional corpus evidence require support-tier, skip,
  diagnostics, runtime, and failure-interpretation policy before
  implementation.
- Low-rank optimality evidence must state whether it checks Frobenius,
  spectral-norm, reconstruction, sparse-output, or drop-tolerance behavior.
- Convergence-budget evidence must state options, iteration cap, tolerance,
  partial-result semantics, and budget-failure meaning before implementation.
- Public solver-selection wording may change only when accepted evidence earns
  bounded user-facing language; otherwise Sprint 130 should publish a
  no-update rationale.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and a focused
  markdown whitespace scan over Sprint 130 files.

## Input Artifact Inventory

| Input | Role in Sprint 130 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 130 | Defines seven Sprint 130 items for partial-SVD residual expansion and solver-selection claim gating. |
| `docs/planning/EPIC_11/SPRINT_130/PLAN.md` | Provides day-level execution order and 166-hour budget. |
| Sprint 124 Day 8 artifact | Defines vector residual, subspace, projection, tolerance, skip, and failure policies for partial-SVD evidence. |
| Sprint 124 Day 9 artifact | Accepts `partial_svd_vector_residual_diag6_k2` and defers subspace, rectangular, corpus, low-rank, and convergence lanes. |
| Sprint 124 Day 10 artifact | Provides the residual scenario matrix for repeated, clustered, rank-deficient, rectangular, corpus, low-rank, convergence, and nonsymmetric lanes. |
| Sprint 124 Day 11 artifact | Provides the residual deferral package and future-owner promotion gates. |
| Sprint 129 Day 14 artifact | Hands off Sprint 130 as the partial-SVD residual expansion and solver-selection claim gate. |
| `tests/test_svd.c` | Main SVD and partial-SVD registration owner. |
| `tests/test_svd_partial_helpers.h` | Primary partial-SVD helper and bounded evidence owner. |
| `tests/svd_external_dense_reference.py` | External dense-reference singular-value helper; protocol expansion requires a metric gate first. |
| `src/sparse_svd_partial.c` | Partial-SVD implementation owner; touched only after evidence and convergence semantics are pinned. |
| `include/sparse_svd.h` | Public partial-SVD API wording owner; not changed without solver-selection or API claim gate approval. |
| `docs/maintainer_guide.md` | Maintainer evidence table and solver-selection wording owner for accepted bounded evidence or no-update rationale. |

## Completed Baseline

| Evidence | Owner | What it proves | Boundary |
| --- | --- | --- | --- |
| `partial_svd_diag6_k2` | `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | Bounded external top-2 singular-value agreement for a square diagonal fixture. | Value-only; no vector, subspace, convergence, low-rank, or solver-selection claim. |
| `partial_svd_tall_diag_8x5_k3` | `tests/test_svd_partial_helpers.h`, `tests/svd_external_dense_reference.py` | Bounded external top-3 singular-value agreement for a tall diagonal fixture. | Value-only rectangular evidence; no vector residual or subspace claim. |
| `partial_svd_vector_residual_diag6_k2` | `tests/test_svd_partial_helpers.h` | Bounded sign-invariant triplet residual and orthogonality evidence for one exact square diagonal fixture. | No repeated, clustered, rank-deficient, SuiteSparse, low-rank optimality, convergence-budget, or broad vector/subspace claim. |
| Internal partial-SVD value, vector, rectangular, rank-deficient, SuiteSparse, reconstruction, low-rank, and timing tests | `tests/test_svd.c`, `tests/test_svd_partial_helpers.h` | Internal regression and smoke coverage for current implementation behavior. | Internal consistency only; not independent external parity or solver-selection proof. |

## Deferred Residual Lanes

| Lane | Current state | Sprint 130 owner |
| --- | --- | --- |
| Rectangular vector residual | Deferred from Sprint 124 after the square exact residual lane landed. | Days 3-4 |
| Nonsymmetric rectangular residual | Deferred pending dense-reference fixture and value/residual boundary. | Days 5-6 |
| Repeated-spectrum subspace | Deferred pending projector or principal-angle metrics. | Days 7-8 |
| Clustered-spectrum subspace | Deferred pending gap, ordering, convergence, and projector policy. | Days 7-8 |
| Rank-deficient subspace | Deferred pending rank threshold, zero singular-value tolerance, range/null-space split, and projection metrics. | Days 9-10 |
| SuiteSparse corpus residual | Deferred pending optional-data, conditioning, residual windows, support tier, and skip behavior. | Days 11-12 |
| Low-rank optimality | Deferred pending Frobenius/spectral/reconstruction metric and sparse-output semantics. | Day 12 |
| Convergence budget | Deferred pending option surface, iteration cap, tolerance, deterministic start or partial-result semantics, and failure meaning. | Day 13 |
| Solver-selection wording | Deferred until evidence earns bounded public wording beyond current workflow guidance. | Day 14 |

## Day-Level Ownership

| Day | Owner Focus | Project-Plan Items |
| --- | --- | --- |
| 1 | Sprint intake, residual dedupe baseline, artifact directory, owner map, duplicate fence, validation boundary | Items 1-7 |
| 2 | Partial-SVD dedupe and metric map | Item 1 |
| 3 | Rectangular residual gate | Item 2 |
| 4 | Rectangular residual evidence or explicit deferral | Item 2 |
| 5 | Nonsymmetric rectangular residual gate | Item 2 |
| 6 | Nonsymmetric rectangular evidence or explicit deferral | Item 2 |
| 7 | Repeated and clustered spectrum policy | Item 3 |
| 8 | Repeated and clustered spectrum evidence or explicit deferral | Item 3 |
| 9 | Rank-deficient subspace gate | Item 4 |
| 10 | Rank-deficient subspace evidence or explicit deferral | Item 4 |
| 11 | SuiteSparse corpus evidence gate | Item 5 |
| 12 | SuiteSparse corpus and low-rank optimality evidence or explicit deferral | Item 5 |
| 13 | Convergence-budget evidence or explicit deferral | Item 6 |
| 14 | Solver-selection wording gate, closeout, non-claim update, and Sprint 131 handoff | Item 7 and Items 1-6 reconciliation |

## Validation Expectations

| Change Type | Required Validation |
| --- | --- |
| Documentation only | `git diff --check` and focused markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_130`. |
| `.c` or `.h` edits | `make format && make lint && make test`. |
| Python external-reference helper edits | `python3 -m py_compile tests/svd_external_dense_reference.py`, focused helper invocation, affected test executable, and `git diff --check`. |
| Partial-SVD helper/test edits | Focused `make build/test_svd && ./build/test_svd`, plus full quality gate if `.c` or `.h` files changed. |
| SuiteSparse or optional-data evidence edits | Present/missing data behavior, skip-path proof, residual diagnostics, support-tier/runtime note, and required focused/full validation. |
| Maintainer or public solver-selection wording edits | Evidence-to-claim traceability, non-claim scan, link/path hygiene, and documentation hygiene. |
| Public API wording edits | Header comment review, maintainer wording gate, focused compile, and full validation if headers change. |

## Scope Boundaries

- Sprint 130 may add bounded partial-SVD residual, subspace, corpus,
  optimality, or convergence evidence only after metric, oracle, tolerance,
  diagnostics, support tier, skip behavior, and failure interpretation are
  explicit.
- Sprint 130 may explicitly defer a lane when the blocker, dependency, future
  owner, and promotion gate are recorded.
- Sprint 130 must not describe internal full-SVD comparison, product-observed
  SuiteSparse values, or timing smoke as independent external evidence.
- Sprint 130 must not update public solver-selection wording from workflow
  guidance into a stronger claim without a direct evidence-to-wording trace.
- Sprint 130 must preserve QR/helper boundaries closed by Sprint 129 unless a
  partial-SVD lane explicitly depends on them and records why.

## Day 1 Notes

- Created the Sprint 130 working-notes baseline.
- Created the Sprint 130 artifact directory and Day 1 artifact.
- Re-read the Sprint 130 project-plan section and mapped Items 1-7 to
  day-level owners.
- Reviewed Sprint 124 partial-SVD vector/subspace semantics, accepted
  `partial_svd_vector_residual_diag6_k2` evidence, residual scenario matrix,
  and deferral package.
- Reviewed Sprint 129 closeout and handoff to preserve QR/helper boundaries
  before partial-SVD work starts.
- Recorded completed-versus-deferred partial-SVD evidence so future days do
  not duplicate Sprint 124 baselines silently.
- Established validation expectations for documentation, partial-SVD tests,
  external dense-reference helper changes, SuiteSparse optional evidence,
  maintainer wording, public solver-selection wording, and public API wording.

## Day 2 Notes

- Converted the Day 1 completed-versus-deferred baseline into a metric,
  tolerance, oracle, diagnostics, and failure-interpretation policy.
- Classified rectangular vector residual, nonsymmetric rectangular residual,
  repeated-spectrum subspace, clustered-spectrum subspace, rank-deficient
  subspace, SuiteSparse corpus residual, low-rank optimality,
  convergence-budget, and solver-selection wording lanes.
- Confirmed current `tests/svd_external_dense_reference.py` output is
  singular-value only; it is not a vector, projector, convergence, or
  low-rank optimality oracle.
- Defined vector-residual evidence as product-owned triplet residuals plus
  `U`/`V` orthogonality, not raw singular-vector equality.
- Defined subspace evidence as projector, principal-angle, or two-way
  projection residual evidence with explicit left/right/range/null-space
  ownership.
- Required fixture-specific tolerances for non-exact rectangular,
  repeated-spectrum, clustered-spectrum, rank-deficient, SuiteSparse,
  low-rank, and convergence-budget lanes.
- Required every later evidence artifact to name the evidence class, fixture,
  metric, tolerance, oracle, diagnostics, failure class, duplicate fence,
  non-claims, and validation plan before implementation.
- Preserved the default no-update posture for public solver-selection wording
  until accepted Sprint 130 evidence earns bounded user-facing language.

## Day 3 Notes

- Applied the Day 2 metric policy to current tall and wide rectangular
  partial-SVD surfaces.
- Confirmed `partial_svd_tall_diag_8x5_k3` is already an external
  singular-value fixture, but value-only and not vector residual evidence.
- Confirmed the existing wide vector smoke checks only `A v` residuals and
  product-owned values; it is not a complete rectangular triplet-residual or
  external evidence lane.
- Accepted `partial_svd_vector_residual_tall8x5_k3` as the Day 4 candidate
  because it adds tall shape triplet-residual evidence beyond the Sprint 124
  square `partial_svd_vector_residual_diag6_k2` baseline while reusing an
  existing singular-value oracle.
- Required the Day 4 lane to check external singular values, both triplet
  residual equations, `U`/`V` orthogonality, and `m/n/k` shape diagnostics
  under `1e-8` exact-diagonal tolerances.
- Deferred wide rectangular residual, existing wide smoke upgrade,
  rectangular low-rank reconstruction, and nonsymmetric rectangular residual
  lanes to their later owners.
- Preserved no public solver-selection update for Day 3 and kept maintainer
  wording changes gated on Day 4 validation.

## Day 4 Notes

- Implemented `partial_svd_vector_residual_tall8x5_k3` in
  `tests/test_svd_partial_helpers.h`.
- Reused the existing `partial_svd_tall_diag_8x5_k3` external singular-value
  fixture; no Python helper protocol changed.
- Registered the new test in `tests/test_svd.c` next to the existing
  partial-SVD external and square vector-residual fixtures.
- Checked the accepted Day 3 metrics: singular-value agreement, both triplet
  residual equations, `U`/`V` orthogonality, and `m/n/k` shape diagnostics.
- Updated `docs/maintainer_guide.md` to name the bounded tall rectangular
  vector-residual fixture while preserving no-claims for broad rectangular,
  nonsymmetric, subspace, low-rank, convergence-budget, and solver-selection
  behavior.
- Deferred wide rectangular residual, existing wide smoke upgrade,
  rectangular low-rank reconstruction, and nonsymmetric rectangular residual
  evidence to their later owners.

## Day 5 Notes

- Reviewed current nonsymmetric rectangular partial-SVD coverage and confirmed
  `test_partial_svd_nonsymmetric` is internal consistency evidence against
  this library's full SVD, not independent external residual evidence.
- Confirmed Day 4's exact diagonal tall residual lane does not cover
  non-diagonal or nonsymmetric rectangular behavior.
- Accepted a staged Day 6 path for `partial_svd_nonsym_rect10x8_k4` using the
  existing deterministic 10x8 non-diagonal matrix from
  `test_partial_svd_nonsymmetric`.
- Required Day 6 to add an external singular-value helper fixture before
  promoting the lane beyond product-internal comparison.
- Preferred Day 6 metrics are external top-4 singular values plus both
  triplet residual equations, U/V orthogonality, and shape diagnostics.
- Required fixture-specific tolerance and top-4 gap diagnostics; exact
  diagonal `1e-8` tolerance cannot be copied onto the nonsymmetric fixture
  without evidence.
- Forbid raw vector equality and keep sign/orientation/multiplicity issues in
  residual or subspace metrics.
- Deferred wide nonsymmetric, nonsymmetric subspace, nonsymmetric low-rank,
  nonsymmetric convergence-budget, and public solver-selection wording to
  later owners.

## Day 6 Notes

- Preflighted the Day 5 `partial_svd_nonsym_rect10x8_k4` candidate and found
  the fourth through seventh singular values form a near-zero clustered tail.
- Narrowed Day 6 implementation to the stable top-3 lane
  `partial_svd_vector_residual_nonsym_rect10x8_k3` to avoid turning clustered
  tail behavior into individual-vector residual evidence.
- Added `partial_svd_nonsym_rect10x8_k3` to the external dense-reference
  helper and SVD fixture whitelist.
- Added a shared 10x8 nonsymmetric fixture builder and reused it in the
  existing internal nonsymmetric partial-SVD test.
- Registered the new nonsymmetric rectangular vector-residual test with
  external top-3 singular-value agreement, both triplet residual equations,
  U/V orthogonality, and shape diagnostics.
- Updated `docs/maintainer_guide.md` with bounded nonsymmetric rectangular
  evidence names while preserving no-claims for broad rectangular,
  nonsymmetric, subspace, low-rank, convergence-budget, and solver-selection
  behavior.
- Deferred the original top-4 candidate, wide nonsymmetric residual,
  nonsymmetric subspace, rank-deficient, low-rank, convergence-budget, and
  public solver-selection lanes to their later owners.

## Day 7 Notes

- Reviewed repeated and clustered spectrum policy from Sprint 124 and current
  SVD/partial-SVD tests.
- Confirmed current repeated-spectrum coverage is full-SVD value-only and does
  not provide partial-SVD projector or principal-angle evidence.
- Confirmed existing partial-SVD vector comparisons are limited to separated
  spectra or internal product cross-checks; they should not be reused for
  repeated or clustered claims.
- Defined repeated/clustered subspace evidence around left and right
  projector errors, optional principal-angle diagnostics, singular-value gap
  policy, triplet residuals, orthogonality, and shape checks.
- Accepted `partial_svd_repeated_diag6_k3_projector` as the lowest-risk Day 8
  implementation path using analytic projectors for
  `diag(7, 7, 7, 3, 2, 1)` with `k=3`.
- Deferred partial selection through repeated multiplicity, clustered
  diagonal evidence, Day 6 near-zero clustered-tail evidence, corpus
  clustered spectra, and public solver-selection wording until their owners
  define containment, gap, budget, rank, corpus, and claim policies.

## Day 8 Notes

- Attempted the Day 7 accepted `partial_svd_repeated_diag6_k3_projector`
  lane for `diag(7, 7, 7, 3, 2, 1)` with `k=3`.
- Focused preflight failed the value/projector evidence gate:
  `sigma=5.000e+00`, `PU=2.000e+00`, and `PV=2.000e+00`, even though triplet
  residuals and orthogonality were below `1e-8`.
- Removed the attempted projector helper/test and registration instead of
  landing a failing or weakened repeated-spectrum evidence lane.
- Left `docs/maintainer_guide.md` without any repeated/clustered SVD evidence
  addition because no Day 8 evidence was accepted.
- Recorded explicit deferrals for the repeated projector lane, partial
  selection through repeated multiplicity, clustered diagonal evidence, the
  Day 6 near-zero clustered tail, corpus clustered spectra,
  rank-deficient subspace behavior, convergence-budget behavior, and public
  solver-selection wording.

## Day 9 Notes

- Reviewed current rank-deficient SVD and partial-SVD coverage across external
  full-SVD values, rank API checks, threshold fixtures, QR/SVD rank
  consistency, and internal partial-SVD zero-slot value tests.
- Confirmed `svd_rankdef_duplicate_5x4` is full-SVD singular-value evidence,
  not partial-SVD range/null-space projector evidence.
- Confirmed `test_partial_svd_rank_deficient` is internal value coverage that
  requests `k=4` and crosses into zero singular slots, so it should not be
  silently upgraded into a subspace claim.
- Defined rank-deficient evidence policy around declared numerical rank,
  positive singular-value lower bounds, zero singular-value tolerance,
  left/right nullity boundaries, range projectors, triplet residuals,
  orthogonality, and shape checks.
- Preferred a Day 10 first lane,
  `partial_svd_rankdef_diag6x4_k2_range_projector`, using an exact diagonal
  6x4 fixture with rank `2`, requested `k=2`, and analytic left/right range
  projectors.
- Deferred zero-crossing `k > rank` evidence, duplicate-column projector
  evidence without a clear left projector oracle, existing partial-SVD
  rank-deficient test upgrades, Day 6 near-zero nonsymmetric tail behavior,
  minimum-norm/pseudoinverse behavior, and public solver-selection wording.

## Day 10 Notes

- Implemented `partial_svd_rankdef_diag6x4_k2_range_projector` for an exact
  6x4 diagonal fixture with positive entries `9` and `6`, expected rank `2`,
  and requested `k=2`.
- Added left and right coordinate range-projector error helpers for the
  current partial-SVD U/Vt storage layout.
- Checked rank with `sparse_svd_rank(A, 1e-8)` before constructing the
  range-projector evidence.
- Kept the accepted evidence range-only: the fixture has right nullity `2`
  and left nullity `4`, but Day 10 does not assert null-space basis or
  projector behavior.
- Registered the new test with the bounded partial-SVD evidence lanes and
  updated `docs/maintainer_guide.md` with a bounded rank-deficient
  range-projector fixture name.
- Deferred `k > rank` zero-crossing, null-space projectors, duplicate-column
  projector evidence, existing `test_partial_svd_rank_deficient` upgrades,
  Day 6 near-zero nonsymmetric tail behavior, minimum-norm/pseudoinverse
  behavior, and public solver-selection wording.

## Day 11 Notes

- Inventoried checked-in SuiteSparse matrices from `tests/data/suitesparse`
  and classified them by shape, nnz, Matrix Market symmetry, runtime tier,
  and evidence readiness.
- Classified existing partial-SVD corpus tests (`nos4`, `west0067`, and
  low-rank corpus-safety checks) as product-regression smoke because their
  expected values come from product full-SVD output or product path-to-path
  comparisons.
- Defined support tiers for local analytic fixtures, checked-in smoke data,
  checked-in expensive data, and optional external corpus data.
- Required future corpus evidence to declare skip behavior, matrix metadata,
  independent oracle source, diagnostics, tolerances, failure class, and
  runtime class before promotion.
- Selected a Day 12 path that avoids overstating corpus support:
  `partial_svd_lowrank_diag6x4_k2_frobenius_optimality`, a local analytic
  dense-reconstruction low-rank fixture with exact discarded-tail Frobenius
  error.
- Deferred SuiteSparse residual parity, SuiteSparse vector/projector parity,
  large-matrix SVD lanes, sparse-output/drop-tolerance optimality, broad
  low-rank optimality, convergence-budget behavior, and public
  solver-selection wording.

## Day 12 Notes

- Applied the Day 11 corpus gate and did not promote SuiteSparse residual,
  vector, projector, or low-rank corpus checks beyond product smoke.
- Implemented `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` as a
  Tier 0 local analytic fixture.
- Used a 6x4 diagonal matrix with singular values `9`, `6`, `3`, and `1`,
  requested `k=2`, and checked dense reconstruction error against the
  independent discarded-tail target `sqrt(3^2 + 1^2) = sqrt(10)`.
- Kept the metric dense-reconstruction-only and paired it with retained
  singular-value, triplet residual, and U/V orthogonality diagnostics.
- Updated `docs/maintainer_guide.md` with the bounded dense low-rank
  Frobenius fixture name while preserving sparse-output/drop-tolerance,
  SuiteSparse, broad optimality, convergence-budget, and solver-selection
  non-claims.
- Deferred SuiteSparse corpus parity, large-matrix SVD lanes, sparse
  low-rank output optimality, broad best-rank approximation wording, and
  convergence-budget behavior.

## Day 13 Notes

- Inventoried the partial-SVD convergence API and found that
  `sparse_svd_opts_t` exposes `max_iter` and `tol`, while `sparse_svd_t`
  does not expose iteration count, achieved tolerance, residual history,
  `n_converged`, or partial-result validity fields.
- Implemented `partial_svd_max_iter_fail_closed_diag6_k2` as the bounded
  convergence-budget lane.
- Used a 6x6 analytic diagonal fixture with requested `k=2`; with
  `max_iter=1`, the fixture returns `SPARSE_ERR_NOT_CONVERGED` and publishes
  no `sigma`, `U`, or `Vt` payload.
- Verified the same fixture recovers under the default budget with expected
  retained singular values and vector residual diagnostics.
- Updated `docs/maintainer_guide.md` with the bounded fail-closed fixture
  and preserved convergence-rate, partial-result, SuiteSparse,
  solver-selection, and platform non-claims.
- Deferred iteration-count evidence, achieved-tolerance reporting,
  partial-result publication, stagnation behavior, clustered/corpus
  convergence budgets, and public solver-selection wording.

## Day 14 Notes

- Reconciled Sprint 130 Items 1-7 against the project-plan checklist and
  separated accepted fixture-bounded evidence from deferred claim lanes.
- Confirmed Sprint 130 earned maintainer-facing evidence for tall rectangular
  vector residuals, nonsymmetric rectangular vector residuals, rank-deficient
  range projectors, local analytic Frobenius low-rank optimality, and
  max-iteration fail-closed behavior.
- Confirmed Sprint 130 did not earn broad public solver-selection wording
  beyond current workflow guidance because repeated/clustered spectra,
  SuiteSparse corpus parity, null-space behavior, sparse-output optimality,
  convergence-rate reporting, partial-result semantics, and platform claims
  remain deferred.
- Published the Day 14 claim-closeout artifact with the final evidence index,
  deferral register, validation package, and Sprint 131 handoff owners.
- Added the Sprint 130 retrospective as the sprint-level closeout index.
- Kept public solver-selection wording unchanged; `docs/maintainer_guide.md`
  remains the only public repository surface refreshed with the new bounded
  evidence names.
