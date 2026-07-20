# Sprint 128 Day 2 Compatible and Wide Residual Semantics Policy

## Purpose

Day 2 defines when compatible zero-residual and wide residual-only QR evidence
can add distinct trust after Sprint 125-127 already accepted bounded
rank-deficient residual, projector, threshold, and exact minimum-norm lanes.

This artifact is policy-only. It does not add tests, external-reference helper
output, Matrix Market data, build membership, maintainer wording, public
solver wording, or claims.

## Inputs Reviewed

| Input | Semantics-Policy Use |
| --- | --- |
| Sprint 128 Plan Day 2 | Requires compatible zero-residual and wide residual-only candidate tables, output semantics, proof value, residual tolerance/diagnostics, and explicit non-claims. |
| Sprint 128 Day 1 artifact | Provides duplicate fences and day-level owners for Sprint 127 carry-forward debt. |
| Sprint 127 Day 2-3 artifacts | Provide compatible zero-residual and wide residual-only trust policy plus explicit deferrals. |
| Sprint 125-126 residual artifacts | Provide completed duplicate-column and dependent-row residual-only baselines. |
| Sprint 125-127 minimum-norm artifacts | Define exact underdetermined, owner-local minimum-norm, SuiteSparse submatrix, exact 3 x 6 and 5 x 10, and QR-vs-SVD boundaries that residual-only evidence must not absorb. |
| Sprint 124-127 Q/economy artifacts | Define Q/economy, sparse-mode, projection, and output-shape semantics that wide residual-only evidence must not imply. |
| `tests/test_qr.c` | Owns deterministic wide QR, Q orthogonality, economy, sparse-mode, rank, nullspace, and refinement coverage. |
| `tests/test_qr_solve.c` | Owns QR solve residual evidence and bounded external QR solve fixtures. |
| `tests/test_colamd.c` | Owns QR minimum-norm, COLAMD, fallback, refinement, QR-vs-SVD-pseudoinverse, exact underdetermined, and SuiteSparse minimum-norm scenarios. |
| `tests/qr_external_dense_reference.py` | Owns current Python standard-library external QR reference protocols. |
| `docs/maintainer_guide.md` QR evidence row | Names maintained QR evidence and the explicit non-claim boundary for residual, minimum-norm, Q/economy, SuiteSparse, and parity claims. |

## Current Evidence Inventory

| Evidence Class | Current Owner | Current Evidence | Day 2 Interpretation |
| --- | --- | --- | --- |
| Duplicate-column residual-only external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_duplicate_5x4_residual_only` compares product residual against a standard-library column-space residual reference. | Completed Sprint 125 residual-only baseline. Do not duplicate as compatible zero-residual or wide residual evidence. |
| Dependent-row residual-only external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_rankdef_dependent_row_4x3_residual_only` compares product residual against a standard-library column-space residual reference. | Completed Sprint 126 second-structure residual-only baseline. Do not duplicate as compatible zero-residual, nullspace, or threshold evidence. |
| Full-rank compatible external fixture | `tests/test_qr_solve.c`, `tests/qr_external_dense_reference.py` | `qr_overdetermined_compatible_5x3` checks solution and near-zero residual for a full-rank compatible system. | Completed compatible full-rank solve evidence. It is not rank-deficient compatible evidence. |
| Deterministic compatible tall solve | `tests/test_qr_solve.c` | `test_qr_solve_overdetermined_compatible_tall` checks compatible tall residual behavior. | Useful baseline; compatible rank-deficient zero-residual evidence must add a distinct rank-deficient diagnostic. |
| Rank-deficient solve smoke | `tests/test_qr_solve.c` | `test_qr_solve_rank_deficient` checks a rank-deficient solve residual under product semantics. | Existing product smoke; not external compatible zero-residual evidence and not minimum-norm evidence. |
| Wide QR factorization and Q evidence | `tests/test_qr.c` | `test_qr_wide`, `test_q_orthogonality_wide`, `test_economy_wide`, `test_sparse_mode_wide`. | Wide shape is covered, but these lanes involve Q/economy/sparse-mode semantics, not residual-only proof. |
| Exact underdetermined and minimum-norm evidence | `tests/test_qr_solve.c`, `tests/test_colamd.c` | Exact 2 x 4, 3 x 6, 5 x 10, owner-local COLAMD/fallback/rank-deficient/refinement, QR-vs-SVD, and `west0067` submatrix evidence. | Separate minimum-norm owner. Residual-only evidence must not assert norm or optimality. |
| SuiteSparse QR controls | `tests/test_qr_solve.c`, Matrix Market fixtures | Checked-in full-rank QR controls and SuiteSparse rank-deficient QR corpus deferrals. | Corpus behavior remains gated by independent expected-rank metadata, support tier, skip behavior, diagnostics, and runtime policy. |

## Compatible Zero-Residual Candidate Table

| Candidate | Candidate Shape | Required Output Semantics | Trust Value | Duplicate/Claim Risk | Day 2 Disposition |
| --- | --- | --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_compatible_zero_residual` | Existing duplicate-column 5 x 4 fixture with RHS in the column space | QR solve returns near-zero residual for a rank-deficient compatible system; no solution-vector or norm oracle. | Low to moderate. Could show compatibility on the same duplicate-column structure as completed rank/residual lanes. | High duplicate risk against existing compatible solve, rank-only, residual-only, and rank-deficient smoke evidence; high solution-selection/minimum-norm confusion risk. | Defer by default. Day 3 may accept only if it proves a distinct diagnostic not already covered. |
| `qr_rankdef_dependent_row_4x3_compatible_zero_residual` | Existing dependent-row 4 x 3 fixture with compatible RHS | QR solve returns near-zero residual for a dependent-row compatible system; rank/dependence are context only. | Low. The incompatible dependent-row residual lane and dependent-row projector lane already add structural evidence. | Overlaps deterministic dependent-row behavior, Sprint 126 incompatible residual evidence, and Sprint 127 projector evidence. | Defer unless Day 3 rejects duplicate-column compatibility and proves a sharper dependent-row diagnostic. |
| New synthetic compatible rank-deficient fixture | New small matrix with pinned rank deficiency and compatible RHS | Near-zero residual only, with independent exact construction. | Unknown until fixture is designed. | Adds new matrix family without need; risks scope creep and duplicate fixture taxonomy. | Defer. Prefer existing fixture builders before adding new families. |
| SuiteSparse compatible zero-residual fixture | Checked-in or optional corpus matrix/submatrix with constructed compatible RHS | Near-zero residual under corpus support-tier semantics. | Potentially useful later. | Requires expected rank/nullity metadata, support tier, optional-data behavior, runtime budget, diagnostics, and corpus claims. | Defer to Days 8-11 corpus and minimum-norm gates. |

## Wide Residual-Only Candidate Table

| Candidate | Candidate Shape | Required Output Semantics | Trust Value | Duplicate/Claim Risk | Day 2 Disposition |
| --- | --- | --- | --- | --- | --- |
| `qr_rankdef_wide_3x5_residual_only` | Existing or small wide 3 x 5 rank-deficient fixture | Residual-only proof over `A*x`, with solution-selection explicitly unspecified and no minimum-norm assertion. | Low until output semantics are pinned. | High risk of implying underdetermined solution selection, nullspace, minimum-norm, exact values, or Q/economy behavior. | Defer by default. Day 3 may only accept if semantics and proof value are fully pinned. |
| `qr_wide_compatible_zero_residual_only` | Wide compatible fixture with RHS in range of A | Near-zero residual only; no statement about which solution is returned. | Low. Zero residual can be true for many returned solutions. | Very high minimum-norm and solution-selection confusion risk. | Defer. Better owned by minimum-norm or Q/economy work after output semantics are named. |
| `qr_wide_incompatible_residual_only` | Wide fixture with inconsistent or structurally constrained RHS | Non-zero residual-only reference. | Usually not meaningful for underdetermined full-row-rank systems because many wide systems are compatible for all RHS. | Can create misleading shape semantics unless rank, row space, and RHS feasibility are pinned. | Defer until rank/nullity and output semantics are explicit. |
| Wide sparse-mode residual-only fixture | Existing sparse-mode wide QR path | Residual-only comparison while sparse-mode output semantics are fenced. | Low for Sprint 128 Item 2. | Risks absorbing Sprint 129 Q/economy/sparse-mode work. | Defer to Sprint 129 Q/economy/sparse-mode owners unless Day 3 proves residual-only value independent of those outputs. |

## Output-Semantics Policy

Accepted residual-only evidence must define the output semantics before code
changes:

| Topic | Required Semantics |
| --- | --- |
| Fixture key | Named key with shape, structural rank-deficient context, RHS construction, expected residual class, and duplicate fence. |
| Returned quantity | Residual norm only, plus optional recomputed true residual as diagnostic. |
| Compatible zero residual | May assert near-zero residual only when the fixture adds distinct rank-deficient diagnostic value beyond existing compatible solve, rank-only, residual-only, and rank-deficient smoke evidence. |
| Wide shape | Must state whether `sparse_qr_solve` or `sparse_qr_solve_minnorm` owns the behavior. Residual-only `sparse_qr_solve` evidence must not imply minimum-norm. |
| Solution vector | Do not assert solution-vector equality, uniqueness, selected free variables, exact values, or compatibility with a dense solver's selected solution. |
| Solution norm | Do not assert norm equality, norm minimization, or comparison against feasible vectors. |
| Nullspace/subspace | Do not assert nullity, raw basis vectors, projector equality, principal angles, or subspace residuals. |
| Q/economy/sparse-mode | Do not assert Q columns, signs, orientation, economy shape, sparse-mode representation, or Sprint 129 output-shape semantics. |
| Rank | May use fixture rank as context only; residual-only evidence is not new rank proof. |
| Corpus/platform | Do not assert SuiteSparse, optional-data, runtime, platform, or support-tier behavior unless the relevant corpus gate accepts the candidate. |

## Residual Tolerance and Diagnostic Policy

| Policy Point | Requirement |
| --- | --- |
| Reference source | Prefer Python standard-library projection or closed-form derivation. Do not add NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependencies. |
| Fixture reuse | Prefer existing duplicate-column or dependent-row fixture builders before adding new families. |
| Zero residual tolerance | Default near-zero bound is `1e-10` for tiny exact compatible fixtures unless Day 3 derives a different fixture-local tolerance. |
| Residual difference tolerance | Default absolute product/reference residual-difference bound is `1e-8` for external residual references. |
| Diagnostics | Print fixture key, expected residual, returned residual, recomputed true residual when available, absolute diff, matrix shape, and claim class. |
| Skip behavior | Preserve existing external-reference helper skip behavior on unsupported Python/platform paths. |
| Failure interpretation | A residual mismatch means only the named fixture's residual contract failed; it is not a broad QR parity conclusion. |

## Trust-Value Decision Rules

A Day 3 compatible or wide residual candidate can proceed only if all of the
following are true:

1. The candidate is non-duplicate relative to Sprint 125-127 residual,
   compatible, rank, projector, threshold, and minimum-norm lanes.
2. The expected residual is independently derived or emitted by the
   standard-library helper.
3. The assertion set is residual-only.
4. The artifact states why the evidence adds trust beyond existing compatible,
   rank-deficient, wide, and minimum-norm tests.
5. The artifact records why the result cannot be misread as minimum-norm,
   nullspace, Q/economy, sparse-mode, SuiteSparse, or parity evidence.
6. Focused validation and full quality requirements are named before any code
   or helper changes.

If any condition is missing, Day 3 must explicitly defer the candidate instead
of weakening the proof boundary.

## Day 3 Implementation or Deferral Checklist

Day 3 should evaluate candidates in this order:

1. First evaluate `qr_rankdef_duplicate_5x4_compatible_zero_residual` because
   it reuses the most established rank-deficient fixture builder.
2. Accept it only if the Day 3 artifact can name a distinct diagnostic beyond
   existing full-rank compatible solve, duplicate-column rank-only,
   duplicate-column residual-only, and rank-deficient solve-smoke evidence.
3. If accepted, assert near-zero residual only; do not assert solution values,
   rank, nullity, nullspace, projector, norm, QR-vs-SVD, Q/economy,
   sparse-mode, SuiteSparse, or backend behavior.
4. If the duplicate-column compatible candidate cannot prove distinct trust,
   explicitly defer all compatible zero-residual candidates.
5. Defer wide residual-only candidates unless Day 3 fully pins
   underdetermined output semantics, solution-selection boundaries,
   Q/economy/sparse-mode fences, and residual-only proof value.
6. If code or helper files change, run focused helper/test validation and the
   full `make format && make lint && make test` gate.
7. If no code changes are made, validate documentation with `git diff --check`
   and Sprint 128 markdown whitespace scans.

## Deferred Candidate Promotion Gates

| Deferred Candidate | Future Owner | Promotion Gate |
| --- | --- | --- |
| Duplicate-column compatible zero-residual fixture | Day 3 or future QR residual owner | Prove zero residual adds trust beyond existing compatible solve behavior, duplicate-column rank-only evidence, duplicate-column residual-only evidence, and rank-deficient solve smoke; assert residual only. |
| Dependent-row compatible zero-residual fixture | Future QR residual owner | Show why compatible dependent-row zero residual adds a new diagnostic after the accepted incompatible dependent-row residual and dependent-row projector lanes. |
| New synthetic compatible fixture | Future QR residual owner | Justify a new fixture family, fixture key, exact compatibility construction, residual tolerance, diagnostics, and duplicate fence. |
| Wide residual-only fixture | Day 4-5 nullspace/subspace owner, Day 10-13 minimum-norm owner, or Sprint 129 Q/economy owner | Define underdetermined output semantics, solution-selection policy, Q/economy boundaries, residual-only proof value, and non-claims. |
| Wide sparse-mode residual fixture | Sprint 129 Q/economy/sparse-mode owner | Define sparse-mode output semantics, economy/Q boundaries, and residual-only proof value. |
| SuiteSparse compatible or wide residual fixture | Days 8-11 corpus owners | Define expected-rank metadata, support tier, optional-data behavior, runtime budget, skip behavior, diagnostics, and validation requirements. |

## Non-Claim Register

Day 2 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, or broad dense-library parity;
- broad QR factorization, QR solve, compatible solve, wide solve,
  rank-deficient solve, nullspace, minimum-norm, Q-basis, economy,
  sparse-mode, reorder, backend, corpus, optional-data, platform, or
  performance parity;
- new residual evidence beyond completed Sprint 125-127 fixtures;
- new rank, nullity, nullspace, projector, subspace, Q-basis, economy, or
  sparse-mode evidence;
- solution-vector uniqueness, solution equality, solution-selection policy,
  solution-norm optimality, or minimum-norm optimality;
- QR-vs-SVD-pseudoinverse oracle behavior or dense-library parity;
- SuiteSparse corpus, optional-data, runtime, platform, or performance
  support;
- global near-rank-deficient threshold, default-threshold, or numerical-rank
  policy;
- generic QR/SVD helper API or helper consolidation behavior;
- package, ABI, public API, CMake, Makefile, CI, CTest, scalability, memory,
  or state-of-the-art behavior.

## Validation Notes

Day 2 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_128`

No `.c`, `.h`, Python helper, build, public API, maintainer, or public wording
files changed, so no code quality gate is required for Day 2.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| No residual candidate proceeds without distinct trust value. | Complete | See candidate tables, trust-value decision rules, and Day 3 checklist. |
| Wide residual-only fixtures have pinned output semantics or are deferred. | Complete | Wide candidates are deferred by default until output semantics are pinned. |
| Residual evidence cannot imply nullspace or minimum-norm behavior. | Complete | See output-semantics policy and non-claim register. |
