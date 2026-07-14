# Sprint 124 Day 2 Rank-Deficient QR Policy Design

## Purpose

Day 2 defines the rank-deficient QR policy required before Sprint 124 can
accept or defer any external rank-deficient QR oracle lane. The policy keeps
rank, nullspace, residual, pseudoinverse, minimum-norm, Q-basis, and economy
evidence separate so one bounded external fixture cannot imply broad QR or
dense-library parity.

This is a policy artifact only. No C source, header, Python helper, build,
CMake, CTest, workflow, public API, or public wording changes are made by Day
2.

## Inputs Reviewed

| Input | Policy Use |
| --- | --- |
| Sprint 124 Plan Day 2 | Requires rank-deficient, near-rank-deficient, and deterministic rank evidence inventory plus rank/nullspace/pseudoinverse/tolerance/skip policy. |
| Sprint 124 Day 1 artifact | Provides duplicate fences and day-level proof owners. |
| Sprint 123 Day 5 QR requirements | Defines QR candidate classes, behavior-specific fixture requirements, basis rules, tolerance rules, and QR external non-claims. |
| Sprint 123 Day 6 QR compatible/rank-deficient decision | Defers rank-deficient QR external evidence until rank threshold, nullspace, minimum-norm, and reference-solver behavior are separated. |
| `tests/test_qr.c` | Owns QR factorization, reconstruction, rank, nullspace, diagonal threshold, Q application, economy, sparse-mode, and reordering evidence. |
| `tests/test_qr_solve.c` | Owns bounded external least-squares fixtures, compatible/incompatible solve behavior, rank-deficient solve residual behavior, and QR solve scenarios. |
| `tests/test_colamd.c` | Owns QR+COLAMD solve behavior, minimum-norm scenarios, refinement, SVD-pseudoinverse comparison, and optional SuiteSparse minimum-norm coverage. |
| `tests/qr_external_dense_reference.py` | Current standard-library external dense least-squares reference helper for bounded full-column-rank QR fixtures. |
| `docs/maintainer_guide.md` | Current QR trust-boundary table and explicit non-claims for rank-deficient, minimum-norm, Q-basis, economy, sparse-mode, reorder, and broad parity evidence. |

## Current QR Rank Evidence Inventory

| Evidence Class | Current Owner | Evidence Summary | External Policy Implication |
| --- | --- | --- | --- |
| Duplicate-column rank deficiency | `tests/test_qr.c` | `test_qr_rank_deficient` expects rank 2 for a 4x3 duplicate-column fixture. | Completed deterministic rank evidence; do not relabel as external oracle parity. |
| Rank-1 nullspace | `tests/test_qr.c` | `test_rank_1_nullspace` checks rank 1 and verifies extracted null vectors satisfy `A*v ~= 0`. | Nullspace evidence exists internally, but external evidence needs basis/subspace semantics. |
| Known nullspace | `tests/test_qr.c` | `test_known_nullspace` verifies a known duplicate-column nullspace vector. | Useful input for nullspace policy; not an external basis parity claim. |
| Rectangular rank-deficient nullspace | `tests/test_qr.c` | `test_rank_rect_deficient` checks 3x5 rank 2 and verifies three nullspace vectors. | Covers wide deterministic nullspace behavior; external lanes must not duplicate it without new trust value. |
| Explicit rank tolerance | `tests/test_qr.c` | `test_rank_explicit_tol` verifies loose tolerance does not increase rank. | Establishes tolerance direction only; does not define a global default external rank threshold. |
| Diagonal threshold fixture | `tests/test_qr.c` | `test_qr_rank_diagonal_threshold_fixture` expects ranks 3, 2, and 1 at thresholds `1e-14`, `1e-10`, and `1e-6`. | Best deterministic input for a future explicit external threshold policy. |
| Dependent-row rank fixture | `tests/test_qr.c` | `test_qr_rank_dependent_row_fixture` verifies rank, reconstruction, and nullspace residual. | Candidate structural model for external design, but only after rank/nullspace claims are separated. |
| Rank-deficient solve residual | `tests/test_qr_solve.c` | Rank-deficient solve path checks solve status and residual behavior. | Residual evidence must not imply minimum-norm, nullspace, or pseudoinverse agreement. |
| Minimum-norm rank-deficient solve | `tests/test_colamd.c` | `test_minnorm_rank_deficient` and related tests cover minimum-norm behavior under QR-minnorm owners. | Belongs to the minimum-norm owner unless Day 4-5 explicitly accepts external evidence. |
| SVD pseudoinverse comparison | `tests/test_colamd.c` | `test_minnorm_vs_pinv` compares QR minimum-norm against SVD pseudoinverse for a bounded case. | Pseudoinverse agreement is not a Day 2 rank-only policy; it must remain separately owned. |

## Rank-Deficient QR Candidate Table

| Candidate | Adds Evidence? | Required Policy Before Acceptance | Day 2 Disposition |
| --- | --- | --- | --- |
| `qr_rankdef_duplicate_5x4_rank_only` | Moderate. Would externally confirm structural duplicate-column rank under a named threshold. | Explicit numerical rank threshold, expected rank, output protocol, and no residual/minimum-norm/nullspace claim. | Candidate for Day 3 if kept rank-only. |
| `qr_rankdef_duplicate_5x4_residual_only` | Limited. Would externally compare least-squares residual but not rank semantics. | Clear statement that residual-only evidence does not assert rank, nullspace, or minimum-norm behavior. | Weak candidate; prefer rank-only or explicitly defer. |
| `qr_rankdef_duplicate_5x4_nullspace` | High. Would validate nullspace residual or subspace behavior. | Basis orientation, subspace/projection metric, rank threshold, and null residual tolerance. | Defer from Day 3 unless basis/subspace policy is complete. |
| `qr_rankdef_duplicate_5x4_minnorm` | High but cross-owner. Would validate minimum-norm solution behavior. | QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, SuiteSparse, norm, and residual policies. | Defer to Days 4-5 minimum-norm owner. |
| diagonal near-rank-deficient threshold fixture | Moderate. Would externally check rank threshold around tiny diagonal values. | Explicit threshold values, expected rank at each threshold, and robust reference rank computation. | Candidate only if Day 3 avoids broad numerical-rank claims. |
| SuiteSparse rank-deficient QR fixture | Potentially broad. | External corpus selection, optional availability, platform policy, and support-tier interpretation. | Reject for Sprint 124 rank policy; too broad for this residual lane. |

## Rank-Threshold Policy

| Policy Point | Sprint 124 Rule |
| --- | --- |
| Default external rank threshold | Do not invent a new global threshold in Sprint 124. Any accepted fixture must name its threshold explicitly. |
| Structural rank fixtures | Prefer exact duplicate-column or dependent-row fixtures where expected rank is mathematically obvious. |
| Near-rank-deficient fixtures | Require named thresholds and expected rank at each threshold; do not infer a global numerical-rank policy. |
| Output protocol | A rank-only external helper may emit `OK 1` plus rank, or a richer protocol only if each value has a named semantic. |
| Product comparison | Compare expected rank directly when the threshold is pinned; avoid using residual success as rank proof. |
| Failure interpretation | A rank mismatch is rank-policy evidence only for the named fixture and threshold, not a broad QR rank claim. |

## Nullspace Policy

| Policy Point | Sprint 124 Rule |
| --- | --- |
| Nullity | Nullity may be asserted as `n - rank` only when rank threshold and expected rank are explicit. |
| Basis vectors | Do not compare raw nullspace vectors externally unless sign, basis ordering, and subspace orientation rules are defined. |
| Residual metric | `||A*v||` can validate a returned vector as a null vector, but it does not prove the entire nullspace basis is equivalent to an external basis. |
| Subspace metric | Future external nullspace evidence should use projection/subspace residuals rather than column equality. |
| Day 3 scope | Day 3 should not implement nullspace external evidence unless it first narrows the claim to nullity or vector residual with no basis-parity implication. |

## Pseudoinverse and Minimum-Norm Separation

| Topic | Sprint 124 Rule |
| --- | --- |
| Rank-only evidence | Must not compare pseudoinverse or minimum-norm solutions. |
| Residual-only evidence | Must explicitly state that a small residual does not prove minimum-norm optimality or pseudoinverse agreement. |
| Minimum-norm evidence | Belongs to Days 4-5 and must name QR solve, COLAMD, SVD-pseudoinverse, fallback, refinement, and optional SuiteSparse owners. |
| Pseudoinverse evidence | May be a trust input only under a behavior-specific minimum-norm owner, not under Day 2 rank policy. |
| Fallback behavior | Overdetermined and square fallback paths remain separate from rank-deficient external rank evidence. |

## Tolerance, Skip, and Failure-Interpretation Policy

| Rule | Requirement |
| --- | --- |
| Reference implementation | Python standard library only for any new bounded helper work; no NumPy, SciPy, LAPACK, BLAS, SuiteSparse, or external package dependency. |
| Fixture size | Prefer tiny fixed fixtures no larger than roughly 6 in either dimension unless a later day justifies otherwise. |
| Rank tolerance | Pin tolerance per fixture; likely candidates are exact structural rank with threshold `0.0` or diagonal threshold checks with explicit `1e-14`, `1e-10`, and `1e-6` style thresholds. |
| Residual tolerance | If residual is reported, compare it separately from rank and state whether it is diagnostic or asserted. |
| Missing Python | Preserve existing external-reference helper skip behavior. |
| Windows behavior | Preserve existing explicit Windows skip for external QR helper lanes unless a future platform sprint promotes it. |
| Helper `ERROR` output | Treat as test failure, not skip. |
| Failure diagnostics | Identify fixture key, reference status, threshold, expected rank, product rank, optional residual, and whether the mismatch is rank, residual, output protocol, helper, or product QR behavior. |

## Affected Owner Map

| Owner | Role in Rank-Deficient QR Policy |
| --- | --- |
| `tests/test_qr.c` | Primary deterministic rank, nullspace, diagonal-threshold, Q-basis, economy, sparse-mode, reorder, and reconstruction owner. |
| `tests/test_qr_solve.c` | QR solve and bounded external least-squares owner; may host a rank-only external fixture only if it does not hide nullspace/minimum-norm semantics. |
| `tests/test_colamd.c` | Minimum-norm, COLAMD, SVD-pseudoinverse, fallback, refinement, and optional SuiteSparse owner; Day 2 must not absorb this into rank-only evidence. |
| `tests/qr_external_dense_reference.py` | External helper owner for tiny standard-library QR references; any new fixture must extend its protocol without broadening dependencies. |
| `docs/maintainer_guide.md` | Maintainer evidence table owner; update only if Sprint 124 changes accepted evidence or non-claims. |
| Public solver-selection docs | No Day 2 update; future update requires evidence-to-claim traceability and Day 13-14 claim gate. |

## Non-Claim Register

Day 2 does not claim:

- LAPACK, NumPy, SciPy, BLAS, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  broad external dense-library parity;
- broad QR factorization, QR solve, least-squares, or rank-deficient parity;
- global QR rank-threshold policy;
- raw nullspace basis, Q-basis, Q-sign, Q-orientation, economy-mode,
  sparse-mode, reorder, or backend parity;
- underdetermined or rank-deficient minimum-norm global optimality;
- SVD-pseudoinverse parity;
- package, ABI, platform, public API, CMake, Makefile, CI, or CTest expansion;
- performance, scalability, memory behavior, or state-of-the-art behavior.

## Day 3 Decision Criteria

Day 3 should evaluate rank-deficient QR external evidence in this order:

1. Treat completed compatible and incompatible external QR lanes as fenced.
2. Prefer a rank-only structural fixture if it can pin expected rank and
   threshold without implying nullspace or minimum-norm behavior.
3. Reject or defer residual-only rank-deficient evidence unless it adds a
   distinct trust value and explicitly avoids rank/nullspace/minimum-norm
   claims.
4. Defer nullspace basis/subspace evidence unless projection/subspace metrics
   are defined.
5. Defer minimum-norm and pseudoinverse evidence to Days 4-5.
6. For accepted work, define fixture key, matrix, threshold, expected rank,
   output protocol, affected owners, diagnostics, and validation commands.
7. For deferred work, name future owner, dependency, and promotion gate.

## Validation Notes

Day 2 changed documentation only. Required validation is:

1. `git diff --check`
2. Focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_124`

No `.c` or `.h` files changed, so the full `make format && make lint &&
make test` gate is not required for Day 2.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Item 1 has explicit decision criteria. | Complete | See candidate table, rank-threshold policy, and Day 3 decision criteria. |
| Rank-deficient QR evidence is not conflated with minimum-norm evidence. | Complete | See pseudoinverse and minimum-norm separation. |
| Every accepted candidate has a clear trust-boundary rationale. | Complete | Day 2 accepts no implementation yet; candidate trust boundaries are named before Day 3 selection. |
