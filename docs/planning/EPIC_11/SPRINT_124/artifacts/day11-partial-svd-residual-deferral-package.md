# Day 11 Partial-SVD Residual Deferral Package

## Decision

Defer new partial-SVD residual scenario implementation for Sprint 124 Day 11.

Day 9 already landed the bounded exact square diagonal vector-residual lane
`partial_svd_vector_residual_diag6_k2`. Day 10 accepted no additional
residual scenario for immediate implementation because the remaining scenarios
need metric, helper, threshold, corpus, convergence-budget, or low-rank
ownership beyond the existing exact diagonal residual protocol.

## No-Code Rationale

| Candidate | Why it is not implemented on Day 11 |
| --- | --- |
| Rectangular vector residual | It is feasible, but it would mostly extend Day 9's exact diagonal residual protocol to shape coverage. Sprint 124 still lacks an explicit choice between tall-only, wide-only, or both, and adding both would risk silently claiming broader rectangular vector parity. |
| Repeated-spectrum subspace | Requires projector or principal-angle reference output; raw vector residuals do not prove basis-ambiguous subspace behavior. |
| Clustered-spectrum subspace/convergence | Requires gap policy, ordering/set policy, iteration budget, and failure interpretation before the fixture can be meaningful. |
| Rank-deficient subspace | Requires rank threshold, zero singular-value tolerance, and a range/null-space evidence split. |
| SuiteSparse corpus vector residual | Requires optional-data skip rules, corpus-specific residual windows, and conditioning notes. |
| Low-rank optimality | Requires a separate dense reconstruction or sparse-output metric and must not be folded into top-k vector residual evidence. |
| Convergence budget | Requires an options surface, iteration cap, deterministic initialization policy, tolerance, and budget-failure semantics. |
| Nonsymmetric rectangular residual | Requires a dense-reference non-diagonal fixture and a value/residual boundary that avoids vector/subspace claims. |

## Affected-Surface Matrix

| Surface | Day 11 action | Reason |
| --- | --- | --- |
| `tests/svd_external_dense_reference.py` | No change | No new reference protocol was accepted. Existing helper remains singular-value only. |
| `tests/test_svd_partial_helpers.h` | No change | Existing Day 9 vector-residual helper remains the only accepted bounded residual lane. |
| `tests/test_svd.c` | No change | No new test registration is justified by Day 10 decisions. |
| `docs/maintainer_guide.md` | No change | Day 9 already names the bounded vector-residual fixture and preserves broad non-claims. |
| Sprint 124 artifacts | Add this deferral package | Carries forward each residual scenario with a future owner and promotion gate. |
| Public examples, package metadata, ABI, CI, CMake, Makefile | No change | No product, build, package, or platform behavior changed. |

## Residual Scenario Handoff

| Residual scenario | Future owner | Promotion gate |
| --- | --- | --- |
| Rectangular vector residual | Partial-SVD vector owner | Pick exactly one bounded shape lane first, define matrix, `k`, dimensions, tolerances, and claim wording before editing tests. |
| Repeated-spectrum subspace | Subspace owner | Add projector or principal-angle protocol for left and right subspaces; forbid raw vector equality. |
| Clustered-spectrum subspace/convergence | Convergence/subspace owner | Define spectral gap, ordered versus set-based value policy, projector tolerance, iteration budget, and failure meaning. |
| Rank-deficient subspace | Rank/subspace owner | Define rank threshold, zero singular-value tolerance, and whether range or null-space projectors are being checked. |
| SuiteSparse vector residual | Corpus owner | Define file availability rules, conditioning notes, per-fixture residual windows, and non-external-oracle wording. |
| Low-rank optimality | Low-rank owner | Choose Frobenius or spectral norm evidence, dense versus sparse-output semantics, and sparse drop-tolerance handling. |
| Convergence budget | Convergence owner | Add deterministic option surface, iteration cap, tolerance, and budget-failure classification. |
| Nonsymmetric rectangular value residual | External value owner | Add a non-diagonal dense-reference fixture without extending vector/subspace claims. |

## Validation Evidence

Day 11 changes documentation only. No focused partial-SVD executable rerun is
required by the sprint validation policy because no `.c`, `.h`, script,
Makefile, CMake, or helper protocol changed.

The required Day 11 validation is:

1. `git diff --check`
2. focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_124`

## Non-Claim Confirmation

Day 11 preserves the following non-claims:

- no LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no broad partial-SVD external parity claim;
- no broad rectangular vector-residual claim beyond the Day 9 square exact
  diagonal fixture;
- no repeated-spectrum, clustered-spectrum, rank-deficient subspace, or
  null-space parity claim;
- no SuiteSparse corpus vector-residual parity claim;
- no low-rank global optimality claim;
- no convergence-budget guarantee;
- no package, ABI, platform, performance, scalability, public API, or
  state-of-the-art claim.

## Day 12 Input

Day 12 should not treat this deferral package as a helper-movement request.
The minimum-norm and Bidiagonal/Golub-Kahan helper ownership work remains
separate from partial-SVD residual scenarios. If helper movement touches SVD
or residual helpers indirectly, it must preserve the behavior-specific owners
named in this package.
