# Sprint 130 Day 2 - Partial-SVD Dedupe And Metric Map

## Purpose

Day 2 turns the Day 1 residual dedupe baseline into an enforceable metric,
tolerance, oracle, diagnostics, and failure-interpretation policy for every
Sprint 130 partial-SVD evidence lane.

This is a policy gate, not an implementation day. No rectangular,
nonsymmetric, repeated-spectrum, clustered-spectrum, rank-deficient,
SuiteSparse, low-rank optimality, convergence-budget, or solver-selection
claim may proceed until the relevant row in this artifact is satisfied by the
later day owner.

## Inputs Reviewed

| Input | Role |
| --- | --- |
| Sprint 130 Day 1 baseline | Provides completed evidence, deferred lanes, duplicate fences, owner map, and validation boundary. |
| Sprint 124 Day 8 partial-SVD semantics | Defines sign-invariant residuals, subspace metrics, tolerance, skip, and failure policy. |
| Sprint 124 Day 9 partial-SVD decision | Establishes `partial_svd_vector_residual_diag6_k2` as the only accepted bounded vector-residual lane. |
| Sprint 124 Day 10 residual scenario matrix | Provides deferred scenario classes and required diagnostics. |
| Sprint 124 Day 11 residual deferral package | Provides future owners and promotion gates for deferred lanes. |
| `tests/svd_external_dense_reference.py` | Current external helper emits singular values only for named fixtures. |
| `tests/test_svd_partial_helpers.h` | Owns current partial-SVD residual, orthogonality, corpus smoke, low-rank, and timing checks. |
| `include/sparse_svd.h` and `src/sparse_svd_partial.c` | Define public option semantics and current implementation convergence surface. |
| `docs/maintainer_guide.md` | Current evidence table and solver-selection wording boundary. |

## Residual Scenario Metric Matrix

| Lane | Shape | Spectrum | Rank behavior | Corpus status | Primary metric | Secondary diagnostics | Oracle | Tolerance posture | Solver-selection impact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rectangular vector residual | Tall or wide, chosen explicitly | Well separated | Full numerical rank for first lane | Local fixture | `||A v_i - sigma_i u_i||_2` and `||A^T u_i - sigma_i v_i||_2` | `U^T U`, `V^T V`, dimensions, ordered singular values | Analytic diagonal or current singular-value external helper plus product-owned residuals | Exact diagonal lane may use `1e-8`; non-diagonal lanes need fixture-specific tolerance | None until multiple shapes and failure modes are covered |
| Nonsymmetric rectangular residual | Rectangular non-diagonal | Prefer well separated | Full numerical rank for first lane | Local fixture | Triplet residuals on both left and right equations | Shape, ordered values, orthogonality, conditioning note | New analytic or dense-reference singular-value fixture; no raw vector oracle by default | Fixture-specific; must justify looser than `1e-8` if not exact | None unless later evidence justifies bounded workflow wording |
| Repeated-spectrum subspace | Square or rectangular, chosen explicitly | Repeated leading values | Usually full rank unless rank policy is also owned | Local fixture | Left and/or right projector distance | Subspace dimension, unordered value multiset, residuals, orthogonality | Dense-reference projector/principal-angle protocol or analytic repeated subspace | Fixture-specific projector tolerance; raw vector equality forbidden | None |
| Clustered-spectrum subspace | Square or rectangular, chosen explicitly | Small declared gap | Usually full rank unless rank policy is also owned | Local fixture | Projector distance or principal-angle bound | Gap size, ordered versus set value policy, residuals, iteration diagnostics | Dense-reference projector/principal-angle protocol plus declared gap | Fixture-specific tolerance tied to spectral gap and algorithm budget | None |
| Rank-deficient subspace | Rectangular first, if chosen | May include zeros or near-zeros | Numerical rank and nullity declared | Local fixture | Range projector, null-space projector, or two-way projection residual | Rank threshold, zero singular-value tolerance, dimensions, residuals | Analytic rank/nullity fixture or dense-reference projector protocol | Threshold-specific; must define positive and zero singular-value windows | None |
| SuiteSparse corpus residual | Matrix-specific | Matrix-specific | Matrix-specific | Checked-in or optional corpus | Corpus-specific triplet residual or singular-value residual | Availability, conditioning, nnz, dimensions, runtime, support tier | External metadata if available; otherwise bounded smoke, not independent oracle | Fixture-specific residual windows; exact diagonal tolerance invalid | At most support-tier wording after Day 14 review |
| Low-rank optimality | Dense local fixture first | Well separated preferred | Usually full rank unless rank policy is owned | Local fixture | Reconstruction error against stated rank-k target | Frobenius or spectral norm, retained singular values, sparse-output/drop-tolerance notes | Analytic low-rank fixture or dense full-SVD reference values | Metric-specific; must state norm and expected bound before implementation | None until global wording is explicitly rejected or earned |
| Convergence budget | Fixture-specific | Prefer clustered or difficult only after policy | Any rank, declared | Local or corpus | Achieved residual and returned status under declared options | Iteration cap, tolerance, returned `k`, partial-result fields, restart/budget notes | Product-independent expected behavior only when deterministic; otherwise bounded API semantics | Budget-specific; non-convergence may be expected only when declared | None without Day 14 wording gate |
| Solver-selection wording | Documentation surface | Evidence-dependent | Evidence-dependent | Evidence-dependent | Evidence-to-wording traceability | Non-claim scan and public wording diff | Accepted Sprint 130 evidence package | N/A | Default no update unless bounded claim is earned |

## Metric Policy

### Value Metrics

- Singular-value comparisons are valid when the fixture has an independent
  analytic or dense-reference value oracle.
- Current external partial-SVD helper output is singular-value only. It does
  not provide singular-vector, projector, convergence, or optimality oracles.
- Ordered value comparison is acceptable for well-separated spectra.
- Repeated or clustered spectra need unordered/set-based value policy plus
  subspace metrics before values can support vector or subspace claims.

### Vector-Residual Metrics

- Vector residual evidence is meaningful when the lane checks both triplet
  equations:
  - `||A v_i - sigma_i u_i||_2`
  - `||A^T u_i - sigma_i v_i||_2`
- `U` and `V` orthogonality checks are required when both vector families are
  requested.
- Raw singular-vector equality is not a pass/fail metric for Sprint 130.
- Sign flips, valid basis rotations, and equivalent ordering inside repeated
  or clustered subspaces are not failures by themselves.

### Subspace and Projection Metrics

- Repeated, clustered, and rank-deficient lanes require projector,
  principal-angle, or two-way projection residual metrics.
- A subspace lane must state whether it covers the left subspace, right
  subspace, range, null space, or both left and right spaces.
- Projection metrics require dimension checks before projector construction.
- Projector metrics must not be mixed with rank-deficient null-space claims
  unless rank and nullity policies are declared in the same artifact.

### Low-Rank Optimality Metrics

- Low-rank evidence must name the norm: Frobenius, spectral, or another
  explicitly justified metric.
- Dense reconstruction evidence and sparse-output evidence are separate. A
  dense optimality result does not prove sparse drop-tolerance behavior.
- A fixture-specific reconstruction check is not a global Eckart-Young or
  production optimality claim unless Day 12 explicitly proves and bounds that
  wording.

### Convergence-Budget Metrics

- Convergence-budget evidence must define `max_iter`, `tol`, requested `k`,
  returned status, and partial-result expectations before implementation.
- A timeout, timing smoke, or successful default run is not convergence-budget
  evidence.
- Budget exhaustion is a failure only when the fixture contract says the
  budget should converge; otherwise it may be the expected behavior being
  checked.

## Tolerance Policy

| Metric | Default policy | When a different tolerance is allowed |
| --- | --- | --- |
| Exact diagonal singular values | `1e-8` absolute difference | Only with documented conditioning or precision reason. |
| Exact diagonal vector residuals | `1e-8` for `A v`, `A^T u`, and orthogonality | Only if the fixture is no longer exact diagonal or uses an iterative budget. |
| Rectangular non-diagonal residuals | Fixture-specific absolute or relative residual window | Required when values are scaled, ill-conditioned, or generated from dense references. |
| Projector or principal-angle metrics | Fixture-specific bound | Required for repeated, clustered, and rank-deficient cases. |
| Rank/nullity thresholds | Paired positive-rank and zero singular-value thresholds | Required whenever `k` crosses numerical rank. |
| SuiteSparse corpus residuals | Matrix-specific window | Required because exact diagonal tolerances are invalid for corpus matrices. |
| Low-rank reconstruction/optimality | Norm-specific expected bound | Required before implementation; must state dense versus sparse-output semantics. |
| Convergence budget | Budget-specific residual and status expectations | Required for every budgeted fixture. |

Tolerances must be stated in the artifact before the code, helper, or public
wording changes land. A tolerance copied from an unrelated evidence class is
not acceptable.

## Oracle Policy

| Oracle source | Acceptable use | Boundary |
| --- | --- | --- |
| Analytic diagonal fixture | Singular values, dimensions, simple residual expectations, and exact rank when declared. | Does not prove non-diagonal, nonsymmetric, clustered, corpus, or optimality behavior. |
| Current `tests/svd_external_dense_reference.py` helper | Bounded singular-value references for named fixtures. | Does not emit vectors, projectors, convergence metadata, or optimality certificates. |
| Future dense-reference helper expansion | Projectors, principal angles, or additional singular-value fixtures only after protocol and parser policy are written. | Helper `ERROR` remains a test infrastructure failure; missing `python3` remains a skip through the existing harness. |
| Product full SVD | Internal consistency cross-check. | Not independent external evidence and not solver-selection proof. |
| Product-observed SuiteSparse output | Smoke diagnostics only. | Not an expected-value oracle unless external metadata is supplied. |
| Published corpus metadata | Corpus rank, conditioning, or spectral facts when source and support tier are recorded. | Must include optional-data and runtime policy. |

## Failure Interpretation Policy

| Failure class | Meaning |
| --- | --- |
| Helper skip | Missing optional execution dependency, such as `python3`, under an existing skip policy. |
| Helper protocol error | Reference generator or parser failed; this is a test infrastructure failure, not a numerical mismatch. |
| Unsupported shape or options | API boundary failure when the fixture requested behavior the implementation does not support. |
| Singular-value mismatch | Bounded value regression for the named fixture only. |
| Vector residual mismatch | Bounded triplet-quality regression; not a raw sign or vector-orientation failure. |
| Orthogonality mismatch | Vector publication quality regression for the named fixture. |
| Projector or principal-angle mismatch | Subspace-quality regression under the declared subspace metric only. |
| Rank/nullity threshold mismatch | Fixture-specific threshold-policy failure. |
| SuiteSparse optional-data skip | Allowed only when the support-tier policy says missing corpus data is optional. |
| SuiteSparse corpus mismatch | Matrix-specific corpus diagnostic failure; not broad SuiteSparse parity. |
| Low-rank reconstruction mismatch | Fixture-specific low-rank metric failure; not global optimality unless that claim is explicitly proven. |
| Convergence-budget exhaustion | Either expected budget-boundary behavior or a failure, depending on the declared fixture contract. |
| Solver-selection wording mismatch | Documentation overclaim when wording is not traceable to accepted bounded evidence. |

## Deferred-Lane Promotion Checklist

Before a later Sprint 130 day may implement or update wording for a lane, its
artifact must answer:

1. Which evidence class is being promoted?
2. What fixture key, matrix, dimensions, `k`, options, support tier, and owner
   files define the lane?
3. What is the primary metric, and what diagnostics are required?
4. What tolerance or threshold applies, and why is it valid for this fixture?
5. What oracle is independent, analytic, product-owned, or smoke-only?
6. What exact failure class should be reported for each possible failure?
7. How does the lane avoid duplicating `partial_svd_vector_residual_diag6_k2`
   or existing internal SVD tests?
8. Which non-claims must be preserved?
9. Which focused validation and full quality gates are required?
10. Does the lane affect maintainer evidence or public solver-selection
    wording? If not, record a no-update rationale.

If any answer is missing, the lane remains deferred with blocker, dependency,
future owner, and promotion gate recorded.

## Day 3 Handoff

Day 3 should apply this policy to rectangular vector-residual candidates. The
lowest-risk path is a single exact diagonal rectangular lane, either tall or
wide, that reuses singular-value oracle coverage where possible and adds only
product-owned triplet residual, orthogonality, and shape diagnostics. Day 3
should not implement both tall and wide lanes unless it first proves distinct
non-duplicative claim value for each.

## Non-Claim Register

Day 2 preserves the following non-claims:

- no LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK,
  vendor-backend, dense-library, external package, or ecosystem parity;
- no broad SVD or partial-SVD external parity claim;
- no broad singular-vector parity beyond the single bounded
  `partial_svd_vector_residual_diag6_k2` fixture;
- no repeated-spectrum, clustered-spectrum, rank-deficient subspace, or
  null-space parity claim;
- no SuiteSparse corpus residual parity or optional-data platform claim;
- no low-rank global optimality claim;
- no convergence-budget, performance, scalability, or state-of-the-art claim;
- no public solver-selection wording readiness beyond current workflow
  guidance.

## Validation

Day 2 changes documentation only. Validation is limited to:

1. `git diff --check`
2. focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_130`
