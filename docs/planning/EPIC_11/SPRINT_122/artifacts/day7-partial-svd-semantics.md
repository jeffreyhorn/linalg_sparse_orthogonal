# Sprint 122 Day 7 Partial-SVD External Semantics Inventory

## Purpose

Day 7 inventories partial-SVD external comparison semantics before Day 8 decides
whether to implement a bounded lane or explicitly defer. Partial-SVD must not be
treated as a direct reuse of full-SVD external singular-value parity because
top-k extraction, vector availability, subspace ambiguity, ordering, tolerance,
and convergence semantics differ.

## Existing Partial-SVD Evidence Inventory

| Evidence Area | Current Owner | Current Oracle Boundary | Day 7 Interpretation |
| --- | --- | --- | --- |
| Partial singular values on diagonal fixtures | `tests/test_svd_partial_helpers.h` | Analytical diagonal values or this library's full SVD | Strong internal regression evidence, not external parity. |
| Partial singular values on dense Hilbert-like fixture | `test_partial_svd_dense_8x8` | This library's full SVD | Useful algorithm regression check with looser windows, not independent dense truth. |
| Partial singular values on tall/wide fixtures | `test_partial_svd_tall`, `test_partial_svd_wide` | This library's full SVD and shape checks | Shape coverage, not external parity. |
| Partial singular values on SuiteSparse fixtures | `test_partial_svd_nos4`, `test_partial_svd_west0067` | This library's full SVD plus optional fixture availability | Corpus smoke, not broad SuiteSparse or external-library parity. |
| Partial rank-deficient behavior | `test_partial_svd_rank_deficient` | This library's full SVD and known rank-deficient structure | Internal top-k/rank regression; threshold semantics remain local. |
| Partial-SVD ordering | `test_partial_svd_descending` | Internal order checks | Ordering proof is local; external references must state tie behavior. |
| Partial-SVD timing smoke | `test_partial_svd_timing` | Runtime observation only | Not performance evidence and not an external comparison input. |
| Partial-SVD vectors | `test_partial_svd_vectors_*` | Orthogonality, `A*v ~= sigma*u`, internal full-SVD vector comparison, reconstruction | Vector evidence is meaningful but sign/subspace semantics make external parity separate. |
| Rectangular low-rank reconstruction | `test_partial_svd_vectors_rectangular_lowrank_recon` | Analytical diagonal tail `sqrt(10)` and `A*v` residual | Strong deterministic fixture; not arbitrary rank-k optimality. |

## Semantic Difference From Full-SVD External Comparisons

| Topic | Full-SVD External Singular-Value Lane | Partial-SVD External Lane Requirement |
| --- | --- | --- |
| Output cardinality | Compare all `min(m,n)` singular values for a fixed small fixture. | Compare exactly top `k`, and define behavior when `k` approaches rank, repeated values, or zero tail. |
| Ordering | Descending full spectrum is expected. | Descending top-k order must be explicit, especially around equal or clustered singular values. |
| Vectors | Current full-SVD external lanes do not compare vectors. | Partial vectors are often the product surface; sign and subspace ambiguity need a separate gate. |
| Subspaces | Not checked in current full-SVD lanes. | Needed for repeated or clustered spectra, but requires angle/projection metrics rather than component equality. |
| Convergence | Full-SVD lane calls deterministic product full SVD. | Partial-SVD may have iteration budgets, restarts, or convergence tolerances that can make external mismatch ambiguous. |
| Tolerance | One fixture-local singular-value tolerance plus zero-tail checks. | Separate tolerances may be needed for top-k values, vector residuals, subspace angle, and convergence residual. |
| Failure interpretation | Singular-value mismatch is the primary failure. | Failure may mean value mismatch, ordering mismatch, vector sign/subspace mismatch, nonconvergence, or unsupported mode. |
| Trust boundary | Pure-Python dense reference validates one fixed full-SVD value surface. | External reference may validate only values unless vector/subspace semantics are explicitly designed. |

## Candidate Partial-SVD External Evidence Classes

| Candidate Class | Adds New Evidence? | Risk | Day 7 Disposition |
| --- | --- | --- | --- |
| `partial_svd_diag6_k2_external_sigma` | Yes. Bounded top-k singular-value comparison against an independent dense reference for a simple diagonal fixture. | Low: analytical-like fixture can duplicate internal coverage unless framed as protocol proof. | Strongest Day 8 candidate if implementation is desired. |
| `partial_svd_rect_lowrank_6x4_k2_external_sigma` | Yes. Matches Sprint 121 rectangular low-rank fixture and validates top-k values externally. | Moderate: overlaps existing analytical tail/reconstruction proof and may blur low-rank optimality. | Candidate only if Day 8 wants rectangular shape evidence. |
| `partial_svd_rankdef_duplicate_k2_external_sigma` | Partial. Could validate top-k values on rank-deficient non-diagonal shape. | Moderate: rank threshold and zero-tail semantics overlap full-SVD rank-deficient lane. | Defer until rank-threshold external policy is explicit. |
| `partial_svd_vectors_external_av_residual` | Partial. External reference could provide expected top-k values while product vectors satisfy `A*v ~= sigma*u`. | Moderate: still not vector parity; external vector equality is not designed. | Defer unless Day 8 explicitly restricts to value reference plus product vector residual. |
| `partial_svd_subspace_external_projection` | Potentially high. | High: repeated/clustered spectra need projection-angle metrics and basis-invariance rules. | Defer to future subspace owner. |
| `partial_svd_suite_sparse_external_values` | Potentially broad. | High: optional data, runtime, and corpus interpretation risk. | Reject for Sprint 122. |
| `partial_svd_convergence_budget_external` | Potentially useful. | High: would require deterministic iteration budgets and convergence diagnostics. | Defer to future iterative/eigensolver-style owner. |

## Vector and Subspace Risk Notes

| Risk | Why It Matters | Required Owner Before Implementation |
| --- | --- | --- |
| Sign ambiguity | Singular vectors may be negated without changing correctness. | Vector comparison owner must compare sign-invariant quantities or residuals. |
| Basis ambiguity | Repeated or clustered singular values can rotate within the same subspace. | Subspace owner must compare projectors or principal angles, not component equality. |
| Top-k boundary ambiguity | If `sigma_k` and `sigma_{k+1}` are close or equal, the selected vectors may vary. | Fixture owner must avoid ambiguous gaps or define accepted subspace behavior. |
| Convergence ambiguity | Partial-SVD algorithms may stop based on residual or iteration budget. | Convergence owner must record budget, residual target, and failure mode. |
| Internal full-SVD dependency | Current partial-SVD value checks mostly compare to this library's full SVD. | External lane owner must state whether it is independent value evidence or only protocol proof. |
| Low-rank claim drift | Top-k agreement does not prove arbitrary low-rank optimality. | Low-rank owner must keep reconstruction/tail claims fixture-local. |

## Day 8 Design Checklist

Day 8 should choose one of these outcomes:

| Outcome | Required Evidence |
| --- | --- |
| Implement `partial_svd_diag6_k2_external_sigma` | Define a fixed fixture key, top-k output protocol, Python standard-library reference, value tolerance, skip behavior, existing-executable test owner, and non-claims. |
| Implement a rectangular top-k value lane | Explain why rectangular shape adds evidence beyond the diagonal protocol lane and avoid low-rank optimality claims. |
| Defer partial-SVD external implementation | Identify which semantic gate is not ready: vector/subspace, ordering, convergence, tolerance, reference trust, or duplicate risk. |
| Reject partial-SVD external lane for Sprint 122 | Explain why existing internal evidence plus full-SVD/QR external lanes are sufficient for this sprint. |

## Non-Claim Register

Day 7 introduces no partial-SVD external parity claim. It also does not claim:

- singular-vector external parity;
- subspace external parity;
- convergence-budget parity;
- broad dense-library, LAPACK, SciPy, NumPy, SuiteSparse, PETSc, Trilinos, or
  Eigen parity;
- arbitrary low-rank or pseudoinverse optimality;
- performance, scalability, package, platform, ABI, public API, or
  state-of-the-art behavior.

## Validation Notes

Day 7 changed documentation only. Required validation is `git diff --check` and
a focused trailing-whitespace scan over `docs/planning/EPIC_11/SPRINT_122`.
The branch already passed `make lint` and `make test` after the Day 4 and Day 6
code changes.

## Completion Criteria Status

| Criterion | Status | Evidence |
| --- | --- | --- |
| Partial-SVD external work is not treated as full-SVD parity reuse. | Complete | Semantic difference table separates value, vector, subspace, ordering, convergence, and tolerance gates. |
| Vector/subspace and convergence risks are explicit. | Complete | See vector and subspace risk notes. |
| No partial-SVD external parity claim is introduced. | Complete | Non-claim register preserves partial-SVD external parity as unearned. |
