# Day 14 Solver-Selection Claim Closeout

## Purpose

Close Sprint 130 by deciding whether the partial-SVD residual, subspace,
optimality, convergence-budget, and corpus outcomes justify a stronger public
solver-selection claim.

## Claim Gate Decision

No public solver-selection wording update was made.

Sprint 130 added useful maintainer-facing evidence for named partial-SVD
fixtures, but the accepted lanes are still fixture-bounded. They do not support
a broader user-facing claim than the current workflow guidance:

- use SVD and partial SVD for singular-value, rank, pseudoinverse, and
  low-rank workflows where their documented behavior applies;
- treat direct, iterative, QR, SVD, and partial-SVD solver selection as a
  workflow and matrix-property decision, not as a universal superiority claim;
- keep benchmarks, corpus smoke checks, and fixture-specific oracle checks out
  of portable performance or ecosystem-parity wording;
- avoid claiming broad vector/subspace, repeated-spectrum,
  clustered-spectrum, rank-deficient null-space, SuiteSparse corpus,
  sparse-output optimality, convergence-rate, partial-result, package,
  platform, or vendor-backend parity.

The earned Sprint 130 evidence belongs in maintainer evidence maps and future
claim gates, not in broader public solver-selection language.

## Project-Plan Reconciliation

| Item | Sprint 130 outcome | Claim impact |
| --- | --- | --- |
| 1. Partial-SVD dedupe and metric map | Completed Day 1 and Day 2 duplicate fences, metric ownership, tolerance policy, oracle boundaries, diagnostics, and failure classes. | No public wording change; this was policy and evidence hygiene. |
| 2. Rectangular and nonsymmetric residual evidence | Added `partial_svd_vector_residual_tall8x5_k3` and `partial_svd_vector_residual_nonsym_rect10x8_k3`; added external value fixture `partial_svd_nonsym_rect10x8_k3`. | No public wording change; evidence is bounded to one tall exact fixture and one stable top-3 nonsymmetric rectangular fixture. |
| 3. Repeated and clustered spectrum evidence | Defined projector policy, attempted `partial_svd_repeated_diag6_k3_projector`, and deferred after focused preflight failed value/projector gates. | No public wording change; no repeated or clustered evidence was accepted. |
| 4. Rank-deficient subspace evidence | Added `partial_svd_rankdef_diag6x4_k2_range_projector` for an exact rank-2 range-projector lane. | No public wording change; range-only evidence does not cover null-space, zero-crossing, pseudoinverse, or minimum-norm behavior. |
| 5. SuiteSparse corpus and low-rank optimality | Deferred SuiteSparse corpus promotion; added local analytic `partial_svd_lowrank_diag6x4_k2_frobenius_optimality`. | No public wording change; local dense Frobenius optimality does not support corpus, sparse-output, drop-tolerance, or broad best-rank claims. |
| 6. Convergence-budget evidence | Added `partial_svd_max_iter_fail_closed_diag6_k2` and verified default-budget recovery on the same fixture. | No public wording change; fail-closed behavior is not convergence-rate, achieved-tolerance, stagnation, or partial-result evidence. |
| 7. Solver-selection wording | Reviewed evidence and deferrals, then chose explicit no-update rationale. | Complete; public wording remains workflow guidance only. |

## Accepted Evidence Index

| Evidence | Artifact | Scope |
| --- | --- | --- |
| `partial_svd_vector_residual_tall8x5_k3` | Day 4 rectangular residual evidence | Tall exact diagonal `8x5`, `k=3`; external singular values, triplet residuals, and orthogonality. |
| `partial_svd_nonsym_rect10x8_k3` | Day 6 nonsymmetric rectangular evidence | External dense-reference singular values for stable top-3 nonsymmetric rectangular fixture. |
| `partial_svd_vector_residual_nonsym_rect10x8_k3` | Day 6 nonsymmetric rectangular evidence | Nonsymmetric rectangular `10x8`, `k=3`; triplet residuals, orthogonality, and shape diagnostics. |
| `partial_svd_rankdef_diag6x4_k2_range_projector` | Day 10 rank-deficient subspace evidence | Exact rank-2 `6x4` diagonal fixture; left and right range-projector evidence only. |
| `partial_svd_lowrank_diag6x4_k2_frobenius_optimality` | Day 12 low-rank optimality evidence | Local analytic dense-reconstruction Frobenius target for discarded diagonal tail. |
| `partial_svd_max_iter_fail_closed_diag6_k2` | Day 13 convergence-budget evidence | Bounded max-iteration failure with no published payload plus default-budget recovery. |

No Sprint 130 evidence was accepted for repeated spectra, clustered spectra,
SuiteSparse corpus parity, sparse-output/drop-tolerance optimality,
rank-deficient null-space behavior, convergence rates, achieved tolerance,
stagnation, or partial-result publication.

## Final Deferral Register

| Deferred lane | Blocker / dependency | Future owner |
| --- | --- | --- |
| Wide rectangular vector residual | Needs independent vector-residual policy and a fixture that checks both triplet equations and orthogonality, not only one product direction. | Sprint 131 rectangular residual owner |
| Repeated-spectrum projector evidence | Day 8 preflight failed value and projector gates; implementation or option semantics must handle repeated leading blocks before claim promotion. | Sprint 131 repeated-spectrum owner |
| Partial selection through repeated multiplicity | Needs containment or subspace policy when `k` cuts through an equal singular-value block. | Sprint 131 repeated-spectrum owner |
| Clustered-spectrum evidence | Needs explicit gap policy, tolerance windows, convergence budget, and projector/principal-angle diagnostics. | Sprint 131 clustered-spectrum owner |
| Nonsymmetric top-4 near-zero tail | Needs rank and clustered-tail semantics before individual-vector or subspace evidence is meaningful. | Sprint 131 nonsymmetric/rank owner |
| `k > rank` zero-crossing behavior | Needs positive/zero singular-value threshold, payload semantics, and range/null-space split. | Sprint 131 rank-deficient owner |
| Rank-deficient null-space projectors | Needs explicit left/right null-space oracle definitions and zero-slot behavior. | Sprint 131 rank-deficient owner |
| Duplicate-column projector evidence | Needs an independent projector oracle that is not ambiguous under duplicate-column orientation. | Sprint 131 rank-deficient owner |
| SuiteSparse corpus residual or vector parity | Needs independent metadata, oracle source, support tier, skip policy, diagnostics, tolerance windows, and runtime class. | Sprint 131 corpus owner |
| Large-matrix SVD lanes | Needs optional-data policy and failure interpretation that separates support, speed, and correctness claims. | Sprint 131 corpus owner |
| Sparse-output and drop-tolerance optimality | Needs metric ownership for sparsity pattern, truncation, drop tolerance, and reconstruction error. | Sprint 131 low-rank owner |
| Broad low-rank optimality | Needs more than one analytic dense fixture and must state Frobenius versus spectral-norm semantics. | Sprint 131 low-rank owner |
| Iteration-count reporting | Current public result object does not expose iteration count. | Sprint 131 convergence API owner |
| Achieved-tolerance reporting | Current public result object does not expose achieved tolerance or residual history. | Sprint 131 convergence API owner |
| Partial-result publication | Current fail-closed behavior publishes no payload on non-convergence; any partial-result support needs API semantics. | Sprint 131 convergence API owner |
| Stagnation and convergence-rate behavior | Needs diagnostics and fixtures that distinguish slow convergence, stagnation, and tolerance success. | Sprint 131 convergence owner |
| Public solver-selection wording | Needs broad evidence that exceeds named fixture boundaries without contradicting non-claims. | Sprint 131 claim-gate owner |

## Maintainer And Public Wording Result

`docs/maintainer_guide.md` was refreshed during Sprint 130 with the bounded
fixture names accepted by Days 4, 6, 10, 12, and 13.

No README, public solver-selection guide, public API header, package,
platform, performance, or ecosystem-parity wording was expanded. Existing
public guidance remains workflow-oriented and does not advertise broad
partial-SVD residual, subspace, corpus, convergence, or optimality parity.

## Validation Package

Sprint 130 validation included focused `test_svd` runs for each implemented
partial-SVD lane and full C quality gates after `.c` or `.h` edits:

- Day 4: focused SVD validation and full `make format && make lint && make
  test`.
- Day 6: Python helper compilation/invocation, focused SVD validation, and
  full `make format && make lint && make test`.
- Day 10: focused SVD validation and full `make format && make lint && make
  test`.
- Day 12: focused SVD validation and full `make format && make lint && make
  test`.
- Day 13: focused SVD validation and full `make format && make lint && make
  test`.

Day 14 is documentation-only. Final Day 14 hygiene:

- `git diff --check` passed.
- `rg -n "[[:blank:]]$" docs/planning/EPIC_11/SPRINT_130` passed with no
  matches.

## Sprint 131 Handoff

1. Repeated and clustered spectra should be first-class algorithm or option
   work before another projector claim attempt.
2. Rank-deficient expansion should split range, null-space, zero-slot,
   duplicate-column, pseudoinverse, and minimum-norm behavior before
   implementation.
3. Corpus work should start by adding independent metadata, support tier,
   skip behavior, runtime class, oracle provenance, diagnostics, and
   failure-interpretation policy.
4. Low-rank work should decide whether the next claim is dense Frobenius,
   spectral-norm, sparse-output, drop-tolerance, or corpus evidence.
5. Convergence work should decide whether to expose iteration counts,
   achieved tolerance, residual history, `n_converged`, or partial-result
   payload semantics before writing success claims.
6. Public solver-selection wording should remain unchanged until one of the
   above owners lands evidence that is broader than named fixture behavior.

## Completion Criteria

- Item 7 is complete.
- All Sprint 130 items are complete or explicitly deferred.
- Public solver-selection wording remains within evidence-supported workflow
  guidance.
- Every unresolved lane has a blocker, dependency, and future owner.
