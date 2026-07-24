# Sprint 130 Retrospective

**Sprint:** 130 - Partial-SVD Residual Expansion & Solver-Selection Claim Gate
**Duration:** 14 days
**Status:** Complete

## Definition Of Done Checklist

- [x] Created Sprint 130 day-by-day plan, working notes, and artifact
      directory.
- [x] Re-read Epic 11 Sprint 130 scope, Sprint 124 residual debt, and Sprint
      129 handoff boundaries.
- [x] Established duplicate fences around existing partial-SVD singular-value,
      vector-residual, rectangular, rank-deficient, SuiteSparse, low-rank, and
      convergence smoke coverage.
- [x] Defined metric ownership for vector residuals, subspace/projector
      evidence, rank-deficient range/null-space splits, SuiteSparse corpus
      evidence, low-rank optimality, convergence-budget behavior, and public
      solver-selection wording.
- [x] Implemented bounded tall rectangular vector-residual evidence:
      `partial_svd_vector_residual_tall8x5_k3`.
- [x] Implemented bounded nonsymmetric rectangular evidence:
      `partial_svd_nonsym_rect10x8_k3` and
      `partial_svd_vector_residual_nonsym_rect10x8_k3`.
- [x] Attempted repeated-spectrum projector evidence and explicitly deferred
      it after value/projector preflight failure.
- [x] Implemented bounded rank-deficient range-projector evidence:
      `partial_svd_rankdef_diag6x4_k2_range_projector`.
- [x] Deferred SuiteSparse corpus claim promotion until independent metadata,
      oracle provenance, runtime, skip, diagnostic, and tolerance policy exist.
- [x] Implemented bounded local analytic Frobenius low-rank evidence:
      `partial_svd_lowrank_diag6x4_k2_frobenius_optimality`.
- [x] Implemented bounded max-iteration fail-closed evidence:
      `partial_svd_max_iter_fail_closed_diag6_k2`.
- [x] Refreshed `docs/maintainer_guide.md` with the accepted bounded
      partial-SVD evidence and non-claims.
- [x] Published a no-update rationale for public solver-selection wording.
- [x] Published the final evidence index, deferral register, validation
      package, closeout artifact, and Sprint 131 handoff notes.
- [x] Ran focused SVD validation for implemented lanes and full
      `make format && make lint && make test` gates after C/header edits.

## What Went Well

1. **Evidence moved from generic residual debt to named fixtures.** Sprint 130
   added concrete tall rectangular, nonsymmetric rectangular, rank-deficient
   range-projector, low-rank Frobenius, and convergence-budget lanes.

2. **Metric gates prevented overclaiming.** The sprint rejected or deferred
   lanes when ambiguity, missing oracle metadata, repeated-spectrum behavior,
   or API reporting gaps would make the claim broader than the evidence.

3. **Maintainer wording now reflects current proof.** The maintainer guide
   names the accepted bounded partial-SVD fixtures and keeps broad parity,
   performance, platform, corpus, and convergence non-claims visible.

4. **The repeated-spectrum failure was contained.** Day 8 removed the failing
   lane rather than weakening the assertion or landing evidence that did not
   meet the policy gate.

5. **The public claim gate stayed evidence-led.** The sprint added
   maintainer-facing confidence but did not convert fixture-local proof into a
   broad solver-selection claim.

## What Did Not Go Well

1. **Repeated and clustered spectra remain unsolved.** The first repeated
   leading-block projector attempt failed value and projector gates, so future
   work needs implementation or option semantics before claim work resumes.

2. **SuiteSparse corpus evidence is still product smoke.** Checked-in corpus
   data is useful for regression coverage, but Sprint 130 did not find an
   independent oracle and metadata package that could justify corpus parity.

3. **Convergence reporting is API-limited.** The library exposes `max_iter`
   and `tol` inputs, but not iteration count, achieved tolerance, residual
   history, `n_converged`, or partial-result status.

4. **Rank-deficient coverage remains range-only.** The accepted fixture proves
   one positive-rank projector case and does not cover null-space,
   zero-crossing, duplicate-column, pseudoinverse, or minimum-norm behavior.

5. **Low-rank optimality remains local and dense.** The accepted Frobenius lane
   is analytic and useful, but it does not prove sparse-output,
   drop-tolerance, corpus, spectral-norm, or broad best-rank behavior.

## Final Metrics

| Metric | Sprint 130 close state |
|---|---:|
| accepted rectangular vector-residual fixtures | 2 |
| accepted nonsymmetric rectangular external value fixtures | 1 |
| accepted repeated/clustered fixtures | 0 |
| accepted rank-deficient projector fixtures | 1 |
| accepted SuiteSparse corpus parity fixtures | 0 |
| accepted low-rank optimality fixtures | 1 |
| accepted convergence-budget fixtures | 1 |
| public solver-selection wording updates | 0 |
| maintainer evidence table updates | 1 |
| Sprint 130 artifact files | 14 |
| retrospective files | 1 |
| full C quality gates after implementation days | passed |
| final diff hygiene | passed |
| final Sprint 130 markdown whitespace scan | passed |
| Day 14 C quality rerun | not required; documentation-only closeout |

## Movement And Claim Outcomes

| Area | Outcome |
|---|---|
| Sprint intake and duplicate fencing | Completed in working notes and Day 1 artifact. |
| Metric policy | Completed Day 2 with evidence-class ownership and non-claim rules. |
| Rectangular vector residual | Added bounded tall fixture; deferred wide and broader rectangular claims. |
| Nonsymmetric rectangular residual | Added bounded stable top-3 fixture; deferred top-4 near-zero tail and broader nonsymmetric claims. |
| Repeated/clustered spectra | Policy completed; evidence deferred after failed repeated-spectrum preflight. |
| Rank-deficient subspace | Added bounded range-projector fixture; deferred null-space and zero-crossing claims. |
| SuiteSparse corpus | Policy completed; parity promotion deferred. |
| Low-rank optimality | Added bounded local dense Frobenius fixture; deferred sparse/drop-tolerance and broad optimality. |
| Convergence budget | Added bounded max-iteration fail-closed fixture; deferred iteration, tolerance, partial-result, and convergence-rate claims. |
| Maintainer evidence | Refreshed with bounded Sprint 130 fixtures and non-claims. |
| Solver-selection wording | No public wording expansion; no-update rationale published. |

## Residual Deferred Debt

Most important carry-forward work:

- Fix or define repeated leading-block behavior before retrying projector
  evidence.
- Define clustered-spectrum gap, ordering, tolerance, and convergence policy.
- Split rank-deficient expansion into range, null-space, zero-crossing,
  duplicate-column, pseudoinverse, and minimum-norm owners.
- Promote SuiteSparse corpus evidence only after independent metadata, oracle
  provenance, support tier, skip behavior, diagnostics, tolerance, and runtime
  policy exist.
- Decide the next low-rank claim class before adding more fixtures:
  Frobenius, spectral norm, sparse output, drop tolerance, or corpus.
- Add convergence reporting semantics before claiming achieved tolerance,
  iteration count, convergence rate, stagnation handling, or partial results.
- Refresh public solver-selection wording only when future evidence supports a
  user-facing claim beyond current workflow guidance.

Still consciously constrained rather than silently solved:

- no LAPACK, NumPy, SciPy, SuiteSparse, PETSc, Trilinos, Eigen, ARPACK, or
  vendor-backend parity claim;
- no broad SVD or partial-SVD vector/subspace parity claim;
- no broad rectangular, nonsymmetric, repeated-spectrum, clustered-spectrum,
  or rank-deficient null-space claim;
- no pseudoinverse, minimum-norm, sparse-output, drop-tolerance, or broad
  low-rank optimality claim;
- no convergence-rate, achieved-tolerance, stagnation, or partial-result
  claim;
- no package-manager distribution, ABI, platform, portable performance,
  scalability, memory, or state-of-the-art claim.

## Key Deliverables

- [PLAN.md](./PLAN.md)
- [WORKING_NOTES.md](./WORKING_NOTES.md)
- [day1-sprint-intake-residual-dedupe-baseline.md](./artifacts/day1-sprint-intake-residual-dedupe-baseline.md)
- [day2-partial-svd-dedupe-metric-map.md](./artifacts/day2-partial-svd-dedupe-metric-map.md)
- [day3-rectangular-residual-gate.md](./artifacts/day3-rectangular-residual-gate.md)
- [day4-rectangular-residual-evidence.md](./artifacts/day4-rectangular-residual-evidence.md)
- [day5-nonsymmetric-rectangular-gate.md](./artifacts/day5-nonsymmetric-rectangular-gate.md)
- [day6-nonsymmetric-rectangular-evidence.md](./artifacts/day6-nonsymmetric-rectangular-evidence.md)
- [day7-repeated-clustered-spectrum-policy.md](./artifacts/day7-repeated-clustered-spectrum-policy.md)
- [day8-repeated-clustered-spectrum-evidence.md](./artifacts/day8-repeated-clustered-spectrum-evidence.md)
- [day9-rank-deficient-subspace-gate.md](./artifacts/day9-rank-deficient-subspace-gate.md)
- [day10-rank-deficient-subspace-evidence.md](./artifacts/day10-rank-deficient-subspace-evidence.md)
- [day11-suitesparse-corpus-gate.md](./artifacts/day11-suitesparse-corpus-gate.md)
- [day12-lowrank-optimality-evidence.md](./artifacts/day12-lowrank-optimality-evidence.md)
- [day13-convergence-budget-evidence.md](./artifacts/day13-convergence-budget-evidence.md)
- [day14-solver-selection-claim-closeout.md](./artifacts/day14-solver-selection-claim-closeout.md)

## Final Status

Sprint 130 is complete. It added bounded partial-SVD residual, projector,
low-rank, and convergence-budget evidence, refreshed maintainer evidence, left
public solver-selection wording unchanged, and handed the remaining
partial-SVD claim debt to Sprint 131 with explicit blockers and owners.
