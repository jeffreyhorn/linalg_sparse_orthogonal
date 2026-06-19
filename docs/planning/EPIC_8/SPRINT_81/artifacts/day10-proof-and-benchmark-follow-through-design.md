# Sprint 81 Day 10 - Proof and Benchmark Follow-Through Design

Date: 2026-06-19  
Branch: sprint-81

## Purpose

Fix the exact proof, benchmark, header, and support-surface follow-through
required after the Day 6 and Day 9 implementation batches without widening
Sprint 81 into generic docs churn or another implementation pass.

## Main Result

Sprint 81 now has one exact Day 11 follow-through contract:

- required surface:
  - `include/sparse_analysis.h`
- strongest support-only wording if the batch truly needs it:
  - `README.md`
  - `docs/maintainer_guide.md`
- lower-value support-only surfaces that do not currently need movement:
  - `benchmarks/README.md`
  - `examples/README.md`

## Strongest Remaining Contradiction

The strongest current contradiction is narrow and explicit:

- `include/sparse_analysis.h` still describes the shared Cholesky CSC
  repeated-run path as a larger-problem-only reuse lane
- that is now stale after Day 9, because the shared repeated-run Cholesky and
  LDL^T paths both stay on the analysis-backed CSC-aware route for all problem
  sizes

## Proof and Benchmark Result

The proof and benchmark side is already in the right place:

- `tests/test_integration.c` already owns the new below-threshold same-pattern
  Cholesky and LDL^T parity proofs
- `benchmarks/bench_refactor_csc.c` only needed the Day 9 comment correction
- no additional benchmark binary or proof-code follow-through is required

## Support-Surface Reading

The support-only docs lane is narrower than a generic cleanup pass:

- `README.md` already stays broadly truthful, but may benefit from a bounded
  wording refresh if the public repeated-run contract reads too
  large-`n`-centric after the header fix
- `docs/maintainer_guide.md` is the strongest policy-side support surface if
  the Day 11 wording batch needs one authoritative ownership refresh
- `benchmarks/README.md` and `examples/README.md` already reconcile cleanly
  with the landed batch and do not currently justify edits

## Preserved Fence

- no more proof-code expansion
- no more benchmark logic changes
- no generic README/tutorial/examples sweep
- no reopening of `src/sparse_matrix.c` or `src/sparse_analysis.c`

## Exit State

- Sprint 81 now knows the exact Day 11 touch set.
- The strongest required follow-through is narrowed to the public repeated-run
  header contract, with README and maintainer wording support-only.
- Day 11 can stay bounded instead of turning into a generic support-surface
  cleanup pass.
