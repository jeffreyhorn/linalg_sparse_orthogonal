# Sprint 93 Day 11: Proof-Surface & Runtime Evidence Design

## Purpose

Freeze one exact Day 12 evidence contract so Sprint 93 can close the remaining
runtime gap from the cleaner Day 10 control seam without reopening broad proof,
benchmark-governance, or support-surface churn.

## Main Result

Sprint 93 now has one exact Day 12 follow-through contract:

- required Day 12 center:
  - `benchmarks/bench_reorder.c`
- directly forced support-only follow-through only if the Day 12 batch truly
  needs them:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `scripts/bench_canonical_report.sh`
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
- retained later surfaces unless the evidence batch exposes a real contract
  mismatch:
  - `README.md`
  - `INSTALL.md`
  - `tests/test_threads.c`
  - `tests/test_omp.c`

## Exact Day 12 Center

The exact Day 12 center is now explicit:

- keep the remaining Sprint 93 gap evidence-owned rather than proof-owned
- use the retained reorder benchmark owner:
  - `bench_reorder --sprint86-slice`
- expose the bounded runtime evidence needed for the touched ND lane after the
  Day 7 runtime reduction and Day 10 control cleanup

The strongest reason for that choice is now explicit:

- the touched proof owners already passed cleanly after the Day 10 landing:
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
- no new correctness contradiction surfaced from the runtime or control-model
  batches
- the remaining gap is not baseline proof trust anymore
- the remaining gap is bounded runtime evidence shape:
  - what Sprint 93 wants to keep reporting about the touched ND lane
  - how the Sprint 86 slice should read after the Day 7 recursion-side
    reduction
  - whether the touched benchmark lane needs a smaller, cleaner emitted shape
    before closeout

## Frozen Reporting Shape

Sprint 93's Day 12 reporting shape is now fixed to:

- bounded workload:
  - keep the evidence centered on `--sprint86-slice`
  - keep the fixture set bounded to the touched reviewed-runtime lane
- bounded fields:
  - preserve the current CSV core:
    - `matrix`
    - `n`
    - `reorder`
    - `nnz_L`
    - `reorder_ms`
    - `factor_ms`
  - add only the smallest extra runtime-evidence fields if the touched lane
    still needs them to read clearly after Day 7 and Day 10
- bounded interpretation:
  - calibrate the touched ND lane against AMD/RCM/none on the retained
    Sprint 86 slice
  - do not reinterpret the result as broad threading maturity, broad graph
    superiority, or generic benchmark supremacy

## Proof Rebalancing Call

The proof-topology call is now explicit:

- a Day 12 proof-owner rebalance is not currently required
- `tests/test_reorder_nd.c` remains heavy, but the Day 10 landing did not
  force new proof movement or expose weakened trust
- proof-owner movement should land only if the evidence batch shows a real
  mismatch that the current reviewed proof surfaces cannot explain or validate

## Strongest Clarification

The strongest Day 11 clarification is now explicit:

- Day 12 should not become another `src/sparse_reorder_nd.c` implementation batch
- Day 12 should not widen into generic proof splitting just because the runtime
  owner is large
- Day 12 should not widen canonical reporting beyond the touched reorder lane
  unless the bounded evidence contract truly forces it
- Day 12 should not reopen public/install wording detached from a real
  benchmark or reporting-surface change

## Exit State

- Sprint 93 now has one exact bounded evidence center.
- Day 12 is fixed to `benchmarks/bench_reorder.c`.
- Proof-owner movement and support-surface wording stay behind real evidence
  changes rather than being pulled forward speculatively.
