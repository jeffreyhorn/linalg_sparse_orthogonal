# Sprint 93 Day 9: Runtime-Control Cleanup Design

## Purpose

Freeze one exact Day 10 cleanup contract so Sprint 93 can sharpen the touched
ND runtime/threading control model without turning the second batch into a
broad graph-policy rewrite, proof-topology pass, or generic threading sweep.

## Main Result

Sprint 93 now has one exact second implementation contract:

- required Day 10 center:
  - `src/sparse_reorder_nd.c`
- directly forced support-only follow-through only if the Day 10 batch truly
  needs them:
  - `src/sparse_reorder_nd_internal.h`
  - `src/sparse_graph_internal.h`
  - `tests/test_reorder_nd.c`
  - `tests/test_graph.c`
  - `benchmarks/bench_reorder.c`
- strongest later surfaces only if runtime-control cleanup exposes a real
  maintained-contract mismatch:
  - `tests/test_threads.c`
  - `tests/test_omp.c`
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`

## Exact Day 10 Target

The exact Day 10 target is now explicit:

- stop treating ND runtime-control as a loose cluster of compatibility parsing
  plus stacked override begin/end calls
- keep the touched cleanup centered on `src/sparse_reorder_nd.c`
- preserve the current runtime-policy results while making the control seam
  smaller and sharper

In practical terms, Day 10 should center on:

- consolidating the ND compatibility/default policy normalization path
- tightening the override-staging seam around:
  - coarsening override
  - coarsen-floor-ratio override
  - coarsening-CV-fallthrough override
  - coarsest-bisection override
  - separator-lift override
- keeping the current typed-policy and compatibility semantics intact:
  - `sparse_reorder_nd_default_policy()` remains the baseline owner
  - typed policy still wins where the shipped contract says it should win
  - current env names and accepted values stay intact
- preserving the touched benchmark/test-only hooks unless the cleanup proves a
  strictly smaller owner can preserve them cleanly:
  - ND profile override
  - ND base-threshold hook

## Strongest Clarification

The strongest Day 9 clarification is now explicit:

- Day 10 should not become a generic graph-policy redesign
- Day 10 should not widen into FM/coarsening algorithm changes
- Day 10 should not remove or reinterpret the benchmark/test-only threshold
  hook as public product work
- Day 10 should not widen early to proof-topology or benchmark/reporting work
- Day 10 should not reopen public or maintainer wording detached from a real
  touched runtime-control movement

## Deferred Behind Day 10

The following remain later unless the runtime-control cleanup truly forces
them:

- proof-surface rebalancing in `tests/test_reorder_nd.c` or `tests/test_graph.c`
- bounded benchmark and runtime-evidence follow-through in
  `benchmarks/bench_reorder.c`
- thread/OMP interpretation work in:
  - `tests/test_threads.c`
  - `tests/test_omp.c`
- README / install / maintainer wording
- broader workflow or package-surface follow-through

## Validation Contract

The validation reading is now fixed:

- if Day 10 changes only touched `*.c` / `*.h` owners:
  - `make format`
  - `make lint`
  - `make test`
- if the cleanup widens materially into benchmark, proof, or support surfaces:
  - `make quality-review-full`

## Exit State

- The second Sprint 93 implementation center is fixed.
- Day 10 will stay code-owned and bounded to ND runtime-control cleanup on the
  touched owner.
- Later proof, benchmark, and support work remains sequenced behind a real
  landed control-model improvement.
