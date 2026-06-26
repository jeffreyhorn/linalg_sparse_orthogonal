# Sprint 92 Day 10: Observability and Proof Design

## Purpose

Freeze one exact Day 11 observability center so Sprint 92 can expose backend
selection, fallback behavior, and bounded repeated-run evidence for the widened
direct-family dense backend surface without reopening broad solver, proof, or
package work.

## Main Result

Sprint 92 now has one exact Day 11 observability contract:

- required Day 11 center:
  - `benchmarks/bench_refactor_csc.c`
- directly forced support-only follow-through only if the Day 11 contract truly
  needs them:
  - `benchmarks/README.md`
  - `README.md`
  - `docs/maintainer_guide.md`
- retained adjacent proof owners only if the benchmark lane exposes a real
  contradiction they must cover:
  - `tests/test_ldlt.c`
  - `tests/test_ldlt_csc.c`
- retained later surfaces unless the benchmark contract truly forces them:
  - `scripts/bench_canonical_report.sh`
  - `Makefile`
  - `CMakeLists.txt`
  - QR/backend follow-through

## Exact Day 11 Center

The exact Day 11 center is now explicit:

- keep the remaining Sprint 92 evidence gap benchmark-owned
- use the retained repeated-run direct benchmark owner:
  - `bench_refactor_csc --indefinite-kkt`
- do not reopen the LDLT proof-owner files unless the benchmark lane exposes a
  real contradiction the current tests do not already cover

The strongest reason for that choice is now explicit too:

- the retained LDLT proof owner already proves:
  - builtin env selection
  - accelerate env selection
  - external env selection
  - actual solve correctness under the widened env contract
- the remaining gap is not basic correctness anymore
- the remaining gap is benchmark-side observability:
  - which dense backend actually ran
  - whether requested external selection fell back to builtin
  - what bounded repeated-run LDLT evidence Sprint 92 wants to keep

## Frozen Reporting Shape

Sprint 92's Day 11 reporting shape is now fixed to:

- backend selection visibility:
  - report the selected dense backend name on the maintained LDLT repeated-run
    benchmark lane
- fallback behavior visibility:
  - make builtin fallback visible when external acceleration is requested but
    not actually selected
- bounded performance evidence:
  - keep the workload bounded to `bench_refactor_csc --indefinite-kkt`
  - keep the evidence local to repeated-run LDLT direct workflow comparison
  - do not reinterpret the result as portable benchmark supremacy or a broad
    dense-kernel comparison matrix

In practical terms, Day 11 should prefer:

- self-describing benchmark output over new workflow or package policy
- one bounded LDLT repeated-run benchmark lane over a broad direct-family
  measurement rewrite
- evidence that the widened backend seam is real and observable over new
  benchmark breadth

## Strongest Clarification

The strongest Day 10 clarification is now explicit:

- Day 11 should not become a QR adoption batch
- Day 11 should not reopen LDLT correctness tests unless benchmark evidence
  shows a real gap
- Day 11 should not widen canonical reporting or package/build surfaces unless
  the benchmark contract truly changes them
- Day 11 should not try to prove portable external-backend superiority

## Exit State

- Sprint 92 now has one exact bounded observability center.
- Day 11 is fixed to `benchmarks/bench_refactor_csc.c`.
- Support-only wording and build/package movement stay behind real output or
  command-contract changes.
