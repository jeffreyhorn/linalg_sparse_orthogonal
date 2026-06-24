# Sprint 86 Retrospective

**Sprint:** 86 — Reordering Scalability & Reviewed Runtime Convergence  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 86 fixed the reviewed-runtime baseline, proof split, and
      implementation-day validation contract before landing runtime work
- [x] the strongest live reviewed-runtime contradiction map was reranked from
      the current tree rather than inherited generically from Sprint 85
- [x] Sprint 86 fixed one explicit first implementation fence centered on:
  - `src/sparse_reorder_nd.c`
- [x] Sprint 86 landed one bounded ND runtime reduction batch:
  - the shipped ND base threshold moved from `128` to `160`
  - the default reviewed/runtime lane now routes more medium cases to the
    leaf-AMD path before full ND expansion
  - the strongest reviewed long pole was materially reduced in code
- [x] Sprint 86 landed one bounded proof-surface rebalancing batch:
  - `tests/test_reorder_nd.c` now uses cached heavy fixtures and grouped
    local runner structure for repeated ND families
  - the proof-owner cleanup stayed inside the retained ND proof owner
- [x] Sprint 86 landed one bounded benchmark/comparison follow-through batch:
  - `bench_reorder --sprint86-slice` now owns the cheap branch-local runtime
    rerun surface for `bcsstk14` and `Pres_Poisson`
  - `make bench-reorder-sprint86` makes that rerun contract explicit without
    widening the canonical maintained benchmark face
- [x] Sprint 86 used bounded follow-through correctly:
  - `tests/test_reorder_nd.c` remained the retained ND proof owner
  - `tests/test_reorder.c`, `tests/test_reorder_amd_qg.c`, and
    `tests/test_graph.c` remained retained adjacent proof owners
  - `README.md`, `docs/maintainer_guide.md`, workflow files, install/export
    proof, and canonical reporting surfaces were correctly not widened where
    the sprint did not change their maintained contract
- [x] Sprint 86 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 86 closed with one explicit Sprint 87-first handoff queue instead
      of another generic Epic 8 runtime summary

## What Went Well

1. **Sprint 86 chose the right first runtime lane.**
   The sprint did not start by splitting tests, widening benchmark policy, or
   reopening Sprint 85 maintainability work. It first reduced the strongest
   runtime contradiction directly on the ND policy seam in
   `src/sparse_reorder_nd.c`.

2. **The Day 6 runtime landing delivered a real measured win.**
   The bounded threshold shift reduced the strongest reviewed long pole from:
   - `test_reorder_nd`: `283.53 sec` -> `138.68 sec`
   - reviewed CMake total: `404.15 sec` -> `234.05 sec`

3. **The runtime fence stayed disciplined.**
   The first landing remained:
   - runtime-owned
   - bounded to the ND orchestration/default seam
   - proof-preserving
   - free of generic graph-family rewrite churn

4. **Proof ownership stayed explicit after the runtime work.**
   Day 9 improved the ND proof-owner structure without redistributing
   correctness ownership into adjacent reorder or graph binaries.

5. **The measurement follow-through moved the right surface.**
   Day 11 did not widen `bench-canonical-report` or create a timing gate. It
   added one explicit branch-local runtime slice under `bench_reorder`, which
   is where the touched Sprint 86 evidence actually belongs.

6. **Sprint 87 now starts from a much smaller reviewed runtime burden.**
   Packaging/ABI/install-export convergence can now proceed on top of a
   materially smaller reviewed baseline instead of carrying the earlier ND
   runtime drag.

## What Didn't Go Well

1. **The proof-surface rebalance did not improve runtime on its own run.**
   Day 9 was a valid cleanup, but it did not beat the Day 6 runtime anchor on
   that machine. The measured value moved to:
   - `test_reorder_nd`: `144.95 sec`
   - reviewed total: `246.07 sec`

2. **The strongest long pole still remains `test_reorder_nd`.**
   Sprint 86 materially reduced it, but it still dominated the Day 13 close:
   - reviewed `test_reorder_nd` = `135.01 sec`
   - reviewed total = `229.94 sec`

3. **The measurement package stayed intentionally branch-local.**
   That was the correct bounded decision, but it means Sprint 86 did not
   widen the canonical maintained benchmark/reporting contract.

4. **The sprint reduced runtime cost, not all reorder/graph complexity.**
   That was the correct bounded result, but it leaves residual adjacent work:
   - `tests/test_reorder_nd.c` remains a large proof owner
   - `tests/test_graph.c` remains a large adjacent proof owner
   - the graph-family implementation remains substantial even after the
     runtime win

5. **Sprint 86 correctly did not touch packaging or cross-platform consumer mechanics.**
   That is a strength for scope discipline, but it also means Sprint 87 still
   inherits the full packaging/ABI/install-export convergence package.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 86 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `229.94 sec` |
| reviewed `test_reorder_nd` time | `135.01 sec` |
| focused `test_reorder` follow-on | `38 / 38` |
| focused `test_reorder_nd` follow-on | `35 / 35` with `1` skip |
| focused `test_reorder_amd_qg` follow-on | `7 / 7` |
| focused `test_graph` follow-on | `61 / 61` |
| branch-local runtime slice | `make bench-reorder-sprint86` passed |
| canonical reporting follow-on | `make bench-canonical-report` passed |

### Runtime reduction headline

| Metric | Sprint 85 close | Day 6 landed | Sprint 86 close |
|---|---:|---:|---:|
| reviewed `test_reorder_nd` | `283.53 sec` | `138.68 sec` | `135.01 sec` |
| reviewed CMake total | `404.15 sec` | `234.05 sec` | `229.94 sec` |

### Sprint 86 artifact package

| Metric | Sprint 86 close state |
|---|---:|
| total artifact files under `SPRINT_86/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-reviewed-runtime-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-reviewed-surface-recheck.md`
  - `day3-reviewed-runtime-long-pole-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-ci-reviewed-path-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-runtime-scalability-boundary.md`
  - `day5-algorithm-proof-runtime-architecture-design.md`
  - `day6-nd-runtime-reduction-batch.md`
  - `day8-proof-surface-rebalancing-design.md`
  - `day9-proof-surface-rebalancing-batch.md`
  - `day10-benchmark-comparison-follow-through-design.md`
  - `day11-benchmark-comparison-follow-through-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 86 close state |
|---|---:|
| implementation `src/` files touched | `2` |
| internal header files touched | `1` |
| proof-owner test files touched | `1` |
| benchmark source files touched | `1` |
| build/command surfaces touched | `1` |
| benchmark-local support docs touched | `1` |
| repo-wide support docs requiring follow-through | `0` |

Notes:

- implementation `src/` files touched:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph.c`
- internal header files touched:
  - `src/sparse_reorder_nd_internal.h`
- proof-owner test files touched:
  - `tests/test_reorder_nd.c`
- benchmark source files touched:
  - `benchmarks/bench_reorder.c`
- build/command surfaces touched:
  - `Makefile`
- benchmark-local support docs touched:
  - `benchmarks/README.md`
- support surfaces intentionally left untouched after recheck:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `scripts/bench_canonical_report.sh`
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`

## Residual Deferred Debt

Sprint 86 deliberately stopped after the highest-value reviewed-runtime
package. The main open work it hands forward is:

- packaging, ABI, install/export, and cross-platform quality convergence
- later bounded iterative/eigensolver maintained external differential
  widening only where bounded evidence still justifies it
- later adjacent reorder/runtime follow-through only where refreshed runtime
  evidence justifies more change beyond the bounded Sprint 86 lane

Still consciously constrained rather than silently “solved”:

- no generic graph-family optimization claim
- no benchmark-governance widening into the canonical maintained surface
- no new timing gate in reviewed CI
- no package/install/export/runtime-package claim broadening
- no proof-owner redistribution out of `tests/test_reorder_nd.c`

Not carried forward as unresolved Sprint 86 debt:

- the baseline/reviewed-surface recheck
- the live reviewed-runtime rerank
- the bounded algorithm / proof runtime architecture contract
- the Day 6 ND runtime reduction landing
- the Day 9 proof-owner/runtime-surface rebalance
- the Day 11 benchmark/comparison follow-through landing
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 87-first handoff queue

## Key Deliverables

1. **One bounded ND runtime reduction landed on the strongest reviewed long pole.**
   `src/sparse_reorder_nd.c` now routes more medium cases through the
   leaf-AMD path first by default, materially reducing the strongest reviewed
   runtime burden.

2. **One bounded ND proof-owner rebalance landed without weakening correctness ownership.**
   `tests/test_reorder_nd.c` now has cached heavy-fixture handling and grouped
   local runners for repeated ND families while remaining the authoritative ND
   reviewed proof owner.

3. **One bounded branch-local runtime evidence surface landed.**
   `bench_reorder --sprint86-slice` and `make bench-reorder-sprint86` now
   provide the cheap rerun surface for the touched Sprint 86 corpus without
   widening the canonical maintained benchmark face.

4. **Sprint 86 closed from a measured reviewed-runtime baseline, not just from optimization prose.**
   The branch ended with a full Day 13 validation sweep, focused reorder/graph
   reviewed reruns, runtime-slice confirmation, canonical report confirmation,
   and an explicit Day 14 handoff queue for Sprint 87 and later Epic 8 lanes.

## Bottom Line

Sprint 86 succeeded because it stayed bounded where the repo most needed it.
It did not pretend to solve every remaining reorder/runtime concern, but it
did materially reduce the strongest reviewed long pole, preserve the retained
ND proof-owner model, and separate branch-local runtime evidence from the
canonical maintained benchmark face. That is enough real runtime convergence
to make Sprint 87 the correct next Epic 8 step: packaging, ABI,
install/export, and cross-platform quality convergence on top of a much
smaller reviewed runtime burden instead of another round of ND runtime
reduction first.
