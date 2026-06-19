# Sprint 81 Retrospective

**Sprint:** 81 — Core Product / Storage Modernization Phase 2  
**Duration:** 14 days (Days 1-14 landed on this branch)  
**Status:** Complete

## Definition Of Done Checklist

- [x] Sprint 81 fixed the product/storage baseline, proof split, and
      implementation-day validation contract before landing code
- [x] the strongest live storage contradiction map was reranked from the
      current tree rather than inherited generically from Sprint 80
- [x] Sprint 81 fixed one explicit first implementation fence centered on:
  - `include/sparse_matrix.h`
  - `src/sparse_matrix.c`
- [x] Sprint 81 landed one bounded compressed-first construction/import batch:
  - `sparse_copy(...)`
  - `sparse_transpose(...)`
  - `sparse_load_mm(...)`
  now use one bulk-build seam instead of rebuilding through repeated
  `sparse_insert(...)` row/column searches
- [x] Sprint 81 landed one bounded repeated-run workflow convergence batch:
  - repeated-run Cholesky and LDL^T no longer fall back through the
    small-problem linked-list `build_permuted_copy(...)` path
  - the analysis-backed CSC-aware route is now used for all problem sizes
  - the symmetry/failure-preservation guard remained intact
- [x] Sprint 81 used bounded follow-through correctly:
  - `include/sparse_analysis.h` was reconciled with the landed repeated-run
    behavior
  - broader README/docs/examples churn was correctly avoided where the tree
    already stayed truthful
- [x] Sprint 81 ran the full validation sweep and closed from one explicit
      validated baseline:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- [x] Sprint 81 closed with one explicit Sprint 82-first handoff queue instead
      of another generic Epic 8 summary

## What Went Well

1. **Sprint 81 moved real product-model code instead of stopping at architecture prose.**
   The sprint landed two meaningful implementation batches:
   - Day 6 reduced the linked-list-first tax on copy/transpose/import
   - Day 9 removed the smaller-problem repeated-run fallback through
     `build_permuted_copy(...)` for Cholesky and LDL^T

2. **The first landing stayed well bounded.**
   Day 6 improved high-value construction/import seams without pretending to
   redesign the public `SparseMatrix` shell. Compatibility callers stayed
   intact while the internals became more compressed-first on touched paths.

3. **The second landing chose the right workflow seam.**
   The sprint did not widen immediately into wrapper churn or generic direct
   solver cleanup. It targeted the real contradiction:
   - repeated-run Cholesky and LDL^T were analysis-backed in principle
   - but still dropped back into a linked-list-first path on smaller problems
   Closing that inconsistency is high leverage for both product coherence and
   future backend work.

4. **Proof ownership stayed disciplined.**
   The sprint used the right proof owners:
   - `tests/test_sparse_matrix.c` for the Day 6 matrix-shell regression surface
   - `tests/test_integration.c` for public repeated-run direct parity and
     failure preservation
   - `benchmarks/bench_refactor_csc.c` only as benchmark-side throughput/proof
     context, not as the oracle owner

5. **The branch avoided low-value support-surface churn.**
   Day 10 and Day 11 narrowed correctly to the stale public header wording in
   `include/sparse_analysis.h`. `README.md`, `docs/maintainer_guide.md`,
   `benchmarks/README.md`, and `examples/README.md` were left alone because
   they already stayed truthful.

6. **Sprint 82 now has a cleaner starting point.**
   Sprint 81 reduced the product/storage contradiction enough that the next
   strongest Epic 8 center can move cleanly to:
   - dense/backend performance ceiling first
   instead of reopening the same linked-list-first workflow ambiguity.

## What Didn't Go Well

1. **The branch had to restore the Epic 8 planning tree before Sprint 81 could even start cleanly.**
   `master` did not carry `docs/planning/EPIC_8/` at the time Sprint 81 began,
   so Day 1 had to restore that planning context from `origin/sprint-80`.
   That did not change the Sprint 81 technical package, but it was operational
   overhead that should not have been necessary.

2. **Sprint 81 did not eliminate the linked-list compatibility shell itself.**
   That was the correct bounded choice, but the broader state-of-the-art gap
   remains open:
   - the public matrix model is still compatibility-first
   - only the highest-value touched seams became more compressed-first

3. **The repeated-run convergence batch was intentionally narrower than the full direct-family queue.**
   Day 9 handled:
   - Cholesky
   - LDL^T
   but explicitly deferred:
   - LU
   That kept the batch credible, but it means one residual direct-workflow
   split still exists for later Epic 8 work.

4. **The sprint did not reopen package/install proof, by design.**
   That is truthful rather than a defect, but it also means Sprint 81’s
   validation story is intentionally narrower than Sprint 80’s:
   - install/export proof was left untouched because no package mechanics moved

5. **The reviewed runtime long pole remains large.**
   `test_reorder_nd` still dominated reviewed runtime. Sprint 81 closed cleanly
   from a strong validation baseline, but it did not reduce that operational
   drag.

## Final Metrics

### Validation and reviewed anchors

| Metric | Sprint 81 close state |
|---|---:|
| standard code-day gate | `make format && make lint && make test` passed |
| strongest reviewed baseline | `make quality-review-full` passed |
| reviewed CMake `ctest -N` anchor | `53` |
| Makefile/CMake parity | `53 vs 53` |
| reviewed CMake `ctest` | `53 / 53` |
| reviewed CMake total time | `405.45 sec` |
| reviewed `test_reorder_nd` time | `277.62 sec` |
| focused `test_sparse_matrix` follow-on | `58 / 58` |
| focused `test_integration` follow-on | `53 / 53` |
| focused `test_chol_csc` follow-on | `147 / 147` |
| focused `test_ldlt` follow-on | `84 / 84` |

### Sprint 81 artifact package

| Metric | Sprint 81 close state |
|---|---:|
| total artifact files under `SPRINT_81/artifacts/` | `15` |
| baseline/audit artifacts | `6` |
| design/follow-through artifacts | `7` |
| validation/closeout artifacts | `2` |

Notes:

- baseline/audit artifacts:
  - `day1-scope-and-storage-baseline.md`
  - `day1-authoritative-inputs.txt`
  - `day2-validation-baseline-and-proof-surface-recheck.md`
  - `day3-storage-conversion-hotspot-audit.md`
  - `day7-post-landing-audit-and-rerank.md`
  - `day12-final-proof-alignment-and-validation-queue.md`
- design/follow-through artifacts:
  - `day4-first-storage-boundary.md`
  - `day5-compressed-first-architecture-design.md`
  - `day6-construction-import-batch1.md`
  - `day8-workflow-convergence-design.md`
  - `day9-workflow-convergence-batch.md`
  - `day10-proof-and-benchmark-follow-through-design.md`
  - `day11-docs-examples-header-alignment-batch.md`
- validation/closeout artifacts:
  - `day13-full-validation-sweep.md`
  - `day14-closeout-and-handoff.md`

### Landed implementation package

| Metric | Sprint 81 close state |
|---|---:|
| implementation `.c` files touched | `3` |
| public header files touched | `2` |
| proof-owner test files touched | `2` |
| benchmark source files touched | `1` |
| support docs/examples requiring follow-through | `0` |

Notes:

- implementation `.c` files touched:
  - `src/sparse_matrix.c`
  - `src/sparse_analysis.c`
  - `benchmarks/bench_refactor_csc.c`
- public header files touched:
  - `include/sparse_matrix.h`
  - `include/sparse_analysis.h`
- proof-owner test files touched:
  - `tests/test_sparse_matrix.c`
  - `tests/test_integration.c`
- support surfaces intentionally left untouched after recheck:
  - `README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `examples/README.md`

## Residual Deferred Debt

Sprint 81 deliberately stopped after the highest-value product/storage package.
The main open work it hands forward is:

- builtin scalar dense/backend performance ceiling
- bounded capability-surface widening on the highest-value solver seams
- later residual direct-workflow convergence beyond the bounded Cholesky and
  LDL^T landing
- later broader linked-list-shell containment only where bounded evidence
  justifies more product-model churn
- later Epic 8 assurance, maintainability, runtime, package/platform, and
  usability lanes in the preserved project-plan order

Still consciously constrained rather than silently “solved”:

- no broad public API redesign
- no fake compressed-only product claim
- no backend or capability spill hidden inside storage work
- no package/platform claim broadening
- no generic docs/examples sweep

Not carried forward as unresolved Sprint 81 debt:

- the baseline/proof-surface recheck
- the live storage contradiction rerank
- the bounded compressed-first architecture contract
- the Day 6 construction/import landing
- the Day 9 repeated-run convergence landing
- the bounded public header follow-through
- the Day 13 full validation sweep
- the Day 14 explicit Sprint 82-first handoff queue

## Key Deliverables

1. **One bounded compressed-first construction/import seam landed.**
   `sparse_copy(...)`, `sparse_transpose(...)`, and `sparse_load_mm(...)` now
   use a bulk-build route instead of repeated linked-list insertion searches.

2. **One bounded repeated-run direct-workflow contradiction was removed.**
   Cholesky and LDL^T now stay on the analysis-backed CSC-aware path for all
   problem sizes instead of reverting to the small-problem linked-list
   `build_permuted_copy(...)` path.

3. **One truthful public contract update landed.**
   `include/sparse_analysis.h` now says directly what the repeated-run Cholesky
   and LDL^T path actually does after the Day 9 implementation batch.

4. **One strong proof-owner split was preserved and extended.**
   Sprint 81 added focused proof where it belonged without widening into a
   broad regression rewrite.

5. **Sprint 81 closed from a measured baseline, not just from architecture prose.**
   The branch ended with a full Day 13 validation sweep and an explicit Day 14
   handoff queue for Sprint 82 and the later Epic 8 lanes.

## Bottom Line

Sprint 81 succeeded because it stayed narrow where the tree most needed it. It
did not try to solve all of Epic 8’s product-model debt at once. Instead, it
reduced the highest-value linked-list-first construction/import costs,
eliminated the smaller-problem repeated-run Cholesky and LDL^T fallback
contradiction, reconciled the public contract, and closed from a full reviewed
baseline. That is enough real product/storage movement to make Sprint 82 the
correct next contradiction center.
