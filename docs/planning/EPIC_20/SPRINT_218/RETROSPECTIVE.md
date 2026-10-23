# Sprint 218 Retrospective

**Sprint:** 218 - Linear-Time Matmul Output Construction

**Duration:** 14 planned days on branch `sprint-218`

**Status:** Six selected items closed locally; review and merge pending

## Source And Definition Of Done

The [Epic 20 project plan](../PROJECT_PLAN.md) assigns six items and 140
estimated hours. The [sprint plan](PLAN.md), [working notes](WORKING_NOTES.md),
and [Day 14 closeout](artifacts/day14-quality-closeout.md) link the daily
evidence. Estimates are not measured hours. Here, complete means the
selected matmul output construction no longer rescans row or column heads
for each retained value, its tested numerical/ownership behavior is
preserved, and the required local quality gates passed. It does not mean
the entire multiplication is linear-time or that its performance is
portable.

- [x] Pinned the physical-index, output-order, threshold, error, and
      ownership contract and three deterministic baseline fixtures.
- [x] Replaced per-entry `sparse_insert` in the matmul flush with private,
      ordered row- and column-tail append after sorting touched columns.
- [x] Added independent dense-oracle, cancellation, duplicate, threshold,
      seeded-shape, traversal, and downstream-consumer regressions.
- [x] Covered explicit matmul result-shell, workspace, first-slab, and
      later-slab allocation failures with NULL output, unchanged inputs,
      stale-output ownership, rollback, and retry checks.
- [x] Retained 60 validated before/after process samples, 180 raw files,
      a SHA-256 manifest, derived CSV, and calibrated timing/memory summary.
- [x] Passed full normal Make and reviewed CMake suites, focused matmul and
      allocation-failure gates, and default-/one-node-slab ASan/UBSan checks.
- [ ] Established portable speed, reduced matmul-only peak memory,
      all-shape behavior, or library-wide/state-of-the-art performance.
      None was a selected deliverable.

## What Went Well

1. **Contract-first work kept the change narrow.** The Day 1 inventory
   identified the repeated sorted-row and column-head scans and recorded
   output order, drop threshold, accumulation order, permutations, and
   failure precedence before implementation. The existing entry builder
   was not reused because its duplicate and zero rules differ from matmul.
2. **The builder removed the selected cost without changing storage.**
   The candidate still accumulates and sorts touched columns, then links
   each retained node through row and column tails in O(1). The result owns
   nodes immediately through its pool; only the success path publishes
   `*C`. Public headers and the library source list did not change.
3. **Tests checked both values and structure.** The focused matmul suite
   grew from 18 baseline tests and 185 assertions to 41 tests and 372262
   assertions in the final normal-build gate. Independent dense products,
   row/column traversals, and consumer checks caught more than a lookup
   into the new output matrix would have established.
4. **Failure injection reached previously invisible sites.** Matmul-specific
   checked allocation now covers the result shell and node slabs as well as
   workspaces. The one-node-slab configuration made every subsequent node
   request injectable; default slabs exercised failures after thousands of
   already-linked nodes. Both sanitizer configurations passed, and the
   normal focused binary reported zero leaked bytes under native `leaks`.
5. **Measurement was reproducible and bounded.** The same fully validating
   driver ran against pinned baseline and candidate binaries. Ten
   alternating-order pairs per fixture all passed value, shape, and `nnz`
   checks. Raw streams and statuses remained intact through Day 14, with
   all 180 hashes revalidated.

## What Didn't Go Well

1. **The first baseline was not directly comparable to the final driver.**
   Day 2's five-sample run used an earlier driver revision and showed wide
   host variability, including a wide-row outlier. Days 11-12 rebuilt both
   revisions with one driver and a paired protocol; the early samples were
   retained as historical baseline context, not mixed into final ratios.
2. **The initial allocation gate missed shell and slab allocations.**
   Workspace injection alone could not demonstrate cleanup after a partially
   linked result. The new private checked path and later-slab sweep closed
   that explicit-site gap, but do not control allocations inside system
   libraries such as `qsort`.
3. **Short-row speed and memory evidence stayed inconclusive.** The banded
   time ranges overlapped and the candidate was slower in three of ten
   pairs. Candidate median whole-process RSS/footprint was 8192 bytes higher
   on banded and rectangular, equal on wide-row; these coarse peaks do not
   isolate matmul allocation. The new `B->cols` column-tail workspace is a
   real cost, so no memory-reduction claim is earned.
4. **Full validation was expensive and host-bounded.** Lint's clang-tidy and
   cppcheck passes and the reviewed CMake nested-dissection test took time.
   CMake emitted nonfatal `-Wdouble-promotion` warnings in existing code.
   Some opt-in/framework fixtures were skipped, and macOS ASan leak
   detection was disabled. Those are recorded as limits, not silently
   counted as covered behavior.

## Final Metrics And Claim

| Metric | Sprint 218 close state |
| --- | --- |
| Planned effort | 140 h across six items and 14 days; actual hours not tracked |
| Daily evidence | 14 primary day artifacts, working notes, this retrospective |
| Focused matmul | 41 tests, 0 failed/skipped, 372262 assertions in final normal gate; baseline was 18 tests and 185 assertions |
| Full normal Make | `make format`, fresh-build `make lint`, and `make test` passed; 59 test binaries; opt-in/framework skips disclosed |
| Reviewed CMake/CTest | Fresh clean build, test-count parity, 60/60 CTests passed; labeled matmul selector 1/1 passed |
| Instrumented focused tests | ASan/UBSan default- and one-node-slab CTests passed 1/1 each; native normal-build `leaks`: 0 leaks, 0 leaked bytes; ASan leak detection disabled |
| Retained measurement | Three fixtures, 10 pairs each, 60 successful invocations, 180 raw files verified by SHA-256 |
| Local kernel medians | Banded 0.0695 vs 0.0610 ms; wide-row 1.314 vs 0.109 ms; rectangular 3.948 vs 1.166 ms (baseline vs candidate) |

The earned result is an **ordered, linear-in-retained-nodes output-linking
stage after sorting** for `sparse_matmul`, plus selected local timing
evidence. On this host, the wide-row and rectangular candidate ranges did
not overlap baseline ranges, and all ten candidate pairs for each were
faster. Banded ranges overlapped. This is not a claim about total matmul
complexity, arbitrary sparsity patterns, reduced kernel memory, cross-host
speed, or state-of-the-art sparse linear algebra.

## Daily Evidence

- [Day 1 contract and cost inventory](artifacts/day1-matmul-contract-cost-inventory.md)
- [Day 2 pinned fixture baseline](artifacts/day2-matmul-baseline.md)
- [Day 3 builder and ownership design](artifacts/day3-builder-ownership-design.md)
- [Day 4 row append](artifacts/day4-row-append-implementation.md)
- [Day 5 column tails](artifacts/day5-column-tail-integration.md)
- [Day 6 builder review and dense oracle](artifacts/day6-builder-review-dense-oracle.md)
- [Day 7 cancellation and determinism](artifacts/day7-cancellation-determinism.md)
- [Day 8 seeded shapes and consumers](artifacts/day8-seeded-shape-consumer-matrix.md)
- [Day 9 allocation-site map](artifacts/day9-allocation-site-failure-map.md)
- [Day 10 rollback, retry, and sanitizer](artifacts/day10-rollback-retry-sanitizer.md)
- [Day 11 failure signoff and measurement protocol](artifacts/day11-failure-closure-measurement-protocol.md)
- [Day 12 selected-kernel profile](artifacts/day12-matmul-profile.md)
- [Day 13 evidence freeze and instrumented checks](artifacts/day13-measurement-freeze-quality.md)
- [Day 14 full quality and closeout](artifacts/day14-quality-closeout.md)

## Residual Handoff

Sprint 219 can take the installed index-width contract on its own gate; it
does not inherit a portable matmul performance claim. Future matmul work
would need broader matrix shapes, kernel-specific allocation measurement,
and independent host/compiler runs before promoting speed or memory claims.
The current focused sanitizer runs are not a full instrumented repository
suite, and opt-in skipped modes are not acceptance evidence for those modes.
No known failed required Sprint 218 gate blocks this selected local closeout.
