# Sprint 218 Plan: Linear-Time Matmul Output Construction

**Sprint Duration:** 14 days
**Goal:** Remove `sparse_matmul`'s repeated sorted-row head scans while
preserving observable results, ownership, and failure behavior.

**Time budget:** 10 hours per day, 140 hours total. No day exceeds 12 hours;
the Epic 20 capacity limit is 168 hours per sprint. Estimates are engineering
hours, not elapsed CI time.

**Primary scope:** `src/sparse_matrix.c`'s matmul output flush, the internal
row/column linkage needed to build its result, `tests/test_matmul.c`, the
existing allocation-failure gate, and selected-kernel measurements. Preserve
the public `SparseMatrix` API and input matrices. A sorted touched-index pass
may remain; the target is linear construction *after sorting*, not a claim
that the complete multiplication is linear in output size.

**Evidence rule:** Pin inputs, toolchain, build mode, timing method, and peak
memory method before recording a baseline. Retain raw before/after samples
and failed as well as passing checks. A local speedup is not portable speed
evidence. A failed, skipped, or unavailable required gate blocks completion.

**Non-goals:** A general storage-format rewrite, a new public matrix type,
unrelated sparse-kernel optimization, universal asymptotic speed claims,
or state-of-the-art performance claims.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 218.1 Baseline And Semantics | 20 h | Days 1-2: 10 + 10 |
| 218.2 Output Builder | 32 h | Days 3-6: 10 + 10 + 10 + 2 |
| 218.3 Correctness Matrix | 28 h | Days 6-8: 8 + 10 + 10 |
| 218.4 Failure Lifecycle | 24 h | Days 9-11: 10 + 10 + 4 |
| 218.5 Controlled Measurement | 18 h | Days 11-13: 6 + 10 + 2 |
| 218.6 Quality Closure | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Matmul Contract And Cost Inventory

**Title:** Matmul Contract And Cost Inventory
**Theme:** Locate the repeated scans and freeze the caller-visible contract.
**Time estimate:** 10 hours (218.1: 10 h)

### Tasks

1. Trace `sparse_matmul`, `sparse_insert`, node creation, row/column headers,
   permutations, `sparse_free`, and allocation helpers. Identify every row
   head scan and column-link scan in the current flush.
2. Record shape and NULL error precedence, `*C = NULL` behavior, zero-drop
   threshold, accumulation order, sorted output, duplicate contributions,
   and non-finite/overflow behavior observed by existing tests.
3. Inventory `test_matmul`, the `matmul-allocation-failure-gate`, Make/CTest
   registration, and existing dense-oracle or CSR/CSC helpers.
4. Start `WORKING_NOTES.md` with a semantics table, owner map, risk ledger,
   and command/evidence log. Do not change the kernel on this day.

### Deliverables

- Current contract and output-construction cost map.
- Working-notes scaffold with test and allocation-site inventory.

### Completion Criteria

- Every current flush allocation and repeated scan has an owner and location.
- Error/output and ordering semantics are explicit enough to test before edits.
- No performance number is presented without a reproducible command.

---

## Day 2: Pinned Fixtures And Baseline

**Title:** Pinned Fixtures And Baseline
**Theme:** Establish comparable pre-change correctness, time, and memory data.
**Time estimate:** 10 hours (218.1: 10 h)

### Tasks

1. Pin at least three deterministic products: a narrow sparse result, a wide
   row with many emitted columns, and a rectangular/dense-result case. Record
   dimensions, input/output nonzeros, seed if used, and expected zero drops.
2. Build a reproducible selected-kernel driver or adapt an existing benchmark
   so setup and measurement phases are separated; prevent stale build objects
   and unrelated work from contaminating comparison.
3. Capture raw repeated timing samples and peak-memory observations for the
   current implementation, including host, compiler, flags, command, and exit
   status. Record variation rather than reducing samples to one speedup.
4. Run the existing matmul test and allocation-failure gate unchanged, and
   retain their output as the pre-change correctness baseline.

### Deliverables

- Pinned fixture manifest and raw pre-change timing/memory record.
- Passing or explicitly blocked baseline tests and semantic snapshot.

### Completion Criteria

- The wide-row fixture actually exercises repeated head scans.
- Each measurement can be rerun against the same fixture and build mode.
- Item 218.1's 20-hour baseline/semantics allocation is closed.

---

## Day 3: Builder Ownership Design

**Title:** Builder Ownership Design
**Theme:** Specify append order, node ownership, and rollback before coding.
**Time estimate:** 10 hours (218.2: 10 h)

### Tasks

1. Compare a row-append builder with a compressed temporary output followed
   by one conversion; choose the smallest internal design compatible with
   existing `SparseMatrix` row and column chains.
2. Specify how sorted touched columns give ascending row order and how rows
   processed in order give ascending column links without rescanning heads.
   Account for permutation state and any input aliasing assumptions.
3. Define ownership of allocated nodes, row/column tails or offsets, and
   temporary arrays at every success and failure point. Keep publication of
   `*C` until construction is complete.
4. Write an explicit complexity and peak-memory bound for the chosen builder,
   separating accumulation/sort cost from output-link cost.

### Deliverables

- Internal builder design with invariants and allocation/rollback diagram.
- Reviewable complexity and memory model for selected fixtures.

### Completion Criteria

- No output node requires a scan from an existing row or column head.
- Cleanup has one unambiguous owner for every partially built node.
- The design preserves the Day 1 public contract without adding public API.

---

## Day 4: Row Append Implementation

**Title:** Row Append Implementation
**Theme:** Replace the quadratic row flush with ordered node emission.
**Time estimate:** 10 hours (218.2: 10 h)

### Tasks

1. Implement the internal append path for non-dropped touched entries after
   the existing deterministic column sort; allocate each output node once.
2. Maintain the current `fabs(acc[col]) >= 1e-15` drop rule and accumulation
   order, including cancellation and NaN comparison behavior.
3. Link rows using a tail or equivalent append state, updating counts and
   matrix bookkeeping consistently with `sparse_insert`'s observable result.
4. Run focused matmul controls and inspect row ordering, duplicate
   contributions, and empty rows; record any temporary failure honestly.

### Deliverables

- Internal ordered row-emission path and focused row-link checks.
- Updated implementation/invariant notes.

### Completion Criteria

- The row flush does not restart at the row head for each emitted entry.
- Focused simple, wide-row, and cancellation cases retain their expected
  values and sorted rows.
- Partial-builder cleanup remains explicit, not deferred to a later refactor.

---

## Day 5: Column Links And Builder Integration

**Title:** Column Links And Builder Integration
**Theme:** Complete matrix linkage without replacing row scans with column scans.
**Time estimate:** 10 hours (218.2: 10 h)

### Tasks

1. Add column-tail or compressed-offset handling so each emitted node joins
   its column in ordered time as result rows are processed.
2. Check column headers, row headers, node cross-links, `nnz`, empty rows and
   columns, and the result's default permutation metadata.
3. Review allocation overflow checks for output dimensions, tail arrays, and
   node creation; free temporary state on every return path.
4. Exercise `sparse_get`, matvec, conversion, and `sparse_free` on products so
   malformed links cannot hide behind dense value comparisons.

### Deliverables

- Fully linked matmul output builder with no per-entry head scans.
- Focused structural and consumer-smoke results.

### Completion Criteria

- Row and column traversal both see the same ordered nodes and `nnz`.
- A partially built output is never returned to callers.
- Existing basic matmul tests pass or a specific blocker is recorded.

---

## Day 6: Builder Review And Dense Oracle

**Title:** Builder Review And Dense Oracle
**Theme:** Freeze the construction path and start independent numerical checks.
**Time estimate:** 10 hours (218.2: 2 h; 218.3: 8 h)

### Tasks

1. Review the builder's final ownership and complexity argument, removing
   remaining per-value `sparse_insert` or head-scan paths (218.2: 2 h).
2. Implement a simple independent dense product oracle for small matrices;
   do not compute expectations through the new sparse output builder.
3. Add rectangular, identity, empty, zero-row/column, and dense-result
   comparisons with exact shapes, values, nonzero counts, and ordering.
4. Run focused tests with both Make and CTest entry points and retain outputs.

### Deliverables

- Reviewed, bounded matmul output builder.
- Initial independent dense-oracle regression matrix.

### Completion Criteria

- Item 218.2's 32-hour builder allocation is closed.
- The independent oracle can detect wrong values and malformed output shape.
- Focused tests do not rely only on the same implementation under test.

---

## Day 7: Cancellation And Duplicate Contributions

**Title:** Cancellation And Duplicate Contributions
**Theme:** Prove the builder preserves accumulation and zero-drop semantics.
**Time estimate:** 10 hours (218.3: 10 h)

### Tasks

1. Add cases where multiple `A(i,j) * B(j,col)` paths contribute to one
   output entry, including exact cancellation and values on both sides of the
   historical drop threshold.
2. Compare values with the independent dense oracle while separately checking
   the expected sparse support after zero dropping; do not require exact
   floating-point equality if the reference sums in a different order.
3. Add wide-row ordering and repeated-run determinism checks using row/column
   traversal or conversion, not only random `sparse_get` lookups.
4. Run the focused matmul suite and inspect any threshold or ordering mismatch
   against the frozen Day 1 semantics.

### Deliverables

- Cancellation, duplicate-contribution, threshold, and ordering regressions.
- Focused comparison results with justified tolerances.

### Completion Criteria

- Accumulated duplicates produce one ordered output entry when retained.
- Cancellation and near-threshold support match pre-change behavior.
- Repeated runs produce the same structure and values for pinned inputs.

---

## Day 8: Seeded Shape And Consumer Matrix

**Title:** Seeded Shape And Consumer Matrix
**Theme:** Close numerical coverage across sparse and dense result shapes.
**Time estimate:** 10 hours (218.3: 10 h)

### Tasks

1. Add deterministic seeded small matrices spanning tall/wide rectangular
   shapes, sparse and dense results, and empty intermediate rows.
2. Compare against the independent oracle and verify `sparse_get`, matvec,
   CSR/CSC conversion where available, traversal order, and unchanged inputs.
3. Check finite large-value and non-finite controls against the recorded
   pre-change semantics without silently broadening public numeric contracts.
4. Run the complete matmul and relevant conversion tests; record coverage and
   close the correctness discrepancy ledger.

### Deliverables

- Seeded oracle suite and downstream structural-consumer checks.
- Correctness matrix with pass/fail evidence by fixture.

### Completion Criteria

- Item 218.3's 28-hour correctness allocation is closed.
- Rectangular, cancellation, duplicate, empty, seeded, and ordering cases
  have explicit expectations and pass.
- Any remaining discrepancy is treated as a blocker, not a speed tradeoff.

---

## Day 9: Allocation-Site Failure Map

**Title:** Allocation-Site Failure Map
**Theme:** Make every new builder allocation observable under fault injection.
**Time estimate:** 10 hours (218.4: 10 h)

### Tasks

1. Enumerate allocation sites for the result matrix, accumulator, flags,
   touched list, tail/offset state, and emitted nodes after the rewrite.
2. Extend `tests/test_matmul.c` and the existing allocation-failure gate so
   injected failure reaches each new site, including mid-row and mid-column
   construction after earlier nodes have linked successfully.
3. For each failure, assert the expected allocation error, `*C == NULL`,
   unchanged inputs, and safe cleanup; also test a stale non-NULL output
   pointer if that behavior is in the Day 1 contract.
4. Retain the allocation-site-to-test mapping and run the focused gate.

### Deliverables

- Exhaustive new-allocation fault-injection matrix.
- Focused failure-gate results and site coverage record.

### Completion Criteria

- Every newly introduced allocation has a deterministic failing fixture.
- No failed call publishes a partially linked output or modifies inputs.
- The existing gate remains registered and actually exercises the new path.

---

## Day 10: Rollback And Retry

**Title:** Rollback And Retry
**Theme:** Validate partial-output cleanup and repeatability after failure.
**Time estimate:** 10 hours (218.4: 10 h)

### Tasks

1. Sweep failure positions across node creation in a wide row and across
   later rows/columns, not only the first allocation of each array.
2. Verify output ownership and deallocation under each failure; use the
   repository's leak/ASan tooling where available and record exact commands.
3. Immediately retry every failed case without injection and compare the
   result against the independent oracle and original inputs.
4. Recheck NULL and shape error precedence, zero-sized dimensions, and
   successful `sparse_free` for output produced after a retry.

### Deliverables

- Mid-build rollback, cleanup, and retry regressions.
- Leak/sanitizer observations tied to exact failure fixtures.

### Completion Criteria

- All injected failure paths leave `*C == NULL` and can be retried.
- No partial-output leak or input mutation is observed by the available gate.
- Public error precedence remains as frozen on Day 1.

---

## Day 11: Failure Closure And Measurement Setup

**Title:** Failure Closure And Measurement Setup
**Theme:** Finish failure coverage and lock the before/after comparison.
**Time estimate:** 10 hours (218.4: 4 h; 218.5: 6 h)

### Tasks

1. Close any remaining fault-injection gaps, rerun the registered allocation
   gate, and sign off the rollback map (218.4: 4 h).
2. Rebuild baseline and candidate from recorded revisions with identical
   compiler, flags, datasets, and timing boundaries; avoid cached or mixed
   instrumented build artifacts (218.5: 6 h).
3. Define warmup, repetition count, peak-memory collection, noise reporting,
   and raw-output retention for the three pinned shapes.
4. Record the exact host/environment and known confounders before comparison.

### Deliverables

- Closed allocation-failure ledger and passing focused gate.
- Reproducible baseline/candidate measurement protocol.

### Completion Criteria

- Item 218.4's 24-hour failure lifecycle allocation is closed.
- The performance comparison uses identical fixtures and build conditions.
- Raw results, not only derived ratios, have a retention destination.

---

## Day 12: Selected-Kernel Before/After Profile

**Title:** Selected-Kernel Before/After Profile
**Theme:** Measure local effect without turning it into a portable claim.
**Time estimate:** 10 hours (218.5: 10 h)

### Tasks

1. Run the pinned narrow, wide-row, and rectangular/dense-result workloads
   repeatedly on both revisions; retain all timing samples and exit statuses.
2. Collect comparable peak-memory observations, output nonzero counts, and
   correctness checks for both revisions so faster-but-wrong is rejected.
3. Summarize median, spread, and selected-shape differences; disclose cases
   with no improvement or memory regression instead of selecting only wins.
4. Review the row-flush complexity change against measured profiles and note
   residual sort, accumulation, and linked-structure costs.

### Deliverables

- Raw before/after timings, memory samples, and fixture manifest.
- Selected-kernel profile with uncertainty and limitations.

### Completion Criteria

- At least three pinned workloads have comparable complete sample sets.
- Every profiled output passes the same correctness checks.
- No portable or all-kernel performance conclusion is drawn from local data.

---

## Day 13: Measurement Freeze And Instrumented Checks

**Title:** Measurement Freeze And Instrumented Checks
**Theme:** Lock evidence and begin final independent quality gates.
**Time estimate:** 10 hours (218.5: 2 h; 218.6: 8 h)

### Tasks

1. Freeze the raw measurement bundle, derived summary, exact commands, and
   selected-kernel wording in the sprint artifact ledger (218.5: 2 h).
2. Run the focused matmul suite and registered allocation-failure gate from
   a clean normal build, then run the repository sanitizer path on changed
   code and fault-injection fixtures (218.6: 8 h).
3. Record toolchain, flags, pass/fail/skip status, and any sanitizer limits;
   do not treat an unavailable sanitizer as passing evidence.
4. Inspect the final diff for accidental public ABI, source-list, or unrelated
   storage-format changes and record remaining risks.

### Deliverables

- Frozen selected-kernel evidence bundle and claim boundary.
- Focused/instrumented quality results and final risk list.

### Completion Criteria

- Item 218.5's 18-hour measurement allocation is closed.
- Required focused and sanitizer gates pass, or promotion is blocked with a
  specific failing command and owner.
- The final diff remains inside the selected matmul ownership boundary.

---

## Day 14: Full Quality And Sprint Closeout

**Title:** Full Quality And Sprint Closeout
**Theme:** Prove Make and CMake parity and close only verified claims.
**Time estimate:** 10 hours (218.6: 10 h)

### Tasks

1. Return to a clean normal build after instrumentation, then run
   `make format && make lint && make test`; retain exact commands and results.
2. Run the reviewed CMake configure/build/CTest parity path and the focused
   `test_matmul` label or selector, recording counts and platform/toolchain.
3. Re-run the allocation-failure gate after the final build, verify output
   ownership and retained raw measurement files, and inspect final status.
4. Update `WORKING_NOTES.md` and the closeout artifact with completed items,
   blockers, evidence links, and calibrated selected-kernel claims.

### Deliverables

- Passing full C-quality, CMake/CTest, sanitizer, and matmul-failure records.
- Final evidence/claim ledger and Sprint 218 closeout notes.

### Completion Criteria

- Item 218.6's 18-hour quality allocation and all six Epic items are closed.
- The repeated head-scan cost is absent, numerical/failure behavior is
  preserved, and raw before/after evidence is retained.
- Any failed or unavailable required gate blocks completion; no broad speed
  or state-of-the-art claim is made.

---

## Sprint Completion Gate

Sprint 218 is complete only when the matmul output path no longer performs a
per-emitted-entry sorted-row or column head scan, deterministic dense-oracle
and allocation-failure regressions pass, the three-shape raw local profile is
retained, and the focused, full Make, CMake/CTest, and sanitizer gates are
green. The result supports only a measured selected-kernel statement on the
recorded host; it does not establish portable library-wide performance.
