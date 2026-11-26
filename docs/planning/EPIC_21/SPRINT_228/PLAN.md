# Sprint 228 Plan: Default Low-Rank Reconstruction Memory

**Sprint Duration:** 14 days

**Goal:** Remove the additional `m×n` reconstruction accumulator from the default sparse low-rank output path while preserving accepted numerical, structural, ownership, and failure behavior.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours. Unattended benchmark, sanitizer, and hosted-CI elapsed time is separate from these estimates.

**Prerequisites:** Use the [Epic 21 Sprint 228 scope](../PROJECT_PLAN.md), [Sprint 227 arithmetic and failure-invariant work](../SPRINT_227/RETROSPECTIVE.md), the existing `sparse_svd_lowrank_sparse()` outer-product route, and the maintained SVD corpus. Preserve the public ABI and both supported index widths.

**Evidence rule:** Freeze current behavior before changing defaults. Retain exact source SHA, compiler and build settings, fixture dimensions/ranks/tolerances, deterministic seeds, allocation-failure indices, raw timing and peak-memory observations, and test logs. Numerical references must be independent of the reconstruction route under test.

**Claim boundary:** This sprint may establish removal of the additional dense reconstruction accumulator from the default sparse low-rank output path and report results for pinned fixtures. It does not establish that full SVD is sparse-memory end to end, remove dense factor storage, prove portable speedups, or support broad state-of-the-art performance claims.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 228.1 Baseline And Contract | 20 h | Days 1–2: 10 + 10 |
| 228.2 Differential Corpus | 26 h | Days 3–5: 10 + 10 + 6 |
| 228.3 Default Path Promotion | 30 h | Days 5–8: 4 + 10 + 10 + 6 |
| 228.4 Failure Lifecycle | 24 h | Days 8–10: 4 + 10 + 10 |
| 228.5 Memory Accounting | 20 h | Days 11–12: 10 + 10 |
| 228.6 Quality Closure | 20 h | Days 13–14: 10 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Inventory Reconstruction Paths and Memory Owners

**Title:** Inventory Reconstruction Paths and Memory Owners

**Theme:** Establish the current implementation, contract, and evidence surface before edits.

**Time estimate:** 10 hours (228.1: 10 h)

### Tasks

1. Trace `sparse_svd_lowrank_sparse()` and its dense and outer-product reconstruction routes through public headers, `src/sparse_svd.c`, matrix construction helpers, tests, benchmarks, and environment-controlled selection.
2. Identify every allocation owned by decomposition, dense factors, reconstruction scratch, sparse output construction, and compatibility selection; record dimensions and overflow checks.
3. Freeze current status codes, output-pointer state, output ordering, duplicate/zero behavior, `drop_tol` semantics, ownership, and cleanup expectations for valid and invalid calls.
4. Capture exact baseline commands and representative allocation counts in `WORKING_NOTES.md` and a day-one artifact.

### Deliverables

- Reconstruction call graph and allocation-ownership inventory.
- Current behavior and error-contract baseline.

### Completion Criteria

- Both reconstruction routes and every material allocation have named owners and cleanup paths.
- The baseline distinguishes reconstruction scratch from storage required by full SVD factors.

---

## Day 2: Freeze Fixtures, Metrics, and Compatibility Contract

**Title:** Freeze Fixtures, Metrics, and Compatibility Contract

**Theme:** Define acceptance before selecting a new default.

**Time estimate:** 10 hours (228.1: 10 h)

### Tasks

1. Pin at least three rectangular and square shapes, multiple requested ranks, and zero/nonzero drop tolerances, including an all-zero and rank-deficient case.
2. Define independent dense-reference construction, elementwise and normwise tolerances, sparse structural checks, deterministic ordering expectations, and non-finite handling.
3. Specify the maintained default and controlled compatibility-selector behavior, including supported selector values, invalid values, and whether the selector is diagnostic or contractual.
4. Freeze measurement methodology for elapsed time, process peak RSS, reconstruction-only allocation bytes, repetitions, warmups, machine metadata, and raw-result retention.

### Deliverables

- Reviewed reconstruction acceptance contract.
- Pinned differential and measurement matrix.

### Completion Criteria

- Every success and failure outcome can be judged without using either production route as its own oracle.
- The contract explicitly limits the memory claim to the additional reconstruction accumulator.

---

## Day 3: Build the Core Differential Harness

**Title:** Build the Core Differential Harness

**Theme:** Compare both routes against an independent reconstruction oracle.

**Time estimate:** 10 hours (228.2: 10 h)

### Tasks

1. Add test helpers that run the current dense route, the outer-product route, and an independent dense reference from accepted SVD factors.
2. Cover square, tall, and wide matrices across rank one, intermediate rank, and maximum accepted rank.
3. Compare statuses, dimensions, logical values, `nnz`, row/column ordering, duplicate absence, and drop-threshold behavior.
4. Record deterministic fixture data and replay information without coupling expected values to traversal order.

### Deliverables

- Core route-to-route and independent-oracle differential harness.
- Shape/rank/tolerance regression set.

### Completion Criteria

- The harness detects meaningful numerical or structural divergence between either production route and the oracle.
- All fixtures are deterministic and safely exercise rectangular dimensions.

---

## Day 4: Extend Degenerate and Numerical Coverage

**Title:** Extend Degenerate and Numerical Coverage

**Theme:** Exercise cancellation, threshold boundaries, and difficult spectra.

**Time estimate:** 10 hours (228.2: 10 h)

### Tasks

1. Add all-zero, empty-pattern, repeated/clustered singular-value, rank-deficient, signed-cancellation, and sparse-output fixtures.
2. Test values immediately below, at, and above the accepted drop threshold using a documented floating-point comparison policy.
3. Add finite extreme-scale cases that expose accumulation-order differences without accepting NaN or infinity as valid output.
4. Verify retained entries and dropped entries against independently computed values and structural invariants.

### Deliverables

- Degenerate and threshold-boundary differential corpus.
- Numerical tolerance rationale and observed baseline results.

### Completion Criteria

- Zero, deficient-rank, cancellation, clustered, and sparse-output cases have explicit expected outcomes.
- Tolerance handling neither requires incidental operation ordering nor masks material errors.

---

## Day 5: Complete Differential Acceptance and Design Promotion

**Title:** Complete Differential Acceptance and Design Promotion

**Theme:** Close corpus gaps and define the maintained implementation seam.

**Time estimate:** 10 hours (228.2: 6 h; 228.3: 4 h)

### Tasks

1. Add deterministic randomized fixtures over bounded dimensions, ranks, densities, signs, and tolerances, with seed replay.
2. Confirm both index-width configurations where size-sensitive logic differs and audit `m×n`, rank, and allocation-size arithmetic for overflow.
3. Design the maintained outer-product default path, sparse-output construction seam, and controlled compatibility selector without changing public ABI.
4. Enumerate default-selection, explicit-compatibility, invalid-selector, and environment-isolation tests.

### Deliverables

- Completed pre-promotion differential corpus.
- Reviewed default-promotion and compatibility design.

### Completion Criteria

- Corpus coverage satisfies the frozen shape, rank, tolerance, and degenerate-case matrix.
- Promotion design identifies all selection, allocation, ownership, and cleanup transitions.

---

## Day 6: Harden the Outer-Product Reconstruction Route

**Title:** Harden the Outer-Product Reconstruction Route

**Theme:** Correct the lower-memory route before making it default.

**Time estimate:** 10 hours (228.3: 10 h)

### Tasks

1. Repair any numerical, ordering, duplicate, drop-threshold, dimension, or error-path discrepancies exposed by the differential corpus.
2. Ensure accumulation and sparse insertion/construction preserve the accepted result contract for rectangular and cancellation-heavy inputs.
3. Add checked size arithmetic and explicit non-finite handling at all newly exercised boundaries.
4. Run focused differential tests after each change and inspect all partially constructed outputs for safe destruction.

### Deliverables

- Contract-conforming lower-memory reconstruction route.
- Focused before/after discrepancy record.

### Completion Criteria

- The outer route meets the accepted numerical and structural contract across the frozen corpus.
- No unchecked size calculation, leaked partial output, or ambiguous ownership remains.

---

## Day 7: Promote the Lower-Memory Route to Default

**Title:** Promote the Lower-Memory Route to Default

**Theme:** Remove the additional `m×n` accumulator from ordinary sparse low-rank reconstruction.

**Time estimate:** 10 hours (228.3: 10 h)

### Tasks

1. Make the corrected outer-product route the maintained default for `sparse_svd_lowrank_sparse()`.
2. Retain the frozen compatibility selector for bounded comparison and rollback diagnostics, with deterministic parsing and documented invalid-value handling.
3. Add tests proving no selector chooses the old route by default and that environment state cannot leak between test cases.
4. Audit callers, examples, benchmarks, and documentation for assumptions about the old default or dense reconstruction scratch.

### Deliverables

- Lower-memory default reconstruction selection.
- Selector and caller regression coverage.

### Completion Criteria

- A normal call cannot allocate the old `m×n` reconstruction accumulator.
- Explicit compatibility behavior remains controlled, tested, and does not alter the public ABI.

---

## Day 8: Finalize Promotion and Begin Failure Injection

**Title:** Finalize Promotion and Begin Failure Injection

**Theme:** Verify selection behavior and establish exhaustive allocation boundaries.

**Time estimate:** 10 hours (228.3: 6 h; 228.4: 4 h)

### Tasks

1. Complete default/compatibility route parity checks for valid, invalid, unset, and repeated selector use.
2. Instrument or inspect the default path to demonstrate absence of an allocation proportional to `m×n` reconstruction scratch.
3. Enumerate every decomposition, factor, accumulator, sparse-node, and output allocation reached by low-rank sparse reconstruction.
4. Start deterministic fail-index sweeps and assert status, null/valid output state, input stability, cleanup, and retry behavior.

### Deliverables

- Default-route selection proof.
- Initial allocation-failure matrix.

### Completion Criteria

- Route selection is deterministic and isolated across repeated calls.
- Failure injection reaches the first allocation through at least one successful completion boundary.

---

## Day 9: Complete Allocation and Numeric Failure Sweeps

**Title:** Complete Allocation and Numeric Failure Sweeps

**Theme:** Prove cleanup and error behavior throughout reconstruction.

**Time estimate:** 10 hours (228.4: 10 h)

### Tasks

1. Sweep every reachable allocation boundary in the default route through success plus one index beyond the last allocation.
2. Add invalid rank, negative/NaN/infinite tolerance, dimension overflow, non-finite factor/output, and sparse-construction failure cases.
3. Verify exact status precedence, output-pointer state, unchanged input, no leaked or double-owned nodes, and safe destruction after partial work.
4. Run leak/sanitizer-focused checks for the new failure cases and retain failing-index evidence.

### Deliverables

- Exhaustive default-route failure and cleanup corpus.
- Allocation-site and numeric-error evidence log.

### Completion Criteria

- Every injected allocation failure cleans up fully and a subsequent valid call succeeds.
- Numeric and argument failures return the frozen status without exposing partial output.

---

## Day 10: Verify Repeatability, Reuse, and Compatibility Failures

**Title:** Verify Repeatability, Reuse, and Compatibility Failures

**Theme:** Close lifecycle behavior across repeated and mixed-route calls.

**Time estimate:** 10 hours (228.4: 10 h)

### Tasks

1. Alternate successful, failed, default, and compatibility calls in one process to detect stale selector, allocator, or result state.
2. Verify repeated reconstruction with the same input, destruction between calls, and retry after allocation or numeric failure.
3. Exercise zero-output and high-drop-tolerance results, ensuring empty matrices remain structurally valid and safely reusable.
4. Complete ownership, cleanup, and error-precedence review and map each frozen contract clause to a test.

### Deliverables

- Repeatability and mixed-route lifecycle suite.
- Contract-to-test closure matrix.

### Completion Criteria

- Prior success or failure cannot affect a later call's route, status, values, or ownership.
- Every failure-lifecycle contract clause has an automated regression.

---

## Day 11: Measure Reconstruction Scratch and Peak Memory

**Title:** Measure Reconstruction Scratch and Peak Memory

**Theme:** Quantify the exact memory improvement without overstating full-SVD behavior.

**Time estimate:** 10 hours (228.5: 10 h)

### Tasks

1. Run the frozen default-versus-compatibility measurement protocol on all pinned shapes, ranks, and tolerances.
2. Record reconstruction-only allocation bytes/counts separately from SVD factors, sparse output nodes, process baseline, and process peak RSS.
3. Retain raw samples, warmups, machine/toolchain metadata, source SHA, commands, and unsuccessful or noisy trials.
4. Check measurements against analytical allocation expectations and investigate unexplained memory growth before accepting results.

### Deliverables

- Raw reconstruction memory packet.
- Component-level allocation and RSS accounting table.

### Completion Criteria

- Evidence demonstrates that the default route lacks the additional `m×n` reconstruction accumulator.
- Dense factor and sparse-output memory remain visible rather than being attributed to the removed scratch.

---

## Day 12: Document Memory Limits and Selected Timing

**Title:** Document Memory Limits and Selected Timing

**Theme:** Turn measurements into precise user and maintainer guidance.

**Time estimate:** 10 hours (228.5: 10 h)

### Tasks

1. Analyze selected timing samples for regressions large enough to block promotion under the frozen methodology; retain raw data rather than portable performance claims.
2. Update `include/sparse_svd.h`, benchmark guidance, and relevant user documentation with the new default, compatibility selector, drop behavior, and error contract.
3. State plainly that full SVD still stores dense factors and that sparse low-rank output memory depends on retained `nnz` and rank.
4. Review all memory wording against measured evidence and remove ambiguous claims such as globally "sparse memory" or universally "faster."

### Deliverables

- Updated public and benchmark documentation.
- Scoped memory/timing interpretation and limitation record.

### Completion Criteria

- Documentation distinguishes removed reconstruction scratch from remaining dense factor and output storage.
- Every quantitative or comparative statement is traceable to retained, pinned evidence.

---

## Day 13: Run Focused, Cross-Build, and Sanitizer Gates

**Title:** Run Focused, Cross-Build, and Sanitizer Gates

**Theme:** Validate the changed C surface across maintained configurations.

**Time estimate:** 10 hours (228.6: 10 h)

### Tasks

1. Run the complete focused SVD, partial-SVD, low-rank, differential, failure-injection, and memory-contract suites.
2. Validate Make and CMake/CTest registration and execution, including 32-bit and 64-bit index configurations where supported.
3. Run the maintained sanitizer and leak checks on the affected corpus and investigate every failure or flaky result.
4. Audit formatting, compiler warnings, public-header documentation, and generated dependency surfaces affected by the default change.

### Deliverables

- Focused/CMake/index-width validation packet.
- Sanitizer and warning-clean evidence.

### Completion Criteria

- All focused, cross-build, index-width, and sanitizer gates pass from a clean tree.
- No unexplained warning, leak, flaky case, or unregistered regression remains.

---

## Day 14: Full Quality Closure and Handoff

**Title:** Full Quality Closure and Handoff

**Theme:** Prove sprint completion and preserve auditable evidence.

**Time estimate:** 10 hours (228.6: 10 h)

### Tasks

1. Run `make format && make lint && make test` from the final source state and rerun any required reviewed CMake parity gate.
2. Verify the final diff, public ABI, selector behavior, test registration, documentation links, and absence of generated or local-only debris.
3. Consolidate commands, exact SHA, raw results, allocation sweeps, memory evidence, limitations, and completion-criterion mapping in `WORKING_NOTES.md` and sprint artifacts.
4. Prepare the sprint handoff with completed work, blocked evidence if any, residual risks, and precise no-claim boundaries.

### Deliverables

- Passing final quality and parity records.
- Auditable Sprint 228 evidence packet and handoff.

### Completion Criteria

- The default sparse low-rank reconstruction path has no additional `m×n` accumulator, and accepted output/error behavior is verified.
- Full SVD's remaining dense factor storage is documented plainly, all required gates pass, and no broader memory or performance claim is made.
