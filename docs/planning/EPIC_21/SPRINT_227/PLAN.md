# Sprint 227 Plan: Sorted Matrix Arithmetic and Permutation Safety

**Sprint Duration:** 14 days

**Goal:** Close the repeated-scan cost in `sparse_add` and make `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` obey one explicit, tested contract for matrices with nonidentity row or column permutations.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours. Hosted queue time and unattended benchmark execution are separate from these estimates.

**Prerequisites:** Use the [Epic 21 Sprint 227 scope](../PROJECT_PLAN.md), [Sprint 218 sorted matmul output work](../../EPIC_20/SPRINT_218/RETROSPECTIVE.md), and [Sprint 219 width-aware build contract](../../EPIC_20/SPRINT_219/RETROSPECTIVE.md). Preserve public ABI and index-width behavior.

**Evidence rule:** Freeze existing behavior before edits. Retain exact commands, source SHA, compiler/build configuration, fixture seeds, allocation-failure indices, benchmark inputs, raw timing and memory output, and pass/fail logs. Dense-oracle and structural checks must be independent of the implementation path under test.

**Claim boundary:** This sprint may establish a selected arithmetic contract, linear output construction for sorted addition, and results on pinned wide-row workloads. It does not establish universal arithmetic speedups, logical-permutation support for unrelated APIs, broad sparse-kernel superiority, or a state-of-the-art performance claim.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 227.1 Contract Freeze | 18 h | Days 1–2: 10 + 8 |
| 227.2 Permutation Guard | 30 h | Days 2–5: 2 + 10 + 10 + 8 |
| 227.3 Sorted Output Builder | 28 h | Days 5–8: 2 + 10 + 10 + 6 |
| 227.4 Failure And Alias Matrix | 26 h | Days 8–11: 4 + 10 + 10 + 2 |
| 227.5 Scaling Evidence | 20 h | Days 11–13: 8 + 10 + 2 |
| 227.6 Quality Closure | 18 h | Days 13–14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Inventory Arithmetic Contracts and Baselines

**Title:** Inventory Arithmetic Contracts and Baselines

**Theme:** Establish the current semantics, costs, and evidence surface before changing code.

**Time estimate:** 10 hours (227.1: 10 h)

### Tasks

1. Trace `sparse_add`, `sparse_add_inplace`, and `sparse_matmul` through public headers, `src/sparse_matrix.c`, internal storage helpers, permutation maps, and current tests.
2. Record current behavior for identity and nonidentity row/column permutations, output ownership, shape/null errors, alias combinations, drop/cancellation behavior, and partial updates.
3. Build independent dense-oracle fixtures for square, rectangular, empty, cancellation-heavy, and wide-row matrices; record deterministic seeds and expected logical results.
4. Capture baseline test commands, selected allocation counts, output invariants, timing, and peak-memory measurements in `WORKING_NOTES.md` and a day-one artifact.

### Deliverables

- Arithmetic behavior and ownership inventory.
- Reproducible dense-oracle and performance baseline packet.

### Completion Criteria

- Every public arithmetic entry point has a documented current-state row, including unsupported or ambiguous permutation behavior.
- Baselines are reproducible from recorded commands and do not rely on the implementation under review as their oracle.

---

## Day 2: Freeze the Compatible Arithmetic Contract

**Title:** Freeze the Compatible Arithmetic Contract

**Theme:** Decide physical-versus-logical semantics before implementation.

**Time estimate:** 10 hours (227.1: 8 h; 227.2: 2 h)

### Tasks

1. Specify index-space semantics for both operands and outputs of add, in-place add, and matmul, including identity and nonidentity permutations.
2. Choose one compatible policy: reject unsupported nonidentity permutations before mutation, or implement logical arithmetic consistently across all three APIs; document why it is source/ABI compatible.
3. Freeze alpha/beta scaling, duplicate combination, exact-zero dropping, ownership, aliasing, allocation failure, and in-place partial-update or rollback behavior.
4. Convert the contract into a decision table with required positive, negative, dense-oracle, and failure fixtures for both operands and every permutation-map variant.

### Deliverables

- Reviewed arithmetic and permutation contract.
- Contract-to-test matrix covering all three APIs.

### Completion Criteria

- The policy has no path that can silently return success for an unsupported permutation state.
- Output, mutation, and error guarantees are precise enough to assert without inspecting implementation details.

---

## Day 3: Build Permutation and Oracle Fixtures

**Title:** Build Permutation and Oracle Fixtures

**Theme:** Make the frozen contract executable before repairing behavior.

**Time estimate:** 10 hours (227.2: 10 h)

### Tasks

1. Add helpers that construct valid nonidentity row-only, column-only, and row-plus-column states without corrupting inverse maps.
2. Add add/in-place/matmul cases with the left operand, right operand, and both operands permuted, including rectangular dimensions where supported.
3. Compare successful cases to independent logical dense arithmetic, or assert the exact pre-mutation error for rejected states under the selected policy.
4. Verify output pointer, destination contents, permutation maps, node counts, and structural invariants after success and failure.

### Deliverables

- Focused permutation regression corpus.
- Independent logical dense-oracle comparison helpers.

### Completion Criteria

- Tests fail against every current behavior that violates the frozen contract and pass for already conforming cases.
- Both operand positions and each supported permutation category are represented.

---

## Day 4: Implement Shared Permutation Validation

**Title:** Implement Shared Permutation Validation

**Theme:** Enforce the selected policy consistently and before mutation.

**Time estimate:** 10 hours (227.2: 10 h)

### Tasks

1. Implement or refine one bounded internal validation seam for row/column permutation compatibility, avoiding duplicate policy logic across APIs.
2. Apply validation in `sparse_add` and `sparse_add_inplace` at the correct point relative to null, shape, alias, allocation, and mutation checks.
3. Preserve the documented error-precedence and output/destination state on rejected calls.
4. Run focused arithmetic tests and inspect all callers for assumptions about the prior physical-index behavior.

### Deliverables

- Shared add and in-place permutation enforcement.
- Caller and error-precedence audit record.

### Completion Criteria

- Add paths enforce the frozen contract before an unsupported state can allocate output or mutate a destination.
- Existing identity-permutation behavior and ABI remain unchanged.

---

## Day 5: Complete Cross-API Permutation Consistency

**Title:** Complete Cross-API Permutation Consistency

**Theme:** Close permutation handling and prepare the sorted-output seam.

**Time estimate:** 10 hours (227.2: 8 h; 227.3: 2 h)

### Tasks

1. Apply and verify the selected policy in `sparse_matmul`, including both operands, shape errors, and output state.
2. Complete add/in-place/matmul permutation regressions for success, rejection, aliases, and unchanged inputs.
3. Update affected public/internal contract comments without implying support beyond the three arithmetic APIs.
4. Design the sorted-add append/link builder, including ownership, cleanup, row/column ordering, and allocation-failure boundaries.

### Deliverables

- Consistent permutation behavior across all three APIs.
- Reviewed sorted-output builder design.

### Completion Criteria

- The complete permutation matrix matches one policy and has no misleading-success case.
- The builder design identifies every allocation and structural invariant before implementation.

---

## Day 6: Implement Sorted Addition Append Construction

**Title:** Implement Sorted Addition Append Construction

**Theme:** Remove per-output-node insertion scans from `sparse_add`.

**Time estimate:** 10 hours (227.3: 10 h)

### Tasks

1. Introduce a bounded internal append/link construction path for monotonically ordered addition output.
2. Replace `sparse_insert` in the sorted merge with direct ownership-safe node construction while maintaining row heads, column heads, and tails.
3. Preserve scaling, duplicate combination, exact-zero dropping, `nnz`, output permutation initialization, and allocation error propagation.
4. Run focused deterministic arithmetic tests and inspect generated structures after each representative row pattern.

### Deliverables

- Initial sorted addition output builder.
- Structural-invariant test evidence for representative merges.

### Completion Criteria

- The sorted merge does not restart a row or column scan for each emitted node.
- Outputs remain sorted, duplicate-free, correctly counted, and safely destroyable on tested success paths.

---

## Day 7: Harden Builder Invariants and Edge Cases

**Title:** Harden Builder Invariants and Edge Cases

**Theme:** Prove correctness across sparse row shapes and cancellation patterns.

**Time estimate:** 10 hours (227.3: 10 h)

### Tasks

1. Exercise empty inputs, empty rows, disjoint patterns, identical columns, full cancellation, scalar-zero operands, wide rows, and rectangular matrices.
2. Validate row and column linked ordering, tail maintenance, node uniqueness, `nnz`, logical reads, copies, and destruction.
3. Compare deterministic and randomized cases against the independent dense oracle under both supported index widths where practical.
4. Review the builder for integer-size calculations, unchecked multiplications, ownership transfers, and cleanup of partially linked nodes.

### Deliverables

- Expanded sorted-builder edge corpus.
- Invariant and overflow/ownership audit notes.

### Completion Criteria

- Edge and randomized fixtures match the oracle and pass structural checks.
- No unchecked size arithmetic or ambiguous partial-node ownership remains in the builder.

---

## Day 8: Finalize Builder Cleanup and Start Failure Sweeps

**Title:** Finalize Builder Cleanup and Start Failure Sweeps

**Theme:** Make the new construction path safe under every allocation boundary.

**Time estimate:** 10 hours (227.3: 6 h; 227.4: 4 h)

### Tasks

1. Finish the builder's rollback/destruction behavior for allocation failure at matrix, permutation-array, and output-node boundaries.
2. Enumerate allocation sites for add and establish a deterministic fail-index sweep through success plus one boundary beyond the final allocation.
3. Assert returned status, output pointer state, unchanged inputs, leak freedom, and successful retry after each injected failure.
4. Confirm the final builder implementation still satisfies the no-repeated-scan criterion by code-path inspection or bounded instrumentation.

### Deliverables

- Completed ownership-safe sorted builder.
- Initial exhaustive allocation-failure sweep.

### Completion Criteria

- Every builder allocation boundary fails cleanly and a later retry succeeds.
- No output node is leaked, double-owned, or left in an invalid row/column structure.

---

## Day 9: Close Add Failure and Alias Semantics

**Title:** Close Add Failure and Alias Semantics

**Theme:** Verify output ownership and input stability under adverse calls.

**Time estimate:** 10 hours (227.4: 10 h)

### Tasks

1. Complete `sparse_add` failure sweeps for null/shape/permutation rejection, allocation failures, cancellation-heavy output, and width-sensitive sizes.
2. Exercise output-pointer alias hazards and repeated calls according to the frozen ownership contract.
3. Verify A and B bytes/logical values, permutation maps, factor-state metadata, and node structures remain unchanged on every failure.
4. Add randomized rectangular and wide-row dense-oracle comparisons with deterministic replay information.

### Deliverables

- Complete out-of-place add failure/ownership matrix.
- Reproducible randomized arithmetic corpus.

### Completion Criteria

- Every documented error path has an asserted output and input state.
- Randomized and adversarial success cases match the independent oracle.

---

## Day 10: Close In-Place Atomicity and Alias Behavior

**Title:** Close In-Place Atomicity and Alias Behavior

**Theme:** Make destination mutation behavior explicit and tested.

**Time estimate:** 10 hours (227.4: 10 h)

### Tasks

1. Exercise `A == B`, shared-pattern cases, cancellation, scaling, empty rows, permutation rejection, and shape errors for `sparse_add_inplace`.
2. Sweep all in-place allocation failures and compare destination state to the frozen rollback or partial-update contract.
3. Repair mutation ordering where required to meet the contract, preserving factor-state and permutation invariants.
4. Verify post-failure destruction, retry, copy, lookup, and subsequent arithmetic operations under sanitizer-friendly fixtures.

### Deliverables

- In-place alias and allocation-failure corpus.
- Implemented and documented mutation/rollback semantics.

### Completion Criteria

- Every failure leaves the destination in exactly the documented state and safely reusable or destroyable.
- Alias cases produce oracle-correct results or the specified pre-mutation error.

---

## Day 11: Integrate the Failure Matrix and Pin Profiles

**Title:** Integrate the Failure Matrix and Pin Profiles

**Theme:** Finish adverse-behavior coverage and define honest scaling evidence.

**Time estimate:** 10 hours (227.4: 2 h; 227.5: 8 h)

### Tasks

1. Run the combined add/in-place/matmul permutation, alias, allocation, cancellation, and structural corpus; close any cross-API discrepancy.
2. Select pinned wide-row and varied-row workloads that isolate sorted-output construction without claiming whole-library performance.
3. Define baseline/candidate build flags, repetitions, warmups, host controls, time metric, peak-memory metric, and raw-output retention.
4. Add or update the maintained arithmetic benchmark/profiling route with correctness checks and deterministic workload identity.

### Deliverables

- Integrated failure and alias matrix result.
- Pre-registered selected scaling protocol and workloads.

### Completion Criteria

- The full adverse corpus passes with the same frozen semantics across APIs.
- The profile protocol can distinguish construction cost while retaining correctness checks and explicit claim limits.

---

## Day 12: Collect Wide-Row Time and Memory Evidence

**Title:** Collect Wide-Row Time and Memory Evidence

**Theme:** Measure the selected construction change without overgeneralizing.

**Time estimate:** 10 hours (227.5: 10 h)

### Tasks

1. Build baseline and candidate revisions with matched compiler, flags, index width, allocator, and threading settings.
2. Run correctness-qualified warmups and repeated measurements on all pinned workloads; retain raw time, peak memory, stdout/stderr, status, and environment data.
3. Compare scaling against emitted `nnz` and row width, inspect noisy or contradictory results, and rerun only under the predeclared policy.
4. Record median/range and memory observations with source and executable hashes; avoid portable or state-of-the-art wording.

### Deliverables

- Raw baseline/candidate timing and peak-memory bundle.
- Selected-workload analysis with limitations.

### Completion Criteria

- Every reported result is correctness-qualified and tied to exact source/build/host identity.
- Negative, neutral, or noisy outcomes are retained rather than converted into unsupported speed claims.

---

## Day 13: Reconcile Evidence and Begin Quality Closure

**Title:** Reconcile Evidence and Begin Quality Closure

**Theme:** Convert implementation and measurements into reviewable contract evidence.

**Time estimate:** 10 hours (227.5: 2 h; 227.6: 8 h)

### Tasks

1. Finalize the selected scaling report and connect each statement to raw evidence, exact workload, source SHA, and claim boundary.
2. Update public/internal arithmetic documentation for permutation, ownership, aliasing, failure, and complexity behavior.
3. Run `make format`, `make lint`, and focused arithmetic/matmul tests; inspect and repair all failures.
4. Run CMake test registration/parity and focused sanitizer checks for the changed arithmetic and allocation paths.

### Deliverables

- Final selected scaling evidence and claim ledger.
- Initial quality, parity, sanitizer, and documentation results.

### Completion Criteria

- Documentation and evidence agree with the implemented contract and do not broaden the measured claim.
- All completed checks pass; unresolved failures remain explicit blockers for Day 14.

---

## Day 14: Run Full Gates and Close Sprint 227

**Title:** Run Full Gates and Close Sprint 227

**Theme:** Prove the exact final tree and hand off only supported claims.

**Time estimate:** 10 hours (227.6: 10 h)

### Tasks

1. Run `make format && make lint && make test` on the final tree and verify no formatter-generated changes remain.
2. Run required CMake/CTest parity, sanitizer, affected installed-consumer/index-width, API documentation, and link checks.
3. Re-run the focused permutation, dense-oracle, allocation-failure, alias, structural, and selected profile checks after final edits.
4. Complete `WORKING_NOTES.md`, the evidence index, residual-risk list, and sprint completion decision; mark any unmet gate blocked rather than complete.

### Deliverables

- Exact-head quality and focused regression packet.
- Sprint 227 completion record and bounded handoff for Sprint 228.

### Completion Criteria

- Sorted addition emits nodes without repeated head scans, all three arithmetic APIs enforce the frozen permutation contract, and numerical/failure behavior matches the independent oracle.
- Every required gate passes on the final source tree; otherwise the sprint records the precise blocker and does not claim closure.

---

## Sprint Completion Checklist

- [x] `sparse_add` uses sorted append/link construction rather than repeated `sparse_insert` scans.
- [x] Add, in-place add, and matmul share one explicit and tested nonidentity-permutation policy.
- [x] Dense-oracle, structural, alias, cancellation, rectangular, wide-row, and exhaustive allocation-failure cases pass.
- [x] Public ABI, source compatibility, index-width behavior, ownership, and documented error states are preserved.
- [x] Selected time/memory evidence is reproducible and limited to pinned workloads without portable speed claims.
- [x] Format, lint, full tests, CMake parity, sanitizers, affected installed-consumer, and documentation gates pass on the final tree.
- [x] `WORKING_NOTES.md` maps each Epic item to exact commands, artifacts, outcomes, and remaining risks.
