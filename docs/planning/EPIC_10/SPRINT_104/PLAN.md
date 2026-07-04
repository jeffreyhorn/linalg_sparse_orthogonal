# Sprint 104 Plan: Performance Backend & Parallel Runtime Modernization

**Sprint Duration:** 14 days
**Goal:** Establish a durable backend and runtime contract for builtin dense
kernels, optional acceleration, OpenMP behavior, and bounded local regression
evidence. This sprint implements the Sprint 104 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 104 begins from:
- the Sprint 100 benchmark templates, evidence contract, and claim discipline
- the Sprint 101 compressed-first storage and API front-door baseline
- the Sprint 102 direct solver oracle and backend robustness work
- the Sprint 103 iterative, eigensolver, and SVD comparison closeout
- existing builtin dense kernels, optional backend hooks, OpenMP controls, and
  benchmark surfaces that need a clearer product contract

The strongest Sprint 104 pressure is to modernize runtime and backend behavior
without claiming portable timing superiority from local measurements. The
sprint must:
- audit dense backend consumers, fallback paths, optional acceleration points,
  and benchmark ownership
- define runtime expectations for builtin kernels, optional acceleration,
  OpenMP, nested parallelism, and observability
- refine backend descriptor and selection surfaces while preserving builtin
  fallback truth
- reduce confusing thread-local and global override behavior where practical
- add bounded local performance sentinels for hot paths
- align benchmark reports, docs, and maintainer guidance with actual evidence
- close with validation, comparison artifacts, and a Sprint 105 handoff queue

**End State:** Sprint 104 leaves behind:
- a backend consumer audit and runtime-contract design record
- a clearer backend descriptor and selection contract
- documented builtin fallback and optional acceleration behavior
- cleaner OpenMP and runtime-control guidance
- bounded local performance regression sentinels for selected hot paths
- benchmark wording aligned with maintained evidence
- validation artifacts and Sprint 105 handoff criteria

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 104 project-plan estimate.

---

## Day 1: Sprint 104 Scope & Runtime Baseline

**Title:** Runtime Baseline
**Theme:** Convert the Sprint 104 project-plan section and prior Epic 10
handoffs into one bounded backend/runtime modernization package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 104 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 benchmark templates and Sprint 102-103 closeout notes
   for backend, comparison, and non-claim constraints.
3. Inventory the Sprint 104 workstreams:
   - backend consumer audit
   - runtime contract design
   - backend descriptor batch
   - OpenMP and threading cleanup
   - performance sentinel batch
   - benchmark reporting alignment
   - validation and closeout
4. Create Sprint 104 working notes and artifacts directory.
5. Record validation expectations for docs-only, benchmark-touch,
   test-touch, source-touch, and workflow-touch days.

### Deliverables
- Sprint 104 workstream inventory
- working-notes baseline
- initial artifacts directory structure
- validation expectation list

### Completion Criteria
- every Sprint 104 project-plan item has day-level ownership
- Sprint 100 evidence and non-claim rules are visible in working notes
- validation expectations are explicit before implementation days begin

---

## Day 2: Backend Consumer Audit

**Title:** Consumer Audit
**Theme:** Profile dense backend consumers, fallback paths, and optional
acceleration points across direct, spectral, and decomposition families
**Time estimate:** 12 hours

### Tasks
1. Inventory source files that call builtin dense kernels, backend selectors,
   decomposition helpers, or optional acceleration hooks.
2. Classify consumers by solver family:
   - direct factorization
   - LDLT and Cholesky CSC paths
   - QR and dense helper paths
   - eigensolver and LOBPCG paths
   - SVD and low-rank paths
   - benchmark and example paths
3. Identify fallback behavior when optional acceleration is unavailable.
4. Identify repeated backend-selection assumptions, hidden global state, and
   ambiguous environment-variable behavior.
5. Write the backend consumer audit artifact.

### Deliverables
- backend consumer inventory
- fallback-path map
- optional acceleration point list
- initial risk and cleanup queue

### Completion Criteria
- direct, eigensolver, SVD, benchmark, and example consumers are represented
- builtin fallback behavior is separated from optional acceleration behavior
- ambiguous runtime-control risks are listed before design starts

---

## Day 3: Runtime Contract Design

**Title:** Runtime Contract
**Theme:** Define runtime expectations for builtin kernels, optional backends,
OpenMP, nested parallelism, and observability
**Time estimate:** 12 hours

### Tasks
1. Define the expected behavior of builtin dense kernels as the portable
   baseline.
2. Define optional backend selection semantics, including unavailable-backend
   fallback and error behavior.
3. Define OpenMP expectations for serial builds, OpenMP builds, nested
   parallelism, and thread-count controls.
4. Define what runtime state should be observable through public diagnostics,
   test diagnostics, benchmark output, or docs.
5. Write the runtime contract design artifact with explicit non-claims.

### Deliverables
- runtime contract design record
- builtin fallback and optional backend behavior table
- OpenMP and nested-parallelism expectation table
- observability and diagnostic-surface notes

### Completion Criteria
- backend behavior is deterministic enough for tests and benchmarks
- local timing evidence is not framed as portable performance superiority
- remaining ambiguity is either assigned to implementation days or deferred

---

## Day 4: Descriptor Surface Boundary

**Title:** Descriptor Boundary
**Theme:** Freeze the backend descriptor and selection-surface changes before
editing source
**Time estimate:** 12 hours

### Tasks
1. Compare the Day 2 consumer audit with the Day 3 runtime contract.
2. Identify descriptor fields, selection helpers, status strings, or diagnostics
   that need refinement.
3. Decide which changes are public API, internal-only, test-support, or
   documentation-only.
4. Define compatibility requirements for existing callers, tests, examples,
   and benchmarks.
5. Write the descriptor boundary artifact with a focused validation plan.

### Deliverables
- descriptor and selection-surface boundary artifact
- compatibility checklist
- implementation sequence
- focused validation command list

### Completion Criteria
- source changes are scoped before implementation starts
- builtin fallback truth remains a first-class contract
- public API changes, if any, have compatibility notes

---

## Day 5: Backend Descriptor Batch

**Title:** Descriptor Batch
**Theme:** Refine or extend backend descriptor and selection behavior without
weakening builtin fallback semantics
**Time estimate:** 12 hours

### Tasks
1. Implement the selected descriptor, selector, status, or diagnostic changes
   from Day 4.
2. Preserve current default behavior for callers that do not opt into optional
   acceleration.
3. Add focused tests for builtin fallback, invalid selection, unavailable
   optional backends, and status reporting.
4. Update examples or benchmark plumbing only where descriptor output needs to
   stay coherent.
5. Run focused validation for touched source and test files.

### Deliverables
- descriptor or selection-surface implementation
- focused fallback and status tests
- updated local diagnostics where needed
- Day 5 validation notes

### Completion Criteria
- builtin fallback remains the default portable truth
- invalid or unavailable backend behavior is explicit and tested
- focused validation passes before broader runtime cleanup begins

---

## Day 6: OpenMP and Threading Audit

**Title:** Threading Audit
**Theme:** Identify confusing thread-local, global override, environment, and
nested-parallelism behavior before cleanup
**Time estimate:** 12 hours

### Tasks
1. Inventory OpenMP compile-time guards, runtime environment controls, and
   public thread-control surfaces.
2. Identify thread-local state, process-global overrides, and test-only
   environment dependencies.
3. Map nested parallelism risks across SpMV, dense kernels, direct solvers,
   eigensolvers, SVD, benchmarks, and examples.
4. Classify cleanup candidates by compatibility risk and validation cost.
5. Write the OpenMP and threading audit artifact.

### Deliverables
- OpenMP and runtime-control inventory
- thread-local vs global behavior map
- nested-parallelism risk table
- cleanup priority list

### Completion Criteria
- cleanup candidates are ranked before code changes begin
- compatibility-sensitive behavior is explicitly protected
- validation needs are known for serial and OpenMP builds

---

## Day 7: OpenMP and Threading Cleanup

**Title:** Threading Cleanup
**Theme:** Reduce confusing runtime-control behavior and document any remaining
threading constraints
**Time estimate:** 12 hours

### Tasks
1. Implement the highest-value low-risk cleanup from the Day 6 audit.
2. Preserve serial-build behavior and existing public option semantics.
3. Add or update focused tests for thread-count controls, fallback behavior, or
   diagnostic output.
4. Update maintainer or user-facing docs for runtime controls touched by the
   cleanup.
5. Run focused validation for the touched runtime-control surface.

### Deliverables
- OpenMP or runtime-control cleanup patch
- focused threading/runtime tests
- updated runtime-control documentation
- Day 7 validation notes

### Completion Criteria
- serial and OpenMP expectations remain coherent
- global and thread-local control behavior is less ambiguous or explicitly
  documented
- focused validation passes before performance sentinels are added

---

## Day 8: Performance Sentinel Design

**Title:** Sentinel Design
**Theme:** Select bounded local regression sentinels for hot paths without
claiming portable timing superiority
**Time estimate:** 12 hours

### Tasks
1. Use the Day 2 audit and existing benchmark surfaces to select hot paths for
   regression sentinels.
2. Define sentinel goals as local regression detection, not cross-platform
   performance claims.
3. Choose deterministic fixtures, warm-up rules, measurement limits, skip
   behavior, and output fields.
4. Define acceptable variance handling and failure thresholds.
5. Write the performance sentinel design artifact.

### Deliverables
- selected sentinel path list
- fixture and measurement design
- variance and threshold policy
- validation and skip-behavior plan

### Completion Criteria
- each sentinel has a clear local-regression purpose
- thresholds are conservative and maintainable
- benchmark wording cannot be mistaken for portable superiority claims

---

## Day 9: Performance Sentinel Batch

**Title:** Sentinel Batch
**Theme:** Add bounded local performance regression sentinels for selected hot
paths
**Time estimate:** 12 hours

### Tasks
1. Implement the selected sentinel tests, benchmark hooks, or maintainer-only
   scripts from Day 8.
2. Keep sentinel output compact, deterministic where practical, and easy to
   compare across local runs.
3. Add skip behavior for missing optional dependencies, unavailable OpenMP, or
   unsuitable local runtime conditions.
4. Run focused sentinel validation and capture representative output.
5. Record limitations and non-claims in Sprint artifacts.

### Deliverables
- bounded local performance sentinels
- representative local sentinel output
- skip and limitation notes
- Day 9 validation notes

### Completion Criteria
- sentinels detect regressions without asserting broad benchmark dominance
- optional backend and OpenMP variance is handled explicitly
- focused validation passes before benchmark reporting alignment starts

---

## Day 10: Benchmark Reporting Audit

**Title:** Reporting Audit
**Theme:** Reconcile benchmark docs, scripts, and artifacts with the refined
backend and runtime contract
**Time estimate:** 12 hours

### Tasks
1. Inventory benchmark scripts, benchmark binaries, maintainer docs, README
   references, and planning artifacts that discuss runtime or performance.
2. Compare each reporting surface to the Day 3 runtime contract and Day 8
   sentinel design.
3. Identify stale wording, overbroad performance claims, missing backend
   disclosure, and unclear OpenMP notes.
4. Define wording rules for builtin fallback, optional acceleration, local
   timing, and CI-reviewed sentinels.
5. Write the benchmark reporting audit artifact.

### Deliverables
- benchmark reporting inventory
- stale wording and claim-risk list
- backend disclosure wording rules
- documentation update plan

### Completion Criteria
- benchmark language is audited before docs are changed
- local timing, optional acceleration, and CI-reviewed evidence are separated
- overbroad or ambiguous claims have explicit replacements

---

## Day 11: Benchmark Reporting Alignment

**Title:** Reporting Alignment
**Theme:** Update benchmark docs, scripts, and artifacts so performance wording
matches actual backend/runtime evidence
**Time estimate:** 12 hours

### Tasks
1. Apply the Day 10 wording rules to selected benchmark and maintainer
   documentation.
2. Update benchmark output labels or artifact templates where they need backend
   or runtime disclosure.
3. Preserve existing benchmark execution behavior unless Day 10 identified a
   direct reporting defect.
4. Run docs, script, or focused benchmark validation appropriate to touched
   files.
5. Record before/after reporting examples in Sprint artifacts.

### Deliverables
- updated benchmark and runtime documentation
- aligned benchmark output labels or templates where needed
- before/after reporting examples
- Day 11 validation notes

### Completion Criteria
- performance wording matches maintained evidence
- optional acceleration and OpenMP context are visible where relevant
- validation for touched docs, scripts, or benchmarks passes

---

## Day 12: Cross-Platform Runtime Review

**Title:** Platform Review
**Theme:** Check that backend/runtime behavior remains coherent across local,
CI, Windows, serial, and optional-acceleration contexts
**Time estimate:** 12 hours

### Tasks
1. Review workflow and build-system surfaces that expose backend, OpenMP, or
   benchmark behavior.
2. Compare local validation expectations with enforced CI lanes and Windows
   reviewed scope.
3. Identify any mismatch between source behavior, CMake/Make targets,
   benchmark docs, and maintainer guidance.
4. Add documentation or narrow workflow/source-list updates only if needed for
   coherence.
5. Write the cross-platform runtime review artifact.

### Deliverables
- cross-platform runtime review artifact
- CI and local validation mapping
- Windows and serial-build scope notes
- coherence update list or explicit no-change decision

### Completion Criteria
- platform-specific runtime assumptions are documented
- CI and local validation surfaces are not overstated
- any cross-platform follow-up is captured before closeout

---

## Day 13: Validation Reconciliation

**Title:** Validation Reconciliation
**Theme:** Run required checks, reconcile artifacts with implementation, and
prepare the Sprint 104 closeout package
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 104 artifacts against implemented source, test, docs,
   benchmark, and workflow changes.
2. Run required validation for the final touched-file set.
3. Re-run focused backend, OpenMP, sentinel, or benchmark validation where
   artifacts depend on representative output.
4. Update working notes with final command results, known limitations, and
   deferred follow-ups.
5. Draft the Sprint 104 closeout and Sprint 105 handoff notes.

### Deliverables
- final validation command log
- artifact-to-implementation reconciliation
- known limitation and deferred follow-up list
- draft closeout and handoff notes

### Completion Criteria
- required validation passes for all touched implementation surfaces
- artifacts do not claim behavior that implementation or tests do not support
- Sprint 105 handoff candidates are concrete and prioritized

---

## Day 14: Sprint 104 Closeout & Handoff

**Title:** Closeout
**Theme:** Close the backend/runtime modernization sprint with validated
evidence, clear non-claims, and a Sprint 105 handoff queue
**Time estimate:** 12 hours

### Tasks
1. Finalize Sprint 104 working notes and closeout artifacts.
2. Summarize backend descriptor, optional acceleration, OpenMP, sentinel, and
   benchmark-reporting changes.
3. Confirm all day-level deliverables have either landed, been documented as
   no-change decisions, or moved to the residual queue.
4. Run final validation required by the touched files if Day 13 changed
   anything.
5. Write Sprint 105 handoff notes focused on reordering, graph, and
   large-matrix scalability dependencies.

### Deliverables
- Sprint 104 closeout artifact
- finalized working notes
- Sprint 105 handoff queue
- final validation summary

### Completion Criteria
- every Sprint 104 project-plan item has a closeout status
- backend/runtime claims are bounded by maintained evidence
- Sprint 105 can start from explicit handoff notes rather than rediscovery
