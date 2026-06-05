# Sprint 55 Plan: Large-Source Decomposition Phase 1

**Sprint Duration:** 14 days  
**Goal:** Reduce the largest remaining solver implementation hotspots by
splitting the two most consequential translation units into cleaner ownership
seams without reopening the validated public lifecycle contracts landed in
Sprints 49-54.
This sprint implements the Sprint 55 section of
`docs/planning/EPIC_5/PROJECT_PLAN.md`.

**Starting Point:** Sprint 54 closed with the public repeated-run solver
lifecycle support boundary fixed and validated. The strongest remaining
maintainability hotspots are now the large eigensolver and iterative
translation units:
- `src/sparse_eigs.c`
- `src/sparse_iterative.c`

The highest-value next step is not broad API expansion. It is bounded
decomposition work that separates durable helper ownership from the outer
orchestration layers, trims stale sprint-history narrative from touched
implementation files, and preserves the full reviewed validation baseline.

**End State:** Sprint 55 leaves behind smaller and clearer eigensolver and
iterative implementation ownership, with the first bounded helper/module
extractions landed, historical implementation commentary reduced where touched,
and the maintained validation baseline still exact.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to
160 hours, matching the Sprint 55 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 55 Scope Audit & Large-Source Baseline

**Title:** Baseline Setup  
**Theme:** Turn the Sprint 55 project-plan items plus the Sprint 54 close state
into a bounded large-source decomposition map  
**Time estimate:** 10 hours

### Tasks
1. Re-read the Sprint 55 section of `docs/planning/EPIC_5/PROJECT_PLAN.md`,
   the Sprint 54 closeout, and the Epic 5 review/todo notes related to large
   source ownership and maintainability.
2. Reconfirm the preserved Sprint 55 constraints:
   - keep the Sprint 49-54 public lifecycle compatibility fences intact
   - reduce implementation size and ownership drift without broad API redesign
   - preserve the strongest local reviewed baseline and truthfulness anchors
3. Define the Sprint 55 workstreams explicitly:
   - `sparse_eigs.c` seam audit
   - eigensolver decomposition batch 1
   - eigensolver decomposition batch 2
   - `sparse_iterative.c` seam audit
   - iterative decomposition batch 1
   - historical comment reduction
   - validation and closeout
4. Record the highest-risk seams for the sprint:
   - extracting helpers that still leak too much outer orchestration state
   - changing file shape without materially improving ownership boundaries
   - stale sprint-history narrative surviving in touched permanent code
   - accidental behavior drift inside reused helper paths
5. Open Sprint 55 working notes and record the initial landing order and
   touched-surface expectations.

### Deliverables
- Sprint 55 scope inventory
- Large-source baseline notes
- Working-notes starting assumptions

### Completion Criteria
- Sprint 55 starts from the Sprint 54 validated solver-lifecycle state rather
  than reopening public API design
- Preserved compatibility and scope fences are explicit before seam audits or
  code patches land
- The large-source decomposition workstreams are named before implementation
  begins

---

## Day 2: Validation Baseline & Touched-Surface Recheck

**Title:** Validation Baseline  
**Theme:** Reconfirm the reviewed local baseline and the exact solver rerun set
Sprint 55 code days must preserve  
**Time estimate:** 10 hours

### Tasks
1. Reconfirm the maintained reviewed baseline surfaces:
   - `make quality-review-full`
   - reviewed CMake parity
   - current truthfulness-anchor counts
2. Reconfirm the mandatory gate for later `*.c` / `*.h` decomposition batches:
   - `make format`
   - `make lint`
   - `make test`
3. Reconfirm the stronger default for substantial implementation ownership
   batches:
   - `make quality-review-full`
4. Refresh the targeted Sprint 55 follow-on binaries most likely to be needed:
   - `./build/test_iterative`
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/test_minres`
   - `./build/example_iterative`
   - `./build/example_eigs`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`
5. Record the authoritative validation boundary for docs-only audit/design days
   versus implementation landing days.

### Deliverables
- Refreshed validation/truthfulness notes
- Sprint 55 rerun list
- Code-day validation checklist

### Completion Criteria
- Sprint 55 uses the same baseline wording and parity anchors as the live repo
- The authoritative iterative/eigensolver rerun set is explicit before
  decomposition work begins
- No validation ambiguity remains around implementation-extraction days

---

## Day 3: `sparse_eigs.c` Seam Audit

**Title:** Eigensolver Audit  
**Theme:** Reduce `src/sparse_eigs.c` to concrete extraction seams before code
movement begins  
**Time estimate:** 10 hours

### Tasks
1. Audit the live ownership inside `src/sparse_eigs.c` and separate:
   - outer public-handle orchestration
   - backend-specific execution paths
   - shared workspace preparation/growth logic
   - residual/selection/reporting helpers
2. Identify which seams can be extracted without changing the public
   eigensolver contract.
3. Rank the candidate seams by:
   - ownership clarity
   - behavioral risk
   - line-count reduction value
   - test and benchmark proof cost
4. Reject seams that would only move code mechanically without improving
   long-term maintainability.
5. Write the seam audit artifact and the ranked landing order.

### Deliverables
- `sparse_eigs.c` seam audit
- Ranked eigensolver extraction targets
- Proposed first extraction boundary

### Completion Criteria
- The eigensolver decomposition problem is reduced to named ownership seams
- The first extraction target is justified by maintainability, not only line
  count
- Sprint 55 can start eigensolver implementation work from a concrete map

---

## Day 4: Eigensolver Decomposition Batch I Design

**Title:** Eigensolver Design  
**Theme:** Freeze the first eigensolver extraction boundary before editing
permanent implementation files  
**Time estimate:** 12 hours

### Tasks
1. Select the Day 3 highest-value eigensolver seam for the first landing.
2. Define the exact source-file ownership split:
   - what remains in `src/sparse_eigs.c`
   - what moves into the new helper/module file
   - what private declarations need to live in a local internal header
3. Define the invariants the extraction must preserve:
   - public handle semantics
   - backend selection/reporting behavior
   - on-demand growth and reuse behavior
   - benchmark and example parity
4. Define the minimal comment policy for the touched implementation:
   - preserve algorithm truth
   - remove stale sprint-history prose
5. Record the design artifact and landing checklist.

### Deliverables
- First eigensolver extraction design
- File-boundary ownership map
- Extraction invariants and checklist

### Completion Criteria
- The first eigensolver extraction boundary is explicit before code movement
- Ownership is defined at file and helper level, not just conceptually
- Comment-cleanup expectations are fixed before touched code is rewritten

---

## Day 5: Eigensolver Decomposition Batch I

**Title:** Eigensolver Batch I  
**Theme:** Land the first bounded helper/module extraction from
`src/sparse_eigs.c`  
**Time estimate:** 12 hours

### Tasks
1. Extract the first owned eigensolver helper/module slice and wire it into the
   remaining orchestration layer.
2. Add or update any needed private internal header declarations.
3. Keep the public API, tests, examples, and benchmark behavior unchanged.
4. Remove stale sprint-history narrative from touched eigensolver
   implementation blocks while preserving useful algorithm commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First landed eigensolver extraction patch
- Updated internal declarations
- Reduced touched-file narrative noise

### Completion Criteria
- A real ownership seam is extracted from `src/sparse_eigs.c`
- The remaining orchestration file is smaller and clearer than before
- Full required validation passes after the extraction

---

## Day 6: Eigensolver Decomposition Batch II Design

**Title:** Eigensolver Batch II Design  
**Theme:** Freeze the second eigensolver extraction or cleanup boundary using
the Day 5 landed state  
**Time estimate:** 12 hours

### Tasks
1. Re-audit the post-Day-5 eigensolver ownership shape.
2. Select the next bounded extraction or cleanup seam that most improves the
   residual orchestration layer.
3. Decide whether the second batch should emphasize:
   - another helper/module split
   - orchestration simplification around already-extracted helpers
   - targeted private-header cleanup
4. Define the exact file and helper changes for the second batch.
5. Record the second-batch design and validation checklist.

### Deliverables
- Second eigensolver batch design
- Updated post-Day-5 seam map
- Second-batch landing checklist

### Completion Criteria
- The second eigensolver batch is shaped by the landed Day 5 reality, not the
  original estimate alone
- The next ownership improvement target is explicit
- Sprint 55 can proceed to the second eigensolver implementation batch cleanly

---

## Day 7: Eigensolver Decomposition Batch II

**Title:** Eigensolver Batch II  
**Theme:** Land the second bounded eigensolver extraction and tighten the
residual orchestration layer  
**Time estimate:** 12 hours

### Tasks
1. Land the second eigensolver helper/module extraction or orchestration
   cleanup chosen on Day 6.
2. Keep the remaining `src/sparse_eigs.c` layer focused on public entry points
   and top-level coordination.
3. Continue removing stale sprint-history narrative from touched permanent
   implementation blocks.
4. Reconfirm the strongest direct proof surfaces stay green.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- Second landed eigensolver decomposition patch
- Cleaner residual `src/sparse_eigs.c` orchestration
- Updated direct proof notes

### Completion Criteria
- The second eigensolver batch materially improves ownership, not just file
  count
- The orchestration layer is cleaner after the extraction
- Full required validation passes after the batch

---

## Day 8: `sparse_iterative.c` Seam Audit

**Title:** Iterative Audit  
**Theme:** Reduce `src/sparse_iterative.c` to concrete extraction seams before
iterative code movement begins  
**Time estimate:** 12 hours

### Tasks
1. Audit the live ownership inside `src/sparse_iterative.c` and separate:
   - public handle orchestration
   - family-specific solver execution paths
   - shared workspace preparation/growth logic
   - residual/reporting/utility helpers
2. Identify which seams can be extracted without changing the public iterative
   handle contract or one-shot entry points.
3. Rank the candidate seams by ownership clarity, behavioral risk, and
   maintainability value.
4. Reject purely mechanical splits that would not improve future reasoning or
   testing.
5. Write the audit artifact and ranked landing order.

### Deliverables
- `sparse_iterative.c` seam audit
- Ranked iterative extraction targets
- Proposed first iterative extraction boundary

### Completion Criteria
- The iterative decomposition problem is reduced to named ownership seams
- The first iterative extraction target is justified by maintainability, not
  only line count
- Sprint 55 can start iterative implementation work from a concrete map

---

## Day 9: Iterative Decomposition Batch I Design

**Title:** Iterative Design  
**Theme:** Freeze the first iterative extraction boundary before editing the
iterative implementation files  
**Time estimate:** 12 hours

### Tasks
1. Select the Day 8 highest-value iterative seam for the first landing.
2. Define the exact source-file ownership split:
   - what remains in `src/sparse_iterative.c`
   - what moves into the new helper/module file
   - what private declarations belong in a local internal header
3. Define the invariants the extraction must preserve:
   - public handle semantics for supported families
   - one-shot compatibility behavior
   - workspace growth and reuse behavior
   - example and benchmark parity
4. Define the minimal comment policy for touched iterative implementation
   blocks.
5. Record the design artifact and landing checklist.

### Deliverables
- First iterative extraction design
- File-boundary ownership map
- Iterative extraction invariants and checklist

### Completion Criteria
- The first iterative extraction boundary is explicit before code movement
- Ownership is defined at file and helper level
- Comment-cleanup expectations are fixed before the iterative landing begins

---

## Day 10: Iterative Decomposition Batch I

**Title:** Iterative Batch I  
**Theme:** Land the first bounded helper/module extraction from
`src/sparse_iterative.c`  
**Time estimate:** 12 hours

### Tasks
1. Extract the first owned iterative helper/module slice and wire it into the
   remaining orchestration layer.
2. Add or update any needed private internal header declarations.
3. Keep the public iterative handle contract and one-shot compatibility
   behavior unchanged.
4. Remove stale sprint-history narrative from touched iterative implementation
   blocks while preserving useful algorithm commentary.
5. Run:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`

### Deliverables
- First landed iterative extraction patch
- Updated internal declarations
- Reduced touched-file narrative noise

### Completion Criteria
- A real ownership seam is extracted from `src/sparse_iterative.c`
- The remaining orchestration file is smaller and clearer than before
- Full required validation passes after the extraction

---

## Day 11: Historical Comment Reduction Sweep

**Title:** Comment Cleanup  
**Theme:** Remove stale sprint-history narrative from touched permanent
implementation files while preserving durable algorithm commentary  
**Time estimate:** 12 hours

### Tasks
1. Re-scan the Sprint 55 touched implementation files for temporary
   sprint-history or landing-history narrative.
2. Remove or compress comments that describe sprint chronology rather than
   durable code meaning.
3. Keep comments that materially help a maintainer understand:
   - invariants
   - ownership boundaries
   - tricky algorithm or fallback behavior
4. Recheck touched internal headers for the same issue.
5. Run:
   - `make format`
   - `make lint`
   - `make test`

### Deliverables
- Historical-comment reduction patch
- Cleaner permanent implementation commentary
- Maintainer-facing comment policy notes

### Completion Criteria
- Touched permanent implementation files no longer carry unnecessary
  sprint-history narrative
- Useful algorithm and ownership commentary is preserved
- Cleanup passes the required validation gate

---

## Day 12: Post-Landing Compatibility Audit

**Title:** Compatibility Audit  
**Theme:** Verify that the Sprint 55 decomposition work preserved public
contracts and improved ownership in the intended way  
**Time estimate:** 10 hours

### Tasks
1. Audit the landed Sprint 55 code against the preserved compatibility fences:
   - no public API redesign
   - no solver support-boundary drift
   - no behavior-visible lifecycle change
2. Confirm the extraction work improved ownership rather than merely moving
   code across files.
3. Recheck the touched examples, benchmarks, and high-signal tests against the
   landed decomposition.
4. Record any residual follow-up items that belong to later large-source
   decomposition phases instead of Sprint 55.
5. Write the compatibility-audit artifact and the Day 13 checklist.

### Deliverables
- Post-landing compatibility audit
- Residual follow-up queue
- Final validation checklist

### Completion Criteria
- The landed Sprint 55 branch still matches the preserved public solver and
  lifecycle fences
- The ownership gains are explicit and defensible
- No blocker-level drift remains before final validation

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Run the full required validation and the targeted large-source
follow-ons from the landed Sprint 55 state  
**Time estimate:** 12 hours

### Tasks
1. Run the required full gate:
   - `make format`
   - `make lint`
   - `make test`
   - `make quality-review-full`
2. Reconfirm reviewed CMake parity and truthfulness anchors.
3. Run the targeted Sprint 55 follow-ons:
   - `./build/test_iterative`
   - `./build/test_minres`
   - `./build/test_eigs`
   - `./build/test_eigs_lobpcg`
   - `./build/example_iterative`
   - `./build/example_eigs`
   - `./build/bench_iterative_reuse`
   - `./build/bench_eigs_reuse`
4. Record representative direct results from the touched proof surfaces.
5. Write the validation artifact and identify any true blocker if one appears.

### Deliverables
- Full validation record
- Updated truthfulness-anchor notes
- Representative direct proof outputs

### Completion Criteria
- All required validation passes from the landed Sprint 55 state
- Reviewed CMake parity and maintained baseline wording remain exact
- No unresolved blocker remains before closeout

---

## Day 14: Closeout & Handoff

**Title:** Sprint Closeout  
**Theme:** Turn the Sprint 55 landed work into a clean handoff for the next
large-source decomposition phase  
**Time estimate:** 12 hours

### Tasks
1. Summarize what Sprint 55 actually changed in:
   - eigensolver ownership
   - iterative ownership
   - permanent implementation commentary quality
   - validation confidence
2. Record the remaining highest-value large-source seams for the next sprint or
   phase without reopening Sprint 55 scope.
3. Reconfirm whether any Epic 5 project-plan adjustments are needed from the
   landed state.
4. Write the closeout artifact and update working notes with final results and
   next-step handoff.
5. Verify the sprint’s delivered hours and deliverables still match the plan.

### Deliverables
- Sprint 55 closeout artifact
- Final working-notes synthesis
- Next-phase decomposition handoff

### Completion Criteria
- Sprint 55 ends with a clear record of what ownership improved and what
  remains for later phases
- The delivered work matches the planned bounded Phase 1 decomposition scope
- The next sprint can start from a clean, validated, and documented handoff
