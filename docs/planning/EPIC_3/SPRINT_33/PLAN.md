# Sprint 33 Plan: Dead-Code Detection Infrastructure & First Cleanup Pass

**Sprint Duration:** 14 days  
**Goal:** Add dead-code-detection infrastructure, reporting, and policy guardrails, then complete the first cleanup pass against definitely-unused internal code without regressing the active test and validation surface. This sprint implements the Sprint 33 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`.

**Starting Point:** Sprint 32 closed with `0` full-tree warnings, `0` dormant-scaffold debt, a live opt-in test policy (`SPARSE_TEST_SLOW`, `SPARSE_TEST_EXPERIMENTAL`), and validated `make lint`, `make test`, `ctest -N`, and full `ctest` parity. Sprint 33 starts from that clean baseline and shifts focus from warning cleanup to dead-code detection infrastructure, reportability, and the first targeted removal pass on definitely-unused internal code.

**End State:** Sprint 33 leaves behind a documented dead-code policy, Makefile-backed dead-code detection and reporting targets, an auditable candidate-public-API review, and a first cleanup pass that removes definitely-unused internal code while preserving the Sprint 32 truthfulness rules, active opt-in coverage, and green validation flows.

**Time budget:** Each day is capped at 12 hours. The day budgets below sum to 124 hours, matching the Sprint 33 estimate in `PROJECT_PLAN.md`.

---

## Day 1: Sprint 33 Scope Audit & Baseline

**Title:** Dead-Code Baseline  
**Theme:** Convert the Sprint 32 handoff into a precise Sprint 33 infrastructure and cleanup inventory  
**Time estimate:** 8 hours

### Tasks
1. Re-read the Sprint 33 section of `docs/planning/EPIC_3/PROJECT_PLAN.md`, the Sprint 32 handoff, and the Sprint 32 retrospective so the sprint stays anchored to the explicit deferred scope.
2. Confirm the Sprint 32 closeout guarantees that must remain true during Sprint 33: zero warning debt, no dormant scaffold debt, and live opt-in test coverage.
3. Inspect the current Makefile, CMake setup, `compile_commands.json` generation path, and any existing static-analysis workflow relevant to dead-code detection.
4. Identify the code areas most likely to contain definitely-unused internal code, prioritizing `tests/`, `benchmarks/`, `examples/`, and private helper paths before any public surface.
5. Open Sprint 33 working notes and record the baseline assumptions, likely tooling entry points, and initial target files for policy, infrastructure, and cleanup work.

### Deliverables
- Sprint 33 baseline inventory
- Initial dead-code tooling entry-point map
- Named first-pass candidate areas for later cleanup

### Completion Criteria
- The sprint starts from a documented Sprint 32 clean baseline
- The likely implementation surfaces for dead-code tooling are identified before edits begin
- The first cleanup pass is scoped toward definitely-unused internal code, not speculative API churn

---

## Day 2: Dead-Code Policy Audit

**Title:** Policy Audit  
**Theme:** Define how dead-code work will preserve Sprint 32 truthfulness guarantees  
**Time estimate:** 8 hours

### Tasks
1. Review the Sprint 32 truthfulness rules and map how they constrain dead-code classification in tests, examples, benchmarks, and private helpers.
2. Distinguish acceptable active opt-in scaffold from actual dead code, with `tests/test_framework_optin.c` treated as a live protection surface rather than removable scaffolding.
3. Identify cases where historical evidence belongs in `docs/planning/` instead of as compiled-but-unused code.
4. Note the risk boundaries for public headers, exported symbols, and documented examples so Sprint 33 does not remove externally meaningful surface accidentally.
5. Write an audit note that defines the policy questions the implementation must answer before cleanup begins.

### Deliverables
- Dead-code policy audit note
- Initial keep/remove/defer classification rules
- Explicit truthfulness constraints for Sprint 33 cleanup

### Completion Criteria
- The dead-code problem is separated from the already-solved dormant-scaffold problem
- The project has a documented basis for preserving active and opt-in coverage
- Public-surface risk boundaries are identified before tooling or cleanup edits begin

---

## Day 3: Dead-Code Policy & Limitations Design

**Title:** Policy Design  
**Theme:** Turn the audit into a project-level dead-code policy and limitations statement  
**Time estimate:** 10 hours

### Tasks
1. Define the Sprint 33 dead-code policy for active code, opt-in code, historical evidence, generated artifacts, and documented public surface.
2. Decide what counts as “definitely-unused internal code” versus “candidate for later review,” with conservative rules for public APIs and documented behaviors.
3. Capture the limitations of the planned tooling inputs, especially likely false positives from `cppcheck` and reachability gaps in `xunused`.
4. Specify what evidence is required before code can be deleted in Sprint 33.
5. Write the policy/design artifact that Day 5 through Day 11 will follow.

### Deliverables
- Sprint 33 dead-code policy
- Limitations and false-positive handling note
- Deletion threshold for definitely-unused internal code

### Completion Criteria
- The policy is specific enough to drive implementation and code review
- Tool limitations are documented before their output is treated as actionable
- The deletion bar is conservative and auditable

---

## Day 4: Tooling Integration Design

**Title:** Tooling Design  
**Theme:** Design the Makefile and reporting flow for dead-code detection  
**Time estimate:** 8 hours

### Tasks
1. Audit how `make lint`, `make test`, and existing build targets produce or rely on compilation metadata.
2. Design the `deadcode` target around `cppcheck --enable=all --quiet src/` and `xunused build/compile_commands.json`, including prerequisites and output flow.
3. Decide whether `deadcode-report` and `deadcode-check` should be separate targets, aliases, or layered wrappers.
4. Choose how raw tool output should be captured, normalized, and summarized so later CI integration is straightforward.
5. Record the implementation plan for Makefile wiring, generated artifacts, and operator-facing usage.

### Deliverables
- Makefile target design for `deadcode`
- Reporting-flow design for `deadcode-report` / `deadcode-check`
- Notes on prerequisites and output normalization

### Completion Criteria
- The tooling design is concrete enough to implement without inventing new scope midstream
- The reporting flow is separated from raw tool invocation
- The design fits the existing project build structure

---

## Day 5: `deadcode` Target Implementation

**Title:** Deadcode Target  
**Theme:** Add the first executable dead-code detection target to the Makefile  
**Time estimate:** 10 hours

### Tasks
1. Implement the `deadcode` Makefile target with the agreed prerequisites and command flow.
2. Ensure the target works from a clean tree and uses the expected compilation database path consistently.
3. Keep the implementation readable and easy to extend for later CI integration rather than embedding brittle shell logic.
4. Run focused validation on the target and capture the initial raw output for later classification.
5. Update Sprint 33 notes with the actual invocation contract and any friction discovered during implementation.

### Deliverables
- Working `make deadcode` target
- Initial captured raw dead-code output
- Implementation notes for follow-on reporting work

### Completion Criteria
- `make deadcode` runs end to end in the intended local workflow
- The target’s prerequisites and assumptions are explicit
- Raw findings are available for later classification rather than discarded

---

## Day 6: Reporting & Classification Design

**Title:** Report Design  
**Theme:** Define how raw dead-code findings become an auditable report  
**Time estimate:** 8 hours

### Tasks
1. Review the raw Day 5 output and separate recurring classes such as private helpers, test-only utilities, benchmark/example leftovers, and public-surface candidates.
2. Decide how `deadcode-report` should present findings, suppress obvious noise, and preserve enough detail for review.
3. Define `deadcode-check` behavior, including whether it is purely informative this sprint or enforces a narrower invariant.
4. Specify the artifact layout and note where generated reports should live for future sprint handoffs or CI wiring.
5. Write the classification design note for the reporting implementation.

### Deliverables
- Dead-code report classification scheme
- Behavior definition for `deadcode-report` and `deadcode-check`
- Artifact/output layout note

### Completion Criteria
- Raw tool output is translated into an actionable reporting model
- The distinction between informative reporting and enforcement is explicit
- The classification scheme is conservative enough to avoid misleading cleanup decisions

---

## Day 7: Reporting Target Implementation

**Title:** Report Wiring  
**Theme:** Implement report-generation targets and produce the first actionable dead-code report  
**Time estimate:** 10 hours

### Tasks
1. Implement `deadcode-report` and the chosen `deadcode-check` behavior in the Makefile and any supporting script paths if needed.
2. Normalize the raw tool findings into a stable report format that can be compared in later sprints.
3. Generate the first report and classify findings into definitely-unused internal code, public-surface review items, and deferred/false-positive buckets.
4. Record which findings are realistic candidates for the Sprint 33 first cleanup pass.
5. Update Sprint 33 notes with the operator workflow for generating and reading the report.

### Deliverables
- Working `make deadcode-report`
- Working `make deadcode-check` behavior
- First classified dead-code report

### Completion Criteria
- The reporting targets run successfully and produce an auditable artifact
- Findings are grouped into actionable categories rather than left as raw scanner noise
- The first cleanup pass has a documented candidate queue

---

## Day 8: Candidate Public API Audit

**Title:** API Audit  
**Theme:** Separate internal cleanup candidates from public-surface review risks  
**Time estimate:** 8 hours

### Tasks
1. Review any report findings that touch exported headers, documented functions, or externally visible CLI/program surface.
2. Cross-check candidate public-surface findings against README, algorithm docs, examples, and installed headers.
3. Mark each such item as keep, defer, or reclassify based on documented contract rather than local reachability alone.
4. Tighten the Sprint 33 cleanup queue so only definitely-unused internal code remains in scope for deletion.
5. Write a short audit note documenting the public-surface review decisions.

### Deliverables
- Candidate-public-API audit note
- Refined cleanup queue limited to definitely-unused internal code
- Deferred list for any public-surface questions

### Completion Criteria
- Public-facing items are not mixed into the first cleanup batch casually
- The cleanup queue is narrowed to low-risk internal removals
- Deferred API questions are documented rather than silently ignored

---

## Day 9: First Cleanup Pass Design

**Title:** Cleanup Design  
**Theme:** Turn the refined queue into incremental deletion batches  
**Time estimate:** 10 hours

### Tasks
1. Group definitely-unused internal findings into small implementation batches across tests, benchmarks, examples, and private helpers.
2. Decide the order that minimizes merge risk and validation cost while still delivering meaningful cleanup.
3. Identify any nearby docs or comments that would become misleading after removal.
4. Define focused validation commands for each batch so cleanup can proceed incrementally rather than as one large risky edit.
5. Write the cleanup-batch design note for the Day 10 and Day 11 implementation work.

### Deliverables
- First cleanup pass batch plan
- File-by-file removal queue
- Validation plan for each cleanup batch

### Completion Criteria
- The deletion work is sequenced into small auditable batches
- Each batch has a clear justification and validation path
- Cleanup remains focused on definitely-unused internal code

---

## Day 10: First Cleanup Pass — Batch I

**Title:** Cleanup Batch I  
**Theme:** Remove the first set of definitely-unused internal code  
**Time estimate:** 8 hours

### Tasks
1. Delete the first cleanup batch from the lowest-risk internal areas identified on Day 9.
2. Update any affected comments, local docs, or notes so they remain truthful after the removals.
3. Rebuild and run focused validation for the touched files after each small deletion set.
4. Capture any false-positive lessons from the batch so the reporting flow can improve.
5. Record the removal results and remaining queue in Sprint 33 notes.

### Deliverables
- First removal batch landed
- Focused validation evidence for Batch I
- Notes on report accuracy versus real code state

### Completion Criteria
- Batch I removes only definitely-unused internal code
- The touched validation flow stays green
- Documentation remains aligned with the edited code

---

## Day 11: First Cleanup Pass — Batch II & Reconciliation

**Title:** Cleanup Batch II  
**Theme:** Finish the first cleanup pass and reconcile the dead-code report  
**Time estimate:** 10 hours

### Tasks
1. Remove the remaining Sprint 33 first-pass cleanup batch from the refined internal queue.
2. Regenerate the dead-code report and compare it against the pre-cleanup baseline.
3. Identify which residual findings are real follow-on work, which are intentional keeps, and which are tooling noise.
4. Update notes and report annotations so Sprint 34 inherits a truthful remaining queue.
5. Confirm that Sprint 32 truthfulness guarantees still hold after the removals.

### Deliverables
- Second cleanup batch landed
- Reconciled post-cleanup dead-code report
- Updated remaining-queue notes

### Completion Criteria
- The planned first cleanup pass is complete
- Residual report findings are categorized, not left ambiguous
- The post-cleanup state is ready for full validation

---

## Day 12: Documentation Refresh

**Title:** Deadcode Docs  
**Theme:** Document the new tooling, policy, and usage expectations  
**Time estimate:** 8 hours

### Tasks
1. Update maintainer-facing docs for the dead-code workflow, including `make deadcode`, reporting targets, and interpretation limits.
2. Document the distinction between active code, opt-in test code, historical evidence, and definitely-unused internal code.
3. Add notes on conservative handling of candidate public APIs and report false positives.
4. Ensure the docs describe the first cleanup pass accurately and do not overstate what the tooling proves.
5. Update Sprint 33 working notes with the final documentation map.

### Deliverables
- Maintainer docs for dead-code tooling and policy
- Updated description of cleanup/reporting limitations
- Sprint 33 notes reflecting the documented workflow

### Completion Criteria
- Operators can run and interpret the dead-code workflow without guessing
- The docs preserve the Sprint 32 truthfulness model explicitly
- The tooling is presented as conservative evidence, not as perfect reachability proof

---

## Day 13: Full Validation Sweep

**Title:** Validation Sweep  
**Theme:** Prove the sprint changes preserve normal validation and the new dead-code workflow  
**Time estimate:** 8 hours

### Tasks
1. Run `make format`, `make lint`, `make test`, `ctest -N`, and full `ctest`.
2. Run the Sprint 33 dead-code targets and confirm they still produce the documented outputs after cleanup.
3. Reconfirm that `tests/test_framework_optin.c` remains live and that Sprint 32 opt-in conventions are unaffected.
4. Record validation timing, test counts, and any dead-code target observations needed for handoff.
5. Update Sprint 33 notes with the final validated end state before closeout docs begin.

### Deliverables
- Full Sprint 33 validation record
- Dead-code target validation record
- Confirmed post-cleanup active/opt-in test state

### Completion Criteria
- All required validation flows pass
- The new dead-code tooling works after the cleanup pass
- The sprint end state is fully measured before handoff

---

## Day 14: Closeout, Handoff & Future Queue

**Title:** Sprint Closeout  
**Theme:** Package the dead-code tooling and cleanup results for Sprint 34 and beyond  
**Time estimate:** 10 hours

### Tasks
1. Write the Sprint 33 handoff summarizing delivered tooling, policy decisions, cleanup results, and validated commands.
2. Write the Sprint 33 retrospective covering what worked, what remained conservative, and where the tooling still has blind spots.
3. Document the residual dead-code queue, false-positive patterns, and any deferred public-surface review items for future sprints.
4. Update `docs/planning/EPIC_3/PROJECT_PLAN.md` if Sprint 33 reveals concrete deferred work that should be routed into Sprint 34 or later.
5. Ensure the closeout explicitly preserves the Sprint 32 and Sprint 33 invariants future work must not regress.

### Deliverables
- `HANDOFF.md`
- `RETROSPECTIVE.md`
- Forward-plan updates for deferred Sprint 34+ work if needed

### Completion Criteria
- Sprint 33 artifacts explain both what changed and what remains intentionally deferred
- Future planning can recover the residual queue without rereading the full sprint history
- The sprint closes with a clear validated baseline for subsequent cleanup work
