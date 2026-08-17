# Sprint 159 Plan: Hosted Oracle And Comparison Freshness Promotion

**Sprint Duration:** 14 days
**Goal:** Promote selected local-only generated oracle and comparison freshness
checks into reviewed hosted evidence without broadening solver claims. This
sprint implements the Sprint 159 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 159 begins from:
- Sprint 157 evidence contracts and claim-target boundaries available;
- Sprint 158 generated API HTML publication policy available;
- existing QR, partial-SVD, oracle, comparison, corpus, and report-index
  freshness targets available;
- current hosted CI already separates reviewed, supplemental, advisory, and
  local-only evidence;
- public claims remain constrained to reviewed hosted evidence and explicit
  non-claims.

The sprint must:
- select only claim-bearing oracle and comparison families for hosted
  promotion;
- leave expensive, advisory, incomplete, or local-only generated families
  explicitly out of reviewed hosted scope;
- measure local runtime and set hosted timeout, artifact-retention, and
  rerun expectations before adding CI work;
- add reviewed hosted freshness checks for selected generated rows;
- publish deterministic summaries or artifacts that make hosted evidence
  inspectable;
- tighten stale, missing, skipped, and failing row semantics for promoted
  families;
- update maintainer, corpus, README, support-tier, and report-family wording;
- leave Sprint 160 with a QR comparison handoff grounded in hosted evidence.

**End State:** Sprint 159 leaves behind:
- selected hosted oracle/comparison family register;
- runtime budget and artifact-retention decision;
- hosted generated freshness CI lane or steps for selected rows;
- deterministic artifact publication or summary output;
- updated normalizer semantics for promoted rows;
- documentation and support-tier wording aligned with hosted evidence;
- Sprint 160 QR comparison handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 159 project-plan estimate.

---

## Day 1: Sprint Intake And Promotion Boundary

**Title:** Promotion Intake
**Theme:** Establish Sprint 159 scope, artifact layout, and hosted-evidence
boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 159 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 157 evidence contracts and Sprint 158 publication policy.
3. Create Sprint 159 working notes and artifact directory structure.
4. Inventory oracle, comparison, report-index, QR, and partial-SVD freshness
   commands that could be promoted.
5. Record explicit non-goals for broad solver claims, broad external parity,
   and expensive local-only generated families.
6. Write the Day 1 promotion-boundary artifact.

### Deliverables
- Sprint 159 working-notes baseline
- artifact directory structure
- candidate freshness command inventory
- hosted-promotion boundary notes
- Day 1 promotion-boundary artifact

### Completion Criteria
- Sprint 159 scope is tied to the Epic 14 project plan
- candidate evidence families are known before selection work begins
- unsupported broad claims are blocked before hosted-promotion work starts

---

## Day 2: Family Selection Register

**Title:** Family Selection
**Theme:** Select claim-bearing generated families for hosted promotion
**Time estimate:** 12 hours

### Tasks
1. Review current oracle, comparison, QR, partial-SVD, corpus, and report-index
   rows.
2. Classify each family as reviewed-hosted candidate, supplemental-hosted,
   advisory-local, or deferred.
3. Tie each reviewed-hosted candidate to a concrete claim or support-tier row.
4. Record why advisory or expensive families remain local-only.
5. Identify minimum generated outputs needed for inspectable hosted evidence.
6. Write the Day 2 family-selection register.

### Deliverables
- selected hosted family register
- advisory/local-only family register
- claim-to-family mapping
- minimum hosted output list
- Day 2 family-selection artifact

### Completion Criteria
- selected families are claim-bearing and narrowly scoped
- local-only families are explicitly documented
- later CI work has an approved target list

---

## Day 3: Runtime Measurement Plan

**Title:** Runtime Plan
**Theme:** Define timing, timeout, and rerun measurements for selected rows
**Time estimate:** 12 hours

### Tasks
1. Identify local commands needed to run selected oracle and comparison rows.
2. Define cold and warm timing measurements for each selected family.
3. Record timeout and retry expectations for hosted runners.
4. Identify generated files, manifests, logs, and summaries produced by each
   command.
5. Define pass, skip, stale, and fail timing criteria before CI changes.
6. Write the Day 3 runtime-measurement plan.

### Deliverables
- runtime measurement matrix
- timeout and rerun policy draft
- generated-output inventory
- timing criteria
- Day 3 runtime-plan artifact

### Completion Criteria
- selected rows have measurable local commands
- hosted runtime expectations are stated before workflow edits
- artifact-retention planning has concrete output names

---

## Day 4: Runtime Budget Evidence

**Title:** Runtime Evidence
**Theme:** Measure selected families and set hosted budget decisions
**Time estimate:** 12 hours

### Tasks
1. Run selected local oracle and comparison freshness commands.
2. Capture runtime, output size, skip behavior, and failure behavior.
3. Compare observed runtime against likely hosted CI budget.
4. Select timeout, retention, and summary-size thresholds.
5. Record any family that must be demoted from hosted promotion.
6. Write the Day 4 runtime-budget artifact.

### Deliverables
- measured runtime table
- output-size and retention table
- hosted timeout decision
- demotion notes, if needed
- Day 4 runtime-budget artifact

### Completion Criteria
- promoted rows fit a realistic hosted budget
- demotions are documented with evidence
- CI implementation can proceed without guessing about runtime

---

## Day 5: CI Surface Design

**Title:** CI Design
**Theme:** Design the hosted freshness lane without disrupting existing tiers
**Time estimate:** 12 hours

### Tasks
1. Review current workflow files and validation lane names.
2. Decide whether selected checks belong in a new job, existing job, or
   supplemental step.
3. Map environment variables, make targets, scripts, and artifacts required by
   the selected families.
4. Preserve existing reviewed, supplemental, advisory, and local-only wording.
5. Define PR failure semantics for stale, missing, skipped, or failing rows.
6. Write the Day 5 CI-surface design artifact.

### Deliverables
- hosted CI design note
- workflow lane placement decision
- required command and environment list
- PR failure-semantics matrix
- Day 5 CI-design artifact

### Completion Criteria
- CI lane placement is justified by support-tier boundaries
- selected checks have concrete commands and outputs
- existing staged/support-tier wording remains coherent

---

## Day 6: Hosted Freshness Implementation

**Title:** Hosted Freshness
**Theme:** Add hosted CI execution for selected oracle/comparison rows
**Time estimate:** 12 hours

### Tasks
1. Implement workflow changes for the selected hosted freshness checks.
2. Wire selected commands to existing scripts or make targets.
3. Ensure the lane fails on selected stale, missing, or invalid generated rows.
4. Preserve local-only handling for non-promoted generated families.
5. Add focused script or workflow documentation where needed.
6. Record the Day 6 implementation notes.

### Deliverables
- hosted freshness workflow changes
- selected command wiring
- non-promoted family guardrails
- implementation notes
- Day 6 hosted-freshness artifact

### Completion Criteria
- selected rows are executable from hosted CI configuration
- non-selected generated families are not accidentally promoted
- failure semantics match the Day 5 design

---

## Day 7: Artifact Publication Design

**Title:** Artifact Design
**Theme:** Define inspectable hosted summaries and artifact retention
**Time estimate:** 12 hours

### Tasks
1. Identify generated indexes, manifests, skips, logs, and comparison reports
   that hosted reviewers need.
2. Decide which outputs are uploaded artifacts and which are deterministic
   console summaries.
3. Define artifact names, retention days, and path structure.
4. Ensure artifact naming distinguishes reviewed hosted rows from advisory
   local-only rows.
5. Add summary expectations for empty, skipped, stale, and failing rows.
6. Write the Day 7 artifact-publication design.

### Deliverables
- artifact and summary design
- retention policy
- hosted output naming convention
- row-state summary expectations
- Day 7 artifact-design document

### Completion Criteria
- reviewers can inspect promoted evidence from hosted runs
- artifact retention is explicit and bounded
- local-only outputs cannot be confused with reviewed hosted evidence

---

## Day 8: Artifact Publication Implementation

**Title:** Artifact Publication
**Theme:** Publish deterministic summaries or upload artifacts for promoted rows
**Time estimate:** 12 hours

### Tasks
1. Implement artifact upload or deterministic summary output for selected rows.
2. Include generated indexes, manifests, skip summaries, and comparison reports
   according to the Day 7 design.
3. Keep artifact paths stable across hosted runners.
4. Ensure failure paths still publish useful diagnostics where feasible.
5. Run local dry-run commands or script checks for summary formatting.
6. Record the Day 8 publication artifact.

### Deliverables
- artifact upload or summary implementation
- stable hosted output paths
- failure diagnostic output
- local dry-run notes
- Day 8 artifact-publication artifact

### Completion Criteria
- promoted rows produce inspectable hosted evidence
- outputs are deterministic enough for review and rerun comparison
- failure diagnostics do not require reproducing the full run locally

---

## Day 9: Normalizer Semantics Audit

**Title:** Semantics Audit
**Theme:** Audit stale, missing, skipped, and failing row behavior
**Time estimate:** 12 hours

### Tasks
1. Review report-index normalizer behavior for selected promoted families.
2. Trace stale, missing, skipped, failing, advisory, and local-only row states.
3. Identify ambiguity between hosted reviewed rows and local-only generated
   families.
4. Draft tightened semantics for promoted rows.
5. Define test cases or fixture updates needed for normalizer behavior.
6. Write the Day 9 normalizer-semantics audit.

### Deliverables
- current row-state behavior map
- ambiguity and gap list
- promoted-row semantics draft
- test or fixture update list
- Day 9 semantics-audit artifact

### Completion Criteria
- selected row states are understood before implementation
- stale and missing behavior cannot silently pass for reviewed rows
- advisory/local-only semantics remain distinct

---

## Day 10: Normalizer Semantics Implementation

**Title:** Semantics Implementation
**Theme:** Tighten promoted-row stale, missing, skip, and failure behavior
**Time estimate:** 12 hours

### Tasks
1. Implement normalizer or report-index changes for promoted-row semantics.
2. Add or update focused tests for stale, missing, skipped, failing, and valid
   selected rows.
3. Preserve advisory/local-only behavior for non-promoted families.
4. Update script help, comments, or docs if command behavior changes.
5. Run targeted script tests or report-index checks.
6. Record the Day 10 semantics-implementation artifact.

### Deliverables
- normalizer/report-index implementation changes
- focused semantics tests or fixtures
- targeted validation output
- updated command behavior notes
- Day 10 semantics-implementation artifact

### Completion Criteria
- promoted rows fail clearly when stale, missing, or invalid
- non-promoted rows retain documented advisory/local behavior
- targeted semantics validation passes locally

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align maintainer, corpus, README, support-tier, and report-family
wording
**Time estimate:** 12 hours

### Tasks
1. Update maintainer guidance for promoted hosted generated freshness checks.
2. Update corpus or report-family docs with selected hosted rows and local-only
   boundaries.
3. Update README or support-tier language if hosted evidence status changed.
4. Ensure docs do not imply broad solver, broad comparison, package, or
   performance claims.
5. Add Sprint 160 handoff notes for QR comparison work.
6. Write the Day 11 documentation-alignment artifact.

### Deliverables
- maintainer documentation updates
- corpus/report-family wording updates
- README or support-tier wording updates, if needed
- Sprint 160 QR comparison handoff draft
- Day 11 docs-alignment artifact

### Completion Criteria
- docs match the promoted hosted evidence surface
- unsupported claims remain explicit non-claims
- Sprint 160 has a concrete comparison handoff

---

## Day 12: Local Validation Pass

**Title:** Local Validation
**Theme:** Run local freshness, script, docs, and targeted quality checks
**Time estimate:** 12 hours

### Tasks
1. Run selected local oracle and comparison freshness commands.
2. Run focused normalizer/report-index tests.
3. Run relevant docs or generated-index freshness checks.
4. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.
5. If only docs/scripts/workflows changed, run targeted checks plus
   whitespace and diff hygiene.
6. Record the Day 12 local-validation artifact.

### Deliverables
- local freshness validation output
- targeted script/test output
- quality-check record
- issue and follow-up list
- Day 12 validation artifact

### Completion Criteria
- promoted freshness commands pass locally
- required quality checks are selected by changed-file type
- remaining failures, if any, are understood before closeout

---

## Day 13: Hosted Readiness Review

**Title:** Hosted Readiness
**Theme:** Review CI, artifacts, semantics, and claims as one hosted evidence
surface
**Time estimate:** 12 hours

### Tasks
1. Re-read workflow changes, artifact output, normalizer semantics, and docs as
   a single reviewer path.
2. Confirm selected rows have hosted execution, deterministic evidence, and
   clear failure behavior.
3. Confirm advisory/local-only families remain out of reviewed hosted claims.
4. Update working notes with residual risks and hosted rerun expectations.
5. Finalize Sprint 160 QR comparison handoff.
6. Write the Day 13 hosted-readiness artifact.

### Deliverables
- hosted-readiness checklist
- selected-row evidence map
- local-only boundary confirmation
- finalized Sprint 160 handoff
- Day 13 readiness artifact

### Completion Criteria
- a reviewer can trace each promoted claim to hosted evidence
- hosted artifacts or summaries are inspectable and bounded
- Sprint 160 handoff is complete enough to start

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 159 artifacts, validation record, and retrospective
inputs
**Time estimate:** 12 hours

### Tasks
1. Re-run final targeted checks needed by the changed-file surface.
2. Update Sprint 159 working notes with final decisions, commands, and outputs.
3. Create or update final closeout artifacts for promoted rows, demoted rows,
   validation, and Sprint 160 handoff.
4. Review changed files for claim wording, stale paths, and unsupported
   evidence assertions.
5. Prepare retrospective inputs from artifacts and working notes.
6. Record the Day 14 closeout artifact.

### Deliverables
- final validation record
- promoted/demoted row closeout notes
- complete working notes
- retrospective input set
- Day 14 closeout artifact

### Completion Criteria
- Sprint 159 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 160 QR comparison handoff is ready

