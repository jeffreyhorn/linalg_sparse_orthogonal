# Sprint 168 Plan: Hosted Performance Publication Lane

**Sprint Duration:** 14 days
**Goal:** Promote one selected performance report into a hosted,
freshness-checked CI lane with methodology-bound claims. This sprint
implements the Sprint 168 section of
`docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Hosted Performance
Publication Date"; the active merged Sprint 168 project-plan section lives in
`docs/planning/EPIC_15/PROJECT_PLAN.md` and is titled "Hosted Performance
Publication Lane".

**Starting Point:** Sprint 168 begins from:
- Sprint 167 evidence ledger, gap-selection gate, claim gates, and closeout
  handoff landed on `master`;
- the Sprint 167 recommendation to start from `bench_refactor_csc` through
  `make bench-canonical-report` as the preferred hosted performance
  publication candidate;
- existing benchmark/report command owners in `Makefile`,
  `scripts/bench_canonical_report.sh`, `benchmarks/README.md`, and `README.md`;
- existing CI lanes that compile benchmarks and run `bench-fast`, but do not
  yet publish a methodology-bound hosted performance report;
- retained non-claims for portable performance superiority, broad backend
  superiority, external-library performance parity, broad platform parity,
  release benchmark proof, and state-of-the-art performance.

The sprint must:
- select the exact benchmark family, command, fixture scope, platform lane,
  runtime budget, and report path for hosted publication;
- confirm local runtime and output stability before CI promotion;
- add methodology metadata needed to interpret the selected report;
- add a strict freshness check for the selected hosted performance report;
- wire the selected report into hosted CI with bounded runtime and clear
  failure output;
- update README/report docs only for the proven scope;
- run the selected benchmark/report checks, docs checks, and full C quality
  gate if source or header files change.

**End State:** Sprint 168 leaves behind:
- a selected hosted performance publication scope;
- a generated report path and freshness check for that scope;
- hosted CI wiring for the selected performance report;
- claim-safe documentation describing only the supported lane;
- Sprint 168 working notes, daily artifacts, and validation records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 168 project-plan estimate.

---

## Day 1: Sprint Intake And Performance Handoff

**Title:** Performance Intake
**Theme:** Establish Sprint 168 scope from Sprint 167 claim gates and
performance handoff
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 168 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 167 Day 12 claim gates and Day 13 Sprint 168 handoff.
3. Create Sprint 168 working notes and artifact directory structure.
4. Record the prompt path/title mismatch and active Epic 15 source artifact.
5. Define the retained performance non-claims and stop conditions for this
   sprint.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 168 working-notes baseline
- artifact directory structure
- source artifact note
- retained performance non-claim list
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 168 scope is tied to the active Epic 15 project plan
- Sprint 167 acceptance gates are carried forward
- no hosted performance claim is made before a selected lane exists

---

## Day 2: Benchmark Surface Inventory

**Title:** Benchmark Inventory
**Theme:** Inventory current benchmark commands, report scripts, docs, and
generated-output conventions
**Time estimate:** 12 hours

### Tasks
1. Review `Makefile` benchmark targets and report-generation targets.
2. Review `scripts/bench_canonical_report.sh` and related performance report
   scripts.
3. Review `benchmarks/README.md` and README performance wording.
4. Inventory current canonical report outputs, manifest/index fields, and
   local-only boundaries.
5. Identify any existing freshness or report-index conventions that can be
   reused.
6. Write the Day 2 benchmark-surface artifact.

### Deliverables
- benchmark command inventory
- report script inventory
- performance documentation inventory
- generated-output convention notes
- Day 2 benchmark-surface artifact

### Completion Criteria
- current benchmark/report owners are known
- local-only versus hosted evidence boundaries are explicit
- reusable report/freshness conventions are identified

---

## Day 3: Candidate Lane Selection

**Title:** Lane Selection
**Theme:** Select the exact benchmark family, fixture scope, command, and
platform lane for hosted publication
**Time estimate:** 12 hours

### Tasks
1. Compare `bench_refactor_csc` with alternative canonical candidates:
   `bench_chol_csc`, `bench_iterative_reuse`, and `bench_eigs_reuse`.
2. Score candidates by runtime, output stability, user value, methodology
   clarity, and claim-risk containment.
3. Select the primary hosted performance lane or record why the Sprint 167
   recommendation needs a narrower alternative.
4. Define the exact command, fixture or fixture subset, platform, compiler,
   thread settings, repeat semantics, and output path.
5. Record explicit out-of-scope performance claims.
6. Write the Day 3 lane-selection artifact.

### Deliverables
- candidate scoring table
- selected performance lane
- command and fixture-scope definition
- platform/toolchain/thread/repeat scope
- Day 3 lane-selection artifact

### Completion Criteria
- one hosted performance publication candidate is selected
- selected evidence boundary is narrow and reviewable
- portable superiority and broad backend claims remain non-claims

---

## Day 4: Runtime Suitability And Local Dry Run

**Title:** Runtime Dry Run
**Theme:** Measure local runtime, output stability, and CI suitability for the
selected lane
**Time estimate:** 12 hours

### Tasks
1. Build the selected benchmark/report target.
2. Run the selected command locally with the planned fixture, repeat, and
   thread settings.
3. Measure wall time and generated output size.
4. Inspect generated CSV, manifest, and index rows for stable fields.
5. Identify runtime risks, flaky output fields, or missing methodology data.
6. Write the Day 4 runtime-suitability artifact.

### Deliverables
- local dry-run command log
- runtime and output-size notes
- stability findings
- CI suitability decision
- Day 4 runtime-suitability artifact

### Completion Criteria
- selected lane has a bounded runtime plan
- unstable output fields are identified before CI wiring
- lane remains suitable or is narrowed explicitly

---

## Day 5: Methodology Metadata Design

**Title:** Metadata Design
**Theme:** Define methodology fields required for hosted report
interpretation
**Time estimate:** 12 hours

### Tasks
1. Compare current report fields with Sprint 167 Day 12 requirements.
2. Define required methodology fields for compiler, flags, CPU, OS, runner,
   thread settings, backend/build mode, repeat count, warmup state, variance
   state, timestamp, branch, commit, command, fixture, threshold policy, and
   claim boundary.
3. Decide which fields belong in CSV rows, `index.tsv`, and `manifest.txt`.
4. Define deterministic formatting and unknown-value behavior.
5. Specify how local and hosted report metadata differ.
6. Write the Day 5 methodology-metadata artifact.

### Deliverables
- methodology field specification
- row/index/manifest ownership map
- deterministic formatting rules
- local versus hosted metadata notes
- Day 5 methodology-metadata artifact

### Completion Criteria
- every required methodology field has an owner
- missing metadata behavior is explicit
- report metadata does not imply performance superiority

---

## Day 6: Report Metadata Implementation

**Title:** Metadata Implementation
**Theme:** Implement methodology metadata for the selected report path
**Time estimate:** 12 hours

### Tasks
1. Update benchmark/report scripts to emit the selected methodology metadata.
2. Preserve existing canonical report behavior for unselected rows.
3. Add environment-variable or command-line hooks needed by hosted CI.
4. Keep generated output under ignored build/report paths unless policy says
   otherwise.
5. Run the selected report command locally.
6. Write the Day 6 metadata-implementation artifact.

### Deliverables
- metadata implementation changes
- selected report output with methodology fields
- compatibility notes for existing report rows
- Day 6 metadata-implementation artifact

### Completion Criteria
- selected report emits required metadata locally
- existing report commands still run for current local workflows
- no generated build/report artifacts are staged unintentionally

---

## Day 7: Freshness Check Design

**Title:** Freshness Design
**Theme:** Design a strict freshness check for the selected hosted performance
report
**Time estimate:** 12 hours

### Tasks
1. Review existing report freshness and normalization checks.
2. Define what makes the selected performance report fresh.
3. Decide whether the freshness check compares schemas, selected rows,
   manifest metadata, report-index entries, or artifact paths.
4. Define failure messages for missing files, stale rows, unsupported
   metadata, and over-broad claim boundaries.
5. Define local and hosted invocation targets.
6. Write the Day 7 freshness-design artifact.

### Deliverables
- freshness criteria
- local/hosted freshness command design
- failure-message requirements
- selected report-index behavior
- Day 7 freshness-design artifact

### Completion Criteria
- selected report freshness has objective pass/fail rules
- failure output will be actionable in CI
- freshness rules do not convert timing into a superiority gate

---

## Day 8: Freshness Check Implementation

**Title:** Freshness Implementation
**Theme:** Implement and validate the selected performance freshness check
**Time estimate:** 12 hours

### Tasks
1. Add or update the freshness script/target for the selected performance
   report.
2. Wire the check into the appropriate Makefile target.
3. Add focused script tests or self-checks if the implementation introduces
   parsing logic.
4. Run the selected report generation and freshness commands locally.
5. Record failure-mode coverage and limitations.
6. Write the Day 8 freshness-implementation artifact.

### Deliverables
- freshness script or target
- focused validation/self-checks
- local report and freshness command results
- failure-mode notes
- Day 8 freshness-implementation artifact

### Completion Criteria
- selected freshness command passes locally
- missing/stale selected report cases fail clearly
- unselected report families are not accidentally promoted

---

## Day 9: CI Lane Design

**Title:** CI Design
**Theme:** Design the hosted performance CI lane with bounded runtime and
clear artifact ownership
**Time estimate:** 12 hours

### Tasks
1. Review current Linux benchmark, report, and artifact-upload workflow
   structure.
2. Select the hosted workflow/job location for the performance lane.
3. Define runner image, compiler, build flags, thread settings, runtime
   budget, artifact upload paths, and failure messages.
4. Decide whether the lane is reviewed or supplemental evidence.
5. Define hosted evidence wording for README/report docs.
6. Write the Day 9 CI-lane-design artifact.

### Deliverables
- CI lane design
- hosted evidence classification
- artifact upload path list
- runtime budget and environment settings
- Day 9 CI-lane-design artifact

### Completion Criteria
- hosted lane design is bounded and reviewable
- artifact ownership is explicit
- hosted evidence classification is not broader than the lane

---

## Day 10: CI Lane Implementation

**Title:** CI Implementation
**Theme:** Wire the selected performance report and freshness check into
hosted CI
**Time estimate:** 12 hours

### Tasks
1. Update the selected GitHub Actions workflow with the performance lane.
2. Add report generation, freshness check, and artifact upload steps.
3. Ensure CI output names the selected performance scope and retained
   non-claims.
4. Avoid widening existing `bench-fast` or canonical-report wording.
5. Validate workflow syntax and local command equivalents where possible.
6. Write the Day 10 CI-implementation artifact.

### Deliverables
- workflow update
- hosted report generation step
- hosted freshness check step
- artifact upload configuration
- Day 10 CI-implementation artifact

### Completion Criteria
- CI lane is wired for the selected report only
- local equivalents pass before relying on hosted CI
- workflow wording stays methodology-bound

---

## Day 11: Claim-Safe Documentation Update

**Title:** Claim-Safe Docs
**Theme:** Update README and benchmark/report docs for the selected hosted
performance lane
**Time estimate:** 12 hours

### Tasks
1. Update README performance evidence wording for the selected hosted lane.
2. Update `benchmarks/README.md` with command, report path, methodology
   interpretation, and hosted/local distinction.
3. Update report-index or maintainer docs if the new lane changes evidence
   ownership.
4. Add explicit non-claim wording for portable superiority, broad backend
   superiority, external-library parity, release proof, and state-of-the-art
   performance.
5. Run targeted claim scans for risky performance wording.
6. Write the Day 11 claim-safe-docs artifact.

### Deliverables
- README performance wording update
- benchmark/report documentation update
- retained performance non-claim wording
- claim-scan notes
- Day 11 claim-safe-docs artifact

### Completion Criteria
- docs describe only the selected hosted performance scope
- local benchmark rows remain distinct from hosted evidence
- no broad performance or state-of-the-art claims are introduced

---

## Day 12: Local Validation Sweep

**Title:** Local Validation
**Theme:** Run selected report, freshness, docs, and required code quality
checks
**Time estimate:** 12 hours

### Tasks
1. Run the selected benchmark/report command.
2. Run the selected performance freshness check.
3. Run relevant script self-checks or Python compile checks if scripts changed.
4. Run targeted docs consistency and claim-scan checks.
5. Run `make format && make lint && make test` if `.c` or `.h` files changed.
6. Write the Day 12 local-validation artifact.

### Deliverables
- local validation command log
- report/freshness validation result
- docs/claim-scan result
- full C quality gate result or skipped-check rationale
- Day 12 local-validation artifact

### Completion Criteria
- selected local checks pass
- skipped checks have explicit reasons
- code/header changes, if any, pass the full C quality gate

---

## Day 13: Hosted Evidence Reconciliation Prep

**Title:** Hosted Prep
**Theme:** Prepare PR-hosted evidence expectations, artifact review steps, and
fallback handling
**Time estimate:** 12 hours

### Tasks
1. Reconcile local validation against the hosted lane design.
2. Define the expected PR CI job name, artifact names, and report paths.
3. Create hosted-evidence review notes for PR validation.
4. Define fallback wording if hosted performance publication fails due to
   runtime, infrastructure, or metadata issues.
5. Confirm all Sprint 168 artifacts and working notes are current.
6. Write the Day 13 hosted-evidence-prep artifact.

### Deliverables
- hosted evidence expectation list
- artifact review checklist
- fallback/deferral wording
- updated working notes
- Day 13 hosted-evidence-prep artifact

### Completion Criteria
- PR reviewers know which hosted job and artifacts to inspect
- fallback handling is documented
- no hosted claim depends on unobserved CI success

---

## Day 14: Sprint Validation And Closeout

**Title:** Sprint Closeout
**Theme:** Finalize Sprint 168 validation, artifacts, and handoff to Sprint
169 methodology hardening
**Time estimate:** 12 hours

### Tasks
1. Run final selected local validation commands.
2. Confirm no generated build/report/cache artifacts are staged
   unintentionally.
3. Reconcile Sprint 168 deliverables against project-plan items 168.1 through
   168.6.
4. Prepare the Sprint 169 handoff for methodology hardening.
5. Record final validation, skipped checks, hosted evidence expectations, and
   residual risks.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables
- final Sprint 168 validation record
- project-plan item reconciliation
- Sprint 169 methodology-hardening handoff
- residual and hosted-evidence register
- Day 14 sprint-closeout artifact

### Completion Criteria
- selected hosted performance lane is implemented or explicitly deferred with
  evidence
- Sprint 168 artifacts match project-plan items
- Sprint 169 can begin from a clear methodology-hardening baseline
