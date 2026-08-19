# Sprint 169 Plan: Performance Methodology Hardening

**Sprint Duration:** 14 days
**Goal:** Turn the selected performance lane into a durable
methodology-bound publication surface. This sprint implements the Sprint 169
section of `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Sprint 169:
Performance Methodology Hardening"; the active merged Sprint 169 project-plan
section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has the same
title.

**Starting Point:** Sprint 169 begins from:

- Sprint 168 selected hosted performance lane for `bench_refactor_csc` on
  `tests/data/suitesparse/nos4.mtx --repeat 1`;
- `make bench-canonical-report` and `make bench-canonical-report-freshness`;
- `scripts/check_bench_canonical_freshness.py` enforcing selected-row
  freshness and unselected-row local-only boundaries;
- hosted CI job `Linux reviewed hosted selected performance freshness`;
- canonical report metadata for runner context, build flags, CPU model,
  support tier, claim boundary, build mode, thread state, repeat semantics,
  baseline, threshold, warmup, variance, and methodology notes;
- Sprint 168 handoff items for warmup/variance policy, matrix-size
  interpretation, report-index integration, artifact readability, and claim
  boundary enforcement.

The sprint must:

- define the repeat-count, warmup, variance, and threshold policy for selected
  performance reports;
- normalize selected performance report fields so they remain stable,
  diff-friendly, and reviewable;
- add or tighten one bounded regression sentinel without making broad speed
  claims;
- index the selected performance evidence from the canonical report index and
  README evidence surfaces;
- document exact platform, runner, backend, fixture, and claim caveats;
- run report generation, freshness checks, focused script/workflow checks, and
  any required source quality gates if `.c` or `.h` files change.

**End State:** Sprint 169 leaves behind:

- a methodology-bound selected performance policy;
- stable selected performance report schema expectations;
- a bounded regression-sentinel decision or implementation;
- indexed documentation for selected performance evidence;
- platform and backend caveats for the selected lane;
- Sprint 169 working notes, daily artifacts, and validation records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 169 project-plan estimate.

---

## Day 1: Sprint Intake And Sprint 168 Handoff

**Title:** Methodology Intake
**Theme:** Establish Sprint 169 scope from Sprint 168 closeout and Epic 15
project-plan item 169.1 through 169.6
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 169 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 168 closeout, retrospective, hosted evidence expectations,
   and Sprint 169 handoff.
3. Create Sprint 169 working notes and artifact directory structure.
4. Record the prompt path/source-artifact mismatch.
5. Define methodology-hardening stop conditions and retained performance
   non-claims.
6. Write the Day 1 methodology-intake artifact.

### Deliverables

- Sprint 169 working-notes baseline
- artifact directory structure
- source artifact note
- Sprint 168 handoff summary
- methodology-hardening stop conditions
- Day 1 methodology-intake artifact

### Completion Criteria

- Sprint 169 scope is tied to the active Epic 15 project plan
- Sprint 168 selected lane is carried forward without reopening selection
- no new portable performance or broad benchmark claim is introduced

---

## Day 2: Current Report Methodology Audit

**Title:** Methodology Audit
**Theme:** Audit selected report fields, generated artifacts, freshness
checks, and hosted CI metadata
**Time estimate:** 12 hours

### Tasks

1. Review `scripts/bench_canonical_report.sh`.
2. Review `scripts/check_bench_canonical_freshness.py`.
3. Review canonical report output shape for CSV, `index.tsv`, and
   `manifest.txt`.
4. Review the hosted performance CI job and summary output.
5. Identify unstable, underdefined, redundant, or missing methodology fields.
6. Write the Day 2 methodology-audit artifact.

### Deliverables

- current report-schema inventory
- freshness-check inventory
- hosted metadata inventory
- missing or weak methodology field list
- Day 2 methodology-audit artifact

### Completion Criteria

- selected report fields are mapped to their owners
- missing methodology semantics are explicit
- unselected canonical rows remain local/advisory

---

## Day 3: Statistical Policy Design

**Title:** Statistical Policy
**Theme:** Define repeat-count, warmup, variance, sample, and threshold
semantics for selected performance evidence
**Time estimate:** 12 hours

### Tasks

1. Define repeat-count policy for the selected hosted lane.
2. Decide whether warmup remains `not_recorded` or becomes an explicit
   measured/pre-run policy.
3. Decide whether variance remains `not_recorded` or becomes a computed field.
4. Define whether the selected lane stays threshold-free or adds a bounded
   regression sentinel outside the publication row.
5. Define how local and hosted statistical policies differ, if at all.
6. Write the Day 3 statistical-policy artifact.

### Deliverables

- repeat-count policy
- warmup policy
- variance policy
- threshold or threshold-free policy
- local versus hosted methodology notes
- Day 3 statistical-policy artifact

### Completion Criteria

- every statistical field has a documented meaning
- threshold-free publication remains distinct from any regression sentinel
- policy avoids broad performance superiority claims

---

## Day 4: Schema Normalization Design

**Title:** Schema Design
**Theme:** Design stable, diff-friendly selected performance report fields
and row-level claim boundaries
**Time estimate:** 12 hours

### Tasks

1. Review selected-row fields for ordering, naming, and stable values.
2. Decide whether to derive `matrix_size` from the selected fixture.
3. Define stable formatting for warmup, variance, repeats, and sample counts.
4. Define manifest agreement requirements for new or normalized fields.
5. Define unselected-row invariants that must remain local-only.
6. Write the Day 4 schema-normalization artifact.

### Deliverables

- selected-row schema normalization design
- matrix-size derivation decision
- manifest agreement map
- unselected-row invariant list
- Day 4 schema-normalization artifact

### Completion Criteria

- schema changes are planned before implementation
- selected and unselected row behavior is explicitly separated
- generated fields remain machine-readable and diff-friendly

---

## Day 5: Statistical And Schema Implementation

**Title:** Policy Implementation
**Theme:** Implement selected report methodology policy and schema
normalization
**Time estimate:** 12 hours

### Tasks

1. Update the canonical report generator for approved statistical fields.
2. Add or preserve deterministic defaults for local runs.
3. Update selected-row metadata and manifest output.
4. Preserve existing command interfaces unless a documented reason requires a
   change.
5. Update freshness checker expectations for normalized fields.
6. Write the Day 5 implementation artifact.

### Deliverables

- generator methodology update
- selected-row schema update
- freshness-checker update
- manifest metadata update
- Day 5 implementation artifact

### Completion Criteria

- selected report output matches the Day 3 and Day 4 policy
- local freshness still has a conservative interpretation
- unselected rows are not promoted by the implementation

---

## Day 6: Report Schema Regression Tests

**Title:** Schema Tests
**Theme:** Add or tighten tests and negative checks for selected performance
schema behavior
**Time estimate:** 12 hours

### Tasks

1. Add focused checker coverage for new required fields.
2. Add negative checks for malformed or missing methodology metadata.
3. Add checks for selected versus unselected row claim boundaries.
4. Validate row width, header stability, and manifest agreement.
5. Run focused script and report-generation checks.
6. Write the Day 6 schema-tests artifact.

### Deliverables

- freshness-check coverage updates
- negative-check coverage notes
- selected/unselected row invariant proof
- focused validation log
- Day 6 schema-tests artifact

### Completion Criteria

- schema regressions fail with clear messages
- malformed selected performance reports are rejected
- focused validation passes locally

---

## Day 7: Regression Sentinel Design

**Title:** Sentinel Design
**Theme:** Design a bounded regression sentinel that detects large local
regressions without becoming a portable performance claim
**Time estimate:** 12 hours

### Tasks

1. Review existing wall-check and performance-sentinel lanes.
2. Decide whether to extend an existing sentinel or add a selected-lane
   sentinel.
3. Define baseline provenance, machine-class caveats, repeat settings, and
   failure output.
4. Define the line between sentinel pass/fail behavior and threshold-free
   publication rows.
5. Record deferral criteria if no safe sentinel can be added this sprint.
6. Write the Day 7 regression-sentinel design artifact.

### Deliverables

- sentinel design decision
- baseline provenance policy
- runtime budget and failure-output design
- publication-versus-sentinel boundary
- Day 7 sentinel-design artifact

### Completion Criteria

- one bounded sentinel path is selected or explicitly deferred
- sentinel wording cannot be read as universal speed evidence
- runtime budget is compatible with local and hosted checks

---

## Day 8: Regression Sentinel Implementation

**Title:** Sentinel Implementation
**Theme:** Implement or tighten the selected bounded regression sentinel
**Time estimate:** 12 hours

### Tasks

1. Implement the selected sentinel change or documented deferral guard.
2. Add Makefile target wiring if needed.
3. Add clear pass/fail output and remediation text.
4. Keep thresholded sentinel output separate from threshold-free publication
   rows.
5. Run focused sentinel validation.
6. Write the Day 8 sentinel-implementation artifact.

### Deliverables

- sentinel implementation or enforceable deferral
- target wiring
- pass/fail output examples
- focused validation log
- Day 8 sentinel-implementation artifact

### Completion Criteria

- sentinel behavior is bounded and reproducible enough for its stated scope
- selected report publication remains threshold-free
- focused sentinel validation passes or deferral is justified

---

## Day 9: Documentation Indexing Design

**Title:** Evidence Index Design
**Theme:** Design how selected performance evidence is linked from README,
benchmark docs, and report indexes
**Time estimate:** 12 hours

### Tasks

1. Review current README evidence surfaces.
2. Review `benchmarks/README.md` selected performance sections.
3. Review generated report-index and canonical report documentation.
4. Define the selected performance evidence link path and wording.
5. Define stale-report and generated-output handling.
6. Write the Day 9 documentation-indexing artifact.

### Deliverables

- README evidence-index design
- benchmark-doc link design
- report-index ownership decision
- stale-report handling notes
- Day 9 documentation-indexing artifact

### Completion Criteria

- selected performance evidence has a discoverable documentation path
- generated output is not accidentally treated as checked-in evidence
- documentation wording remains claim-safe

---

## Day 10: Documentation Indexing Implementation

**Title:** Evidence Indexing
**Theme:** Implement selected performance evidence indexing and README
evidence-table updates
**Time estimate:** 12 hours

### Tasks

1. Update README evidence surfaces for the selected performance lane.
2. Update `benchmarks/README.md` with methodology policy and report-index
   interpretation.
3. Update maintainer documentation for report generation and freshness
   ownership.
4. Add report-index link or explicit deferral if generated-output policy
   prevents checked-in indexing.
5. Run targeted documentation claim scans.
6. Write the Day 10 documentation-indexing artifact.

### Deliverables

- README evidence updates
- benchmark documentation updates
- maintainer-guide updates
- report-index link or deferral note
- Day 10 documentation-indexing artifact

### Completion Criteria

- selected performance evidence is easier to find
- docs distinguish hosted evidence from local/generated output
- claim scan finds no unsupported performance broadening

---

## Day 11: Platform And Backend Caveats

**Title:** Platform Caveats
**Theme:** Document exact platform, runner, backend, build, and fixture
constraints for the selected performance lane
**Time estimate:** 12 hours

### Tasks

1. Document Linux hosted runner scope and CPU variability.
2. Document compiler and build-flag interpretation.
3. Document serial/OpenMP and backend constraints.
4. Document selected fixture and matrix-size interpretation.
5. Preserve non-claims for Windows/macOS parity and portable performance.
6. Write the Day 11 platform-caveats artifact.

### Deliverables

- platform caveat notes
- backend/build caveat notes
- fixture and matrix-size caveats
- retained platform non-claims
- Day 11 platform-caveats artifact

### Completion Criteria

- selected lane constraints are explicit
- users cannot infer broad platform parity from the selected lane
- docs remain consistent with CI metadata

---

## Day 12: Integrated Local Validation

**Title:** Local Validation
**Theme:** Run selected report generation, freshness, sentinel, schema, and
documentation checks locally
**Time estimate:** 12 hours

### Tasks

1. Run syntax checks for changed shell and Python scripts.
2. Run selected local performance report freshness.
3. Run hosted-mode local metadata validation.
4. Run sentinel validation or deferral guard checks.
5. Run targeted documentation claim scans and `git diff --check`.
6. Write the Day 12 local-validation artifact.

### Deliverables

- local validation command log
- hosted-mode local validation log
- sentinel validation or deferral evidence
- claim-scan results
- Day 12 local-validation artifact

### Completion Criteria

- selected methodology policy passes local checks
- generated report output remains ignored unless intentionally published
- all required focused checks pass

---

## Day 13: Hosted Evidence Prep And PR Review Checklist

**Title:** Hosted Evidence Prep
**Theme:** Prepare reviewer-facing hosted CI expectations, artifact review
steps, and fallback wording
**Time estimate:** 11 hours

### Tasks

1. Reconcile local validation with hosted CI expectations.
2. Define expected hosted summary output.
3. Define hosted artifact review checklist for new methodology fields.
4. Document fallback handling for hosted infrastructure, runtime, schema, or
   sentinel failures.
5. Record what evidence becomes active only after PR CI passes.
6. Write the Day 13 hosted-evidence-prep artifact.

### Deliverables

- hosted evidence checklist
- expected summary output notes
- artifact review checklist
- fallback wording
- Day 13 hosted-evidence-prep artifact

### Completion Criteria

- reviewers know what hosted evidence to inspect
- failure handling does not broaden claims
- hosted proof remains conditional until PR CI passes

---

## Day 14: Sprint Validation And Closeout

**Title:** Sprint Closeout
**Theme:** Finalize Sprint 169 validation, project-plan reconciliation, and
handoff to Sprint 170
**Time estimate:** 11 hours

### Tasks

1. Run final selected local validation commands.
2. Confirm whether any `.c` or `.h` files changed and run the full C quality
   gate if required.
3. Confirm no generated build/report/cache artifacts are staged
   unintentionally.
4. Reconcile Sprint 169 deliverables against project-plan items 169.1 through
   169.6.
5. Prepare Sprint 170 handoff notes for shared-library ABI decision work.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 169 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 170 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- Sprint 169 methodology-hardening deliverables are reconciled
- all required quality checks pass or the sprint stops for user input
- Sprint 170 can begin from a clear performance-methodology baseline
