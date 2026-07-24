# Sprint 131 Plan: Numerical Corpus, Coverage Architecture & Report Indexes

**Sprint Duration:** 14 days
**Goal:** Turn scattered numerical fixtures, coverage, benchmark, dead-code,
and guardrail outputs into a clearer recurring assurance architecture after
the Sprint 124-130 residual QR, partial-SVD, and helper claim gates are
resolved.

**Starting Point:** Sprint 131 begins from:
- Sprint 118 planning templates and recurring artifact conventions
- Sprint 120-130 oracle taxonomy, external-reference decisions, residual
  evidence gates, and no-claim registers
- existing Matrix Market fixtures, generated families, known matrices,
  external-reference scripts, benchmark outputs, coverage outputs, dead-code
  outputs, large-matrix reports, and guardrail diagnostics
- Sprint 125-130 optional-corpus decisions and support-tier boundaries
- current maintainer-guide evidence tables and public solver-selection
  non-claim posture

The sprint must:
- inventory numerical inputs without silently promoting smoke fixtures into
  reviewed oracle evidence
- define corpus tags for matrix structure, numerical properties, solver
  ownership, optional availability, and support tier
- design report indexes for benchmark, coverage, dead-code, large-matrix, and
  oracle outputs without changing benchmark semantics
- re-rank coverage gaps by risk and decide which remain supplemental versus
  reviewed
- implement or explicitly defer the first generated report/index artifact
  with validation and owner notes
- publish residual assurance gaps and future-owner promotion criteria

**End State:** Sprint 131 leaves behind:
- numerical corpus inventory and taxonomy
- report index design and first generated report/index artifact or explicit
  implementation deferral
- coverage architecture decision package
- validation logs for affected report-generation, docs, scripts, or tests
- corpus/report ownership map
- residual assurance queue and Sprint 132 handoff notes

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 131 project-plan estimate.

---

## Day 1: Sprint Intake and Assurance Baseline

**Title:** Assurance Intake
**Theme:** Establish Sprint 131 scope, owner map, artifact structure, and
duplicate fences around existing evidence outputs
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 131 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 118 templates and Sprint 120-130 oracle, residual, and
   report-related artifacts.
3. Create the Sprint 131 working-notes baseline and artifact directory.
4. Inventory candidate source areas for corpus, coverage, benchmark,
   dead-code, large-matrix, oracle, and guardrail outputs.
5. Map Sprint 131 project-plan Items 1-7 to day-level owners.
6. Record duplicate fences so existing smoke tests, timing outputs, and
   optional corpus paths are not reclassified silently.

### Deliverables
- Sprint 131 working-notes baseline
- artifact directory structure
- source-area intake list
- item-to-day owner map
- duplicate and non-claim boundary notes

### Completion Criteria
- every Sprint 131 project-plan item has a day-level owner
- existing Sprint 120-130 evidence boundaries are preserved
- corpus, report, coverage, and validation surfaces are visible before design
  or implementation begins

---

## Day 2: Numerical Fixture Inventory

**Title:** Fixture Inventory
**Theme:** Inventory checked-in numerical fixtures and generated matrix
families without assigning new support claims
**Time estimate:** 12 hours

### Tasks
1. Inventory Matrix Market fixtures under checked-in test data directories.
2. Inventory generated matrix families used by solver, graph, SVD, QR,
   eigenvalue, and integration tests.
3. Record dimensions, symmetry, definiteness, rank hints, conditioning hints,
   sparsity pattern, known matrix family, and current test owner where
   available.
4. Separate local analytic fixtures, checked-in corpus fixtures, optional
   external data, and generated stress cases.
5. Identify missing metadata required before any fixture can become reviewed
   corpus evidence.
6. Write the numerical fixture inventory artifact.

### Deliverables
- Matrix Market fixture inventory
- generated-family inventory
- metadata completeness notes
- support-tier boundary notes
- missing-metadata queue

### Completion Criteria
- every checked-in numerical corpus source has an owner or explicit unknown
  owner
- generated fixtures are not confused with independent external corpus
  evidence
- missing metadata is recorded as a blocker, not left implicit

---

## Day 3: External-Reference and Expected-Failure Inventory

**Title:** Oracle Inventory
**Theme:** Inventory external-reference scripts, oracle fixtures, expected
failures, skips, and optional-corpus decisions
**Time estimate:** 12 hours

### Tasks
1. Inventory external-reference helper scripts and their fixture keys.
2. Map each helper fixture to its solver family, output type, oracle source,
   tolerance policy, and current test owner.
3. Inventory expected-failure tests, skip paths, optional-data gates, and
   platform-specific exclusions.
4. Review Sprint 125-130 optional-corpus and claim-gate decisions for
   reusable support-tier language.
5. Identify fixtures whose names imply broader claims than their oracle
   supports.
6. Write the external-reference and expected-failure inventory artifact.

### Deliverables
- external-reference helper inventory
- expected-failure and skip-path inventory
- optional-corpus decision map
- fixture-name versus claim-boundary notes
- support-tier gaps

### Completion Criteria
- every external-reference helper fixture has a declared output class
- expected failures and skips have failure interpretation notes
- optional corpus decisions are traceable to Sprint 125-130 evidence gates

---

## Day 4: Corpus Taxonomy Policy

**Title:** Taxonomy Policy
**Theme:** Define corpus tags for structure, numerical properties, solver
ownership, optional availability, and support tier
**Time estimate:** 12 hours

### Tasks
1. Define structural tags for shape, symmetry, definiteness, rank,
   singularity, graph pattern, ordering, and storage format.
2. Define numerical tags for scale, conditioning, clustering, repeated
   spectra, nullity, sparsity density, and known exact solutions.
3. Define ownership tags for solver family, report owner, oracle owner,
   validation owner, and documentation owner.
4. Define availability tags for local analytic, checked-in smoke, checked-in
   reviewed, checked-in expensive, optional local, and optional external data.
5. Define support-tier tags that separate smoke, reviewed, supplemental,
   experimental, deferred, and unsupported evidence.
6. Write the corpus taxonomy policy artifact.

### Deliverables
- corpus tag dictionary
- ownership taxonomy
- availability taxonomy
- support-tier taxonomy
- tag promotion and demotion rules

### Completion Criteria
- tags are specific enough to drive report indexes
- support tiers do not imply unsupported public claims
- every tag family has promotion and deferral semantics

---

## Day 5: Corpus Tagging Dry Run

**Title:** Tagging Dry Run
**Theme:** Apply the taxonomy to representative fixtures and refine gaps
before any index is generated
**Time estimate:** 12 hours

### Tasks
1. Select representative fixtures across direct solvers, iterative solvers,
   QR, SVD, partial SVD, eigensolvers, graph partitioning, and integration
   tests.
2. Apply Day 4 taxonomy tags manually to the selected fixtures.
3. Identify ambiguous tags, missing metadata, unsupported support-tier
   transitions, and naming conflicts.
4. Refine taxonomy language where the dry run exposes ambiguity.
5. Define the minimum metadata required for a generated index row.
6. Write the corpus tagging dry-run artifact.

### Deliverables
- representative tagged-fixture table
- taxonomy refinements
- ambiguity and missing-metadata register
- minimum generated-index row schema
- promotion checklist for reviewed corpus rows

### Completion Criteria
- representative fixtures can be tagged without changing test semantics
- ambiguous tags have blockers and future owners
- generated-index schema has required and optional fields

---

## Day 6: Report Index Requirements

**Title:** Index Requirements
**Theme:** Design requirements for benchmark, coverage, dead-code,
large-matrix, oracle, and guardrail report indexes
**Time estimate:** 12 hours

### Tasks
1. Inventory existing benchmark outputs, coverage reports, dead-code reports,
   large-matrix logs, oracle artifacts, and guardrail diagnostics.
2. Define report-index audiences: maintainer triage, release readiness,
   coverage review, corpus ownership, and claim-gate review.
3. Define required fields for report path, generation command, freshness,
   owner, support tier, input corpus, output class, and failure meaning.
4. Decide whether each report family should be generated, curated, or
   documented as deferred.
5. Define non-goals for benchmark semantics, public performance claims, and
   CI policy changes.
6. Write the report index requirements artifact.

### Deliverables
- report-family requirements matrix
- generated versus curated decision table
- index field schema
- owner and freshness policy
- non-goal and no-claim notes

### Completion Criteria
- every report family has an index strategy or explicit deferral
- index requirements do not change benchmark or CI semantics
- freshness and owner fields have clear interpretation

---

## Day 7: Report Index Design

**Title:** Index Design
**Theme:** Design the first report/index artifact and its generation or
curation workflow
**Time estimate:** 12 hours

### Tasks
1. Choose the first report/index artifact candidate using Day 6 requirements.
2. Define source inputs, output location, schema, sorting, stable row
   identity, and regeneration command.
3. Define how generated rows reference corpus tags, support tiers, validation
   commands, and residual gaps.
4. Define stale-output, missing-input, optional-data, and unsupported-report
   behavior.
5. Identify files that would be touched by an implementation versus a
   documentation-only deferral.
6. Write the report index design artifact.

### Deliverables
- selected first index candidate
- source-to-output design
- schema and sorting rules
- regeneration and stale-output policy
- implementation or deferral checklist

### Completion Criteria
- first index candidate has a clear implementation path or blocker
- source inputs and output paths are deterministic
- missing or optional inputs have explicit behavior

---

## Day 8: Coverage Gap Architecture

**Title:** Coverage Architecture
**Theme:** Re-rank coverage gaps by risk and decide reviewed versus
supplemental coverage boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory current coverage-related outputs, report-generation scripts, and
   documented coverage gaps.
2. Classify coverage gaps by solver family, user-facing workflow, numerical
   risk, platform risk, corpus availability, and claim impact.
3. Separate reviewed coverage from supplemental, smoke, optional, expensive,
   and experimental coverage.
4. Define risk ranking criteria and owner labels for future coverage work.
5. Identify coverage gaps that should block report-index claims versus those
   that should remain residual debt.
6. Write the coverage architecture artifact.

### Deliverables
- coverage gap inventory
- risk ranking rubric
- reviewed versus supplemental coverage decision table
- coverage owner map
- residual coverage queue

### Completion Criteria
- coverage gaps are ranked by risk, not by convenience
- reviewed coverage does not absorb optional or smoke-only paths silently
- every residual coverage gap has blocker and future-owner notes

---

## Day 9: Dead-Code and Guardrail Report Architecture

**Title:** Guardrail Reports
**Theme:** Define how dead-code, unused surface, stale artifact, and guardrail
outputs fit into recurring assurance
**Time estimate:** 12 hours

### Tasks
1. Inventory existing dead-code, unused-code, stale-artifact, and guardrail
   checks.
2. Classify each output by tool source, stability, false-positive risk,
   ownership, and actionability.
3. Decide which outputs can enter a generated index and which need curated
   notes or explicit deferral.
4. Define suppression, waiver, known-false-positive, and stale-report
   policies.
5. Link guardrail outputs back to coverage and corpus taxonomy where
   applicable.
6. Write the dead-code and guardrail architecture artifact.

### Deliverables
- dead-code and guardrail output inventory
- actionability and false-positive policy
- suppression and waiver policy
- index eligibility decision table
- residual guardrail queue

### Completion Criteria
- guardrail outputs have clear owner and actionability semantics
- false positives are not treated as reviewed failures
- index eligibility is explicit for each output family

---

## Day 10: First Index Implementation or Deferral

**Title:** First Index
**Theme:** Implement or explicitly defer the first generated report/index
artifact without changing report semantics
**Time estimate:** 12 hours

### Tasks
1. Re-check the Day 7 first-index candidate against Day 8 and Day 9
   architecture decisions.
2. Implement the selected generated index if the source inputs, schema,
   output path, and validation behavior are ready.
3. Otherwise publish a concrete implementation deferral with blockers and
   future owner.
4. Keep benchmark, coverage, dead-code, and test semantics unchanged.
5. Record touched files, regeneration command, and expected diagnostics.
6. Write the Day 10 implementation or deferral artifact.

### Deliverables
- first generated report/index artifact or explicit implementation deferral
- regeneration command or blocker list
- touched-file and validation notes
- unchanged-semantics statement
- residual implementation queue

### Completion Criteria
- generated index is deterministic if implemented
- no benchmark, coverage, dead-code, or test semantics change silently
- every deferral has blocker, dependency, and future owner

---

## Day 11: Index Validation and Freshness Policy

**Title:** Validation Policy
**Theme:** Validate the first index path and define recurring freshness,
drift, and failure behavior
**Time estimate:** 12 hours

### Tasks
1. Run the first index generation or dry-run validation path where available.
2. Validate missing-input, stale-output, optional-data, and unsupported-report
   behavior.
3. Define how freshness is represented without requiring CI changes.
4. Define drift detection responsibilities for generated and curated rows.
5. Record docs hygiene, script checks, or focused tests required by touched
   files.
6. Write the index validation and freshness policy artifact.

### Deliverables
- index validation results or dry-run notes
- freshness and drift policy
- missing-input and optional-data behavior
- validation command log
- residual validation queue

### Completion Criteria
- implemented or deferred index path has validated behavior
- freshness does not imply stronger CI or release guarantees than supported
- validation commands and gaps are reproducible

---

## Day 12: Coverage and Report Ownership Map

**Title:** Ownership Map
**Theme:** Publish recurring owners for corpus, coverage, reports, indexes,
guardrails, and claim gates
**Time estimate:** 12 hours

### Tasks
1. Consolidate owner labels from corpus taxonomy, report index requirements,
   coverage architecture, guardrail architecture, and validation policy.
2. Map owners to files, scripts, report outputs, artifacts, and future sprint
   queues.
3. Identify orphaned reports, orphaned corpus rows, and orphaned validation
   gaps.
4. Define promotion criteria for moving supplemental or smoke outputs into
   reviewed recurring assurance.
5. Update maintainer-facing planning artifacts if evidence supports an
   ownership wording refresh.
6. Write the coverage and report ownership artifact.

### Deliverables
- corpus/report/coverage owner map
- orphaned-output register
- supplemental-to-reviewed promotion criteria
- maintainer wording update or no-update rationale
- future-owner queue

### Completion Criteria
- every recurring assurance area has an owner or explicit orphan status
- no orphaned output lacks blocker and future owner
- maintainer wording changes, if any, trace directly to accepted decisions

---

## Day 13: Validation Batch and Assurance Queue

**Title:** Validation Batch
**Theme:** Run final affected checks and publish the residual assurance queue
for corpus and report architecture
**Time estimate:** 12 hours

### Tasks
1. Run all affected docs, script, report-generation, focused test, and
   hygiene checks required by Sprint 131 changes.
2. If code, script, or generated-index files changed, run the appropriate
   focused checks and record outputs.
3. Reconcile all residual corpus, coverage, report, dead-code, large-matrix,
   oracle, and guardrail gaps.
4. Classify each residual gap by blocker, dependency, support tier, claim
   impact, and future owner.
5. Prepare Day 14 closeout inputs.
6. Write the validation batch and residual assurance queue artifact.

### Deliverables
- final validation command log
- affected-check results
- residual assurance queue
- support-tier and claim-impact classification
- Day 14 closeout inputs

### Completion Criteria
- required checks have passed or the sprint stops with a blocker
- every residual gap has blocker, dependency, and future owner
- validation evidence is sufficient for closeout

---

## Day 14: Sprint Closeout and Report Index Handoff

**Title:** Closeout Index
**Theme:** Publish final corpus/report ownership, no-claim boundaries,
validation package, and Sprint 132 handoff
**Time estimate:** 12 hours

### Tasks
1. Reconcile every Sprint 131 item against the project-plan checklist.
2. Review all accepted taxonomy, architecture, index, ownership, and
   validation outcomes for public or maintainer claim impact.
3. Publish final corpus/report ownership and residual assurance gaps.
4. Update maintainer-facing wording only if accepted evidence supports a
   bounded ownership claim beyond current guidance.
5. Otherwise publish an explicit no-update rationale.
6. Write Sprint 131 closeout, retrospective inputs, and Sprint 132 handoff
   notes.

### Deliverables
- final Sprint 131 closeout artifact
- corpus/report ownership index
- residual assurance queue
- validation package
- maintainer wording update or no-update rationale
- Sprint 132 handoff notes

### Completion Criteria
- all Sprint 131 deliverables are present or explicitly deferred
- public and maintainer wording matches only earned evidence
- no unresolved corpus, coverage, report, or guardrail item lacks blocker,
  dependency, and future-owner notes
