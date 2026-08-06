# Sprint 138 Plan: Maintained Numerical Corpus Architecture

**Sprint Duration:** 14 days
**Goal:** Build the maintained numerical corpus architecture and first durable
oracle/report lane before adding broad fixture volume. This sprint implements
the Sprint 138 section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 138 begins from:
- the completed Sprint 137 evidence contracts and Sprint 138 readiness
  handoff
- selected corpus/oracle scope from
  `docs/planning/EPIC_12/SPRINT_137/artifacts/day7-gap-selection-decision.md`
- corpus/oracle row templates from
  `docs/planning/EPIC_12/SPRINT_137/artifacts/day8-corpus-oracle-evidence-templates.md`
- report/freshness templates from
  `docs/planning/EPIC_12/SPRINT_137/artifacts/day9-report-index-freshness-templates.md`
- quality rules from
  `docs/planning/EPIC_12/SPRINT_137/artifacts/day11-quality-surface-map.md`
- public claim boundaries from
  `docs/planning/EPIC_12/SPRINT_137/artifacts/day12-public-claim-freeze.md`

The sprint must:
- define the maintained matrix taxonomy for corpus fixtures
- create durable manifest, generated-matrix, optional-data, skip/defer, and
  expected-result layout under maintained paths
- define and implement durable oracle row semantics
- add the first sustained deterministic corpus lane
- implement explicit optional-data handling without treating skips as pass
  evidence
- run focused validation and required quality checks for touched surfaces
- document corpus ownership, row interpretation, stale-report assumptions, and
  Sprint 139 QR handoff requirements

**End State:** Sprint 138 leaves behind:
- maintained corpus taxonomy
- corpus storage and manifest layout
- deterministic generated fixture lane
- first sustained oracle/report command
- skip/defer semantics for optional external data
- corpus maintainer documentation
- Sprint 139 QR fixture handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 138 project-plan estimate.

---

## Day 1: Sprint 138 Scope & Corpus Contract Setup

**Title:** Corpus Scope
**Theme:** Convert Sprint 137 handoff artifacts into a bounded Sprint 138
implementation package
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 138 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Re-read Sprint 137 Day 7, Day 8, Day 9, Day 11, Day 12, Day 13, and Day 14
   artifacts.
3. Create Sprint 138 working notes and artifact directory structure.
4. Map Sprint 138 Items 1-7 to day-level owners.
5. Record initial touched-surface validation expectations for docs, scripts,
   tests, build files, corpus data, generated reports, and `.c`/`.h` changes.
6. Define the sprint-level non-claims that must remain frozen while the corpus
   lane is implemented.

### Deliverables
- Sprint 138 working-notes baseline
- artifact directory structure
- Sprint 137 handoff inventory
- item-to-day owner map
- initial validation expectation register
- sprint-level non-claim register

### Completion Criteria
- every Sprint 138 project-plan item has a day-level owner
- Sprint 137 evidence contracts are visible before implementation begins
- validation and public-claim boundaries are documented before files change

---

## Day 2: Fixture Taxonomy Draft

**Title:** Taxonomy Draft
**Theme:** Define the maintained matrix classes and fixture metadata model for
the corpus
**Time estimate:** 12 hours

### Tasks
1. Draft matrix classes for symmetry, definiteness, rank, rectangularity,
   conditioning, scaling, sparsity pattern, graph shape, and expected failures.
2. Map classes to existing solver families and Sprint 139-140 QR/SVD needs.
3. Define required fixture metadata fields using the Sprint 137 Day 8
   template.
4. Identify which classes are in scope for the first durable fixture lane.
5. Identify classes that remain future residuals.
6. Write the taxonomy draft artifact.

### Deliverables
- maintained matrix-class taxonomy draft
- fixture metadata field map
- first-lane fixture class selection
- out-of-scope class list
- QR/SVD dependency notes

### Completion Criteria
- taxonomy covers the project-plan matrix classes
- the first fixture lane is narrow enough to close in Sprint 138
- out-of-scope fixture classes are explicit residuals

---

## Day 3: Taxonomy Review & Claim Boundaries

**Title:** Taxonomy Gate
**Theme:** Review the taxonomy against solver evidence, support tiers, and
claim boundaries before implementation
**Time estimate:** 12 hours

### Tasks
1. Compare the taxonomy draft against current tests, examples, and data
   fixtures.
2. Confirm the first durable lane supports Sprint 139 QR handoff without
   broadening to external corpus parity.
3. Confirm partial-SVD and report-index handoff fields remain available.
4. Record fixture-local claim scopes and non-claims for each selected class.
5. Finalize the taxonomy and promotion gates.
6. Write the taxonomy review artifact.

### Deliverables
- finalized maintained corpus taxonomy
- fixture-class promotion gates
- fixture-local claim boundary table
- QR/SVD/report handoff notes
- taxonomy residual list

### Completion Criteria
- selected taxonomy does not imply broad corpus completeness
- QR, partial-SVD, and report dependencies remain supported
- claim boundaries are written before storage layout begins

---

## Day 4: Corpus Storage Layout Design

**Title:** Storage Design
**Theme:** Design maintained paths for manifests, deterministic fixtures,
generated-matrix metadata, optional data, skips, expected results, and reports
**Time estimate:** 12 hours

### Tasks
1. Choose maintained repository paths for corpus manifests and deterministic
   fixture metadata.
2. Choose generated-matrix metadata and expected-result paths.
3. Choose optional external-data location, environment-variable, and skip/defer
   paths.
4. Choose oracle/report output paths that can feed Sprint 141 report
   normalization.
5. Define naming rules for fixture keys, generator keys, oracle row IDs, and
   report row IDs.
6. Write the storage layout design artifact.

### Deliverables
- corpus path layout
- fixture-key and generator-key naming rules
- optional-data location policy
- expected-result layout
- oracle/report output path design

### Completion Criteria
- all corpus row types have maintained paths
- optional-data paths cannot be mistaken for bundled fixtures
- report output paths can be normalized later without changing row meaning

---

## Day 5: Corpus Storage Layout Implementation

**Title:** Storage Layout
**Theme:** Create the maintained corpus directories, manifest skeletons, and
layout documentation
**Time estimate:** 12 hours

### Tasks
1. Add maintained corpus directory structure.
2. Add manifest skeletons for fixture rows, generated-matrix rows,
   optional-data rows, and expected-result rows.
3. Add README or maintainer notes explaining layout ownership.
4. Add validation-friendly placeholders for the first durable fixture lane.
5. Keep generated outputs out of source control unless the sprint explicitly
   chooses a maintained artifact.
6. Run docs/schema hygiene checks for the new layout.

### Deliverables
- corpus directory structure
- manifest skeleton files
- layout documentation
- first-lane placeholders
- initial layout validation notes

### Completion Criteria
- the maintained storage layout exists in the repository
- skeleton rows match the Day 8 templates
- no generated or optional external data is accidentally committed

---

## Day 6: Oracle Row Schema Design

**Title:** Oracle Schema
**Theme:** Define durable oracle row fields and comparison semantics before
writing report commands
**Time estimate:** 12 hours

### Tasks
1. Define oracle row schema fields for family, fixture, operation, command,
   expected result, observed result, tolerance, support tier, skip reason, and
   source commit.
2. Define supported comparison kinds for value, residual norm, rank, nullity,
   subspace distance, status, diagnostic, and local measurement rows.
3. Define tolerance kinds and required tolerance values.
4. Define comparison status values and failure classes.
5. Define row serialization and validation expectations.
6. Write the oracle schema design artifact.

### Deliverables
- oracle row schema
- comparison-kind table
- tolerance policy
- failure-class table
- row serialization notes

### Completion Criteria
- oracle rows can represent the first corpus lane without ambiguity
- skip/defer/unsupported statuses are distinct from pass/fail
- row fields preserve fixture-local claim boundaries

---

## Day 7: Oracle Schema Implementation

**Title:** Oracle Rows
**Theme:** Implement the initial oracle row schema, validation helper, and
expected-result skeleton
**Time estimate:** 12 hours

### Tasks
1. Add the oracle row schema to maintained corpus paths.
2. Add expected-result skeleton rows for the first durable fixture lane.
3. Add a lightweight schema validation helper or script if needed.
4. Add focused validation for missing required fields and invalid status
   values.
5. Document how oracle rows are generated, reviewed, and updated.
6. Write the oracle implementation artifact.

### Deliverables
- initial oracle schema implementation
- expected-result skeleton rows
- schema validation helper or documented validation path
- oracle ownership notes
- implementation validation notes

### Completion Criteria
- initial oracle rows can be validated mechanically
- expected-result rows are present for the first durable lane
- schema validation does not treat skipped rows as passes

---

## Day 8: First Deterministic Fixture Lane Design

**Title:** Fixture Lane
**Theme:** Design the first sustained deterministic fixture lane and generator
metadata
**Time estimate:** 12 hours

### Tasks
1. Select the exact first deterministic fixture family from the finalized
   taxonomy.
2. Define generator algorithm, parameters, seed policy, canonical format, hash
   policy, and change policy.
3. Define expected structure, value, rank, nullity, conditioning, and behavior
   metadata.
4. Define the validation command and oracle rows that will exercise the lane.
5. Define QR handoff fields needed by Sprint 139.
6. Write the deterministic fixture lane design artifact.

### Deliverables
- selected first fixture family
- generator metadata design
- expected fixture metadata
- validation command design
- Sprint 139 QR handoff fields

### Completion Criteria
- the first fixture lane is deterministic and reproducible by design
- fixture metadata can be checked against the manifest schema
- the lane supports Sprint 139 without claiming broad corpus coverage

---

## Day 9: First Corpus Lane Implementation

**Title:** Lane Landing
**Theme:** Add the first deterministic corpus fixture lane, manifest rows, and
expected-result data
**Time estimate:** 12 hours

### Tasks
1. Implement the deterministic fixture generator or generated fixture metadata.
2. Add fixture manifest rows for the selected lane.
3. Add expected-result rows for the selected lane.
4. Add oracle rows or row-generation scaffolding for the selected lane.
5. Add focused tests or script checks for fixture determinism and manifest
   completeness.
6. Write the implementation artifact.

### Deliverables
- first deterministic fixture lane
- manifest rows
- expected-result rows
- oracle row scaffolding
- focused validation notes

### Completion Criteria
- the first fixture lane exists under maintained paths
- deterministic metadata can be validated
- the lane has expected-result and oracle-row ownership

---

## Day 10: Maintained Oracle/Report Command

**Title:** Oracle Command
**Theme:** Add the first maintained command that validates the corpus lane and
emits durable oracle/report evidence
**Time estimate:** 12 hours

### Tasks
1. Design the first maintained corpus/oracle command interface.
2. Implement the command as a script, Make target, test helper, or focused test
   path consistent with repository patterns.
3. Emit oracle/report rows with command, commit, fixture key, expected result,
   observed result, tolerance, support tier, and status metadata.
4. Ensure output can feed Sprint 141 report normalization.
5. Add docs or maintainer notes for command usage and row interpretation.
6. Write the command implementation artifact.

### Deliverables
- maintained corpus/oracle command
- oracle/report output path
- row metadata output
- command usage notes
- report-normalization handoff notes

### Completion Criteria
- the command validates the first corpus lane
- emitted rows include required provenance and interpretation fields
- report rows do not imply release, performance, or broad correctness proof

---

## Day 11: Optional Data & Skip Semantics

**Title:** Skip Semantics
**Theme:** Implement explicit optional-data skip/defer semantics without false
pass evidence
**Time estimate:** 12 hours

### Tasks
1. Define optional-data availability states for available, unavailable,
   disabled, unsupported platform, missing license, and network unavailable.
2. Implement optional-data skip/defer rows or validation behavior.
3. Ensure optional external data paths are configurable but not required for
   default validation.
4. Add checks that skipped optional data is not counted as numerical pass
   evidence.
5. Document pass, skip, defer, unsupported, and xfail interpretation.
6. Write the skip/defer semantics artifact.

### Deliverables
- optional-data state model
- skip/defer row implementation
- false-pass guard
- optional-data documentation
- validation notes

### Completion Criteria
- unavailable optional data produces skip/defer evidence only
- default validation passes without optional external data
- skip/defer wording preserves corpus and external-parity non-claims

---

## Day 12: Focused Validation & Quality Gates

**Title:** Validation
**Theme:** Run focused corpus/oracle validation and the required quality checks
for touched surfaces
**Time estimate:** 12 hours

### Tasks
1. Run the new corpus/oracle command and capture results.
2. Run schema, manifest, generator, optional-data, and oracle row validation.
3. Run script syntax checks for touched scripts.
4. Run docs link/path and whitespace checks for touched docs.
5. If any `.c` or `.h` files changed, run
   `make format && make lint && make test`.
6. Write the focused validation artifact.

### Deliverables
- validation command log summary
- corpus/oracle validation results
- docs/script quality results
- C quality result if required
- validation residuals

### Completion Criteria
- all required checks for touched surfaces pass
- skipped optional data is reported separately from pass evidence
- validation results are sufficient for Sprint 138 closeout

---

## Day 13: Documentation & Sprint 139 Handoff

**Title:** QR Handoff
**Theme:** Document corpus ownership, row interpretation, stale-report
assumptions, and Sprint 139 QR fixture requirements
**Time estimate:** 12 hours

### Tasks
1. Update maintainer-facing corpus documentation.
2. Document fixture manifest, generated-matrix, optional-data, expected-result,
   oracle row, and report output ownership.
3. Document stale-report assumptions and Sprint 141 report-index dependencies.
4. Define Sprint 139 QR fixture handoff requirements.
5. Record remaining corpus/oracle residuals and non-claims.
6. Write the documentation and handoff artifact.

### Deliverables
- corpus maintainer documentation
- row interpretation guidance
- stale-report assumptions
- Sprint 139 QR handoff requirements
- corpus residual register

### Completion Criteria
- maintainers can update the corpus lane without redefining row semantics
- Sprint 139 QR work has clear fixture and oracle prerequisites
- docs preserve fixture-local claim boundaries

---

## Day 14: Sprint 138 Closeout

**Title:** Closeout
**Theme:** Verify deliverables, publish final validation evidence, and close
Sprint 138 with implementation residuals
**Time estimate:** 12 hours

### Tasks
1. Verify all Sprint 138 deliverables exist or are explicitly deferred.
2. Re-run required docs, script, corpus/oracle, and C quality checks as
   selected by touched surfaces.
3. Confirm no optional-data skip is counted as pass evidence.
4. Publish final Sprint 138 validation summary and residual register.
5. Publish final Sprint 139 QR readiness criteria.
6. Update Sprint 138 working notes and write closeout notes.

### Deliverables
- final Sprint 138 deliverable checklist
- validation result summary
- residual register
- Sprint 139 QR readiness criteria
- completed working notes

### Completion Criteria
- maintained corpus taxonomy, layout, first lane, oracle/report command, and
  skip/defer semantics are present or explicitly deferred
- validation matches touched surfaces and passes
- Sprint 139 has clear QR fixture, oracle, tolerance, and claim-boundary inputs
