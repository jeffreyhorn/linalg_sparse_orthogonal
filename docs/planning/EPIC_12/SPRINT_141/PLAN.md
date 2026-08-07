# Sprint 141 Plan: Report Index Normalization & Freshness Gates

**Sprint Duration:** 14 days
**Goal:** Normalize maintained report metadata across corpus, benchmark,
sentinel, guardrail, coverage, dead-code, package, and oracle lanes where row
meaning can be preserved honestly. This sprint implements the Sprint 141
section of `docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 141 begins from:
- the Sprint 138 maintained corpus schema, report command, generator metadata,
  and skip/defer conventions
- the Sprint 139 QR report rows, stale-report guidance, and fixture-local
  proof-owner pattern
- the Sprint 140 partial-SVD report rows, edge-case corpus fixture, oracle
  comparison semantics, and report-normalization handoff
- existing benchmark, sentinel, guardrail, coverage, dead-code, package,
  install, and corpus artifacts
- existing maintainer, benchmark, corpus, package, and solver documentation

The sprint must:
- inventory report families before designing shared metadata
- define a common metadata contract without flattening incompatible row
  meanings
- implement a maintained normalized index generator for report families that
  can be normalized honestly
- add a stale-report gate that identifies old generated reports without
  claiming local measurements as release proof
- update docs so users and maintainers understand report semantics, freshness,
  support tier, skips, and non-claims
- run report generators/checks and affected quality gates
- hand off runtime/backend rows that need deeper governance to Sprint 142

**End State:** Sprint 141 leaves behind:
- report family inventory
- shared report metadata contract
- normalized report index generator
- stale-report/freshness gate
- updated report interpretation docs
- validation evidence for touched scripts, generated indexes, and docs
- Sprint 142 runtime/backend governance handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 141 project-plan estimate.

---

## Day 1: Report Family Intake

**Title:** Report Intake
**Theme:** Establish Sprint 141 scope, inherited report surfaces, and evidence
boundaries before normalizing metadata
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 141 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 138, Sprint 139, and Sprint 140 artifacts for report rows,
   generated outputs, skip/defer semantics, and stale-report guidance.
3. Create Sprint 141 working notes and artifact directory structure.
4. Inventory candidate report families: corpus, oracle, benchmark,
   performance sentinel, large-matrix guardrail, coverage, dead-code, package,
   install, and CI summary lanes.
5. Map Sprint 141 Items 1-7 to day-level owners.
6. Record initial report-normalization boundaries, non-claims, and stop
   conditions.

### Deliverables
- Sprint 141 working-notes baseline
- artifact directory structure
- initial report family inventory
- item-to-day owner map
- normalization boundaries and non-claim register

### Completion Criteria
- every Sprint 141 project-plan item has a day-level owner
- candidate report families are visible before schema design begins
- report families that may not preserve row meaning are explicitly flagged

---

## Day 2: Report Family Inventory

**Title:** Family Inventory
**Theme:** Build the canonical inventory of maintained and generated report
families, commands, owners, and row meanings
**Time estimate:** 12 hours

### Tasks
1. Inspect scripts, tests, docs, workflows, ignored generated outputs, and
   committed fixture/manifest files for report-producing commands.
2. Record each report family's producer command, input artifacts, output path,
   row identity, row meaning, owner, support tier, and regeneration policy.
3. Separate source-controlled evidence rows from local generated measurement
   outputs.
4. Identify reports with non-normalizable semantics, optional-data dependency,
   platform specificity, or runtime/backend specificity.
5. Draft the report family inventory artifact.
6. Record unknowns and candidate questions for the metadata contract.

### Deliverables
- report family inventory artifact
- command-to-output map
- generated-vs-source-controlled report distinction
- report family risk and unknowns list

### Completion Criteria
- benchmark, sentinel, guardrail, coverage, dead-code, package, corpus, and
  oracle outputs are accounted for
- each family has a documented row meaning or a recorded reason it cannot be
  normalized yet
- inventory is concrete enough to drive schema design

---

## Day 3: Metadata Contract Design

**Title:** Contract Design
**Theme:** Define common report metadata fields and preserve family-specific
semantics without overclaiming
**Time estimate:** 12 hours

### Tasks
1. Define common metadata fields for report family, row ID, generator, command,
   commit, platform, compiler, configuration, support tier, freshness, and
   skip/defer reason.
2. Define row-meaning fields that distinguish fixture result, benchmark
   measurement, sentinel threshold, package proof, coverage summary, and
   advisory documentation rows.
3. Specify required, optional, generated-only, source-controlled, and
   family-specific fields.
4. Define freshness semantics for committed manifests versus generated local
   reports.
5. Define validation errors, warnings, and defer states.
6. Write the shared metadata contract artifact.

### Deliverables
- shared report metadata contract
- required/optional field matrix
- row-meaning taxonomy
- freshness and skip/defer semantics
- validation severity model

### Completion Criteria
- the contract can represent Sprint 138-140 corpus/oracle rows without losing
  meaning
- benchmark and sentinel measurements are not framed as release proof
- non-normalizable families have an explicit defer path

---

## Day 4: Index Generator Design

**Title:** Generator Design
**Theme:** Design the normalized report index generator, input discovery rules,
and output format
**Time estimate:** 12 hours

### Tasks
1. Choose generator ownership, CLI shape, output path, and integration with
   existing scripts.
2. Define input discovery rules for source-controlled manifests and generated
   report files.
3. Define normalized index output columns and deterministic ordering.
4. Define how missing optional reports, local-only reports, and stale reports
   appear in the index.
5. Design unit, smoke, and golden-output tests for the generator.
6. Write the generator design artifact.

### Deliverables
- normalized index generator design
- input discovery and deterministic ordering rules
- output schema draft
- test strategy and golden-output plan
- implementation checklist

### Completion Criteria
- the generator design maps directly to the metadata contract
- optional and generated-only report handling is deterministic
- tests can verify behavior without requiring platform-specific measurements

---

## Day 5: Metadata Contract Implementation

**Title:** Contract Implementation
**Theme:** Add the schema/config surface needed by the normalized report index
without broadening claims
**Time estimate:** 12 hours

### Tasks
1. Add report-family metadata definitions, schema files, or configuration rows
   according to the Day 3 contract.
2. Encode support tier, row meaning, generator, command, freshness, and
   skip/defer vocabulary.
3. Add validation helpers for required fields and deterministic row identity.
4. Preserve existing corpus/oracle schema compatibility unless a narrow
   extension is required.
5. Add focused tests or schema checks for the contract surface.
6. Write the implementation artifact.

### Deliverables
- implemented metadata contract surface
- report-family definitions/configuration
- validation helper or schema updates
- focused validation checks
- contract implementation artifact

### Completion Criteria
- the metadata contract can be validated mechanically
- existing corpus/oracle checks continue to pass
- no field implies unsupported performance, package, or platform claims

---

## Day 6: Index Generator Implementation

**Title:** Generator Implementation
**Theme:** Implement the normalized report index generator for honestly
normalizable report families
**Time estimate:** 12 hours

### Tasks
1. Implement the generator CLI and deterministic output writer.
2. Parse configured report-family metadata and discover source-controlled
   report inputs.
3. Emit normalized rows for corpus, oracle, package, benchmark, sentinel,
   guardrail, coverage, and dead-code families where row meaning is preserved.
4. Emit defer or skip rows for non-normalizable and unavailable report
   families.
5. Add focused tests for deterministic ordering, missing files, family
   filtering, and generated-only inputs.
6. Write the generator implementation artifact.

### Deliverables
- normalized report index generator
- deterministic normalized index output
- focused generator tests
- defer/skip handling
- generator implementation artifact

### Completion Criteria
- the generator produces stable output from the current repository state
- unsupported report families are represented as defer/skip rows, not
  fabricated proof
- tests cover normal, missing, and deferred report-family paths

---

## Day 7: Corpus And Oracle Index Integration

**Title:** Corpus Integration
**Theme:** Integrate Sprint 138-140 corpus and oracle rows into the normalized
report index
**Time estimate:** 12 hours

### Tasks
1. Connect corpus fixture, manifest, expected-result, and oracle report rows to
   the normalized index.
2. Verify QR and partial-SVD report rows retain fixture-local row meaning.
3. Preserve corpus skip/defer semantics for optional data and generated
   reports.
4. Add checks that row IDs, fixture keys, generator keys, support tiers, and
   claim scopes are stable.
5. Update corpus-specific tests or docs if integration reveals ambiguous
   terminology.
6. Write the corpus/oracle integration artifact.

### Deliverables
- corpus/oracle normalized index rows
- QR and partial-SVD row-preservation evidence
- corpus skip/defer validation
- updated corpus checks or wording as needed
- integration artifact

### Completion Criteria
- Sprint 138-140 rows are represented without changing their claim scope
- generated oracle outputs are not committed as release proof
- corpus validation and normalized index checks agree on row identity

---

## Day 8: Benchmark And Sentinel Index Integration

**Title:** Runtime Indexing
**Theme:** Normalize benchmark, performance sentinel, and large-matrix
guardrail metadata while preserving measurement boundaries
**Time estimate:** 12 hours

### Tasks
1. Inventory benchmark and sentinel command outputs against the Day 3 metadata
   contract.
2. Add normalized index rows for benchmark and sentinel artifacts whose row
   meanings are stable.
3. Mark local measurement, optional matrix, platform-specific, and runtime
   backend rows with explicit support tier and freshness semantics.
4. Add defer rows for runtime/backend semantics that belong in Sprint 142.
5. Add tests or sample fixtures that verify measurement rows are not promoted
   into release claims.
6. Write the benchmark/sentinel integration artifact.

### Deliverables
- benchmark and sentinel normalized index rows
- large-matrix guardrail row treatment
- local-measurement non-claim evidence
- Sprint 142 runtime/backend defer list
- integration artifact

### Completion Criteria
- benchmark and sentinel rows are discoverable through the normalized index
- platform/runtime-specific rows have explicit support/freshness semantics
- Sprint 142 handoff captures rows that need deeper runtime governance

---

## Day 9: Coverage, Dead-Code, And Package Index Integration

**Title:** Quality Indexing
**Theme:** Normalize coverage, dead-code, package, install, and pkg-config
metadata where row meaning is stable
**Time estimate:** 12 hours

### Tasks
1. Inventory coverage, dead-code, package, install, CMake package, and
   pkg-config proof outputs.
2. Add normalized index rows for stable quality and package report families.
3. Define row meanings for install proof, package metadata proof, stale
   generated report, and advisory-only quality report.
4. Preserve platform-specific package semantics and static-first non-claims.
5. Add tests or validation fixtures for missing or stale quality/package
   reports.
6. Write the quality/package integration artifact.

### Deliverables
- coverage/dead-code/package normalized rows
- install and package row-meaning definitions
- platform/static-first non-claim treatment
- stale/missing validation coverage
- integration artifact

### Completion Criteria
- quality and package report families are represented without overstating
  freshness or platform coverage
- install/package proof rows distinguish source-controlled checks from local
  generated output
- stale or missing rows produce deterministic validation messages

---

## Day 10: Freshness Gate Design

**Title:** Gate Design
**Theme:** Define a stale-report gate that detects old generated reports
without converting local measurements into release proof
**Time estimate:** 12 hours

### Tasks
1. Define freshness inputs: generator version, command, source file hashes,
   commit, timestamp policy, platform, compiler, and configuration.
2. Define stale, current, missing, skipped, deferred, and unsupported states.
3. Decide which report families fail the gate, warn, or remain advisory.
4. Define CI/local behavior, exit codes, and maintainer override or defer
   policy.
5. Design tests for stale metadata, changed inputs, missing outputs, optional
   data, and advisory-only families.
6. Write the freshness gate design artifact.

### Deliverables
- freshness gate design
- state and severity model
- CI/local behavior matrix
- stale/missing/optional-data test plan
- non-claim wording for local measurements

### Completion Criteria
- stale detection is mechanical and reproducible
- advisory measurements are never treated as broad release proof
- gate behavior is strict only where report freshness can be asserted honestly

---

## Day 11: Freshness Gate Implementation

**Title:** Gate Implementation
**Theme:** Implement freshness validation and integrate it with the normalized
report index
**Time estimate:** 12 hours

### Tasks
1. Implement freshness validation for normalized report index rows.
2. Add CLI flags for check-only, report-only, family filtering, and advisory
   output where appropriate.
3. Add tests for current, stale, missing, skipped, deferred, and unsupported
   rows.
4. Integrate freshness checks with existing validation scripts or Make targets
   where appropriate.
5. Ensure generated local report paths remain ignored unless intentionally
   source-controlled.
6. Write the gate implementation artifact.

### Deliverables
- freshness gate implementation
- CLI and validation integration
- freshness tests
- updated ignore/generated-output behavior if needed
- gate implementation artifact

### Completion Criteria
- stale rows are detected deterministically
- check behavior matches the Day 10 severity model
- generated measurement outputs remain local unless explicitly source-owned

---

## Day 12: Documentation Alignment

**Title:** Documentation Alignment
**Theme:** Update maintainer, benchmark, corpus, package, and report docs with
normalized semantics and non-claims
**Time estimate:** 12 hours

### Tasks
1. Update maintainer docs with normalized report index generation, freshness
   checks, regeneration policy, and troubleshooting.
2. Update benchmark and performance docs with measurement row semantics,
   freshness limits, platform fields, and non-claim wording.
3. Update corpus/oracle docs with normalized index integration and skip/defer
   semantics.
4. Update package/install docs with report rows, static-first package proof,
   and platform-specific boundaries.
5. Update README or tutorial references only where report discovery affects
   user-facing workflow.
6. Write the documentation alignment artifact.

### Deliverables
- updated maintainer report-index guidance
- benchmark and sentinel report semantics
- corpus/oracle documentation updates
- package/install documentation updates
- documentation alignment artifact

### Completion Criteria
- maintainers can regenerate and check report indexes from documented commands
- docs distinguish source-controlled evidence from local measurements
- user-facing docs avoid unsupported state-of-the-art, performance, or package
  claims

---

## Day 13: Validation And Quality Gates

**Title:** Validation Pass
**Theme:** Run report generators, freshness checks, affected script tests, docs
checks, and required quality gates
**Time estimate:** 12 hours

### Tasks
1. Run normalized report index generation and compare deterministic output.
2. Run freshness gate checks for current, skipped, deferred, stale, and missing
   cases.
3. Run corpus schema/oracle validation and any report-family script tests.
4. Run docs checks or link/reference checks available in the repository.
5. If C or header files changed, run `make format && make lint && make test`;
   otherwise run applicable script/docs validation and targeted quality gates.
6. Write the validation pass artifact with commands, results, and residual
   risks.

### Deliverables
- report index generation evidence
- freshness gate validation evidence
- corpus/oracle and script test evidence
- docs validation evidence
- quality gate summary artifact

### Completion Criteria
- all required checks for touched surfaces pass
- generated outputs are either intentionally source-controlled or ignored
- residual risks are documented with owners or Sprint 142 handoff entries

---

## Day 14: Closeout And Sprint 142 Handoff

**Title:** Closeout
**Theme:** Finalize normalized report evidence, close Sprint 141 artifacts, and
hand off runtime/backend governance rows to Sprint 142
**Time estimate:** 12 hours

### Tasks
1. Re-run final report index and freshness checks after documentation updates.
2. Review all Sprint 141 artifacts for consistency with implemented behavior.
3. Confirm normalized report index rows do not overclaim platform, runtime,
   backend, package, benchmark, or corpus evidence.
4. Prepare the Sprint 142 runtime/backend handoff for deferred rows and
   governance decisions.
5. Update working notes with final validation, changed files, decisions,
   deferred work, and known risks.
6. Write the closeout validation summary artifact.

### Deliverables
- final normalized report index evidence
- final freshness gate evidence
- Sprint 141 closeout summary
- Sprint 142 runtime/backend handoff
- updated working notes

### Completion Criteria
- Sprint 141 deliverables are present and traceable to Items 1-7
- validation evidence is current and reproducible
- Sprint 142 receives only the rows that require deeper runtime/backend
  governance rather than unresolved Sprint 141 normalization work
