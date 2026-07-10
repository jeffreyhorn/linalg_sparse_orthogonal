# Sprint 118 Plan: Epic 11 Baseline, Residual Conversion & Product Truth Freeze

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-10 baseline, convert the final Epic 10 residual
queue into Epic 11 owners, and define the claim/evidence rules for the next
hardening cycle.

**Starting Point:** Sprint 118 begins from:
- the completed Epic 10 retrospective and post-epic residual queue
- Sprint 117 final validation, comparison, residual, and non-claim artifacts
- the Epic 11 Codex review and gap-closure todo
- the Epic 11 project plan for Sprints 118-127

The sprint must:
- reconfirm reviewed Make/CMake parity, source-list, CTest count, install,
  package, benchmark, coverage, and CI-tier surfaces
- convert Sprint 117 and Epic 10 residuals into dependency-ordered Epic 11
  owners with duplicate fences
- freeze current product truth for compressed-first workflows, mutable-shell
  compatibility, solver-family behavior, package/platform support, benchmark
  evidence, and public non-claims
- capture current source/test hotspot metrics for large source and giant-test
  owners
- refresh evidence templates for source movement, oracle expansion,
  performance sentinels, package/ABI decisions, and adoption cleanup
- audit public claims for drift against the final Epic 10 evidence and Epic 11
  candidate claims
- close with artifacts, working notes, and handoff requirements for Sprints
  119-127

**End State:** Sprint 118 leaves behind:
- a post-Epic-10 baseline package
- an Epic 11 residual owner map
- a current product truth map
- source/test hotspot metrics
- refreshed evidence templates
- public-claim drift audit results
- Sprint 119-127 handoff requirements

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 118 project-plan estimate.

---

## Day 1: Sprint Intake and Artifact Skeleton

**Title:** Baseline Intake
**Theme:** Establish Sprint 118 scope, inputs, artifact structure, and day-level owners
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 118 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Re-read the Epic 11 review and gap-closure todo.
3. Re-read the Epic 10 retrospective and Sprint 117 closeout artifacts.
4. Create Sprint 118 working notes and artifact directories.
5. Map every Sprint 118 project-plan item to day-level owners.
6. Write the sprint intake artifact with scope boundaries and validation
   expectations.

### Deliverables
- Sprint 118 working-notes baseline
- artifact directory structure
- day-level owner map
- input artifact inventory
- scope and validation boundary notes

### Completion Criteria
- every Sprint 118 project-plan item has a day-level owner
- required Epic 10 and Epic 11 input artifacts are identified
- no Sprint 119-127 implementation work is silently pulled into the baseline sprint

---

## Day 2: Reviewed Baseline Validation Inventory

**Title:** Validation Inventory
**Theme:** Inventory reviewed and supplemental validation surfaces before running checks
**Time estimate:** 12 hours

### Tasks
1. Inspect Makefile, CMake, workflow, install, package, benchmark, coverage,
   and source-list validation targets.
2. Identify the reviewed baseline commands required for Sprint 118.
3. Identify supplemental commands that document current product truth without
   expanding reviewed claims.
4. Record expected CTest counts, staged exclusions, and platform-specific
   boundaries.
5. Write the validation inventory artifact.

### Deliverables
- reviewed validation command matrix
- supplemental validation lane inventory
- expected-count and exclusion notes
- platform support boundary notes
- Day 3 execution checklist

### Completion Criteria
- all baseline commands are known before execution
- reviewed and supplemental evidence are clearly separated
- staged exclusions are documented as current truth, not hidden failures

---

## Day 3: Baseline Quality Recheck Execution

**Title:** Quality Recheck
**Theme:** Reconfirm current Make/CMake/source-list/test/package evidence
**Time estimate:** 12 hours

### Tasks
1. Run the reviewed baseline quality checks selected on Day 2.
2. Run source-list and CTest registration checks as applicable.
3. Run focused install/package or benchmark smoke checks where they are part of
   current reviewed truth.
4. Capture command outputs, versions, counts, and exclusions.
5. Investigate any mismatch before recording it as evidence.
6. Write the baseline quality recheck artifact.

### Deliverables
- baseline command output summary
- Make/CMake parity evidence
- source-list and CTest count evidence
- install/package/benchmark smoke evidence
- mismatch or follow-up notes

### Completion Criteria
- Item 1 is complete
- current baseline evidence is reproducible from named commands
- any failure or mismatch is either fixed immediately or captured as an explicit blocker

---

## Day 4: CI Tier and Platform Truth Freeze

**Title:** Platform Truth
**Theme:** Freeze CI-tier, platform, package, and install support boundaries
**Time estimate:** 12 hours

### Tasks
1. Review current Linux, macOS, and Windows workflow definitions.
2. Compare reviewed lanes against supplemental lanes and staged exclusions.
3. Reconcile package/install support wording with current validation evidence.
4. Identify platform claims that need downgrade, fence, or future sprint owners.
5. Write the CI-tier and platform truth artifact.

### Deliverables
- CI-tier support map
- platform validation boundary table
- package/install claim map
- staged-exclusion register
- future-sprint owner candidates

### Completion Criteria
- current platform truth is explicit and evidence-backed
- package/install claims do not exceed validation
- Sprint 124-125 handoff candidates are visible

---

## Day 5: Residual Queue Intake and Deduplication

**Title:** Residual Intake
**Theme:** Convert final Epic 10 residuals into a deduplicated Epic 11 intake list
**Time estimate:** 12 hours

### Tasks
1. Extract post-Epic residuals, future-epic candidates, optional work, and
   explicit non-claims from Epic 10 closeout artifacts.
2. Extract related work from the Epic 11 review and todo.
3. Remove duplicate or already-scheduled items from the intake list.
4. Classify each residual by source owner, proof owner, oracle, performance,
   package/platform, adoption, or claim category.
5. Write the residual intake and duplicate-fence artifact.

### Deliverables
- raw residual intake list
- duplicate-fence table
- category map
- already-covered work list
- unresolved residual candidate list

### Completion Criteria
- Item 2 has a deduplicated starting inventory
- no completed Epic 10 work is reintroduced as unresolved debt
- every residual candidate has a category and evidence source

---

## Day 6: Residual Owner and Dependency Map

**Title:** Residual Owners
**Theme:** Assign Epic 11 owners, dependencies, and proof gates to residual work
**Time estimate:** 12 hours

### Tasks
1. Assign each residual candidate to a likely Sprint 119-127 owner.
2. Identify dependencies between source-boundary, oracle, performance,
   package/platform, docs, and claim work.
3. Define proof gates required before implementation or public-claim expansion.
4. Identify residuals that should remain future-epic candidates.
5. Write the Epic 11 residual owner map.

### Deliverables
- residual owner table
- dependency graph or ordered list
- proof-gate checklist
- future-epic deferral notes
- Sprint 119-127 handoff candidate list

### Completion Criteria
- Item 2 is complete
- residual work is assigned or explicitly deferred
- no residual depends on work scheduled after it without a documented prerequisite

---

## Day 7: Product Truth Map Design

**Title:** Truth Map Design
**Theme:** Define the product-truth categories and evidence sources to freeze
**Time estimate:** 12 hours

### Tasks
1. Define truth-map categories for compressed-first storage, mutable-shell
   compatibility, solver families, package/platform support, benchmark
   evidence, and public non-claims.
2. Identify source files, headers, tests, docs, and artifacts that define each
   current truth category.
3. Decide which claims are baseline truth, candidate Epic 11 claims, or
   explicit non-claims.
4. Draft the product truth map structure.
5. Write the truth-map design artifact.

### Deliverables
- product-truth category list
- evidence-source inventory
- baseline/candidate/non-claim classification rules
- truth-map template
- Day 8 completion checklist

### Completion Criteria
- Item 3 has a complete structure before content fill-in
- each truth category has named evidence sources
- candidate claims remain fenced until future proof exists

---

## Day 8: Product Truth Map Completion

**Title:** Truth Map
**Theme:** Publish the current compressed-first, solver, package, benchmark, and non-claim truth
**Time estimate:** 12 hours

### Tasks
1. Fill in the compressed-first and mutable-shell truth sections.
2. Fill in direct, iterative, eigensolver, SVD, QR, graph, reorder, and Matrix
   Market truth sections.
3. Fill in package/platform and benchmark truth sections.
4. Fill in public non-claim and candidate-claim sections.
5. Cross-check the truth map against Day 3-4 validation evidence.
6. Write the completed product truth map artifact.

### Deliverables
- current product truth map
- baseline claim list
- candidate claim list
- explicit non-claim list
- evidence cross-reference table

### Completion Criteria
- Item 3 is complete
- every current truth entry cites an evidence source
- public claims and non-claims are ready for Day 12 drift audit

---

## Day 9: Source and Test Hotspot Metric Collection

**Title:** Hotspot Metrics
**Theme:** Capture current file-size, responsibility, and ownership metrics
**Time estimate:** 12 hours

### Tasks
1. Collect current source, header, test, benchmark, example, and documentation
   file counts.
2. Collect largest source and test owner line counts.
3. Identify source files with mixed responsibilities or extraction pressure.
4. Identify giant tests with proof-owner density or hidden fixture coupling.
5. Write the hotspot metric collection artifact.

### Deliverables
- repository file-count summary
- largest source/test owner table
- mixed-responsibility source list
- giant-test proof-owner list
- raw command transcript or reproducibility notes

### Completion Criteria
- Item 4 has current numeric evidence
- metric commands are reproducible
- downstream source-boundary and proof-owner sprints have ranked targets

---

## Day 10: Hotspot Interpretation and Owner Handoff

**Title:** Hotspot Owners
**Theme:** Turn hotspot metrics into Sprint 119-123 handoff guidance
**Time estimate:** 12 hours

### Tasks
1. Interpret Day 9 metrics against Epic 11 review findings.
2. Separate high-risk owners from acceptable large-but-coherent owners.
3. Map eigensolver, direct/iterative, SVD, QR, corpus, and report-index
   follow-through candidates to future sprints.
4. Define proof requirements before any source movement or giant-test split.
5. Write the hotspot owner handoff artifact.

### Deliverables
- ranked hotspot owner map
- source-movement prerequisite list
- giant-test split prerequisite list
- Sprint 119-123 handoff notes
- no-move or defer candidates

### Completion Criteria
- Item 4 is complete
- source/test movement candidates are evidence-ranked
- future sprints receive proof requirements, not broad refactor mandates

---

## Day 11: Evidence Template Refresh Design

**Title:** Template Design
**Theme:** Design evidence templates for Epic 11 implementation and closeout work
**Time estimate:** 12 hours

### Tasks
1. Review Sprint 100-117 evidence templates and closeout artifacts.
2. Identify template gaps for source movement, oracle expansion, performance
   sentinels, package/ABI decisions, and adoption cleanup.
3. Define required sections for proof values, validation, drift, non-claims,
   and handoff.
4. Draft updated template outlines.
5. Write the evidence template design artifact.

### Deliverables
- existing-template inventory
- template-gap list
- refreshed template outlines
- required evidence field list
- Day 12 implementation checklist

### Completion Criteria
- Item 5 has a bounded update design
- templates preserve evidence visibility and non-claim discipline
- future sprints can use the templates without rediscovering required fields

---

## Day 12: Evidence Template Refresh Implementation

**Title:** Template Refresh
**Theme:** Publish refreshed evidence templates for Sprint 119-127 work
**Time estimate:** 12 hours

### Tasks
1. Create or update source-movement evidence template.
2. Create or update oracle-expansion evidence template.
3. Create or update performance-sentinel evidence template.
4. Create or update package/ABI decision evidence template.
5. Create or update adoption-cleanup evidence template.
6. Write usage notes and future-sprint handoff rules.

### Deliverables
- refreshed source-movement template
- refreshed oracle-expansion template
- refreshed performance-sentinel template
- refreshed package/ABI decision template
- refreshed adoption-cleanup template
- template usage notes

### Completion Criteria
- Item 5 is complete
- each template includes proof, validation, drift, and non-claim fields
- future sprint handoffs reference the refreshed templates

---

## Day 13: Public Claim Drift Audit

**Title:** Claim Drift
**Theme:** Recheck public and support docs against final Epic 10 truth and Epic 11 candidates
**Time estimate:** 11 hours

### Tasks
1. Audit README, install docs, algorithm docs, benchmark docs, solver-selection
   docs, Matrix Market docs, examples, and maintainer docs for public claims.
2. Compare each claim against the Day 8 product truth map.
3. Identify unsupported, partially supported, stale, or candidate-only claims.
4. Define edits, fences, or future sprint owners for any drift.
5. Write the public claim drift audit artifact.

### Deliverables
- public/support claim audit table
- unsupported or stale claim list
- candidate-only claim list
- edit/fence/future-owner recommendations
- Sprint 126-127 adoption and claim handoff notes

### Completion Criteria
- Item 6 is complete
- public wording does not silently exceed the current truth map
- unsupported claims have a cleanup owner or explicit non-claim disposition

---

## Day 14: Sprint Closeout and Handoff Package

**Title:** Handoff Package
**Theme:** Publish Sprint 118 artifacts, working notes, validation summary, and Sprint 119-127 requirements
**Time estimate:** 11 hours

### Tasks
1. Review all Sprint 118 artifacts for consistency, duplicate fences, and
   evidence links.
2. Summarize validation evidence, product truth, residual owners, hotspot
   metrics, refreshed templates, and claim audit results.
3. Create Sprint 119-127 handoff requirements.
4. Identify residual deferred debt created by Sprint 118.
5. Update working notes and closeout checklist.
6. Write the Sprint 118 closeout artifact.

### Deliverables
- complete Sprint 118 artifact index
- validation and product-truth summary
- residual owner handoff package
- hotspot and template handoff package
- claim-drift handoff package
- residual deferred debt list

### Completion Criteria
- Item 7 is complete
- every Sprint 118 deliverable has an artifact or explicit deferral
- Sprint 119 can begin with clear prerequisites, evidence gates, and non-claim boundaries
