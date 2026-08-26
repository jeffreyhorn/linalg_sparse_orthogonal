# Sprint 181 Plan: Selected Report Target Manifest

**Sprint Duration:** 14 days
**Goal:** Centralize selected report target metadata so workflows, guards,
docs, and freshness checks stop duplicating target lists by hand. This sprint
implements the Sprint 181 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_181/PLAN.md` and implements the Sprint 181
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 181 begins from:

- the Sprint 177 evidence matrix and acceptance gate for selected report target
  ownership;
- the Sprint 180 closeout state, which leaves package-manager claims stable
  while Sprint 181 focuses on report metadata;
- current report-family metadata in `tests/corpus/manifests/report_families.tsv`;
- current oracle, comparison, package, CI, documentation, benchmark, sentinel,
  guardrail, dead-code, and coverage report rows;
- current report normalizer and freshness checks in
  `scripts/normalize_report_index.py`;
- current workflow and guard tests that duplicate selected target lists by
  hand;
- maintainer-guide report-index documentation and support-tier wording.

The sprint must:

- inventory selected oracle, comparison, performance, artifact, expected-row,
  and support-tier metadata across workflows, tests, manifests, and docs;
- design a canonical selected report target manifest with duplicate detection,
  required files, expected row counts, generator commands, artifact patterns,
  freshness policies, support tiers, owners, claim scopes, and non-claims;
- refactor workflow and report guards to read the manifest instead of copying
  selected target lists;
- keep YAML guard logic scoped to exact jobs and artifact upload blocks while
  taking expected target data from the manifest;
- update maintainer and report-index documentation to explain the manifest as
  the selected target authority;
- validate report normalizer behavior, selected workflow guards, freshness
  checks, Python syntax, and documentation whitespace.

**End State:** Sprint 181 leaves behind:

- one canonical selected report target manifest;
- manifest-driven report and workflow guard behavior;
- reduced target-list duplication across docs, tests, scripts, and workflows;
- clear duplicate, missing-row, stale-row, unsupported-row, and expected-count
  failure diagnostics;
- updated maintainer/report-index documentation;
- Sprint 181 working notes, daily artifacts, validation records, and
  retrospective inputs.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 181 project-plan estimate.

---

## Day 1: Report Target Intake

**Title:** Target Intake
**Theme:** Establish Sprint 181 scope, artifact layout, inherited report
evidence, and manifest success criteria
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 181 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review Sprint 177 evidence matrix, quality surface map, and selected report
   target acceptance gate.
3. Review Sprint 180 closeout and confirm package-manager work does not change
   Sprint 181 report-target scope.
4. Create Sprint 181 working notes and artifact directory structure.
5. Define inventory fields for selected target family, subfamily, generator
   command, artifact pattern, expected row count, support tier, freshness
   policy, owner, claim scope, and non-claims.
6. Write the Day 1 report-target-intake artifact.

### Deliverables

- Sprint 181 working-notes baseline
- artifact directory structure
- inherited gate and evidence notes
- selected report target inventory field list
- Day 1 report-target-intake artifact

### Completion Criteria

- Sprint 181 scope is tied to the Epic 16 project plan
- inherited evidence and acceptance-gate requirements are explicit
- manifest design work starts from shared target metadata fields

---

## Day 2: Current Report Surface Inventory

**Title:** Report Surface Inventory
**Theme:** Inventory existing report-family rows, generated artifacts,
freshness commands, and docs claim surfaces
**Time estimate:** 12 hours

### Tasks

1. Inspect `tests/corpus/manifests/report_families.tsv` and classify current
   report families by source-controlled, generated-local, hosted-CI, and
   documentation rows.
2. Inventory selected oracle, comparison, package, CI, documentation,
   benchmark, sentinel, guardrail, dead-code, and coverage rows.
3. Inspect `scripts/normalize_report_index.py` for current normalization,
   duplicate detection, freshness, required-generated, and expected-count
   behavior.
4. Inventory README, maintainer guide, benchmark docs, INSTALL, and planning
   docs that describe selected report targets or support tiers.
5. Record current generated artifact paths under `build/`, `coverage/`, and
   workflow upload scopes.
6. Write the Day 2 report-surface-inventory artifact.

### Deliverables

- report-family row inventory
- selected target and generated artifact inventory
- normalizer/freshness behavior notes
- documentation claim-surface inventory
- Day 2 report-surface-inventory artifact

### Completion Criteria

- every selected report target surface is accounted for before schema design
- current duplicate target-list owners are visible
- unsupported or advisory rows are separated from selected proof rows

---

## Day 3: Workflow And Guard Duplication Audit

**Title:** Duplication Audit
**Theme:** Find duplicated selected target lists across workflow guards, tests,
scripts, and documentation
**Time estimate:** 12 hours

### Tasks

1. Inspect selected workflow guard tests, including comparison and report
   freshness workflow tests.
2. Inspect CI workflow YAML for report-freshness jobs, artifact uploads,
   platform scope, and selected versus unselected report lanes.
3. Inspect report normalizer tests for embedded selected targets, expected
   rows, generated artifact paths, and freshness assumptions.
4. Compare duplicated lists against `report_families.tsv` rows and maintainer
   guide report sections.
5. Classify duplication as target identity, expected count, artifact path,
   generator command, support tier, or workflow upload scope.
6. Write the Day 3 workflow-and-guard-duplication artifact.

### Deliverables

- duplicate target-list inventory
- workflow/job/upload scope notes
- guard/test embedded expectation inventory
- candidate manifest-owned fields
- Day 3 workflow-and-guard-duplication artifact

### Completion Criteria

- all known duplicate selected target lists have owner files
- YAML guard scope boundaries are explicit before refactor
- manifest-owned fields are distinguished from guard-owned structural checks

---

## Day 4: Manifest Schema Design

**Title:** Schema Design
**Theme:** Design the canonical selected report target manifest format and
validation contract
**Time estimate:** 12 hours

### Tasks

1. Choose a source-controlled manifest path and file format consistent with
   existing manifest conventions.
2. Define required fields for family, subfamily, target key, generator
   command, artifact pattern, expected rows, freshness policy, support tier,
   owner, workflow scope, claim scope, and non-claims.
3. Define duplicate detection rules for family/subfamily/target-key
   collisions.
4. Define missing required file, missing generated row, stale generated row,
   unsupported support-tier, and expected-row-count diagnostics.
5. Define how manifest data maps to `report_families.tsv` without creating
   conflicting authorities.
6. Write the Day 4 manifest-schema-design artifact.

### Deliverables

- selected report target manifest schema
- canonical manifest path decision
- duplicate and missing-row validation rules
- support-tier and freshness-field rules
- Day 4 manifest-schema-design artifact

### Completion Criteria

- schema fields cover project-plan item 181.2
- manifest authority and existing report-family metadata roles do not conflict
- validation failures are specific enough for maintainers to fix rows quickly

---

## Day 5: Manifest Prototype

**Title:** Manifest Prototype
**Theme:** Add the first source-controlled selected target manifest and
populate baseline rows
**Time estimate:** 12 hours

### Tasks

1. Add the selected report target manifest at the Day 4 path.
2. Populate selected oracle rows with generator commands, expected row counts,
   artifact patterns, support tiers, owners, claim scopes, and non-claims.
3. Populate selected comparison rows for the maintained QR, partial-SVD, and
   LU comparison targets.
4. Populate selected package, CI, documentation, benchmark, sentinel,
   guardrail, dead-code, and coverage target rows where Sprint 181 keeps them
   in scope.
5. Add comments or docs only where the manifest format requires local
   explanation.
6. Write the Day 5 manifest-prototype artifact.

### Deliverables

- source-controlled selected target manifest
- populated selected target baseline rows
- row ownership and support-tier notes
- Day 5 manifest-prototype artifact

### Completion Criteria

- manifest exists with all selected Sprint 181 target categories represented
- rows carry enough metadata to replace duplicated lists later in the sprint
- unselected rows are not promoted to selected proof status

---

## Day 6: Manifest Parser And Schema Checks

**Title:** Parser And Schema Checks
**Theme:** Add parser support and focused validation for manifest shape,
required fields, and duplicates
**Time estimate:** 12 hours

### Tasks

1. Extend or add Python manifest-loading code using structured TSV/CSV parsing
   consistent with existing report tooling.
2. Validate required fields, allowed support tiers, allowed freshness policies,
   artifact pattern presence, generator command presence, and owner presence.
3. Add duplicate target-key detection with diagnostics naming every duplicate
   row.
4. Add malformed-row and unsupported-value tests.
5. Keep parser output stable for downstream guard refactors.
6. Write the Day 6 parser-and-schema-checks artifact.

### Deliverables

- manifest parser or loader
- schema validation checks
- duplicate detection tests
- focused parser diagnostics
- Day 6 parser-and-schema-checks artifact

### Completion Criteria

- malformed manifest rows fail clearly
- duplicate selected targets fail clearly
- parser output can drive report and workflow guards

---

## Day 7: Report Normalizer Refactor Design

**Title:** Normalizer Refactor Design
**Theme:** Plan how report normalizer and freshness checks consume
manifest-owned expectations
**Time estimate:** 12 hours

### Tasks

1. Map manifest rows to current report normalizer family filtering and
   generated-row freshness checks.
2. Identify embedded expected row counts and required-generated target lists
   that should move behind manifest access.
3. Define compatibility behavior for advisory rows and unselected generated
   report families.
4. Define error messages for missing, stale, duplicate, or unsupported
   manifest-backed rows.
5. Decide whether CLI flags need to reference selected target keys, families,
   or support tiers.
6. Write the Day 7 normalizer-refactor-design artifact.

### Deliverables

- normalizer manifest-integration design
- expected-count migration map
- freshness behavior compatibility notes
- CLI and diagnostic decisions
- Day 7 normalizer-refactor-design artifact

### Completion Criteria

- report normalizer refactor is scoped before implementation
- selected target expectations have a manifest migration path
- existing advisory behavior remains intentional

---

## Day 8: Report Guard Refactor Batch 1

**Title:** Guard Refactor Batch 1
**Theme:** Update report normalizer tests and freshness checks to consume the
selected target manifest
**Time estimate:** 12 hours

### Tasks

1. Implement manifest-driven selected target loading in report normalizer or
   companion guard code.
2. Replace duplicated expected selected oracle target lists with manifest
   lookups.
3. Replace duplicated expected selected comparison target lists with manifest
   lookups.
4. Add tests for missing selected generated rows, duplicate manifest rows, and
   stale selected rows.
5. Preserve current local-only and advisory freshness semantics.
6. Write the Day 8 report-guard-refactor-batch-1 artifact.

### Deliverables

- manifest-driven normalizer/freshness checks
- updated report normalizer tests
- selected oracle and comparison expectation migration
- Day 8 report-guard-refactor-batch-1 artifact

### Completion Criteria

- selected oracle and comparison checks read manifest-owned expectations
- tests cover missing, duplicate, stale, and unsupported selected rows
- existing report freshness commands remain runnable

---

## Day 9: Report Guard Refactor Batch 2

**Title:** Guard Refactor Batch 2
**Theme:** Extend manifest-driven checks to package, CI, documentation,
benchmark, sentinel, guardrail, dead-code, and coverage rows
**Time estimate:** 12 hours

### Tasks

1. Refactor remaining report guard tests to read manifest-owned target
   metadata where target lists are duplicated.
2. Validate selected package proof-owner rows and support-tier expectations
   from the manifest.
3. Validate selected CI and documentation report rows without treating docs or
   workflow metadata as generated pass evidence.
4. Validate selected benchmark, sentinel, guardrail, dead-code, and coverage
   rows according to their freshness policies.
5. Add regression tests for unsupported support tiers and missing artifact
   patterns.
6. Write the Day 9 report-guard-refactor-batch-2 artifact.

### Deliverables

- broader manifest-driven report guard coverage
- package/CI/documentation target checks
- benchmark/sentinel/guardrail/dead-code/coverage target checks
- Day 9 report-guard-refactor-batch-2 artifact

### Completion Criteria

- remaining duplicated selected target lists are reduced or justified
- advisory rows remain advisory and do not manufacture pass evidence
- support-tier and freshness-policy checks are manifest-driven

---

## Day 10: Workflow Scope Checks

**Title:** Workflow Scope Checks
**Theme:** Keep YAML guards scoped to exact report jobs and artifact upload
blocks while consuming manifest-owned expectations
**Time estimate:** 12 hours

### Tasks

1. Inspect Linux, macOS, and Windows workflow report-freshness job scopes.
2. Refactor workflow guard tests to load selected target expectations from the
   manifest.
3. Keep YAML parsing scoped to exact jobs, commands, artifact names, and upload
   blocks.
4. Add tests that fail on missing selected report jobs, wrong artifact upload
   scopes, broad generated-report uploads, and unselected report execution.
5. Preserve explicit non-claims for Windows report freshness and unselected
   report families.
6. Write the Day 10 workflow-scope-checks artifact.

### Deliverables

- manifest-driven workflow scope guard updates
- exact job and artifact upload block checks
- workflow drift regression tests
- Day 10 workflow-scope-checks artifact

### Completion Criteria

- workflow guards use manifest-owned expectations without widening YAML scan
  scope
- missing or broadened workflow report lanes fail clearly
- selected hosted/local report boundaries remain explicit

---

## Day 11: Documentation Alignment

**Title:** Documentation Alignment
**Theme:** Update maintainer and report-index docs to explain the selected
target manifest as authority
**Time estimate:** 12 hours

### Tasks

1. Update maintainer-guide report sections to name the selected target manifest
   as the target-list authority.
2. Update report-index schema/docs to describe manifest fields, support tiers,
   expected row counts, freshness policies, and ownership.
3. Remove or reduce duplicated selected target lists in docs where the
   manifest can be referenced instead.
4. Preserve claim boundaries for local-only generated reports, hosted selected
   lanes, Windows report freshness, package-manager support, performance, and
   external-library parity.
5. Update Sprint 181 working notes with documentation changes and remaining
   duplicated-list exceptions.
6. Write the Day 11 documentation-alignment artifact.

### Deliverables

- maintainer-guide report manifest documentation
- report-index schema/documentation updates
- reduced duplicated docs target lists
- Day 11 documentation-alignment artifact

### Completion Criteria

- maintainers know where selected target authority lives
- docs and guard behavior describe the same manifest contract
- public support wording does not widen report, platform, package, or
  performance claims

---

## Day 12: Failure Diagnostics And Drift Tests

**Title:** Diagnostics And Drift Tests
**Theme:** Harden missing, duplicate, stale, unsupported, and workflow-drift
failure modes
**Time estimate:** 12 hours

### Tasks

1. Add or refine tests that inject duplicate manifest keys and require
   row-specific diagnostics.
2. Add or refine tests for missing required generated files and expected-row
   count mismatches.
3. Add or refine tests for stale generated rows and stale generator command
   metadata.
4. Add or refine tests for unsupported support tiers, freshness policies, and
   artifact patterns.
5. Add or refine workflow drift tests for exact job and artifact upload block
   ownership.
6. Write the Day 12 diagnostics-and-drift-tests artifact.

### Deliverables

- duplicate-row diagnostics coverage
- missing/stale/expected-count diagnostics coverage
- unsupported-value diagnostics coverage
- workflow drift diagnostics coverage
- Day 12 diagnostics-and-drift-tests artifact

### Completion Criteria

- failure cases name the manifest row or workflow block that must be fixed
- diagnostics distinguish stale generated data from missing generated data
- tests cover the highest-risk drift paths introduced by manifest ownership

---

## Day 13: Integrated Validation Sweep

**Title:** Integrated Validation
**Theme:** Run report normalizer, selected workflow guard, freshness, Python
compile, and documentation checks
**Time estimate:** 12 hours

### Tasks

1. Run report normalizer tests for manifest parsing, duplicate detection,
   expected rows, selected target filtering, and freshness.
2. Run selected workflow guard tests.
3. Run oracle, comparison, and relevant report freshness checks.
4. Run Python compile checks for changed Python tooling and tests.
5. Run package/static/package-manager guards if touched documentation or
   support-tier wording can affect their claim boundaries.
6. Run `git diff --check` and record any formatting corrections.

### Deliverables

- integrated validation output summary
- fixed validation failures if any
- remaining risk notes
- Day 13 integrated-validation artifact

### Completion Criteria

- all relevant report, workflow, freshness, Python, docs, and whitespace checks
  pass or blockers are explicit
- no changed docs or guards widen unsupported report/package/platform claims
- Sprint 181 is ready for closeout review

---

## Day 14: Closeout And Handoff

**Title:** Closeout And Handoff
**Theme:** Reconcile Sprint 181 deliverables, validation records, residual
risks, and Sprint 182 handoff notes
**Time estimate:** 12 hours

### Tasks

1. Reconcile all Sprint 181 project-plan items against produced artifacts and
   changed files.
2. Confirm selected report target manifest, guard refactors, workflow scope
   checks, docs alignment, and validation records are consistent.
3. Record residual risks, intentionally duplicated exceptions, and unsupported
   report/platform/package/performance claims.
4. Prepare Sprint 181 retrospective inputs.
5. Prepare Sprint 182 handoff notes for the Windows report freshness decision.
6. Write the Day 14 closeout-and-handoff artifact.

### Deliverables

- Sprint 181 closeout artifact
- project-plan item reconciliation
- residual risk and duplicated-exception list
- Sprint 182 handoff notes
- retrospective inputs

### Completion Criteria

- Sprint 181 deliverables are validated and reconciled
- selected report target manifest is the documented target-list authority
- Sprint 182 can begin from clear Windows report freshness decision inputs
