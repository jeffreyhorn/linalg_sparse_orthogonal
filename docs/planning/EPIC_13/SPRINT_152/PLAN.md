# Sprint 152 Plan: Generated Report Freshness Publication

**Sprint Duration:** 14 days
**Goal:** Promote selected generated report freshness checks for claim-bearing
families without converting local generated rows into broad release proof. This
sprint implements the Sprint 152 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 152 begins from:
- Sprint 141 report-index architecture and freshness checks available
- Sprint 150 QR generated report rows available
- Sprint 151 partial-SVD generated report rows available
- normalized corpus/oracle report-index commands available
- generated-local rows currently treated as advisory/local-only evidence
- strict generated-row freshness warnings intentionally documented as residuals

The sprint must:
- select generated report families needed for current claim-bearing surfaces
- define which generated families require generation, strict freshness, or
  advisory handling
- stabilize generation commands, paths, metadata fields, and failure messages
- strengthen report-index freshness checks for selected families
- decide whether any generated freshness gates run in hosted CI
- align benchmark, corpus, report-schema, and maintainer documentation
- regenerate selected reports, validate freshness behavior, and record
  residual generated families
- leave Sprint 153 a clean ABI/package evidence handoff

**End State:** Sprint 152 leaves behind:
- selected generated freshness gates
- stabilized report generation commands
- explicit generated report artifact policy
- updated report and maintainer documentation
- residual generated families recorded with owner candidates
- Sprint 153 ABI handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 152 project-plan estimate.

---

## Day 1: Sprint Intake And Generated-Report Baseline

**Title:** Freshness Intake
**Theme:** Establish Sprint 152 scope, artifact structure, and the current
generated report/freshness baseline
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 152 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 141 report-index architecture artifacts, Sprint 150 QR
   report closeout, and Sprint 151 partial-SVD report handoff.
3. Create Sprint 152 working notes and artifact directory structure.
4. Inventory generated report producers, report-index manifests, freshness
   checks, ignored build artifacts, CI workflows, and docs that mention
   generated reports.
5. Identify current generated report claim boundaries, advisory warnings,
   strict freshness behavior, and proof owners.
6. Define stop conditions for promoting stale, unowned, optional, platform-only,
   or local-only generated rows into broad release proof.

### Deliverables
- Sprint 152 working-notes baseline
- artifact directory structure
- generated report producer inventory
- current freshness policy snapshot
- stop-condition register

### Completion Criteria
- Sprint 152 scope is tied to current repository files and prior sprint
  handoffs
- every current generated report family has an owner or is marked unowned
- stop conditions are explicit before generated family selection begins

---

## Day 2: Generated Family Candidate Audit

**Title:** Family Audit
**Theme:** Audit candidate generated report families and classify their current
claim value, freshness risk, and promotion readiness
**Time estimate:** 12 hours

### Tasks
1. Audit corpus/oracle generated families, including QR and partial-SVD rows.
2. Audit benchmark, sentinel, coverage, dead-code, guardrail, and skip-report
   generated families.
3. Identify command, path, metadata, determinism, platform, and freshness gaps
   for each family.
4. Score each family by claim value, stability, local/CI suitability, artifact
   cost, and failure clarity.
5. Identify generated families that should remain advisory or deferred.
6. Write the generated family candidate audit artifact.

### Deliverables
- generated family candidate list
- per-family readiness and risk table
- command/path/metadata gap inventory
- advisory/deferred candidate list
- promotion-readiness scorecard

### Completion Criteria
- candidate families are compared with concrete repository evidence
- each family has a claim-value and promotion-risk score
- selection inputs are ready for Day 3 without implementation bias

---

## Day 3: Generated Family Selection

**Title:** Family Selection
**Theme:** Select generated report families for Sprint 152 closure and define
their claim scopes, non-claims, and rollback rules
**Time estimate:** 12 hours

### Tasks
1. Select the generated families that Sprint 152 will promote or strengthen.
2. Define whether each selected family is required, strict, advisory, or
   deferred.
3. Define support tier, proof owner, claim scope, non-claims, and artifact
   policy for each selected family.
4. Map selected families to generation commands, freshness checks, report-index
   rows, tests, CI decisions, and documentation updates.
5. Define rollback rules for nondeterminism, stale metadata, noisy CI behavior,
   platform drift, optional data, or broad claim leakage.
6. Write the generated family selection artifact.

### Deliverables
- selected generated family list
- required/strict/advisory/deferred classification
- claim-scope and non-claim register
- implementation map for Days 4-11
- rollback criteria

### Completion Criteria
- selected families can be completely closed within Sprint 152
- every promoted freshness claim has a matching proof owner and validation
  command
- unsupported generated-report claims are explicit before policy design starts

---

## Day 4: Freshness Policy Design

**Title:** Policy Design
**Theme:** Define the generated report freshness contract before command or
gate implementation changes
**Time estimate:** 12 hours

### Tasks
1. Review current `--check-freshness`, `--strict-generated`, and any
   `--require-generated` semantics.
2. Define freshness states for missing, current, stale, advisory, strict,
   source-controlled, and optional generated rows.
3. Define required metadata fields: command, commit, branch, platform,
   compiler, configuration, support tier, artifact path, row count, and
   failure message.
4. Define which mismatch classes should warn, fail locally, fail in CI, or
   remain deferred.
5. Define claim boundaries for local-only generated reports versus
   release/hosted evidence.
6. Write the freshness policy design artifact.

### Deliverables
- generated freshness state model
- required metadata field contract
- local/CI failure policy matrix
- generated-local versus release-proof boundary
- policy design artifact

### Completion Criteria
- freshness policy is precise enough to implement without changing meaning
- advisory versus required behavior is explicit per selected family
- generated rows cannot silently become broad release proof

---

## Day 5: Generator Stabilization Design

**Title:** Stabilization Design
**Theme:** Design stable generated report commands, output paths, manifests,
metadata fields, and failure messages for selected families
**Time estimate:** 12 hours

### Tasks
1. Inspect current generator commands and ignored output paths for selected
   families.
2. Design canonical command strings and path normalization rules.
3. Design metadata normalization for commit, branch, platform, compiler,
   configuration, support tier, artifact path, and row counts.
4. Design stale-output cleanup behavior before regeneration.
5. Design failure-message wording that names the family, row, mismatch, and
   remediation command.
6. Write the generator stabilization design artifact.

### Deliverables
- canonical command and path design
- metadata normalization design
- stale-output cleanup design
- failure-message design
- implementation checklist for Day 6

### Completion Criteria
- selected generators have stable command/path/metadata expectations
- stale generated files cannot contaminate current report-index checks
- failure messages are actionable before implementation starts

---

## Day 6: Generator Stabilization Implementation

**Title:** Stabilization Batch
**Theme:** Implement command, path, metadata, stale-output, and failure-message
stabilization for selected generated families
**Time estimate:** 12 hours

### Tasks
1. Update selected generator scripts to use canonical command metadata.
2. Normalize output paths and report artifact paths.
3. Stabilize commit, branch, platform, compiler, configuration, support tier,
   and row-count metadata.
4. Ensure selected generators remove stale output before writing current rows.
5. Improve failure messages for missing, stale, mismatched, and deferred rows.
6. Run focused generator and report-index checks.

### Deliverables
- stabilized generator command metadata
- normalized output and artifact paths
- stale-output cleanup implementation
- improved freshness failure messages
- focused validation result

### Completion Criteria
- selected generated families emit stable metadata
- stale output cleanup is implemented or explicitly deferred
- focused generator/report checks pass

---

## Day 7: Freshness Gate Design

**Title:** Gate Design
**Theme:** Design strengthened report-index freshness gates for the selected
families without widening their claim scope
**Time estimate:** 12 hours

### Tasks
1. Inspect current report-index normalization and freshness test coverage.
2. Design required/advisory/deferred assertions for selected generated families.
3. Design stale, missing, mismatched-command, mismatched-commit, mismatched-path,
   and mismatched-row-count test cases.
4. Define CLI behavior for required generated families and strict generated
   comparison.
5. Define compatibility behavior for historical source-controlled rows and
   ignored local generated artifacts.
6. Write the freshness gate design artifact.

### Deliverables
- freshness gate design
- report-index test matrix
- CLI behavior design
- compatibility notes
- Day 8 implementation checklist

### Completion Criteria
- every selected family has explicit freshness assertions
- failure behavior is deterministic and scoped
- compatibility boundaries are clear before code changes

---

## Day 8: Freshness Gate Implementation

**Title:** Gate Implementation
**Theme:** Implement strengthened freshness checks and report-index tests for
selected generated families
**Time estimate:** 12 hours

### Tasks
1. Update report-index normalization or freshness-check code for selected
   families.
2. Add or strengthen tests for required, advisory, strict, stale, missing, and
   mismatched generated rows.
3. Add proof that deferred families do not accidentally become required.
4. Verify current generated rows pass the intended policy.
5. Verify intentionally stale or missing rows fail only when policy requires
   failure.
6. Run focused Python/report validation.

### Deliverables
- strengthened freshness-check implementation
- expanded report-index freshness tests
- required/advisory/deferred proof cases
- focused validation result

### Completion Criteria
- selected generated families have executable freshness policy coverage
- stale and missing rows fail or warn according to policy
- focused Python/report checks pass

---

## Day 9: CI And Artifact Policy Design

**Title:** CI Policy
**Theme:** Decide hosted CI and artifact handling for selected generated
freshness gates without converting local rows into broad release proof
**Time estimate:** 12 hours

### Tasks
1. Audit existing CI lanes, supplemental jobs, artifact uploads, and generated
   report commands.
2. Decide whether selected freshness gates run locally only, in hosted CI, or
   both.
3. Define artifact upload, retention, ignore, and path policies.
4. Define CI failure modes for required, advisory, and deferred generated
   families.
5. Define platform/compiler limits and non-claims.
6. Write the CI/artifact policy artifact.

### Deliverables
- CI generated freshness policy
- artifact upload/ignore/retention design
- local-only versus hosted-CI matrix
- platform/compiler non-claim register
- implementation checklist for Day 10

### Completion Criteria
- each selected family has a local/CI policy
- artifact paths and retention choices are explicit
- hosted CI evidence cannot imply unsupported platform, package, ABI, or
  performance claims

---

## Day 10: CI And Artifact Policy Implementation

**Title:** CI Implementation
**Theme:** Implement selected CI or artifact-policy follow-through for
generated freshness checks
**Time estimate:** 12 hours

### Tasks
1. Update CI workflow commands only for selected generated freshness gates.
2. Add artifact upload or ignore-policy updates where selected.
3. Add local scripts or make targets if needed to keep CI and local commands
   aligned.
4. Preserve advisory handling for deferred generated families.
5. Validate affected workflow syntax and local command paths.
6. Record CI/artifact implementation evidence.

### Deliverables
- CI workflow or local-command updates
- artifact policy implementation
- workflow/local command validation result
- CI implementation artifact

### Completion Criteria
- selected CI/artifact policy is implemented or explicitly deferred
- local and CI command names remain aligned
- no generated family receives stronger hosted evidence than policy allows

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align benchmark, corpus, report-schema, maintainer, and handoff docs
with the selected generated freshness policy
**Time estimate:** 12 hours

### Tasks
1. Update corpus/report documentation for selected generated freshness gates.
2. Update benchmark or sentinel documentation if those families are selected.
3. Update maintainer guidance with regeneration, freshness, stale-output, and
   failure-remediation commands.
4. Update report schema docs for any new metadata, policy, or failure fields.
5. Update README or high-level docs only where current claim wording changes.
6. Search active docs for stale advisory/strict/required wording.

### Deliverables
- updated generated report docs
- updated maintainer regeneration guidance
- updated schema documentation
- stale wording search result
- documentation alignment artifact

### Completion Criteria
- docs match implemented generated freshness policy
- active docs do not overclaim generated-local evidence
- stale generated-report wording is removed or intentionally historical

---

## Day 12: Integrated Regeneration And Policy Validation

**Title:** Regeneration Proof
**Theme:** Regenerate selected reports and validate normalization, freshness,
strictness, required behavior, and advisory boundaries together
**Time estimate:** 12 hours

### Tasks
1. Regenerate selected generated report families using canonical commands.
2. Run report-index normalization checks for selected families.
3. Run freshness checks for required, strict, advisory, and deferred families.
4. Run focused unit tests for report-index and generator behavior.
5. Run documentation stale-wording and whitespace checks.
6. Record the integrated regeneration and policy validation artifact.

### Deliverables
- regenerated selected reports under ignored output paths
- normalization validation result
- freshness validation result
- focused unit-test result
- integrated validation artifact

### Completion Criteria
- selected generated freshness gates pass in their intended mode
- advisory warnings are expected and documented
- no stale or unowned generated rows are counted as claim-bearing evidence

---

## Day 13: Full Quality Gate And Residual Review

**Title:** Quality Gate
**Theme:** Run required quality gates, review residual generated families, and
prepare final closeout evidence
**Time estimate:** 10 hours

### Tasks
1. Determine whether `.c` or `.h` files changed during Sprint 152.
2. Run `make format && make lint && make test` if `.c` or `.h` files changed.
3. If only scripts/docs/tests changed, run focused Python/report/doc checks and
   record why the C gate is not required.
4. Review residual generated families and assign owner candidates.
5. Run final `git diff --check`, stale-reference checks, and generated cache
   cleanup.
6. Record Day 13 quality-gate and residual-review artifact.

### Deliverables
- full quality-gate or focused-gate evidence
- residual generated-family register
- final whitespace/stale-reference results
- Day 13 validation artifact

### Completion Criteria
- all required quality checks pass
- unresolved failures are fixed or explicitly escalated before closeout
- residual generated families are assigned to later sprint candidates

---

## Day 14: Closeout And Sprint 153 Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 152 artifacts, validation status, residuals, and the
Sprint 153 ABI/package handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 152 artifacts and ensure links point to current paths.
3. Prepare Sprint 152 retrospective inputs: selected generated freshness gates,
   policy decisions, validation, claim changes, residuals, and follow-up risks.
4. Write the Sprint 153 ABI/package handoff.
5. Run final `git status`, whitespace, stale-reference, schema, report-index,
   and freshness checks.
6. Record closeout summary.

### Deliverables
- finalized Sprint 152 working notes
- complete Sprint 152 artifact set
- generated freshness policy and validation summary
- Sprint 153 ABI/package handoff
- final closeout checklist

### Completion Criteria
- Sprint 152 generated freshness publication is ready for retrospective
- generated-report residuals are explicit and assigned
- branch is clean except for intentional Sprint 152 changes
- generated-report evidence boundary is clear
- Sprint 153 ABI/package handoff is prepared
