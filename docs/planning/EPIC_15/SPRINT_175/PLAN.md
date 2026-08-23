# Sprint 175 Plan: Cross-Platform Report Freshness Promotion

**Sprint Duration:** 14 days
**Goal:** Promote one selected report freshness path beyond Linux or formally
close the cross-platform deferral with precise blockers. This sprint implements
the Sprint 175 section of `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md`, but the active merged Sprint 175
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the title "Sprint 175: Cross-Platform Report Freshness Promotion".

**Starting Point:** Sprint 175 begins from:

- selected report-index and freshness gates for comparison families;
- Sprint 173 local-only generated API HTML freshness and staging policy;
- Sprint 174 selected QR, partial-SVD, and LU generated comparison freshness;
- documented Windows and macOS platform support tiers;
- package-manager, static package, shared-library ABI, performance,
  external-library parity, release, and state-of-the-art non-claim guards.

The sprint must:

- classify generated reports by Linux, macOS, Windows, local-only, and hosted
  status;
- select exactly one freshness path for macOS or Windows promotion, or select
  formal deferral with enforceable blocker documentation;
- fix platform path, newline, shell, executable, or generated-output
  assumptions in the selected path;
- update CI, documentation, report indexes, or deferral gates to match the
  selected decision;
- run platform-neutral validation and any feasible local checks;
- preserve precise claim boundaries for report freshness support.

**End State:** Sprint 175 leaves behind:

- a cross-platform report freshness matrix;
- one promoted report freshness lane or a formal deferral closure;
- updated platform-tier documentation and report freshness guidance;
- validation artifacts showing the selected lane or deferral is enforceable;
- Sprint 175 working notes, daily artifacts, and closeout records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 175 project-plan estimate.

---

## Day 1: Sprint Intake And Report Freshness Boundary

**Title:** Freshness Intake
**Theme:** Establish Sprint 175 scope, source-artifact mismatch, inherited
report freshness surfaces, and platform non-claims
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 175 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Record the prompt path/source-artifact mismatch in working notes.
3. Review Sprint 173 and Sprint 174 retrospectives for generated-output and
   selected comparison freshness boundaries.
4. Inventory current generated report freshness commands, report-family
   manifests, and selected report-index rows.
5. Create Sprint 175 working notes and artifact directory structure.
6. Write the Day 1 freshness-intake artifact.

### Deliverables

- Sprint 175 working-notes baseline
- artifact directory structure
- inherited report freshness and platform-boundary summary
- source artifact note
- Day 1 freshness-intake artifact

### Completion Criteria

- Sprint 175 scope is tied to the active Epic 15 project plan
- existing generated report freshness paths are visible
- unsupported platform, hosted, package, ABI, performance, release, and
  state-of-the-art claims remain protected

---

## Day 2: Generated Report Family Inventory

**Title:** Report Inventory
**Theme:** Inventory generated report families, commands, outputs, freshness
checks, and current platform assumptions
**Time estimate:** 12 hours

### Tasks

1. Locate generated report commands, Make targets, scripts, and CI references.
2. Inventory generated report outputs and whether each path is tracked,
   ignored, local-only, hosted, or artifact-only.
3. Map report-family manifest rows to generator commands and selected
   freshness checks.
4. Identify reports that currently depend on Linux-only tools, paths, shells,
   executables, or filesystem assumptions.
5. Capture current Windows and macOS support-tier notes for each report family.
6. Write the Day 2 generated-report inventory artifact.

### Deliverables

- generated report command inventory
- generated output staging map
- report-family manifest cross-reference
- current platform-assumption list
- Day 2 generated-report inventory artifact

### Completion Criteria

- every selected generated report path has an owner and command
- local-only, hosted, and artifact-only states are not conflated
- platform assumptions are listed before selecting a promotion candidate

---

## Day 3: Platform Gap Matrix

**Title:** Platform Matrix
**Theme:** Classify generated reports by Linux, macOS, Windows, local-only,
hosted, and support-tier status
**Time estimate:** 12 hours

### Tasks

1. Build a matrix for selected report families across Linux, macOS, and
   Windows.
2. Classify each report as reviewed, supplemental, staged, local-only,
   blocked, or unselected for each platform.
3. Separate local-generation support from hosted-publication support.
4. Identify exact blocker classes: shell, path, compiler, dependency,
   executable, newline, temp directory, generated-output, or CI permission.
5. Rank candidate promotion lanes by closure value and risk.
6. Write the Day 3 platform-gap matrix artifact.

### Deliverables

- cross-platform report freshness matrix
- blocker taxonomy
- candidate lane ranking
- local-only versus hosted-publication distinction
- Day 3 platform-gap matrix artifact

### Completion Criteria

- platform gaps are explicit and comparable
- candidate lanes are ranked before decision
- no report freshness support tier is promoted without evidence

---

## Day 4: Promotion Or Deferral Decision

**Title:** Promotion Decision
**Theme:** Select one report freshness lane for promotion or choose formal
deferral with enforceable blockers
**Time estimate:** 12 hours

### Tasks

1. Review the Day 3 platform matrix and ranking.
2. Select one candidate report freshness path for macOS or Windows promotion,
   or select formal cross-platform deferral.
3. Define the selected lane's exact report family, command, output path,
   platform, support tier, and non-claims.
4. Define required path normalization, CI, documentation, and validation work.
5. Record rejected alternatives and why they are deferred.
6. Write the Day 4 promotion-or-deferral decision artifact.

### Deliverables

- selected promotion or deferral decision
- selected report/platform support scope
- rejected alternative list
- implementation and validation checklist
- Day 4 decision artifact

### Completion Criteria

- exactly one path is selected for complete closure
- claim scope and non-claims are documented before implementation
- implementation work has a clear validation target

---

## Day 5: Path And Execution Assumption Audit

**Title:** Path Audit
**Theme:** Audit selected lane path, newline, shell, executable, temp-dir, and
generated-output assumptions
**Time estimate:** 12 hours

### Tasks

1. Trace every command invoked by the selected freshness lane.
2. Audit path handling for POSIX-only separators, absolute paths, relative
   paths, spaces, drive letters, and temporary directories.
3. Audit shell and executable assumptions, including Python, Make, Bash,
   PowerShell, CMake, compiler, and helper script invocation.
4. Audit newline and text encoding assumptions in generated TSV and Markdown
   outputs.
5. Identify minimal normalization changes needed for the selected lane.
6. Write the Day 5 path-and-execution audit artifact.

### Deliverables

- selected lane execution trace
- path normalization audit
- shell/executable dependency audit
- text output portability audit
- Day 5 path audit artifact

### Completion Criteria

- platform blockers are traced to exact commands or assumptions
- normalization work is scoped before code or workflow edits
- generated-output staging remains local-only unless explicitly promoted

---

## Day 6: Normalization Design

**Title:** Normalization Design
**Theme:** Design minimal platform-safe changes for the selected freshness
lane or deferral guard
**Time estimate:** 12 hours

### Tasks

1. Convert the Day 5 audit into a concrete implementation design.
2. Define script, Makefile, CI, or documentation changes needed for the
   selected lane.
3. Define expected generated report outputs and freshness behavior after the
   change.
4. Define failure messages for unsupported platforms or blocked deferral
   states.
5. Define focused tests or dry-run checks for path and output behavior.
6. Write the Day 6 normalization-design artifact.

### Deliverables

- implementation design
- expected output and freshness behavior
- unsupported-platform failure policy
- focused validation plan
- Day 6 normalization-design artifact

### Completion Criteria

- implementation choices are documented before edits
- deferral and promotion paths have enforceable outcomes
- validation scope matches the selected lane

---

## Day 7: Normalization Implementation

**Title:** Normalize Lane
**Theme:** Implement selected path, script, Makefile, or deferral-guard changes
for the report freshness lane
**Time estimate:** 12 hours

### Tasks

1. Apply the Day 6 design to the selected scripts, Make targets, workflows, or
   deferral documentation.
2. Preserve existing Linux freshness behavior while changing only the selected
   cross-platform lane.
3. Add focused tests or assertions for normalized paths, outputs, or
   deferral-guard behavior.
4. Confirm generated reports remain ignored local artifacts unless the Day 4
   decision selected a different publication path.
5. Run focused local checks for the changed implementation.
6. Write the Day 7 normalization-implementation artifact.

### Deliverables

- implemented normalization or deferral guard
- focused test or assertion updates
- local check results
- generated-output staging confirmation
- Day 7 implementation artifact

### Completion Criteria

- selected lane behavior is implemented or explicitly guarded
- existing selected report freshness behavior is preserved
- no unsupported platform or hosted claims are introduced

---

## Day 8: CI Lane Or Deferral Gate Integration

**Title:** Gate Integration
**Theme:** Wire the selected promotion lane or deferral closure into CI,
Make targets, manifests, or report indexes
**Time estimate:** 12 hours

### Tasks

1. Identify the exact CI, Make, manifest, or report-index integration point.
2. Add the selected platform freshness lane or deferral gate with precise
   support-tier wording.
3. Update expected test/report counts if selected rows or artifacts change.
4. Add comments explaining staged exclusions and platform blockers where
   applicable.
5. Run workflow syntax or local validation checks available on the current
   platform.
6. Write the Day 8 gate-integration artifact.

### Deliverables

- CI, Make, manifest, or report-index integration
- updated expected counts or support-tier comments
- local validation notes
- platform-blocker comments if deferring
- Day 8 gate-integration artifact

### Completion Criteria

- selected lane or deferral is enforceable by source-controlled checks
- support-tier comments match actual validation
- generated report freshness is not silently broadened

---

## Day 9: Documentation Tier Update

**Title:** Tier Documentation
**Theme:** Update user and maintainer documentation for the selected platform
freshness support tier
**Time estimate:** 12 hours

### Tasks

1. Update README freshness and platform support wording for the selected lane.
2. Update maintainer guide report freshness workflows and platform-tier notes.
3. Update corpus/report-index documentation and schema notes if selected rows
   or artifact ownership changed.
4. Preserve local-only, hosted, package, ABI, performance, and state-of-the-art
   non-claims.
5. Add or update command examples for the selected freshness path.
6. Write the Day 9 documentation-tier artifact.

### Deliverables

- updated README/platform freshness wording
- updated maintainer guide workflow notes
- updated report-index or corpus documentation
- command examples and non-claims
- Day 9 documentation artifact

### Completion Criteria

- docs match the implemented or deferred support tier
- users can identify which platforms are reviewed, staged, blocked, or
  local-only
- unsupported claims remain explicit

---

## Day 10: Report Index And Manifest Reconciliation

**Title:** Index Reconciliation
**Theme:** Reconcile selected report rows, artifacts, manifest ownership, and
freshness expectations after platform changes
**Time estimate:** 12 hours

### Tasks

1. Review `scripts/normalize_report_index.py` selected rows and artifacts.
2. Review `tests/corpus/manifests/report_families.tsv` support-tier rows.
3. Reconcile generated report output paths and freshness provenance.
4. Update tests if selected rows, artifact counts, or support tiers changed.
5. Run report-index focused checks.
6. Write the Day 10 report-index reconciliation artifact.

### Deliverables

- selected row and artifact reconciliation
- manifest support-tier reconciliation
- focused test updates if needed
- report-index validation results
- Day 10 reconciliation artifact

### Completion Criteria

- selected rows and artifacts match source-controlled ownership
- support tiers are internally consistent
- report-index freshness checks fail closed for missing or stale outputs

---

## Day 11: Cross-Platform Claim Review

**Title:** Claim Review
**Theme:** Review platform, hosted, package, ABI, performance, and release
claims after selected lane changes
**Time estimate:** 12 hours

### Tasks

1. Scan maintained docs, manifests, scripts, and workflow comments for stale
   platform freshness wording.
2. Confirm the selected lane does not imply unearned hosted publication,
   release evidence, package-manager support, shared-library ABI support, or
   performance claims.
3. Confirm generated comparison and API freshness boundaries remain separate.
4. Run package-manager and static package/shared ABI deferral guards if
   touched surfaces include adoption, package, ABI, or platform wording.
5. Record remaining non-claims and deferrals.
6. Write the Day 11 cross-platform claim-review artifact.

### Deliverables

- claim-scan record
- non-claim and deferral record
- package/ABI guard results when relevant
- stale wording fix list
- Day 11 claim-review artifact

### Completion Criteria

- maintained documentation does not overstate platform support
- deferral wording is enforceable and current
- package, ABI, performance, release, and state-of-the-art claims remain
  bounded

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the selected lane, report freshness, focused tests, and
platform-neutral hygiene checks together
**Time estimate:** 12 hours

### Tasks

1. Run the selected report freshness command or deferral gate.
2. Run focused tests for changed scripts, report indexes, manifests, and
   documentation guards.
3. Run platform-neutral checks available locally.
4. If `.c` or `.h` files changed, run `make format && make lint && make test`.
5. Capture validation output and any remaining platform limitations.
6. Write the Day 12 integrated-validation artifact.

### Deliverables

- integrated validation command list
- focused test results
- platform-neutral hygiene results
- full C quality gate results if required
- Day 12 validation artifact

### Completion Criteria

- all required checks pass
- any remaining platform limitation is documented and intentionally deferred
- no generated output is staged unintentionally

---

## Day 13: Maintenance And Handoff Review

**Title:** Maintenance Review
**Theme:** Review maintainability, future platform work, and Sprint 176
handoff boundaries
**Time estimate:** 12 hours

### Tasks

1. Review Sprint 175 artifacts and working notes from Day 1 through Day 12.
2. Confirm the selected lane or formal deferral can be maintained by future
   contributors.
3. Identify fragile assumptions, manual review points, and future automation
   opportunities.
4. Prepare Sprint 176 handoff notes for final validation and Epic 15 closeout.
5. Re-run focused stale wording or support-tier scans if docs changed after
   Day 11.
6. Write the Day 13 maintenance-review artifact.

### Deliverables

- maintainability review
- future automation notes
- Sprint 176 handoff
- final stale wording scan if needed
- Day 13 maintenance-review artifact

### Completion Criteria

- future maintainers can understand the selected lane or deferral
- Sprint 176 has clear validation and closeout boundaries
- unsupported claims remain excluded

---

## Day 14: Sprint Closeout And Final Freshness Record

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 175 deliverables, final validation, generated
output staging, and Sprint 176 handoff
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 175 outcomes against project-plan items 175.1 through
   175.6.
2. Confirm the platform gap matrix, selected promotion or deferral, path
   normalization, tier docs, and validation artifacts are complete.
3. Re-run final hygiene and required focused checks.
4. Confirm generated report outputs are staged or ignored according to the
   selected architecture.
5. Prepare Sprint 176 handoff notes for Epic 15 final validation and claim
   recalibration.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 175 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 176 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- one cross-platform report freshness lane is promoted or formally deferred
  with precise blockers
- platform-tier documentation matches implemented evidence
- Sprint 176 can begin from clear report freshness and claim boundaries
