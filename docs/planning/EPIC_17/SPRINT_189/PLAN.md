# Sprint 189 Plan: PowerShell Validation Ownership

**Sprint Duration:** 14 days
**Goal:** Close the PowerShell validation environment gap by adding an owned
local/hosted validation command for Windows report workflow material.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 189 estimate in the Epic 17
project plan.

**Primary scope:** Inventory PowerShell and Windows report workflow surfaces,
design and implement an owned validation command with local `pwsh`
availability handling, wire the selected hosted Windows validation lane, add
guards against stale ownership or accidental freshness promotion, document the
remaining non-claims, and run the required validation gates.

**Non-goals:** Promoting Windows report freshness, publishing hosted report
artifacts, claiming broad Windows parity, changing solver behavior, changing
the static-first package contract, adding package-manager support, or treating
local PowerShell availability as hosted Windows evidence.

---

## Day 1: Sprint Intake and PowerShell Baseline

**Title:** PowerShell Validation Intake
**Theme:** Establish Sprint 189 scope, owner files, current validation gaps,
and non-claim boundaries.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 189 section of the Epic 17 project plan and the Sprint
   187 Windows acceptance gates.
2. Inventory prior Windows report freshness deferral artifacts, especially
   Sprint 182 outputs and Epic 16 residuals.
3. Locate PowerShell workflow snippets, Windows CI jobs, report scripts,
   manifest references, artifact names, and maintainer documentation.
4. Run read-only discovery for local `pwsh`, Windows-adjacent scripts, report
   schema checks, and selected workflow guards.
5. Create `WORKING_NOTES.md` with baseline state, owner surfaces, risks,
   validation expectations, and Day 2 audit questions.

### Deliverables

- Sprint 189 working-notes scaffold.
- PowerShell validation owner-surface inventory.
- Current local `pwsh` availability record.
- Windows report freshness non-claim baseline.

### Completion Criteria

- Sprint scope is traceable to items 189.1 through 189.6.
- PowerShell validation ownership is separated from report freshness
  promotion.
- Current local and hosted evidence gaps are explicit before implementation.

---

## Day 2: PowerShell Surface Audit

**Title:** PowerShell Surface Map
**Theme:** Build the detailed map of workflow snippets, scripts, artifacts,
and assumptions the validation command must cover.
**Time estimate:** 12 hours

### Tasks

1. Audit `.github/workflows/` for Windows jobs, PowerShell invocations,
   report-related steps, artifact uploads, and shell assumptions.
2. Audit report scripts, selected target manifests, normalization commands,
   freshness commands, and schema tests for Windows-specific assumptions.
3. Audit maintainer docs and sprint artifacts for PowerShell validation,
   hosted Windows evidence, and local skip wording.
4. Record all artifact names and expected output paths that a validation
   command should keep stable.
5. Classify each surface as required validation input, hosted-only evidence,
   local optional evidence, or retained non-goal.

### Deliverables

- PowerShell surface audit artifact.
- Workflow/report artifact name inventory.
- Assumption map for Windows report validation.
- Day 3 command-design requirements.

### Completion Criteria

- All known PowerShell and Windows report validation surfaces are mapped.
- Unsupported report freshness and artifact publication claims remain
  excluded.
- The validation command requirements are concrete enough to implement.

---

## Day 3: Validation Command Design

**Title:** Command Contract Design
**Theme:** Define the owned local/hosted validation command contract,
availability behavior, and failure semantics.
**Time estimate:** 12 hours

### Tasks

1. Decide whether the owner command should be a shell wrapper, Python script,
   Make target, or a combination matching existing project patterns.
2. Define local behavior when `pwsh` is available: parse selected PowerShell
   material, run safe dry-run checks, and validate artifact-name assumptions.
3. Define local behavior when `pwsh` is unavailable: exit with an explicit
   skip/unavailable status and record blocker evidence without failing
   unrelated local validation.
4. Define hosted Windows CI behavior where `pwsh` should be available and
   validation failures must fail the job.
5. Specify command output, exit-code interpretation, owner docs, and required
   follow-up guards.

### Deliverables

- Validation command design artifact.
- Exit-code and availability contract.
- Hosted Windows behavior contract.
- Implementation checklist for Day 4.

### Completion Criteria

- The command has one clear owner and stable invocation.
- Local unavailable states cannot be mistaken for hosted evidence.
- Hosted Windows failures are planned to fail closed.

---

## Day 4: Local Validation Command Scaffold

**Title:** Local Command Scaffold
**Theme:** Implement the first owned validation command with local availability
handling and stable diagnostics.
**Time estimate:** 12 hours

### Tasks

1. Add the selected validation script or Make target using existing repository
   style and naming conventions.
2. Detect local `pwsh` availability and emit explicit pass, fail, or
   unavailable diagnostics.
3. Wire read-only validation inputs from the Day 2 surface map.
4. Add initial checks for selected PowerShell syntax, workflow references, and
   required artifact-name anchors.
5. Record command behavior and changed surfaces in working notes.

### Deliverables

- Initial owned PowerShell validation command.
- Local `pwsh` available/unavailable behavior.
- Stable command diagnostics.
- Updated working notes.

### Completion Criteria

- The command is runnable locally.
- Missing local `pwsh` produces documented unavailable evidence.
- Syntax or anchor drift produces a clear failure.

---

## Day 5: Workflow Snippet Validation

**Title:** Workflow Snippet Coverage
**Theme:** Expand validation to cover selected Windows CI PowerShell snippets
and shell assumptions.
**Time estimate:** 12 hours

### Tasks

1. Add checks for Windows workflow steps that invoke PowerShell or rely on
   PowerShell-compatible syntax.
2. Verify selected workflow commands reference existing scripts, report
   targets, manifests, or artifact paths.
3. Guard shell declarations so PowerShell validation is not silently moved to
   an incompatible shell.
4. Ensure hosted-only workflow expectations are separated from local dry-run
   behavior.
5. Update the Day 5 artifact and working notes with added coverage.

### Deliverables

- Workflow snippet validation coverage.
- Shell/command reference guard.
- Hosted/local distinction record.
- Updated working notes.

### Completion Criteria

- Selected PowerShell workflow material is owned by the validation command.
- Workflow command drift fails with actionable diagnostics.
- Hosted-only claims remain out of local proof wording.

---

## Day 6: Report Artifact and Manifest Validation

**Title:** Report Artifact Guards
**Theme:** Validate Windows report artifact names, selected manifest
references, and report-script assumptions.
**Time estimate:** 12 hours

### Tasks

1. Add checks for selected Windows report artifact names and upload/download
   references.
2. Validate that referenced report manifests, schemas, and normalization
   commands exist.
3. Guard against stale artifact names that would disconnect workflow outputs
   from selected report checks.
4. Confirm package, oracle, comparison, coverage, and Windows report families
   remain in their intended lanes.
5. Record changed validation coverage in the Day 6 artifact.

### Deliverables

- Artifact-name guard coverage.
- Manifest/report reference validation.
- Stale artifact-name failure evidence.
- Updated working notes.

### Completion Criteria

- Selected artifact names are source-controlled and guarded.
- Manifest and report references resolve.
- The guard does not imply report freshness or artifact publication.

---

## Day 7: Local `pwsh` Available Path

**Title:** Local PowerShell Parse Path
**Theme:** Exercise and harden the local validation path when `pwsh` is
available.
**Time estimate:** 12 hours

### Tasks

1. If local `pwsh` is available, run the new validation command through its
   parse or dry-run path and record exact evidence.
2. If local `pwsh` is unavailable, run the unavailable path and verify the
   command explains the blocker without claiming validation success.
3. Add fixture or self-test coverage for available/unavailable path decisions
   when practical without requiring PowerShell in all environments.
4. Confirm local output is stable enough for maintainers and CI logs.
5. Update docs or artifacts if local behavior differs from Day 3 design.

### Deliverables

- Local PowerShell validation run record.
- Available/unavailable behavior evidence.
- Optional local self-test fixture.
- Updated working notes.

### Completion Criteria

- The local command produces deterministic pass/fail/unavailable output.
- Missing `pwsh` is not treated as proof success.
- Available `pwsh` validation does not require hosted artifacts.

---

## Day 8: Hosted Windows CI Wiring

**Title:** Hosted Windows Lane
**Theme:** Add the hosted Windows validation lane while keeping freshness and
publication claims deferred.
**Time estimate:** 12 hours

### Tasks

1. Add the validation command to the selected Windows CI workflow or job.
2. Ensure hosted Windows execution fails closed when PowerShell validation
   fails.
3. Keep report generation, freshness promotion, and artifact publication
   unchanged unless explicitly required for validation ownership.
4. Add comments or naming that distinguish validation ownership from report
   freshness evidence.
5. Record hosted CI wiring and expected evidence in the Day 8 artifact.

### Deliverables

- Hosted Windows validation workflow wiring.
- Fail-closed hosted validation behavior.
- CI evidence expectation record.
- Updated working notes.

### Completion Criteria

- Hosted Windows now owns the selected PowerShell validation command.
- Workflow changes do not promote report freshness.
- Local and hosted evidence expectations are documented.

---

## Day 9: Guard Tests and Drift Checks

**Title:** Ownership Guard Tests
**Theme:** Add tests or scripts that fail on stale validation ownership,
unsupported artifact names, or accidental freshness promotion.
**Time estimate:** 12 hours

### Tasks

1. Add focused guard coverage for the new validation command, workflow wiring,
   and artifact-name assumptions.
2. Add checks that fail if docs claim Windows report freshness from validation
   ownership alone.
3. Add checks that fail if hosted artifact publication is implied without
   selected evidence.
4. Add local skip/unavailable wording checks for environments without `pwsh`.
5. Update working notes with guard ownership and expected failure modes.

### Deliverables

- PowerShell validation ownership guard.
- Freshness non-promotion guard.
- Hosted artifact non-claim guard.
- Updated working notes.

### Completion Criteria

- Drift in validation ownership fails in a targeted way.
- Accidental report freshness promotion is guarded.
- Guard coverage can run in the local development environment.

---

## Day 10: Maintainer Documentation

**Title:** Maintainer Validation Docs
**Theme:** Document the new PowerShell validation owner, local skip semantics,
hosted evidence, and retained non-claims.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` with the new validation command, expected
   local `pwsh` behavior, hosted Windows behavior, and failure interpretation.
2. Document when maintainers must run the PowerShell validation command after
   workflow, report, manifest, or artifact-name changes.
3. Explain that local unavailable output is blocker evidence, not pass
   evidence.
4. Keep Windows report freshness, artifact publication, and broad Windows
   parity as retained non-claims.
5. Record documentation changes and validation needs in working notes.

### Deliverables

- Updated maintainer PowerShell validation guidance.
- Local unavailable-state explanation.
- Hosted evidence expectations.
- Documentation change record.

### Completion Criteria

- Maintainers know exactly which command owns PowerShell validation.
- Local skip/unavailable behavior is unambiguous.
- Docs do not promote Windows report freshness.

---

## Day 11: Windows Support and Report Documentation

**Title:** Windows Claim Calibration
**Theme:** Align user-facing and report-facing docs with the new validation
owner and retained Windows/report non-claims.
**Time estimate:** 12 hours

### Tasks

1. Update Windows support documentation to mention the validation owner only at
   the exact level earned.
2. Update report documentation or selected report notes where they reference
   Windows PowerShell validation, artifact names, or freshness expectations.
3. Audit README and INSTALL for broad Windows parity, report freshness, or
   hosted artifact publication wording.
4. Add explicit revisit criteria for report freshness promotion if needed.
5. Record claim-calibration results in the Day 11 artifact.

### Deliverables

- Updated Windows/report documentation.
- Claim-calibrated user-facing wording.
- Revisit criteria for freshness promotion.
- Updated working notes.

### Completion Criteria

- Documentation consistently separates validation ownership from freshness.
- User-facing wording does not overstate Windows parity.
- Report freshness remains deferred unless separately proven.

---

## Day 12: Integrated Windows-Adjacent Validation

**Title:** Windows Validation Gate
**Theme:** Run all selected PowerShell, workflow, report, docs, and C-gate
checks required by the changed surfaces.
**Time estimate:** 12 hours

### Tasks

1. Run the new PowerShell validation owner command and record pass, fail, or
   unavailable output.
2. Run workflow, selected report manifest, schema, normalization, and freshness
   checks required by the changed report surfaces.
3. Run documentation hygiene and generated-output checks.
4. Run `make format && make lint && make test` if any `.c` or `.h` files
   changed during the sprint.
5. Stop and fix any failing validation before claim audit.

### Deliverables

- Integrated Windows-adjacent validation record.
- PowerShell validation result.
- Report/schema/manifest validation results.
- C quality gate result when required.

### Completion Criteria

- All required checks for changed surfaces pass.
- Any unavailable local PowerShell state is documented and does not promote
  hosted evidence.
- The sprint can enter claim audit without unresolved validation failures.

---

## Day 13: Claim Audit and Residual Decision

**Title:** PowerShell Claim Audit
**Theme:** Decide whether PowerShell validation ownership is closed and record
remaining Windows/report residuals.
**Time estimate:** 12 hours

### Tasks

1. Compare final command, workflow, guard, report, and documentation evidence
   against Sprint 187 Windows acceptance gates.
2. Decide whether Sprint 189 closes PowerShell validation ownership or retains
   a bounded residual.
3. Audit touched docs for unsupported Windows report freshness, artifact
   publication, package, shared-library, dynamic ABI, or broad Windows parity
   claims.
4. Record remaining residuals, revisit criteria, and future non-goals.
5. Prepare retrospective inputs and PR-ready summary notes.

### Deliverables

- Final PowerShell validation claim audit.
- Closure or residual decision record.
- Retained non-claim list.
- Retrospective and PR summary inputs.

### Completion Criteria

- The sprint has one clear final state: owned PowerShell validation, or a
  guarded residual blocker.
- Claim wording is consistent across touched Windows/report docs.
- Remaining Windows report questions are explicit and bounded.

---

## Day 14: Sprint Retrospective and PR Handoff

**Title:** Sprint 189 Closeout
**Theme:** Package final evidence, retrospective inputs, and PR-ready notes.
**Time estimate:** 10 hours

### Tasks

1. Review all Sprint 189 artifacts, working notes, changed files, and
   validation records against items 189.1 through 189.6.
2. Confirm PowerShell command behavior, hosted wiring, guard coverage,
   documentation, and validation results are internally consistent.
3. Check for stale TODOs, unresolved blockers, committed generated outputs,
   broken links, and unsupported Windows/report claims.
4. Update `WORKING_NOTES.md` with final closeout results and retrospective
   inputs.
5. Produce review-ready notes summarizing validation ownership, hosted/local
   evidence, retained non-goals, validation, and residuals.

### Deliverables

- Review-ready Sprint 189 working notes.
- Final Sprint 189 closeout summary.
- Retrospective inputs.
- PR-ready PowerShell validation and claim-boundary notes.

### Completion Criteria

- All Sprint 189 project-plan items have evidence or residual disposition.
- Required validation has passed or the sprint stops before PR handoff.
- The branch is ready for retrospective creation and PR preparation.
