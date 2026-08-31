# Sprint 190 Plan: Windows Selected Report Freshness Decision

**Sprint Duration:** 14 days
**Goal:** Promote one Windows-safe selected report freshness lane or renew the
formal deferral with stronger guard evidence.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 190 estimate in the Epic 17
project plan.

**Primary scope:** Select the smallest credible Windows-safe selected report
freshness candidate, prove whether it can run under the reviewed Windows CI
contract, implement the selected lane or renew the formal deferral, update the
selected report manifest and guard coverage, calibrate documentation claims,
and run the required validation gates.

**Non-goals:** Broad Windows report freshness, Windows Makefile parity,
package-manager support, shared-library support, runtime-loader behavior,
dynamic ABI compatibility, performance-claim expansion, or promotion of more
than one selected Windows report lane.

---

## Day 1: Sprint Intake and Decision Baseline

**Title:** Windows Freshness Intake
**Theme:** Establish the Sprint 190 decision scope, prior deferral state, and
candidate-selection constraints.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 190 section of the Epic 17 project plan and map items
   190.1 through 190.6 to owner files and expected artifacts.
2. Review Sprint 182, Sprint 186, Sprint 187, Sprint 188, and Sprint 189
   residuals that mention Windows report freshness, selected targets, hosted
   workflows, and PowerShell validation ownership.
3. Inventory current selected report freshness lanes, generator commands,
   workflow jobs, artifact names, manifest metadata, and local runtime
   assumptions.
4. Identify Windows-specific blockers, including tool availability, shell
   behavior, path handling, runtime limits, artifact policy, and claim
   boundaries.
5. Create `WORKING_NOTES.md` with the sprint baseline, candidate list,
   decision criteria, risks, and Day 2 audit questions.

### Deliverables

- Sprint 190 working-notes scaffold.
- Windows selected report freshness baseline.
- Candidate lane inventory.
- Decision criteria and blocker list.

### Completion Criteria

- Sprint scope is traceable to items 190.1 through 190.6.
- Prior formal deferral evidence is understood before promotion is attempted.
- Candidate selection has explicit acceptance and rejection criteria.

---

## Day 2: Selected Report Candidate Audit

**Title:** Candidate Lane Audit
**Theme:** Audit the selected report lanes and choose the best candidate for a
bounded Windows decision.
**Time estimate:** 12 hours

### Tasks

1. Compare selected oracle, comparison, and benchmark freshness lanes by
   generator complexity, expected rows, artifact size, dependency surface, and
   Windows portability risk.
2. Inspect generator scripts for POSIX assumptions, path assumptions,
   subprocess usage, compiler requirements, timeout risk, and external-tool
   dependencies.
3. Inspect selected manifest rows for workflow platforms, workflow jobs,
   artifact names, generator commands, and freshness-check ownership.
4. Rank candidate lanes using the Day 1 criteria and identify the smallest
   lane that could credibly produce hosted Windows evidence.
5. Document the selected candidate and at least one fallback deferral path.

### Deliverables

- Candidate audit artifact.
- Ranked Windows freshness candidate table.
- Selected candidate or provisional rejection rationale.
- Fallback deferral path.

### Completion Criteria

- The selected lane is bounded to one report freshness family and exact
  generator command.
- Runtime and dependency risks are explicit.
- The fallback deferral path is concrete enough to implement if promotion
  fails.

---

## Day 3: Runtime and Artifact Feasibility Probe

**Title:** Feasibility Probe
**Theme:** Prove whether the selected candidate can run within the Windows CI
contract without broadening claims.
**Time estimate:** 12 hours

### Tasks

1. Dry-run the candidate generator locally where possible and record expected
   inputs, outputs, row counts, and freshness artifacts.
2. Inspect whether the candidate depends on binaries or build products that
   are already produced by the reviewed Windows CMake lane.
3. Define a Windows-safe invocation sequence using CMake, Python, and shell
   semantics available on `windows-2022`.
4. Estimate hosted runtime and artifact size, including a fail-safe timeout
   target.
5. Decide whether to continue toward promotion or pivot to a renewed
   deferral, and record the evidence.

### Deliverables

- Feasibility probe artifact.
- Candidate command sequence.
- Runtime and artifact-size estimate.
- Promote-or-defer decision checkpoint.

### Completion Criteria

- The candidate has a documented hosted Windows execution model.
- Unsupported dependencies are either removed from scope or trigger deferral.
- The decision checkpoint is backed by concrete evidence.

---

## Day 4: Decision Record Draft

**Title:** Decision Record
**Theme:** Draft the selected Windows report freshness decision and define the
exact implementation contract.
**Time estimate:** 12 hours

### Tasks

1. Draft a decision record for either promotion of the selected lane or renewal
   of the formal deferral.
2. Specify generator command, workflow job, expected output files, artifact
   names, row-count expectations, timeout limit, and reviewed claim boundary.
3. Define manifest-field changes needed to represent the selected outcome.
4. Define validation behavior for stale reports, missing artifacts, wrong
   platform metadata, and unsupported Windows promotion claims.
5. Review the decision record against existing Epic 16 and Epic 17 claim
   language.

### Deliverables

- Windows selected report freshness decision draft.
- Exact implementation contract.
- Manifest update checklist.
- Guard update checklist.

### Completion Criteria

- The decision has one clear outcome: promote one lane or renew deferral.
- Every promoted or deferred surface has an owner and validation path.
- Claim boundaries remain narrower than the evidence.

---

## Day 5: Workflow Implementation Scaffold

**Title:** Workflow Scaffold
**Theme:** Implement the initial workflow surface for the selected outcome.
**Time estimate:** 12 hours

### Tasks

1. If promotion is selected, add the Windows workflow job or step sequence for
   the chosen freshness lane using reviewed Windows-safe commands.
2. If deferral is renewed, add the stronger deferral artifact and workflow
   guard anchor that prevents accidental report generation or artifact upload.
3. Set job names, artifact names, timeouts, shell declarations, and command
   comments to match the Day 4 contract.
4. Keep Windows report freshness isolated from broad Windows parity,
   package-manager support, shared libraries, and runtime-loader claims.
5. Record changed workflow surfaces in working notes.

### Deliverables

- Workflow scaffold for the selected outcome.
- Timeout and artifact naming contract.
- Updated working notes.
- Initial workflow drift risk list.

### Completion Criteria

- Workflow changes are tightly scoped to the selected decision.
- The workflow does not imply unsupported Windows capabilities.
- Deferral or promotion behavior is reviewable from source.

---

## Day 6: Manifest Metadata Updates

**Title:** Manifest Metadata
**Theme:** Update selected report target metadata so Windows status is
machine-checkable.
**Time estimate:** 12 hours

### Tasks

1. Update selected report manifest fields for the selected Windows outcome,
   including workflow platform, workflow file, workflow job, artifact, and
   generator ownership.
2. Add or adjust schema validation for Windows platform metadata, selected
   freshness status, and deferral-only rows.
3. Ensure manifest metadata does not make unreviewed Windows claims for other
   selected rows.
4. Add tests for valid promoted metadata or valid renewed-deferral metadata.
5. Document the manifest changes and validation expectations.

### Deliverables

- Updated selected report manifest metadata.
- Schema or manifest validation updates.
- Focused manifest tests.
- Manifest update artifact.

### Completion Criteria

- Windows status is represented by structured metadata, not prose alone.
- Schema checks reject unsupported Windows freshness promotion.
- Existing selected target rows remain coherent.

---

## Day 7: Freshness Guard Implementation

**Title:** Freshness Guard
**Theme:** Add the guard that checks the selected Windows lane or enforces the
renewed deferral.
**Time estimate:** 12 hours

### Tasks

1. Implement or update the freshness guard for the selected Windows scope.
2. Validate expected report files, row counts, selected target identity,
   generated metadata, and artifact names for a promoted lane.
3. For a renewed deferral, validate absence of Windows selected freshness
   commands, artifact uploads, and platform metadata.
4. Add clear diagnostics for stale output, missing generator commands, wrong
   workflow jobs, and unexpected artifact publication.
5. Run the focused guard locally and record pass, fail, or unavailable states.

### Deliverables

- Selected Windows freshness or deferral guard.
- Guard diagnostics for drift scenarios.
- Focused guard test coverage.
- Updated working notes.

### Completion Criteria

- The selected Windows outcome is enforced by a command, not only by
  documentation.
- Drift failures identify the broken file or metadata field.
- Local unavailable states cannot be mistaken for hosted evidence.

---

## Day 8: Hosted Validation Integration

**Title:** Hosted Integration
**Theme:** Wire the selected decision into hosted CI with bounded runtime and
clear evidence semantics.
**Time estimate:** 12 hours

### Tasks

1. Integrate the selected guard or freshness command into the appropriate
   hosted workflow lane.
2. Add job-level or command-level timeout controls that keep the selected lane
   within the sprint runtime contract.
3. Ensure artifacts are uploaded only if promotion is selected and only under
   the approved artifact name.
4. Preserve existing Windows PowerShell validation ownership and selected
   report freshness non-claim guards where applicable.
5. Record hosted evidence expectations and review any branch-only limitations.

### Deliverables

- Hosted workflow integration.
- Runtime timeout controls.
- Artifact upload or no-upload policy.
- Hosted evidence artifact.

### Completion Criteria

- CI invokes the selected Windows decision path automatically.
- Artifact publication behavior matches the decision record.
- Existing Windows validation ownership remains intact.

---

## Day 9: Deterministic Test Expansion

**Title:** Deterministic Tests
**Theme:** Expand tests so the Windows decision is stable across local and
hosted environments.
**Time estimate:** 12 hours

### Tasks

1. Add deterministic tests for selected workflow metadata, artifact names,
   generator commands, and platform fields.
2. Add negative tests for accidental promotion of unselected report lanes.
3. Add tests for stale row counts, missing selected artifacts, unsupported
   Windows platform metadata, and claim-boundary regressions.
4. Ensure local tests do not depend on ambient Windows tools unless the test
   explicitly models hosted Windows availability.
5. Run focused tests and update working notes with the new regression surface.

### Deliverables

- Deterministic test coverage for the decision.
- Negative drift tests.
- Updated working notes.
- Test-output evidence.

### Completion Criteria

- The test suite catches the main ways the Windows decision could drift.
- Tests behave predictably on Linux, macOS, and Windows runners.
- Hosted-only assumptions are isolated from local test outcomes.

---

## Day 10: Documentation and Claim Calibration

**Title:** Claim Calibration
**Theme:** Update public and maintainer docs to match the selected Windows
freshness outcome without overstating support.
**Time estimate:** 12 hours

### Tasks

1. Update README support-tier wording for the selected Windows outcome.
2. Update `INSTALL.md`, maintainer guide, and corpus/report docs with exact
   command names, artifact names, and evidence interpretation.
3. If promotion is selected, state the one promoted lane and explicitly keep
   other Windows report freshness surfaces out of scope.
4. If deferral is renewed, state the renewed blocker, owner guard, and next
   promotion criteria.
5. Add or update claim-boundary markers so unsupported wording fails tests.

### Deliverables

- Claim-calibrated public docs.
- Maintainer command documentation.
- Report evidence interpretation notes.
- Updated claim-boundary tests.

### Completion Criteria

- Documentation claims are no broader than the implemented evidence.
- Maintainers can run and interpret the selected command.
- Unsupported Windows report freshness claims remain guarded.

---

## Day 11: Report Index and Freshness Evidence

**Title:** Report Evidence
**Theme:** Regenerate or verify the selected report evidence for the chosen
Windows decision.
**Time estimate:** 12 hours

### Tasks

1. For a promoted lane, regenerate the selected report output and verify
   freshness normalization locally where possible.
2. For a renewed deferral, verify that selected report indexes remain
   unchanged and that no Windows freshness artifact is implied.
3. Check report index entries, manifest rows, generated timestamps or
   freshness metadata, and workflow artifact references.
4. Record the exact commands run, expected hosted evidence, and any local
   limitations.
5. Update the Day 11 artifact with report evidence and residual risks.

### Deliverables

- Regenerated or verified selected report evidence.
- Report index and manifest consistency notes.
- Command log for freshness validation.
- Updated residual-risk record.

### Completion Criteria

- Report evidence matches the selected decision record.
- Freshness checks fail on stale or mismatched selected output.
- No unselected report family is accidentally promoted.

---

## Day 12: Integrated Validation Pass

**Title:** Integrated Validation
**Theme:** Run the integrated validation surface and close implementation
gaps before final audit.
**Time estimate:** 12 hours

### Tasks

1. Run selected workflow tests, manifest schema tests, Windows validation
   ownership tests, and report freshness or deferral guards.
2. Run documentation link or formatting checks that cover changed planning,
   maintainer, and report files.
3. If any `.c` or `.h` files changed, run `make format && make lint &&
   make test`.
4. Investigate failures, fix root causes, and rerun the failing checks.
5. Record validation commands, results, changed files, and any remaining
   residuals in working notes.

### Deliverables

- Integrated validation log.
- Fixed validation failures.
- Updated working notes.
- Residual risk list for Day 13.

### Completion Criteria

- All required focused checks pass or have explicit unavailable semantics.
- Required C quality gates pass if C or header files changed.
- Remaining residuals are decision-level issues, not untriaged failures.

---

## Day 13: Final Claim and Residual Audit

**Title:** Final Audit
**Theme:** Audit the implementation against the decision record, project plan,
and public claim boundaries.
**Time estimate:** 12 hours

### Tasks

1. Compare implemented files against the Day 4 decision record and confirm
   every promised guard, manifest field, workflow command, and doc update
   exists.
2. Audit README, INSTALL, maintainer guide, corpus docs, sprint artifacts, and
   project-plan references for overbroad Windows report freshness claims.
3. Audit CI workflows for accidental artifact uploads, unselected generator
   commands, wrong runners, wrong shell declarations, and missing timeouts.
4. Update residual records to close `R186-WIN-REPORT-FRESHNESS` if promotion
   is complete, or renew it with sharper evidence if deferral remains.
5. Prepare the Day 14 closeout checklist.

### Deliverables

- Final claim audit artifact.
- Residual decision update.
- CI/workflow audit notes.
- Day 14 closeout checklist.

### Completion Criteria

- Claims, guards, and evidence all describe the same Windows outcome.
- Residual status is explicit and justified.
- No unsupported Windows report freshness lane is left ambiguous.

---

## Day 14: Sprint Closeout

**Title:** Sprint Closeout
**Theme:** Finalize Sprint 190 artifacts, validation evidence, and handoff for
review.
**Time estimate:** 12 hours

### Tasks

1. Create the final Sprint 190 closeout artifact summarizing the selected
   outcome, implementation, changed surfaces, validation, and residuals.
2. Update `WORKING_NOTES.md` with final command results, decisions, and
   follow-up candidates for Sprint 191.
3. Re-run the final required validation set from Day 12 and record results.
4. Check changed files, documentation formatting, manifest consistency, and
   branch cleanliness.
5. Prepare the retrospective inputs and review-ready summary.

### Deliverables

- Day 14 sprint closeout artifact.
- Final working notes.
- Final validation evidence.
- Review-ready handoff summary.

### Completion Criteria

- Sprint 190 has a complete promote-or-defer decision with implementation
  evidence.
- All required validation has passed or documented unavailable semantics.
- The branch is ready for retrospective, commit, push, and pull request.
