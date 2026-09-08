# Sprint 199 Plan: Selected Windows Cholesky Freshness Promotion

**Sprint Duration:** 14 days
**Goal:** Promote the guarded selected Windows Cholesky comparison freshness
lane only if hosted evidence, manifest metadata, workflow guards, and
documentation agree.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 199 estimate in the Epic 18
project plan.

**Primary scope:** Review hosted Windows evidence for the selected
`cholesky-spd-tridiag-5` comparison target, decide whether the selected target
manifest can be promoted or must be explicitly re-deferred, harden
normalizer/selected-target filtering for Windows paths and freshness
diagnostics, align workflow and PowerShell guards, calibrate public and
maintainer documentation, and run the focused validation gates.

**Non-goals:** Broad Windows report freshness, selected oracle freshness,
selected benchmark freshness, QR incompatible Windows comparison promotion,
Linux/macOS comparison promotion, general package-manager support,
Homebrew/core readiness, bottles, Linuxbrew support, shared-library package
support, dynamic ABI compatibility, runtime-loader behavior, or broad
state-of-the-art claims.

---

## Day 1: Promotion Intake and Evidence Map

**Title:** Windows Freshness Intake
**Theme:** Establish Sprint 199 scope, evidence owners, and the current
selected Windows Cholesky claim boundary before changing code or docs.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 199 Epic 18 project-plan section and map items 199.1
   through 199.6 to expected artifacts.
2. Inventory owner surfaces for selected Windows freshness: selected target
   manifest, comparison report artifacts, normalizer, external comparison
   runner, Windows workflow, PowerShell validator, README, INSTALL, corpus
   docs, and maintainer guide.
3. Review Sprint 190 and Sprint 191 artifacts for prior Windows Cholesky
   workflow wiring, selected target constraints, and retained non-claims.
4. Identify current validation commands and classify which ones require hosted
   Windows evidence versus local deterministic checks.
5. Create `WORKING_NOTES.md` with item checklist, evidence ledger, validation
   matrix, risk register, and open questions.

### Deliverables

- Sprint 199 working-notes scaffold.
- Owner-surface inventory for selected Windows Cholesky freshness.
- Item-to-artifact traceability map.
- Initial claim-boundary record.

### Completion Criteria

- Every Sprint 199 item has an identified owner artifact.
- Current selected Windows freshness state is known before edits begin.
- Broad Windows, benchmark, oracle, and QR freshness claims remain non-goals.

---

## Day 2: Hosted Artifact Inventory

**Title:** Hosted Evidence Inventory
**Theme:** Inspect hosted Windows evidence for the exact selected Cholesky
target, artifact paths, row IDs, and workflow metadata.
**Time estimate:** 12 hours

### Tasks

1. Locate the latest relevant hosted Windows workflow runs and artifacts for
   `cholesky-spd-tridiag-5`.
2. Record workflow name, job name, run IDs, commit SHA, generator arguments,
   artifact names, artifact paths, timestamps, and exit status.
3. Verify that evidence is tied to the reviewed branch or merged baseline and
   not to a stale or unrelated workflow run.
4. Compare hosted artifact paths with selected manifest target metadata.
5. Record any missing, stale, ambiguous, or platform-mismatched evidence.

### Deliverables

- Hosted Windows Cholesky evidence inventory.
- Artifact path and row ID table.
- Workflow/run provenance record.
- Initial evidence gap list.

### Completion Criteria

- The selected target has a concrete hosted evidence record or a documented
  absence.
- Artifact provenance is specific enough for later manifest decisions.
- Evidence from unrelated targets or platforms is rejected.

---

## Day 3: Evidence Semantics Review

**Title:** Evidence Semantics
**Theme:** Decide what the hosted artifacts prove and what they do not prove.
**Time estimate:** 12 hours

### Tasks

1. Inspect generated comparison rows for target key, platform, build system,
   library path, artifact path, freshness status, and timestamp fields.
2. Confirm whether the hosted evidence proves the selected Windows Cholesky
   freshness lane end to end.
3. Identify evidence semantics that remain local-only, source-controlled-only,
   stale, or unavailable.
4. Define the exact promotion threshold for item 199.1.
5. Produce a Day 3 evidence-semantics artifact with promote/re-defer inputs.

### Deliverables

- Evidence semantics artifact.
- Promotion threshold checklist.
- Freshness-field interpretation notes.
- Gap and ambiguity classification.

### Completion Criteria

- The sprint has a clear, evidence-backed interpretation of hosted artifacts.
- Missing or ambiguous hosted evidence cannot be accidentally promoted.
- Claim vocabulary is selected-target and Windows-Cholesky scoped.

---

## Day 4: Manifest Promotion Decision

**Title:** Manifest Decision
**Theme:** Promote or explicitly re-defer selected target metadata based only
on the reviewed hosted evidence.
**Time estimate:** 12 hours

### Tasks

1. Review `tests/corpus/manifests/selected_report_targets.tsv` and related
   manifest docs for the current `cholesky-spd-tridiag-5` disposition.
2. Apply the Day 3 promotion threshold to the selected Windows evidence.
3. If evidence is sufficient, update manifest metadata to the exact promoted
   selected Windows freshness state.
4. If evidence is insufficient, record an explicit re-deferral with the exact
   missing evidence.
5. Update working notes with the manifest decision and changed-surface impact.

### Deliverables

- Manifest promotion or re-deferral decision artifact.
- Updated selected target manifest if promotion is earned.
- Residual evidence checklist if promotion is deferred.
- Working-notes decision entry.

### Completion Criteria

- Manifest state matches the reviewed hosted evidence.
- No unreviewed target, platform, or freshness family is promoted.
- The decision can be audited from source-controlled evidence.

---

## Day 5: Windows Path Normalization Tests

**Title:** Path Normalization
**Theme:** Harden selected comparison filtering for Windows artifact paths and
mixed path separators.
**Time estimate:** 12 hours

### Tasks

1. Review existing `normalize_report_index.py` tests for selected comparison
   artifact matching and Windows path handling.
2. Add or update tests covering backslash artifact paths, forward-slash paths,
   absolute paths, relative paths, and suffix matching.
3. Confirm target-specific filtering preserves only intended selected rows.
4. Add negative cases for near-match paths that must not pass.
5. Run focused normalizer tests and record results.

### Deliverables

- Windows artifact-path normalization tests.
- Positive and negative selected filtering cases.
- Focused normalizer validation log.
- Updated working-notes evidence.

### Completion Criteria

- Windows path separators cannot drop valid selected rows.
- Near-match artifact paths cannot create false freshness evidence.
- Focused tests fail before or meaningfully guard the intended behavior.

---

## Day 6: Missing Row and Stale Artifact Diagnostics

**Title:** Freshness Diagnostics
**Theme:** Ensure selected Windows freshness diagnostics report missing rows,
stale artifacts, and target mismatches clearly.
**Time estimate:** 12 hours

### Tasks

1. Review normalizer diagnostics for selected target missing-row behavior.
2. Add or update tests for missing selected comparison rows, stale artifact
   timestamps, wrong target keys, and wrong platform/build-system rows.
3. Ensure failure messages name the selected target and artifact path.
4. Confirm diagnostics distinguish unavailable evidence from stale evidence.
5. Record focused test coverage and remaining diagnostic residuals.

### Deliverables

- Missing-row diagnostic tests.
- Stale-artifact diagnostic tests.
- Wrong-target and wrong-platform negative cases.
- Diagnostic behavior record.

### Completion Criteria

- Selected Windows freshness failures are actionable and target-specific.
- Stale and missing evidence cannot silently pass.
- Diagnostic wording supports reviewer triage.

---

## Day 7: Normalizer Implementation Hardening

**Title:** Normalizer Hardening
**Theme:** Implement any required normalizer fixes revealed by Days 5 and 6
without broadening freshness scope.
**Time estimate:** 12 hours

### Tasks

1. Patch selected comparison artifact matching to normalize path separators
   and preserve exact/suffix semantics.
2. Patch selected-target argument behavior if CLI invocation can silently
   ignore freshness-target filters.
3. Keep changes limited to selected comparison freshness behavior.
4. Run focused normalizer, selected manifest, and report-index tests.
5. Update working notes with changed files and validation impact.

### Deliverables

- Hardened normalizer behavior.
- Focused normalizer test results.
- CLI misuse protection if required.
- Updated evidence ledger.

### Completion Criteria

- Tests prove Windows path matching and target filtering behavior.
- CLI selected-target misuse cannot silently skip intended checks.
- No broad report freshness claim is introduced.

---

## Day 8: Workflow Command Alignment

**Title:** Workflow Alignment
**Theme:** Align the Windows workflow command, artifact names, and selected
freshness lane with the manifest decision.
**Time estimate:** 12 hours

### Tasks

1. Review `.github/workflows/windows-ci.yml` selected Cholesky comparison
   commands and artifact upload names.
2. Ensure generator arguments match `cholesky-spd-tridiag-5`, CMake/MSVC
   build-system expectations, and selected manifest metadata.
3. Align artifact names and paths with normalizer freshness expectations.
4. Keep selected workflow scope bounded to the chosen Cholesky target.
5. Record workflow changed-surface impact and expected hosted verification.

### Deliverables

- Windows workflow alignment record.
- Updated workflow command or artifact metadata if required.
- Hosted verification checklist.
- Working-notes workflow entry.

### Completion Criteria

- Workflow commands match selected target metadata exactly.
- Artifact names and paths are compatible with normalizer filtering.
- No unrelated Windows workflow lane is promoted.

---

## Day 9: PowerShell Ownership Guard Update

**Title:** PowerShell Guard
**Theme:** Update PowerShell validation ownership so selected Windows Cholesky
freshness promotion is guarded and reviewable.
**Time estimate:** 12 hours

### Tasks

1. Review `scripts/validate_windows_powershell.py` and related tests for
   selected comparison freshness workflow ownership.
2. Add or update guard expectations for workflow command, artifact name,
   selected target key, and evidence wording.
3. Preserve existing Windows non-claims for Makefile parity, `pkg-config`
   execution parity, broad report freshness, package managers, shared
   libraries, and dynamic ABI.
4. Run focused PowerShell validation tests locally where possible.
5. Record guard evidence and any hosted-only validation residuals.

### Deliverables

- Updated PowerShell validator or guard tests if required.
- Selected Cholesky ownership validation record.
- Retained Windows non-claim checklist.
- Focused guard logs.

### Completion Criteria

- PowerShell ownership guards the selected freshness workflow path.
- Guard wording matches the manifest decision.
- Existing Windows support boundaries remain intact.

---

## Day 10: Report Index and Manifest Gate Integration

**Title:** Gate Integration
**Theme:** Connect manifest, normalizer, and workflow guard behavior into one
selected freshness validation path.
**Time estimate:** 12 hours

### Tasks

1. Run selected target manifest tests and update them if the manifest schema or
   promoted status changes.
2. Run report-index comparison freshness checks for the selected Cholesky
   target.
3. Ensure generated report index outputs do not claim broad Windows freshness.
4. Add an artifact recording pass/fail behavior for promoted and re-deferred
   states.
5. Update working notes with validation command ownership.

### Deliverables

- Integrated selected freshness gate record.
- Manifest and normalizer test results.
- Report-index comparison freshness result.
- Updated validation matrix.

### Completion Criteria

- Selected manifest, normalizer, and workflow guard behavior agree.
- Failure paths identify the selected target and missing/stale evidence.
- Broad freshness families remain unclaimed.

---

## Day 11: Public Documentation Calibration

**Title:** Public Docs
**Theme:** Update README, INSTALL, and corpus docs with only the earned
selected Windows Cholesky freshness claim or explicit re-deferral.
**Time estimate:** 12 hours

### Tasks

1. Update README support/readiness wording for the selected Windows Cholesky
   freshness disposition.
2. Update INSTALL support matrix and Windows interpretation bullets.
3. Update corpus or selected target documentation with manifest evidence,
   freshness scope, and retained non-claims.
4. Avoid wording that implies broad Windows report freshness, selected oracle
   freshness, selected benchmark freshness, or QR incompatible freshness.
5. Run docs-focused checks and record results.

### Deliverables

- README freshness wording update.
- INSTALL support matrix update.
- Corpus/manifest docs update.
- Public documentation validation record.

### Completion Criteria

- Public docs match the manifest decision and validation evidence.
- Unsupported Windows freshness families remain explicit non-claims.
- Documentation points reviewers to the correct owner surfaces.

---

## Day 12: Maintainer and Planning Alignment

**Title:** Maintainer Alignment
**Theme:** Align maintainer guidance, sprint status, and residual queues with
the final selected Windows Cholesky disposition.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` with selected Windows freshness owner
   surfaces, primary gates, and residual interpretation.
2. Update Sprint 199 working notes with item status for 199.1 through 199.6.
3. Record any residuals for broad Windows freshness, QR incompatible
   comparisons, oracle freshness, benchmark freshness, or hosted evidence
   gaps.
4. Update Epic 18 planning status only if the branch has enough evidence.
5. Run docs and guard checks affected by maintainer/planning edits.

### Deliverables

- Maintainer guide update.
- Sprint 199 item status ledger.
- Residual queue or handoff notes.
- Maintainer/planning validation record.

### Completion Criteria

- Maintainer guidance matches public docs and guard behavior.
- Residuals are explicit and owner-scoped.
- Planning language does not overstate Sprint 199 completion.

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run all applicable Sprint 199 validation gates and record exact
results before closeout.
**Time estimate:** 11 hours

### Tasks

1. Run selected manifest tests, normalizer tests, external comparison runner
   tests, PowerShell validator tests, and selected freshness checks.
2. Run package/static/docs guards if public support wording or maintainer
   guidance changed.
3. Run `make format && make lint && make test` if any `.c` or `.h` files
   changed.
4. Run `git diff --check` and scan for generated artifacts that must not be
   staged.
5. Create the Day 13 integrated validation artifact with command output
   summaries and residuals.

### Deliverables

- Integrated validation artifact.
- Command result matrix.
- Changed-surface quality-gate decision.
- Generated artifact review.

### Completion Criteria

- Required local gates pass before closeout.
- Hosted-only residuals are explicitly named.
- Quality-check requirements match the actual changed file types.

---

## Day 14: Closeout and Retrospective Inputs

**Title:** Closeout Review
**Theme:** Finalize Sprint 199 evidence, claim boundaries, and retrospective
inputs.
**Time estimate:** 11 hours

### Tasks

1. Review every Sprint 199 artifact for consistency with the manifest
   decision, hosted evidence interpretation, and validation results.
2. Finalize item status for 199.1 through 199.6.
3. Verify no generated report, workflow, cache, or proof artifacts are staged
   unless intentionally source-controlled.
4. Record final retained non-claims and any follow-on residuals.
5. Create the Day 14 closeout artifact and retrospective source notes.

### Deliverables

- Sprint 199 closeout review artifact.
- Final item status table.
- Retained non-claim checklist.
- Retrospective input notes.

### Completion Criteria

- Sprint 199 can be reviewed from source-controlled artifacts.
- Promotion or re-deferral is evidence-backed and internally consistent.
- The branch is ready for retrospective preparation and PR review.
