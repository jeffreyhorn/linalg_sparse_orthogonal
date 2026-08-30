# Sprint 188 Plan: Homebrew Proof Completion

**Sprint Duration:** 14 days
**Goal:** Close the selected Homebrew local proof blocker by resolving
standalone license metadata and proving the full local formula workflow.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 188 estimate in the Epic 17
project plan.

**Primary scope:** Decide and implement the standalone license metadata needed
for the local Homebrew proof, harden the Homebrew formula proof workflow, keep
package guards aligned with the exact proven support level, calibrate package
documentation, and run the required validation gates.

**Non-goals:** Claiming Homebrew/core readiness, bottle support, Linuxbrew
support, public tap maintenance, binary package distribution, other package
managers, shared-library package support, dynamic ABI stability, broad package
manager support, or any solver behavior/API change.

---

## Day 1: Sprint Intake and Package Baseline

**Title:** Package Proof Intake
**Theme:** Establish the Sprint 188 scope, owner files, and current blocker
state.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 188 section of the Epic 17 project plan and the Sprint
   187 package acceptance gates.
2. Inventory package proof owners: root license metadata, Homebrew template,
   proof script, package guards, install docs, and maintainer docs.
3. Run the current package guard/proof discovery commands in observation mode
   and record whether the blocker is license metadata, local environment, or a
   proof-script failure.
4. Create `WORKING_NOTES.md` with baseline state, validation expectations,
   local tool availability, risks, and open questions.
5. Draft the Day 2 license strategy decision checklist.

### Deliverables

- Sprint 188 working-notes scaffold.
- Package owner surface inventory.
- Current Homebrew proof blocker record.
- Day 2 license strategy checklist.

### Completion Criteria

- Sprint scope is traceable to items 188.1 through 188.6.
- Current proof state is known before metadata edits begin.
- Package support non-goals are recorded in working notes.

---

## Day 2: License Strategy Decision

**Title:** License Strategy
**Theme:** Decide the approved standalone license metadata path.
**Time estimate:** 12 hours

### Tasks

1. Inspect current repository licensing references, package metadata, docs, and
   formula placeholders for license assumptions.
2. Decide whether Sprint 188 will add an approved root `LICENSE`, `COPYING`,
   or `NOTICE` file, or record an alternate formula license strategy.
3. Select the exact Homebrew license identifier that should populate
   `SPARSE_HOMEBREW_LICENSE`.
4. Record archive inclusion requirements for the selected standalone metadata
   file.
5. Update working notes with the decision, rationale, risks, and required
   follow-up edits.

### Deliverables

- License strategy decision record.
- Selected standalone license metadata owner.
- Selected Homebrew license identifier.
- Archive inclusion requirements.

### Completion Criteria

- Formula metadata has an approved license source.
- The selected strategy can be validated by the local Homebrew proof script.
- Unsupported package-provider claims remain explicitly out of scope.

---

## Day 3: Metadata Implementation

**Title:** License Metadata Implementation
**Theme:** Add or update the standalone license and formula metadata inputs.
**Time estimate:** 12 hours

### Tasks

1. Add or update the selected root license metadata file according to the Day 2
   decision.
2. Update Homebrew proof documentation or script inputs so
   `SPARSE_HOMEBREW_LICENSE` has an exact expected value.
3. Confirm the local source archive includes the selected license metadata and
   excludes generated proof outputs.
4. Add or update lightweight checks that prevent formula rendering from using
   missing or placeholder license metadata.
5. Update working notes with changed files and validation required by the
   changed surface.

### Deliverables

- Implemented standalone license metadata.
- Formula/proof metadata alignment.
- Archive inclusion check plan or implementation.
- Updated working notes.

### Completion Criteria

- The proof script can find the standalone license metadata.
- Formula rendering has an accurate license identifier input.
- No generated proof artifacts are committed.

---

## Day 4: Formula Template Audit

**Title:** Formula Template Coherence
**Theme:** Verify the Homebrew template still represents only local static
source formula proof.
**Time estimate:** 12 hours

### Tasks

1. Audit `packaging/homebrew/sparse-lu-ortho.rb.in` for required placeholders:
   homepage, local archive URL, SHA-256, version, and license metadata.
2. Confirm the formula builds with CMake and installs only the maintained
   static archive package surface.
3. Confirm the formula rejects or avoids shared-library artifacts and dynamic
   ABI wording.
4. Review `test do` coverage for exact-version `find_package(Sparse ...)` and
   `Sparse::sparse_lu_ortho`.
5. Record any template corrections needed before proof-script hardening.

### Deliverables

- Formula template audit record.
- Placeholder and static-package checklist.
- `test do` coverage review.
- Template correction list or confirmation of no changes needed.

### Completion Criteria

- The template remains source-controlled input, not a committed installable
  formula.
- Formula behavior is static-first and local-proof-scoped.
- Provider, bottle, tap, and shared-library claims are absent.

---

## Day 5: Proof Script Render and Archive Hardening

**Title:** Render and Archive Proof
**Theme:** Harden the first half of the Homebrew proof workflow.
**Time estimate:** 12 hours

### Tasks

1. Review proof-script discovery for project root, formula template, version,
   required tools, and SHA-256 command.
2. Harden placeholder validation so unresolved formula variables fail with
   clear diagnostics.
3. Ensure the temporary source archive includes required source/package inputs
   and standalone license metadata.
4. Ensure archive checksum injection is deterministic and logged clearly.
5. Confirm temporary render outputs remain untracked and are cleaned unless
   diagnostics are explicitly requested.

### Deliverables

- Hardened render/archive proof behavior.
- Clear diagnostics for missing tools, missing metadata, and unresolved
  placeholders.
- Temporary-output cleanup policy.

### Completion Criteria

- Render/archive failures stop before misleading install attempts.
- A successful render has no unresolved placeholders.
- Generated proof outputs stay outside version control.

---

## Day 6: Proof Script Install Surface Hardening

**Title:** Install Surface Proof
**Theme:** Validate installed static package artifacts and reject unsupported
shared surfaces.
**Time estimate:** 12 hours

### Tasks

1. Harden Homebrew install-from-source handling for the rendered local formula.
2. Verify installed static archive, headers, CMake package files, and
   `sparse.pc`.
3. Reject shared-library artifacts, shared selectors, export macros, and
   unsupported ABI wording in installed metadata.
4. Improve failure cleanup so partial installs, taps, prefixes, caches, and
   temporary build trees do not contaminate later proof attempts.
5. Record validation output expectations in working notes.

### Deliverables

- Hardened install-surface validation.
- Static package artifact checklist.
- Shared-artifact rejection checks.
- Cleanup and retry notes.

### Completion Criteria

- The proof fails clearly when the installed package surface is incomplete.
- Shared-library or ABI surfaces cannot be counted as package proof.
- Failed proof attempts leave the local environment clean enough to retry.

---

## Day 7: Downstream Consumer Test Proof

**Title:** Formula Test Proof
**Theme:** Prove the formula `test do` downstream CMake consumer path.
**Time estimate:** 12 hours

### Tasks

1. Review the Homebrew `test do` block and downstream CMake consumer example.
2. Ensure the test uses exact-version `find_package(Sparse ...)` and links
   `Sparse::sparse_lu_ortho`.
3. Ensure the test builds and runs a minimal executable that exercises the
   installed package surface.
4. Harden `brew test` invocation, diagnostics, and cleanup behavior.
5. Update working notes with pass/fail interpretation for `brew test`.

### Deliverables

- Downstream CMake consumer proof.
- Hardened `brew test` diagnostics.
- Formula test completion notes.

### Completion Criteria

- The local formula proof includes a working downstream consumer test.
- Test failure blocks package support promotion.
- The test does not imply public tap, bottle, or binary package support.

---

## Day 8: Full Local Homebrew Proof Run

**Title:** End-to-End Homebrew Proof
**Theme:** Run and stabilize the complete local proof sequence.
**Time estimate:** 12 hours

### Tasks

1. Run the full proof command with the selected
   `SPARSE_HOMEBREW_LICENSE` value.
2. Diagnose and fix any render, archive, checksum, install, test, uninstall,
   or cleanup failure.
3. Confirm accepted exit states: `0` as proof success, `2` as unavailable
   blocker, and other nonzero values as proof failures.
4. Verify no temporary formula, archive, tap, cache, build, log, install, or
   bottle output is staged or accidentally retained.
5. Record the proof result and evidence boundary in working notes.

### Deliverables

- Full local Homebrew proof result.
- Failure diagnosis or pass evidence.
- Temporary-output cleanup confirmation.
- Support promotion decision input for Day 9.

### Completion Criteria

- The proof outcome is unambiguous and reproducible.
- A failed proof is fixed or recorded as a blocker.
- Support wording is not promoted until guard checks also pass.

---

## Day 9: Package Guard Alignment

**Title:** Package Guard Calibration
**Theme:** Align guard scripts with the proven local Homebrew proof state.
**Time estimate:** 12 hours

### Tasks

1. Run package-manager and static-package guards against the current proof
   state.
2. Update guard expectations so local Homebrew wording is allowed only when
   proof evidence exists.
3. Preserve rejection of unselected provider recipes, public tap wording,
   bottle claims, binary package claims, shared-library support, and dynamic
   ABI support.
4. Update any package report normalization checks only if package report
   metadata changes.
5. Record guard results and any follow-up fixes in working notes.

### Deliverables

- Passing package-manager deferral guard.
- Passing static-package deferral guard.
- Package report metadata decision.
- Guard calibration notes.

### Completion Criteria

- Guard checks match the actual proof state.
- Unsupported package-manager wording still fails validation.
- Static-first installed package policy remains enforced.

---

## Day 10: README and INSTALL Calibration

**Title:** User-Facing Package Docs
**Theme:** Update public package wording to the exact earned support level.
**Time estimate:** 12 hours

### Tasks

1. Update `README.md` package/support wording to keep source install and static
   package support first.
2. Update `INSTALL.md` with the exact Homebrew proof boundary, required proof
   command, license metadata expectation, and retained non-claims.
3. If the proof remains blocked, document the blocker and keep Homebrew
   unclaimed as a user-facing install route.
4. Verify docs do not imply Homebrew/core, bottles, Linuxbrew, public taps,
   other package managers, shared libraries, or dynamic ABI support.
5. Update working notes with claim-boundary decisions.

### Deliverables

- Calibrated README package wording.
- Calibrated INSTALL package workflow.
- Public non-claim wording for unsupported provider surfaces.

### Completion Criteria

- Public docs match the actual Day 8 and Day 9 proof/guard state.
- Users can distinguish source/static install support from local proof
  evidence.
- Unsupported package claims are absent.

---

## Day 11: Homebrew and Maintainer Documentation

**Title:** Package Maintainer Docs
**Theme:** Document the template/proof workflow and maintainer obligations.
**Time estimate:** 12 hours

### Tasks

1. Update `packaging/homebrew/README.md` with the local template/proof
   workflow, temporary artifacts, license metadata requirement, and cleanup
   behavior.
2. Update `docs/maintainer_guide.md` with the proof command, package guards,
   support promotion rule, and retained non-claims.
3. Document local unavailable states, especially missing `brew` or missing
   license metadata, as blocker evidence rather than pass evidence.
4. Record when maintainers must run install checks or report normalization
   checks in addition to the package guards.
5. Update working notes with documentation review results.

### Deliverables

- Updated Homebrew package README.
- Updated maintainer package proof guidance.
- Documentation validation checklist.

### Completion Criteria

- Maintainers can rerun the proof and understand its pass/block/fail states.
- Documentation describes generated proof outputs as temporary and uncommitted.
- Claim promotion requires proof and guards, not template existence alone.

---

## Day 12: Integrated Package Validation

**Title:** Package Validation Gate
**Theme:** Run all selected package proof, guard, install, and documentation
checks.
**Time estimate:** 12 hours

### Tasks

1. Run the Homebrew local proof command with the selected license identifier.
2. Run package-manager and static-package deferral guards.
3. Run install/downstream checks selected by changed package or documentation
   surfaces.
4. Run report index package checks only if package report metadata changed.
5. Run `make format && make lint && make test` if any `.c` or `.h` files
   changed during the sprint.

### Deliverables

- Integrated validation record.
- Passing package guard results.
- Homebrew proof result.
- C quality gate result when required.

### Completion Criteria

- All required checks for changed surfaces pass.
- Any unavailable local proof state is explicitly documented and does not
  promote support.
- The sprint can enter claim audit without unresolved validation failures.

---

## Day 13: Claim Audit and Residual Decision

**Title:** Package Claim Audit
**Theme:** Decide whether Homebrew proof support is promoted or residualized.
**Time estimate:** 12 hours

### Tasks

1. Compare final proof, guard, install, and documentation evidence against the
   Sprint 187 package gates.
2. Decide whether Sprint 188 closes the blocker with a passing local proof or
   retains a blocker/residual with stronger documentation.
3. Audit all touched docs for unsupported package-manager, shared-library,
   dynamic ABI, or provider registry claims.
4. Record remaining residuals, revisit criteria, and future package-provider
   non-goals.
5. Prepare retrospective inputs and PR-ready summary notes.

### Deliverables

- Final package claim audit.
- Closure or residual decision record.
- Retained non-claim list.
- Retrospective and PR summary inputs.

### Completion Criteria

- The sprint has one clear final state: proven local static formula proof, or
  guarded residual blocker.
- Claim wording is consistent across touched package docs.
- Remaining package questions are explicit and bounded.

---

## Day 14: Sprint Retrospective and PR Handoff

**Title:** Sprint 188 Closeout
**Theme:** Package final evidence, retrospective inputs, and PR-ready notes.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 188 artifacts, working notes, changed files, and
   validation records against items 188.1 through 188.6.
2. Confirm license metadata, formula template, proof script, package guards,
   documentation, and validation results are internally consistent.
3. Check for stale TODOs, unresolved blockers, committed generated outputs,
   broken links, and unsupported package claims.
4. Update `WORKING_NOTES.md` with final closeout results and retrospective
   inputs.
5. Produce review-ready notes summarizing proof state, support wording,
   retained non-goals, validation, and any residuals.

### Deliverables

- Review-ready Sprint 188 working notes.
- Final Sprint 188 closeout summary.
- Retrospective inputs.
- PR-ready proof, guard, and claim-boundary notes.

### Completion Criteria

- All Sprint 188 project-plan items have evidence or residual disposition.
- Required validation has passed or the sprint stops before PR handoff.
- The branch is ready for retrospective creation and PR preparation.
