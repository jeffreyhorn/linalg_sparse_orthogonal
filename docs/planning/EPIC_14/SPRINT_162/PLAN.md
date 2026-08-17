# Sprint 162 Plan: Windows Package Parity Decision Closure

**Sprint Duration:** 14 days
**Goal:** Decide and close the remaining Windows package parity gap for
`pkg-config` and Makefile support without confusing it with CMake install
validation. This sprint implements the Sprint 162 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references the older Epic 12 project-plan
path, but the current Sprint 162 project-plan section lives in
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 162 begins from:
- current Windows CMake reviewed subset and install/downstream lane green;
- existing Linux/macOS Make install and `pkg-config` proof available;
- static-first package decision preserved;
- Sprint 161 comparison evidence explicitly separated from package, platform,
  ABI, and release proof;
- public package wording constrained by static-first and non-claim boundaries.

The sprint must:
- audit Windows CMake install proof separately from Unix Make install and
  `pkg-config` proof;
- decide whether to promote Windows `pkg-config`, Windows Makefile parity,
  both, or neither;
- implement the selected proof or stronger retained non-claim guard;
- align Windows CI names, comments, expected counts, and support-tier docs;
- add or update downstream Windows package evidence for the selected decision;
- update INSTALL, README, maintainer guide, and package metadata comments;
- leave Sprint 163 with a performance-publication handoff grounded in package
  evidence boundaries.

**End State:** Sprint 162 leaves behind:
- Windows package parity product decision;
- implemented proof or stronger retained non-claim;
- updated support-tier docs and CI comments;
- downstream Windows package evidence matching the decision;
- affected install/CMake/package validation record;
- Sprint 163 performance handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 162 project-plan estimate.

---

## Day 1: Sprint Intake And Package Surface Inventory

**Title:** Sprint Intake
**Theme:** Establish Sprint 162 scope, artifact layout, and current package
evidence surfaces
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 162 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 143, Sprint 144, Sprint 149, and Sprint 161 package and
   handoff artifacts.
3. Create Sprint 162 working notes and artifact directory structure.
4. Inventory CMake install/export, Make install/uninstall, `pkg-config`,
   Windows CI, Linux/macOS install proof, downstream examples, and package
   docs.
5. Record explicit non-goals for package-manager availability,
   shared-library ABI, dynamic-loader behavior, broad platform support, and
   performance claims.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 162 working-notes baseline
- artifact directory structure
- package surface inventory
- non-goal and assumption register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 162 scope is tied to the Epic 14 project plan
- current package proof owners are identified
- package parity work is separated from solver comparison evidence

---

## Day 2: Windows Package Audit

**Title:** Windows Audit
**Theme:** Compare Windows CMake install proof with Unix Make and `pkg-config`
proofs
**Time estimate:** 12 hours

### Tasks
1. Inventory current Windows CMake install/downstream workflow behavior.
2. Inventory Linux and macOS Make install, uninstall, and `pkg-config`
   validation behavior.
3. Compare installed files, package metadata, exact-version handling,
   downstream example proof, and unsupported-artifact checks.
4. Identify Windows-specific blockers for `pkg-config` and Makefile parity.
5. Separate technical blockers from product-scope decisions.
6. Write the Day 2 Windows package audit artifact.

### Deliverables
- Windows CMake proof map
- Linux/macOS Make and `pkg-config` proof map
- parity delta table
- blocker and decision register
- Day 2 audit artifact

### Completion Criteria
- parity gaps are concrete and source-backed
- CMake install proof is not confused with Make or `pkg-config` proof
- blockers are classified before any product decision is made

---

## Day 3: Package Metadata And Static-First Boundary Review

**Title:** Metadata Review
**Theme:** Review installed metadata and static-first non-claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Inspect CMake package templates, installed target metadata, and version
   files.
2. Inspect `sparse.pc` generation and static archive link flags.
3. Inspect install/uninstall scripts and package validation tests.
4. Identify wording or metadata that could imply shared-library ABI,
   package-manager availability, dynamic loading, or broad platform support.
5. Define exact unsupported-surface checks needed if Windows parity is
   retained as a non-claim.
6. Write the Day 3 metadata-boundary artifact.

### Deliverables
- static-first metadata inventory
- unsupported-wording audit
- retained non-claim guard candidate list
- package metadata risk register
- Day 3 metadata-boundary artifact

### Completion Criteria
- static-first package boundary is explicit
- unsupported ABI/package-manager wording is identified
- later implementation can choose proof or guard without ambiguity

---

## Day 4: Product Decision Matrix

**Title:** Product Decision
**Theme:** Decide whether to promote Windows `pkg-config`, Windows Makefile
parity, both, or neither
**Time estimate:** 12 hours

### Tasks
1. Build the decision matrix for four options: promote Windows `pkg-config`,
   promote Windows Makefile parity, promote both, or retain both non-claims.
2. Score each option by maintainer cost, CI availability, user value,
   portability risk, and documentation complexity.
3. Define the selected product decision and the rationale.
4. Define the proof or retained non-claim requirements implied by the
   decision.
5. Define rollback criteria if the selected proof is not feasible.
6. Write the Day 4 product-decision artifact.

### Deliverables
- Windows package parity decision matrix
- selected product decision
- proof or retained non-claim requirements
- rollback criteria
- Day 4 product-decision artifact

### Completion Criteria
- one explicit Windows package product decision is selected
- Makefile parity and `pkg-config` parity are treated independently
- decision scope is narrow enough to close in the sprint

---

## Day 5: Selected Proof Or Guard Design

**Title:** Proof Design
**Theme:** Design the selected proof path or stronger rejection guard
**Time estimate:** 12 hours

### Tasks
1. Convert the Day 4 decision into exact files, scripts, workflow steps, and
   docs that must change.
2. Define expected installed artifacts, package metadata assertions,
   downstream consumer behavior, and exact-version behavior.
3. Define failure diagnostics for missing package files, unsupported shared
   artifacts, stale metadata, unsupported wording, and command absence.
4. Define support-tier wording for the selected Windows package surface.
5. Define affected local and hosted validation commands.
6. Write the Day 5 proof-or-guard design artifact.

### Deliverables
- selected proof or retained-guard design
- expected artifact and assertion list
- support-tier wording map
- validation command map
- Day 5 design artifact

### Completion Criteria
- implementation has exact acceptance criteria
- unsupported Windows surfaces are guarded or explicitly excluded
- downstream evidence requirements are known before edits begin

---

## Day 6: Implementation Pass One

**Title:** Implementation I
**Theme:** Implement the selected package proof or retained non-claim guard
foundation
**Time estimate:** 12 hours

### Tasks
1. Update package scripts, CMake metadata, Makefile targets, or CI helpers
   required by the selected decision.
2. Add or update unsupported-surface checks for retained non-claims.
3. Preserve Linux/macOS install and `pkg-config` proof behavior.
4. Preserve existing Windows CMake install/downstream proof behavior.
5. Run focused local checks for touched package scripts where possible.
6. Write the Day 6 implementation artifact.

### Deliverables
- first implementation patch set
- retained non-claim or selected proof checks
- focused local command output
- Day 6 implementation artifact

### Completion Criteria
- selected package decision has an executable implementation path
- existing static-first install proof is not weakened
- unsupported package surfaces fail clearly

---

## Day 7: Implementation Pass Two

**Title:** Implementation II
**Theme:** Complete package proof behavior, metadata checks, and diagnostics
**Time estimate:** 12 hours

### Tasks
1. Complete any remaining script, workflow, metadata, or helper changes.
2. Add exact diagnostics for expected Windows package behavior and retained
   non-claims.
3. Ensure package checks do not infer shared-library ABI, package-manager,
   runtime-loader, or broad platform support.
4. Verify generated or installed metadata remains static-first.
5. Update or add focused tests for script and metadata behavior.
6. Write the Day 7 implementation-completion artifact.

### Deliverables
- completed implementation changes
- focused package metadata tests or checks
- diagnostic coverage notes
- Day 7 implementation-completion artifact

### Completion Criteria
- selected proof or retained non-claim guard is complete locally
- diagnostics are reviewable and actionable
- unrelated package surfaces remain unchanged

---

## Day 8: CI Alignment

**Title:** CI Alignment
**Theme:** Align Windows workflow lanes, comments, expected counts, and
support-tier wording
**Time estimate:** 12 hours

### Tasks
1. Review Windows CI workflow names, comments, expected CTest counts, install
   paths, and downstream proof steps.
2. Update workflow wording to match the Day 4 product decision.
3. Add or adjust Windows package checks only for the selected supported
   surface.
4. Preserve staged exclusions and unsupported-surface notes for unpromoted
   package behavior.
5. Validate workflow syntax and expected-count comments where possible.
6. Write the Day 8 CI-alignment artifact.

### Deliverables
- updated Windows CI wording or checks
- expected-count and staged-exclusion notes
- workflow validation notes
- Day 8 CI-alignment artifact

### Completion Criteria
- CI wording matches the selected Windows package decision
- unsupported Windows package surfaces remain explicit
- CMake install proof is not overstated as Make or `pkg-config` parity

---

## Day 9: Downstream Consumer Evidence

**Title:** Downstream Evidence
**Theme:** Add or update downstream Windows package consumer proof for selected
metadata and exact-version behavior
**Time estimate:** 12 hours

### Tasks
1. Review current CMake installed example and exact-version downstream checks.
2. Add or update downstream checks required by the selected product decision.
3. Ensure downstream checks verify static archive metadata and reject
   unsupported shared-library artifacts.
4. Add exact-version or metadata assertions that match the supported surface.
5. Keep unselected Windows Makefile or `pkg-config` behavior out of pass
   evidence.
6. Write the Day 9 downstream-evidence artifact.

### Deliverables
- downstream consumer proof updates
- exact-version or metadata assertion notes
- static-first support-tier evidence
- Day 9 downstream-evidence artifact

### Completion Criteria
- downstream evidence matches the Windows package decision
- package metadata support is verified at consumer boundary
- unselected parity surfaces remain non-claims

---

## Day 10: Focused Tests And Local Validation

**Title:** Focused Validation
**Theme:** Run targeted package, install, CMake, script, and metadata checks
**Time estimate:** 12 hours

### Tasks
1. Run package/install checks affected by the selected implementation.
2. Run CMake configure/build/install/downstream checks available locally.
3. Run script or metadata tests added during the sprint.
4. Run schema or documentation checks touched by package metadata changes.
5. Run `git diff --check` and trailing-whitespace scans.
6. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.

### Deliverables
- focused validation command output
- package/install check output
- script and metadata test output
- changed-file quality-gate decision
- Day 10 validation artifact

### Completion Criteria
- local validation matches the changed-file surface
- failures are fixed or documented before docs alignment
- full C quality gate is run if C/header files changed

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Update INSTALL, README, maintainer guide, package comments, and
support-tier wording
**Time estimate:** 12 hours

### Tasks
1. Update INSTALL and README wording for the selected Windows package decision.
2. Update maintainer guidance for Windows package proof, retained non-claims,
   and validation commands.
3. Update package metadata comments and report-family wording where needed.
4. Verify docs do not imply unselected `pkg-config`, Makefile,
   package-manager, shared-library ABI, dynamic-loader, or broad platform
   support.
5. Align Sprint 163 performance handoff wording with package boundaries.
6. Write the Day 11 docs-alignment artifact.

### Deliverables
- updated INSTALL/README/maintainer/package docs
- support-tier wording checklist
- unsupported-claim scan notes
- Day 11 docs-alignment artifact

### Completion Criteria
- docs match the implemented Windows package decision
- unselected package surfaces remain explicit non-claims
- static-first package boundary remains clear

---

## Day 12: Cross-Platform Package Validation

**Title:** Cross-Platform Validation
**Theme:** Re-run affected Linux, macOS, and Windows package confidence paths
where available
**Time estimate:** 12 hours

### Tasks
1. Run Linux/macOS Make install and `pkg-config` checks where available
   locally.
2. Run CMake install/export/downstream checks where available locally.
3. Run Windows-specific local or syntax checks that can be executed from the
   current environment.
4. Record hosted-only Windows checks that must be verified by CI.
5. Re-run focused tests and docs hygiene after validation-driven fixes.
6. Write the Day 12 cross-platform-validation artifact.

### Deliverables
- cross-platform validation record
- hosted-only verification checklist
- package evidence support-tier notes
- Day 12 validation artifact

### Completion Criteria
- available local package confidence paths pass
- hosted-only Windows expectations are explicit
- package proof remains static-first and bounded

---

## Day 13: Evidence And Claim Review

**Title:** Evidence Review
**Theme:** Review package proof, retained non-claims, CI wording, and docs as
one evidence surface
**Time estimate:** 12 hours

### Tasks
1. Trace each positive Windows package claim to scripts, workflow steps,
   downstream checks, docs, and validation evidence.
2. Trace each retained non-claim to unsupported-surface guards or explicit
   documentation.
3. Review CI wording for CMake, Makefile, `pkg-config`, static archive,
   exact-version, package-manager, shared-library ABI, and platform claims.
4. Review diffs for stale paths, ambiguous package terminology, and unsupported
   evidence assertions.
5. Finalize Sprint 163 performance publication handoff.
6. Write the Day 13 evidence-review artifact.

### Deliverables
- claim-to-evidence trace
- retained non-claim trace
- CI and docs wording review
- Sprint 163 performance handoff
- Day 13 evidence-review artifact

### Completion Criteria
- Windows package decision is reviewable end to end
- positive package wording is bounded by actual proof
- Sprint 163 handoff is ready

---

## Day 14: Closeout And Retrospective Prep

**Title:** Closeout
**Theme:** Finalize Sprint 162 artifacts, validation record, and retrospective
inputs
**Time estimate:** 10 hours

### Tasks
1. Re-run final targeted checks required by the changed-file surface.
2. Update Sprint 162 working notes with final decisions, commands, and
   outputs.
3. Finalize closeout artifacts for the product decision, selected proof or
   retained guard, validation, and Sprint 163 handoff.
4. Review changed files for claim wording, stale paths, and unsupported
   package evidence assertions.
5. Prepare retrospective inputs from artifacts and working notes.
6. Record the Day 14 closeout artifact.

### Deliverables
- final validation record
- selected proof or retained-guard closeout notes
- complete working notes
- retrospective input set
- Day 14 closeout artifact

### Completion Criteria
- Sprint 162 deliverables are complete and traceable
- validation status is recorded with exact commands
- Sprint 163 performance publication handoff is ready
