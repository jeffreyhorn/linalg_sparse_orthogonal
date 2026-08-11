# Sprint 149 Plan: Windows Install-Validation Parity Decision

**Sprint Duration:** 14 days
**Goal:** Decide and implement the reviewed Windows install-validation support
tier without confusing it with Unix Makefile or `pkg-config` parity. This
sprint implements the Sprint 149 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 149 begins from:
- Sprint 148 Windows staged-test portability closure merged into `master`
- Windows CMake CTest subset updated and enforced in CI
- current Windows supplemental CMake install/downstream lane available
- Linux/macOS reviewed static-first Make install and `pkg-config` proofs
- static-first package metadata, CMake package files, and public support-tier
  wording already established
- Sprint 148 install-parity handoff available for follow-through

The sprint must:
- compare Windows CMake install proof against Linux/macOS reviewed package proof
- decide whether Windows install validation can be promoted, renamed, split, or
  explicitly rejected as reviewed parity
- keep CMake package proof separate from Unix Makefile and `pkg-config` parity
- strengthen Windows checks for static `.lib`, headers, CMake metadata, version
  behavior, and absence of shared-library artifacts
- preserve static-first, no-shared-ABI, and unsupported-package-manager
  boundaries
- update workflow names, comments, report rows, README, INSTALL, and maintainer
  guidance to match the evidence
- run local feasible checks and full quality gates if `.c` or `.h` files change
- leave Sprint 150 a clean QR corpus handoff

**End State:** Sprint 149 leaves behind:
- Windows install-validation product decision
- promoted, renamed, split, or explicitly rejected Windows package lane
- strengthened Windows CMake package and downstream consumer proof
- updated support-tier docs and report rows
- validation evidence and residual list
- Sprint 150 QR corpus handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 149 project-plan estimate.

---

## Day 1: Sprint Intake And Install Baseline

**Title:** Install Intake
**Theme:** Establish Sprint 149 scope, artifact structure, and current install
validation baseline across Linux, macOS, and Windows
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 149 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 148 Day 14 handoff and platform-support artifacts.
3. Create Sprint 149 working notes and artifact directory structure.
4. Inventory Linux/macOS reviewed install workflows, `tests/test_install.sh`,
   and static-first package proof checks.
5. Inventory Windows supplemental CMake install/downstream workflow commands and
   current pass/fail expectations.
6. Define evidence fields for package files, downstream consumers, version
   behavior, and unsupported claims.

### Deliverables
- Sprint 149 working-notes baseline
- artifact directory structure
- cross-platform install-proof inventory
- Windows supplemental lane snapshot
- evidence template for install-validation parity

### Completion Criteria
- Sprint scope is tied to current repository files and Sprint 148 handoff
- Linux/macOS reviewed proof and Windows supplemental proof are separated
- every install-validation evidence category has an owner and recording format

---

## Day 2: Windows Package Audit

**Title:** Package Audit
**Theme:** Compare the Windows CMake install/downstream proof with reviewed
Linux/macOS static-first package validation
**Time estimate:** 12 hours

### Tasks
1. Inspect Windows install workflow steps for configure, build, install,
   package-file checks, and downstream example execution.
2. Compare Windows checks against `tests/test_install.sh` coverage on
   Linux/macOS.
3. Identify Windows-equivalent checks for static archive, header count, CMake
   package files, `sparse.pc`, version config, and no shared artifacts.
4. Identify checks that are Unix-specific and must remain non-parity:
   Makefile install, shell install script behavior, and `pkg-config` execution.
5. Record proof gaps, duplicate checks, and misleading wording risks.
6. Write the Windows package audit artifact.

### Deliverables
- Windows package audit table
- Linux/macOS reviewed proof comparison
- Windows-equivalent check list
- Unix-only non-parity list
- wording-risk register

### Completion Criteria
- each Windows install check is classified as equivalent, supplemental,
  missing, or intentionally non-parity
- unsupported Unix Makefile and `pkg-config` parity claims remain explicit
- remaining proof gaps are concrete enough to drive Day 3 criteria

---

## Day 3: Promotion Criteria

**Title:** Criteria Gate
**Theme:** Define exact criteria for reviewed Windows install-validation parity,
explicit rejection, or a narrower promoted support tier
**Time estimate:** 12 hours

### Tasks
1. Define candidate outcomes: reviewed Windows CMake install validation,
   supplemental-only lane, split reviewed/supplemental lanes, or explicit
   rejection.
2. Define must-pass evidence for static `.lib`, installed headers, CMake
   package metadata, downstream consumer, exact version, mismatch version, and
   no shared artifacts.
3. Define explicit non-goals for Makefile install parity, package-manager
   claims, shared-library ABI, and `pkg-config` execution on Windows.
4. Define failure semantics for missing package files, unexpected DLLs,
   unsupported wording, and mismatched version behavior.
5. Map each criterion to a workflow command or local review artifact.
6. Write the promotion criteria artifact.

### Deliverables
- Windows install-validation decision criteria
- must-pass evidence checklist
- explicit non-goal register
- failure-semantics table
- workflow-to-criterion mapping

### Completion Criteria
- promotion cannot occur without a named evidence row for every required check
- rejected or supplemental-only outcomes have concrete wording requirements
- support-tier language is ready for implementation without ambiguity

---

## Day 4: Product Decision

**Title:** Product Decision
**Theme:** Decide the Windows install-validation support tier and implementation
path before changing CI
**Time estimate:** 12 hours

### Tasks
1. Apply Day 3 criteria to the current Windows supplemental lane evidence.
2. Choose promotion, renaming, splitting, or explicit rejection for the Windows
   install-validation lane.
3. Decide which checks must be added before the selected outcome is earned.
4. Define rollback rules for hosted Windows failures or support-tier drift.
5. Map the decision to workflow names, comments, report rows, README, INSTALL,
   and maintainer guide updates.
6. Write the product decision artifact.

### Deliverables
- Windows install-validation product decision
- implementation target list
- rollback criteria
- docs/report update map
- Sprint 149 decision record

### Completion Criteria
- the selected outcome is traceable to Day 3 criteria
- no CI or documentation change claims more than the selected outcome supports
- remaining unsupported Windows package surfaces are named

---

## Day 5: Workflow Implementation Design

**Title:** Workflow Design
**Theme:** Design workflow edits for the selected Windows install-validation
decision
**Time estimate:** 12 hours

### Tasks
1. Inspect `.github/workflows` for Windows install/downstream job ownership,
   triggers, labels, and comments.
2. Decide exact job names, step names, and environment variables for the
   selected product outcome.
3. Design any split between reviewed and supplemental Windows install proof.
4. Define hosted-only evidence expectations that cannot be fully proven
   locally.
5. Define workflow syntax and command-review checks before implementation.
6. Write the workflow implementation design artifact.

### Deliverables
- workflow edit design
- job and step naming plan
- reviewed/supplemental split plan, if needed
- hosted evidence expectations
- workflow syntax review checklist

### Completion Criteria
- workflow edits are planned before modifying CI files
- job names and comments express the support tier precisely
- hosted-only residual risk is recorded before implementation

---

## Day 6: Workflow Implementation

**Title:** Workflow Update
**Theme:** Implement the selected Windows install/downstream workflow changes
without widening unsupported package claims
**Time estimate:** 12 hours

### Tasks
1. Edit the Windows workflow job names, comments, and step labels according to
   the Day 5 design.
2. Add or split Windows install/downstream checks required by the Day 4 product
   decision.
3. Preserve existing Windows CMake configure/build/test lanes.
4. Keep Unix Makefile and `pkg-config` parity out of Windows reviewed wording.
5. Run local workflow syntax and command-shape review where feasible.
6. Record implementation details in working notes.

### Deliverables
- updated Windows workflow lane
- reviewed/supplemental support-tier wording in CI
- local syntax or command-shape review evidence
- implementation notes

### Completion Criteria
- workflow names and comments match the Day 4 decision
- no unsupported Windows Makefile, package-manager, or shared-library claim is
  introduced
- hosted validation requirements are ready for Day 13 closeout

---

## Day 7: Package Metadata Check Design

**Title:** Metadata Design
**Theme:** Design stronger Windows CMake package, static archive, header,
version, and unsupported-artifact checks
**Time estimate:** 12 hours

### Tasks
1. Review installed Windows package layout for static `.lib`, headers, CMake
   package files, and `sparse.pc`.
2. Define checks for CMake imported-target metadata that reject shared-library
   imports and unsupported DLL references.
3. Define `sparse.pc` text checks that preserve static-first wording without
   requiring Windows `pkg-config` execution.
4. Define header-count, version-file, exact-version, and mismatch-version
   expectations.
5. Decide how failure messages should distinguish package metadata issues from
   downstream build issues.
6. Write the metadata check design artifact.

### Deliverables
- Windows package metadata check design
- shared-artifact rejection rules
- static-first `sparse.pc` text-check rules
- version behavior checklist
- failure-message plan

### Completion Criteria
- each strengthened check has a concrete command or PowerShell assertion
- text checks avoid unsupported package-manager or shared-ABI wording
- downstream consumer proof remains separate from package-file inspection

---

## Day 8: Package Metadata Implementation

**Title:** Metadata Checks
**Theme:** Implement stronger Windows package metadata assertions in the
install/downstream lane
**Time estimate:** 12 hours

### Tasks
1. Add static `.lib`, DLL absence, header-count, and package-file assertions as
   needed.
2. Add CMake package metadata checks for imported-target type and shared
   artifact references.
3. Add `sparse.pc` static-first description and unsupported-wording checks
   without invoking Windows `pkg-config`.
4. Add version-file and exact package-version setup required for Day 9 consumer
   proof.
5. Review all PowerShell path handling for spaces, Windows separators, and
   multiline output.
6. Record before/after check coverage in the Day 8 artifact.

### Deliverables
- strengthened Windows package metadata checks
- static-first package text assertions
- shared-artifact rejection assertions
- before/after coverage table
- updated working notes

### Completion Criteria
- Windows install proof checks all required installed package artifacts
- shared-library artifacts and unsupported wording fail explicitly
- local static review shows path handling is robust

---

## Day 9: Downstream Consumer Proof Design

**Title:** Consumer Design
**Theme:** Design maintained Windows downstream consumer checks for normal,
exact-version, and mismatch-version behavior
**Time estimate:** 12 hours

### Tasks
1. Review `examples/cmake_example` and current installed-package consumer
   behavior.
2. Define the normal downstream consumer configure, build, run, and output
   checks.
3. Define exact-version consumer behavior using installed
   `SparseConfigVersion.cmake`.
4. Define mismatch-version behavior that must fail closed.
5. Define output matching that handles PowerShell arrays and multiline program
   output.
6. Write the downstream consumer proof design artifact.

### Deliverables
- downstream consumer proof design
- exact-version proof plan
- mismatch-version fail-closed plan
- output-matching rules
- source/build directory layout plan

### Completion Criteria
- consumer proof covers configure, build, run, and expected output
- version mismatch semantics are explicit and fail closed
- multiline output and `$LASTEXITCODE` handling are specified

---

## Day 10: Downstream Consumer Implementation

**Title:** Consumer Proof
**Theme:** Implement maintained Windows downstream consumer checks for normal
and versioned CMake package usage
**Time estimate:** 12 hours

### Tasks
1. Add or strengthen the normal installed CMake example configure/build/run
   proof.
2. Add exact-version consumer configure/build/run proof.
3. Add mismatch-version configure failure proof.
4. Normalize PowerShell output capture, text matching, and `$LASTEXITCODE`
   reset behavior.
5. Keep generated temporary source trees outside the repository.
6. Record implementation details and expected hosted evidence.

### Deliverables
- maintained Windows downstream consumer proof
- exact-version consumer proof
- mismatch-version fail-closed proof
- robust PowerShell output handling
- implementation artifact

### Completion Criteria
- downstream proof exercises installed package files rather than in-tree targets
- exact-version consumer builds and runs through the installed package
- mismatch-version consumer fails configure as expected

---

## Day 11: Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align README, INSTALL, maintainer guide, and report-family rows with
the selected Windows install-validation support tier
**Time estimate:** 12 hours

### Tasks
1. Update INSTALL Windows package wording to match the Day 4 decision.
2. Update README support-tier wording without claiming Unix Makefile,
   `pkg-config`, package-manager, or shared-library parity on Windows.
3. Update maintainer guidance for Windows CMake install/downstream proof and
   hosted validation expectations.
4. Update report-family rows or planning artifacts that describe package proof
   tiers.
5. Preserve static-first and no-shared-ABI wording.
6. Write the documentation alignment artifact.

### Deliverables
- updated public install/support wording
- updated maintainer package-proof guidance
- updated report/planning rows
- documentation alignment artifact
- unsupported-claim checklist

### Completion Criteria
- public docs describe exactly the selected Windows install-validation tier
- unsupported package-manager, shared-library, and `pkg-config` claims remain
  absent
- maintainers can identify which Windows lane is reviewed or supplemental

---

## Day 12: Local Validation And Syntax Review

**Title:** Local Validation
**Theme:** Run feasible local checks for workflow syntax, package wording,
documentation consistency, and changed files
**Time estimate:** 12 hours

### Tasks
1. Run workflow YAML syntax or structural checks available in the local
   toolchain.
2. Run focused searches for unsupported Windows install claims, package-manager
   claims, shared-library wording, and stale supplemental/reviewed labels.
3. Run documentation lint or whitespace checks used by the repository.
4. Run affected local Make/CMake checks that do not require hosted Windows.
5. Run full quality gates if any `.c` or `.h` files changed.
6. Write the local validation artifact.

### Deliverables
- local validation log
- workflow syntax or structure evidence
- unsupported-claim search results
- focused affected-check results
- quality-gate status

### Completion Criteria
- local validation commands and results are recorded
- any `.c` or `.h` change has the required full quality gate result
- hosted-only Windows evidence remains clearly marked for Day 13

---

## Day 13: Integrated Evidence Review

**Title:** Evidence Review
**Theme:** Review integrated Sprint 149 evidence, hosted CI expectations, and
remaining residuals before closeout
**Time estimate:** 12 hours

### Tasks
1. Re-read Sprint 149 artifacts for consistency across decision, workflow,
   metadata, consumer, docs, and validation rows.
2. Compare final workflow names and docs against the Day 4 product decision.
3. Review hosted Windows CI status if available and record pass/fail links or
   pending evidence requirements.
4. Re-run focused stale-wording searches after documentation updates.
5. Identify any residuals that must move to Sprint 150 or later.
6. Write the integrated evidence review artifact.

### Deliverables
- integrated evidence review
- hosted CI evidence status
- stale-wording search results
- residual list
- Sprint 150 handoff candidates

### Completion Criteria
- Sprint 149 claims are backed by explicit evidence or marked residual
- hosted-only proof gaps are not hidden as local success
- Sprint 150 QR work is not blocked by package-lane ambiguity

---

## Day 14: Sprint Closeout And Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 149 artifacts, working notes, residuals, and Sprint
150 QR corpus handoff
**Time estimate:** 10 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 149 artifacts and ensure links point to current paths.
3. Prepare Sprint 149 retrospective inputs: completed work, validation, claim
   changes, residuals, and follow-up risks.
4. Write the Sprint 150 QR corpus handoff.
5. Run final `git status`, whitespace, and stale-reference checks.
6. Record closeout summary.

### Deliverables
- finalized Sprint 149 working notes
- complete Sprint 149 artifact set
- Sprint 149 residual and validation summary
- Sprint 150 QR corpus handoff
- final closeout checklist

### Completion Criteria
- Sprint 149 product decision and evidence are ready for retrospective
- residuals are explicit and assigned to later sprint candidates
- branch is clean except for intentional Sprint 149 changes
