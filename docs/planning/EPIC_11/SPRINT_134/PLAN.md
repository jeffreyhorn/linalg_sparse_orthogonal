# Sprint 134 Plan: Cross-Platform Install, Windows Staged Lanes & CI Tier Follow-Through

**Sprint Duration:** 14 days
**Goal:** Advance platform support where feasible and make Linux/macOS/Windows
install and staged validation tiers even more explicit.

**Starting Point:** Sprint 134 begins from:
- Sprint 133 static-first package/ABI decision and local package proof gates
- current CI tier model for Linux, macOS, and Windows
- local Make install/`pkg-config` proof in `tests/test_install.sh`
- local CMake install/export and `find_package(Sparse)` proof in
  `tests/test_cmake_install.sh`
- static-first deferral proof in `scripts/static_package_deferral_check.sh`
- current workflow comments, staged Windows exclusions, CTest counts, and
  platform support wording

The sprint must:
- re-audit Linux install CI, macOS install/export parity, Windows install
  validation, Windows thread/fuzz/property staging, and Windows Makefile gaps
- decide whether Linux install proof should be promoted to reviewed CI or stay
  local/supplemental
- add, strengthen, or explicitly defer reviewed macOS CMake install/export
  parity
- design and implement or defer MSVC install/downstream consumer proof with
  exact CTest count implications
- revisit Windows staged test exclusions and CTest membership without
  silently widening support tiers
- run workflow-equivalent local checks, package proof, documentation hygiene,
  and required quality gates for touched surfaces
- publish the final platform tier and staged-exclusion register

**End State:** Sprint 134 leaves behind:
- platform gap audit
- Linux install CI decision package
- macOS install/export parity decision package
- Windows install validation design or deferral package
- Windows staged-lane decision package
- updated support-tier docs and workflow comments if implementation changes
  require them
- validation and non-claim evidence
- final platform tier and staged-exclusion register

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 134 project-plan estimate.

---

## Day 1: Platform Install Sprint Intake

**Title:** Platform Intake
**Theme:** Establish Sprint 134 scope, artifact structure, platform tiers, and
claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 134 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Sprint 133 package/ABI closeout, retrospective, and static-first
   proof gates.
3. Inventory current Linux, macOS, and Windows workflow files, expected CTest
   counts, staged exclusions, install proofs, and support docs.
4. Map Sprint 134 project-plan Items 1-7 to day-level owners.
5. Create the Sprint 134 working-notes baseline and artifact directory.
6. Record claim fences for reviewed CI, supplemental CI, local proof, staged
   exclusions, deferred install parity, and unsupported platform behavior.

### Deliverables
- Sprint 134 working-notes baseline
- artifact directory structure
- platform workflow and validation surface inventory
- item-to-day owner map
- support-tier and non-claim boundary notes

### Completion Criteria
- every Sprint 134 project-plan item has a day-level owner
- Sprint 133 static-first package truth is preserved before platform changes
- Linux, macOS, and Windows support tiers are visible before decisions begin

---

## Day 2: Platform Gap Audit

**Title:** Gap Audit
**Theme:** Re-audit Linux install CI, macOS install/export parity, Windows
install validation, Windows staged tests, and Windows Makefile gaps
**Time estimate:** 12 hours

### Tasks
1. Inspect Linux workflow install/package coverage and compare it with local
   package proof scripts.
2. Inspect macOS workflow install/package coverage, including supplemental
   Make install/`pkg-config` lanes and any CMake install/export gaps.
3. Inspect Windows workflow coverage, CMake build/test registration, and
   staged exclusions for thread, fuzz, property, and Makefile lanes.
4. Compare workflow comments, maintainer docs, INSTALL, README, and prior
   sprint artifacts for tier wording drift.
5. Record current CTest counts, expected exclusions, and install proof owners.
6. Write the platform gap audit artifact.

### Deliverables
- Linux install CI gap list
- macOS install/export parity gap list
- Windows install-validation and staged-exclusion gap list
- Windows Makefile gap notes
- support wording drift queue

### Completion Criteria
- each platform has reviewed, supplemental, local, staged, and deferred
  evidence classified
- CTest counts and staged exclusions are recorded before changes
- install parity gaps are separated from package-contract gaps

---

## Day 3: Linux Install CI Decision

**Title:** Linux CI Decision
**Theme:** Decide whether Linux install proof should become reviewed CI or
remain local/supplemental
**Time estimate:** 12 hours

### Tasks
1. Review Linux workflow runtime budget, existing reviewed lanes, and local
   package proof cost.
2. Compare `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
   `scripts/static_package_deferral_check.sh` against reviewed CI criteria.
3. Identify risks of CI promotion: duration, tool availability, flakiness,
   artifact paths, and failure triage ownership.
4. Decide whether to promote, partially promote, or explicitly defer Linux
   install proof.
5. Define validation required for the selected Linux decision.
6. Write the Linux install CI decision artifact.

### Deliverables
- Linux reviewed-CI promotion decision
- runtime and toolchain risk notes
- selected implementation or deferral plan
- workflow comment update queue
- validation plan for Linux install proof

### Completion Criteria
- Linux install proof has an explicit support tier
- reviewed CI promotion is either scoped for implementation or deferred with
  evidence
- no local package proof is silently described as reviewed CI

---

## Day 4: Linux Install CI Implementation or Deferral

**Title:** Linux CI Batch
**Theme:** Implement the selected Linux install CI decision or publish an
explicit deferral with docs alignment
**Time estimate:** 12 hours

### Tasks
1. If selected, update Linux workflows to run the bounded install/package proof
   lane.
2. If deferred, update workflow comments or support docs to keep Linux install
   proof local and evidence-bounded.
3. Preserve static-first support and avoid adding package-manager or shared
   library claims.
4. Run workflow-equivalent local commands for any changed Linux install lane.
5. Update maintainer or install docs if support-tier wording changes.
6. Write the Linux implementation or deferral artifact.

### Deliverables
- workflow update or explicit deferral artifact
- Linux install proof command evidence
- docs or workflow comment updates
- residual Linux CI queue
- rollback notes

### Completion Criteria
- selected Linux decision is reflected in code/docs
- workflow-equivalent local evidence exists for touched Linux surfaces
- support wording still distinguishes reviewed, supplemental, and local proof

---

## Day 5: macOS Install/Export Parity Audit

**Title:** macOS Audit
**Theme:** Audit macOS static-first Make install, pkg-config, and CMake
install/export parity against current support tiers
**Time estimate:** 12 hours

### Tasks
1. Review macOS CI workflow lanes, Apple Clang ownership, Homebrew GCC
   supplemental lanes, and existing install proof.
2. Compare macOS Make install/`pkg-config` confidence with CMake
   install/export proof requirements.
3. Identify toolchain constraints for `cmake --install`, `find_package`, and
   pkg-config behavior on macOS runners.
4. Compare docs and maintainer guide wording with observed macOS support tiers.
5. Record macOS parity options: reviewed CMake install/export, supplemental
   proof, or explicit deferral.
6. Write the macOS install/export parity audit artifact.

### Deliverables
- macOS install/export parity audit
- Apple Clang and supplemental lane notes
- CMake install/export gap list
- toolchain and runtime risk notes
- macOS support wording queue

### Completion Criteria
- macOS Make install and CMake install/export support are separated
- parity blockers and feasible proof paths are visible
- macOS support tier wording remains evidence-bounded

---

## Day 6: macOS Install/Export Decision

**Title:** macOS Decision
**Theme:** Decide whether to add, strengthen, or defer reviewed macOS CMake
install/export parity
**Time estimate:** 12 hours

### Tasks
1. Evaluate Day 5 macOS parity options against runtime budget, tool
   availability, and support ownership.
2. Decide whether macOS CMake install/export proof should be reviewed,
   supplemental, local-only, or deferred.
3. Define exact commands and expected pass criteria for any selected macOS
   implementation.
4. Define docs and workflow comment updates for the selected macOS tier.
5. Record rollback and failure-triage expectations.
6. Write the macOS install/export decision artifact.

### Deliverables
- macOS CMake install/export decision
- selected command plan or deferral rationale
- docs and workflow update plan
- triage and rollback notes
- residual macOS package queue

### Completion Criteria
- macOS install/export parity has an explicit decision
- selected support tier is reflected in planned validation
- no macOS install parity is implied without matching proof

---

## Day 7: macOS Install/Export Implementation or Deferral

**Title:** macOS Batch
**Theme:** Implement selected macOS install/export changes or publish explicit
deferral and support-tier documentation
**Time estimate:** 12 hours

### Tasks
1. If selected, update macOS workflows with bounded CMake install/export proof.
2. If deferred, update comments/docs to state the reviewed/supplemental/local
   split clearly.
3. Preserve existing Apple Clang, Homebrew GCC, sanitizer, and supplemental
   confidence lanes unless intentionally changed.
4. Run workflow-equivalent local checks for touched macOS package surfaces.
5. Update maintainer or install docs if support-tier wording changes.
6. Write the macOS implementation or deferral artifact.

### Deliverables
- macOS workflow or documentation update
- local workflow-equivalent evidence
- support-tier wording updates
- residual macOS install/export queue
- rollback notes

### Completion Criteria
- selected macOS decision is implemented or explicitly deferred
- workflow-equivalent evidence exists for touched macOS surfaces
- macOS support claims remain narrower than Linux unless reviewed proof exists

---

## Day 8: Windows Install Validation Audit and Design

**Title:** Windows Install
**Theme:** Design MSVC install/downstream consumer proof or explicit deferral
with exact CTest count implications
**Time estimate:** 12 hours

### Tasks
1. Review Windows workflow, CMake generator use, install support, and current
   CTest membership.
2. Identify whether an MSVC install/downstream consumer proof can run within
   current Windows CI constraints.
3. Map expected changes to CTest counts, exclusions, generated install paths,
   and downstream consumer commands.
4. Identify Windows Makefile parity gaps separately from CMake-first install
   validation.
5. Decide whether to implement Windows install proof, stage it, or explicitly
   defer.
6. Write the Windows install validation design artifact.

### Deliverables
- Windows install validation design or deferral
- MSVC downstream consumer proof sketch
- exact CTest count impact notes
- Windows Makefile parity gap notes
- validation and support-tier plan

### Completion Criteria
- Windows install validation has an implementation or deferral decision
- CTest count implications are explicit before workflow edits
- Windows Makefile parity is not conflated with CMake-first support

---

## Day 9: Windows Install Validation Implementation or Deferral

**Title:** Windows Install Batch
**Theme:** Implement selected Windows install/downstream proof or publish
explicit deferral with support-tier alignment
**Time estimate:** 12 hours

### Tasks
1. If selected, update Windows CMake/CTest workflow or tests for install and
   downstream consumer proof.
2. If deferred, update docs/workflow comments to keep Windows install parity
   staged or unsupported.
3. Preserve existing reviewed Windows CMake subset unless intentionally
   modified.
4. Update expected CTest counts and staged-exclusion notes if membership
   changes.
5. Run local workflow-equivalent checks available on the current platform and
   document any Windows-only validation limits.
6. Write the Windows install implementation or deferral artifact.

### Deliverables
- Windows install proof implementation or deferral artifact
- workflow/test/docs updates if selected
- CTest count and exclusion updates
- local or bounded validation evidence
- Windows validation residual queue

### Completion Criteria
- selected Windows install decision is reflected in changed surfaces
- expected test membership impact is documented
- any Windows-only validation gap is explicit rather than hidden

---

## Day 10: Windows Staged Test Re-Audit

**Title:** Windows Staging Audit
**Theme:** Revisit Windows thread, fuzz, property, and staged-exclusion lanes
against current CTest membership
**Time estimate:** 12 hours

### Tasks
1. Inventory current Windows CTest membership and expected test count.
2. Re-audit staged exclusions for thread, fuzz, property, and integration
   tests.
3. Compare exclusion rationale with source/test maturity and workflow
   constraints.
4. Identify whether any staged test can move into the reviewed Windows subset.
5. Record tests that must remain staged, with blockers and future proof gates.
6. Write the Windows staged test re-audit artifact.

### Deliverables
- Windows CTest membership inventory
- staged-exclusion rationale table
- candidate promotion list
- blocker and proof-gate notes
- support wording update queue

### Completion Criteria
- every Windows staged exclusion has current rationale
- CTest count and candidate promotions are explicit
- staged tests are not silently described as reviewed

---

## Day 11: Windows Staged Lane Follow-Through

**Title:** Windows Staging Batch
**Theme:** Implement selected Windows staged-lane promotions or reinforce
explicit staged exclusions
**Time estimate:** 12 hours

### Tasks
1. If selected, update CMake/CTest or workflow membership for specific Windows
   staged-lane promotions.
2. If no promotions are selected, update docs/comments with current staged
   exclusion rationale.
3. Update expected CTest count documentation and validation notes.
4. Preserve thread/fuzz/property support boundaries unless proof is added.
5. Run local CMake registration checks and any available focused validation.
6. Write the Windows staged lane follow-through artifact.

### Deliverables
- staged-lane workflow/test updates or explicit no-op deferral
- expected CTest count update
- focused registration evidence
- support-tier documentation updates
- residual staged-lane queue

### Completion Criteria
- selected Windows staged-lane decision is implemented or explicitly deferred
- test membership and support docs agree
- no staged test is promoted without validation ownership

---

## Day 12: Support-Tier Documentation and CI Comment Alignment

**Title:** Tier Docs
**Theme:** Align README, INSTALL, maintainer guide, workflow comments, and
Sprint artifacts with final platform decisions
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 134 decisions from Days 3-11.
2. Update README, INSTALL, maintainer guide, workflow comments, and planning
   artifacts where support tiers changed or were clarified.
3. Ensure Linux, macOS, and Windows wording distinguishes reviewed,
   supplemental, local-only, staged, deferred, and unsupported evidence.
4. Preserve Sprint 133 static-first package/ABI non-claims.
5. Run support-claim drift scans across docs and workflow comments.
6. Write the support-tier documentation alignment artifact.

### Deliverables
- updated support-tier docs or explicit no-change record
- workflow comment alignment notes
- claim drift scan evidence
- platform support truth table
- residual wording queue

### Completion Criteria
- public and maintainer docs agree with implemented platform decisions
- no platform install parity or staged-lane claim is overstated
- Sprint 133 static-first package contract remains intact

---

## Day 13: Integrated Platform Validation

**Title:** Platform Validation
**Theme:** Run affected workflow-equivalent local checks, package proof, staged
lane checks, and required quality gates
**Time estimate:** 12 hours

### Tasks
1. Run syntax and focused checks for any changed shell scripts, workflows, or
   build/test files.
2. Run package proof gates affected by platform support changes:
   `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
   `scripts/static_package_deferral_check.sh` as applicable.
3. Run local CMake/CTest registration checks for touched Windows staged-lane
   or install-validation surfaces.
4. Run `make format && make lint && make test` if any `.c` or `.h` files were
   changed.
5. Run docs and workflow hygiene checks.
6. Write the integrated platform validation artifact.

### Deliverables
- integrated validation log
- workflow-equivalent evidence
- package proof evidence
- CTest count and staged-lane evidence
- unresolved validation residuals

### Completion Criteria
- every touched platform surface has matching validation evidence
- required quality gates pass or blockers are explicit
- validation evidence is ready for closeout and PR review

---

## Day 14: Platform Tier Closeout and Handoff

**Title:** Tier Closeout
**Theme:** Publish final platform tier, staged-exclusion register, residual
queues, and Sprint 135 handoff notes
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 134 artifacts, implementation changes, validation logs,
   workflow comments, and documentation updates.
2. Publish final Linux, macOS, and Windows support-tier truth.
3. Publish final staged-exclusion register with owners, blockers, and proof
   gates.
4. Record install parity, Windows Makefile, package-manager, and CI promotion
   residuals.
5. Prepare Sprint 135 handoff notes and PR review summary material.
6. Write the platform tier closeout and handoff artifact.

### Deliverables
- Sprint 134 closeout artifact
- final platform tier table
- staged-exclusion register
- residual install/platform queue
- Sprint 135 handoff notes
- PR review summary material

### Completion Criteria
- final platform support truth is clear to users and maintainers
- staged exclusions have owners, blockers, and support-tier boundaries
- Sprint 134 can close without unresolved workflow or support wording drift
