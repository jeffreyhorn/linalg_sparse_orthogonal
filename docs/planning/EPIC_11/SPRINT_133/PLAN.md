# Sprint 133 Plan: Package, ABI & Shared-Library Product Decision

**Sprint Duration:** 14 days
**Goal:** Decide whether Epic 11 adds shared-library/dynamic ABI support or
explicitly preserves static-first support, then implement the selected product
contract.

**Starting Point:** Sprint 133 begins from:
- Epic 10 static-first package support truth and install/export proof
- Sprint 118 package/ABI residual owner map and platform-package residuals
- Sprint 131 corpus/report index ownership and evidence-boundary decisions
- Sprint 132 backend/runtime governance, sentinel metadata, and non-claim
  boundaries
- current public headers, CMake install/export files, pkg-config output,
  Make install scripts, package documentation, and downstream consumer tests

The sprint must:
- audit public headers, symbol exposure, install layout, versioning,
  package metadata, and downstream consumer expectations
- decide whether shared-library/dynamic ABI support becomes an Epic 11 product
  contract or remains explicitly deferred behind static-first support
- implement the selected build/install contract without silently changing ABI,
  packaging, or support-tier claims
- add ABI/symbol/version proof if shared support is selected, or add static
  deferral checks if static-first remains selected
- strengthen downstream CMake and pkg-config consumer proof for the selected
  contract
- run install, package, source/build, and required quality gates for touched
  surfaces
- publish package/ABI support truth and residual package-manager work

**End State:** Sprint 133 leaves behind:
- package/ABI product decision artifact
- shared-library design or explicit static-first deferral artifact
- build/install contract implementation or enforcement update
- ABI/symbol/version proof or static-first deferral proof
- downstream CMake/pkg-config consumer validation
- updated README, install, maintainer, and package support documentation
- package/ABI residual queue and Sprint 134 handoff notes

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 133 project-plan estimate.

---

## Day 1: Package and ABI Sprint Intake

**Title:** ABI Intake
**Theme:** Establish Sprint 133 scope, artifact structure, decision gates, and
static-first claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 133 section of
   `docs/planning/EPIC_11/PROJECT_PLAN.md`.
2. Review Epic 10 static-first package truth, Sprint 118 package/ABI residuals,
   and Sprint 131-132 report/runtime support boundaries.
3. Inventory candidate package, ABI, install, versioning, and downstream
   consumer surfaces.
4. Map Sprint 133 project-plan Items 1-7 to day-level owners.
5. Create the Sprint 133 working-notes baseline and artifact directory.
6. Record duplicate fences so static install proof, package-manager support,
   and dynamic ABI support are not promoted silently.

### Deliverables
- Sprint 133 working-notes baseline
- artifact directory structure
- package/ABI source-area intake list
- item-to-day owner map
- static-first and dynamic-ABI non-claim boundary notes

### Completion Criteria
- every Sprint 133 project-plan item has a day-level owner
- inherited static-first support truth is preserved before new decisions
- package, ABI, install, and downstream consumer surfaces are visible before
  design or implementation begins

---

## Day 2: Public Header and Symbol Exposure Audit

**Title:** Header Audit
**Theme:** Audit exported headers, public names, version macros, and symbol
exposure risk before deciding the product contract
**Time estimate:** 12 hours

### Tasks
1. Inventory installed public headers and headers included by downstream
   consumer examples.
2. Identify public structs, enums, typedefs, macros, inline helpers, and
   function declarations that would become ABI-sensitive under shared-library
   support.
3. Review symbol visibility defaults, static inline behavior, and dependency
   exposure across public headers.
4. Identify version or feature macros currently available to consumers.
5. Record headers that are install-facing but not intended to be ABI contracts.
6. Write the public header and symbol exposure audit artifact.

### Deliverables
- public header inventory
- ABI-sensitive declaration map
- symbol visibility risk notes
- version and feature macro inventory
- non-public or install-accidental header queue

### Completion Criteria
- every installed header has intended support status or explicit unknown status
- ABI-sensitive declarations are separated from source-compatibility-only
  declarations
- shared-library risk is recorded before the design decision

---

## Day 3: Install Shape and Package Metadata Audit

**Title:** Install Audit
**Theme:** Audit Make install, CMake export, pkg-config metadata, static
library layout, and package documentation
**Time estimate:** 12 hours

### Tasks
1. Inspect Make install scripts, CMake install/export rules, pkg-config
   generation, and install tests.
2. Inventory installed libraries, headers, CMake package files, pkg-config
   fields, and documented install commands.
3. Compare actual install outputs with README, INSTALL, maintainer, and
   package-support documentation.
4. Identify current assumptions about static libraries, transitive
   dependencies, compiler flags, and link interfaces.
5. Record install-shape gaps that affect either static-first enforcement or
   shared-library support.
6. Write the install shape and package metadata audit artifact.

### Deliverables
- install layout inventory
- CMake export and pkg-config metadata map
- documentation drift notes
- static-link interface assumptions
- package metadata gap queue

### Completion Criteria
- install outputs and documented package support are reconciled
- current static-first behavior is distinguishable from accidental omission of
  shared support
- metadata gaps have owners or explicit deferral status

---

## Day 4: Downstream Consumer Expectation Audit

**Title:** Consumer Audit
**Theme:** Audit local and installed downstream consumer workflows that depend
on the package contract
**Time estimate:** 12 hours

### Tasks
1. Review CMake install/export consumer tests and pkg-config consumer tests.
2. Inventory local examples, smoke programs, benchmark builds, and maintainer
   scripts that model downstream usage.
3. Identify whether consumers rely on static archive names, include layout,
   compile definitions, transitive libraries, or build-tree assumptions.
4. Define what a downstream consumer must be able to prove for static-first
   support.
5. Define additional proof required before shared-library support could become
   reviewed.
6. Write the downstream consumer expectation audit artifact.

### Deliverables
- downstream consumer workflow inventory
- CMake and pkg-config proof map
- static consumer contract notes
- shared-library consumer proof requirements
- consumer gap queue

### Completion Criteria
- downstream expectations are evidence-backed, not inferred from package names
- static consumer proof and shared consumer proof are separated
- Day 5 can make a product decision with known consumer impact

---

## Day 5: Shared-Library Product Decision

**Title:** Product Decision
**Theme:** Decide whether Sprint 133 implements shared-library ABI support or
preserves explicit static-first support with stronger deferral proof
**Time estimate:** 12 hours

### Tasks
1. Review Day 2-4 audit artifacts and inherited package support boundaries.
2. Define decision criteria for shared-library support: ABI stability,
   versioning, symbol visibility, install behavior, downstream proof,
   validation cost, and support burden.
3. Apply the criteria to current package and source surfaces.
4. Select either shared-library implementation or static-first deferral as the
   Sprint 133 product contract.
5. Identify documentation and validation changes required by the selected
   contract.
6. Write the package/ABI product decision artifact.

### Deliverables
- package/ABI product decision
- decision criteria and evidence table
- selected contract statement
- rejected alternative notes
- implementation and validation touch-point list

### Completion Criteria
- the shared-library versus static-first decision is explicit
- support wording follows evidence rather than aspiration
- implementation days have a single selected contract to execute

---

## Day 6: Selected Contract Design

**Title:** Contract Design
**Theme:** Design either shared-library ABI support or static-first deferral
enforcement in implementation-ready detail
**Time estimate:** 12 hours

### Tasks
1. Translate the Day 5 decision into build, install, package metadata, and
   documentation requirements.
2. If shared support is selected, design library type controls, symbol
   visibility, ABI versioning, install names, and consumer metadata.
3. If static-first remains selected, design enforcement checks, error wording,
   package documentation, and explicit deferral markers.
4. Identify source, build-system, script, test, and documentation files to
   change.
5. Define rollback and validation criteria for the selected contract.
6. Write the selected contract design artifact.

### Deliverables
- selected contract design
- file-level implementation map
- ABI/version or deferral-check design
- downstream validation plan
- rollback and blocker notes

### Completion Criteria
- implementation work is sequenced and bounded
- shared and static-first paths are not mixed after the decision
- validation expectations are known before code or script changes begin

---

## Day 7: Build and Install Contract Batch

**Title:** Build Contract
**Theme:** Implement the first build/install contract batch for the selected
package decision
**Time estimate:** 12 hours

### Tasks
1. Apply selected build-system changes for static-first enforcement or
   shared-library support.
2. Update install/export rules, library targets, compile definitions, or
   package metadata needed by the selected contract.
3. Preserve existing static install behavior unless the Day 5 decision
   explicitly changes it.
4. Update local install scripts or generated package files touched by the
   contract.
5. Run focused build/install smoke checks for touched surfaces.
6. Write the Day 7 implementation artifact.

### Deliverables
- first build/install implementation batch
- focused build/install smoke evidence
- changed-surface notes
- static compatibility notes
- implementation residual queue

### Completion Criteria
- selected package contract is represented in build/install behavior
- existing static consumers are not broken without an explicit decision
- remaining implementation work is narrow and documented

---

## Day 8: Package Metadata and Documentation Batch

**Title:** Package Metadata
**Theme:** Align README, INSTALL, maintainer, CMake, and pkg-config package
truth with the selected contract
**Time estimate:** 12 hours

### Tasks
1. Update package-facing README and install documentation for the selected
   static-first or shared-library contract.
2. Update maintainer guidance for package support boundaries, validation
   commands, and release review expectations.
3. Update CMake package or pkg-config metadata documentation if the selected
   contract changes consumer interpretation.
4. Remove or reword ambiguous dynamic-library, ABI, or package-manager
   language.
5. Run documentation and metadata focused checks.
6. Write the package metadata and documentation artifact.

### Deliverables
- updated package support documentation
- package metadata wording changes
- maintainer support-truth update
- ambiguity cleanup notes
- focused documentation validation evidence

### Completion Criteria
- public support wording matches the Day 5 decision
- package metadata does not imply unsupported ABI or package-manager claims
- maintainers have clear validation commands for the selected contract

---

## Day 9: ABI, Symbol, or Deferral Proof Design

**Title:** Proof Design
**Theme:** Design the proof mechanism for ABI/symbol/version support or
static-first deferral enforcement
**Time estimate:** 12 hours

### Tasks
1. Review the selected contract and Day 7-8 changes.
2. If shared support is selected, design symbol export checks, ABI version
   checks, install-name checks, and downstream shared-link validation.
3. If static-first remains selected, design checks that prove shared-library
   support is not advertised or accidentally emitted.
4. Define expected outputs, failure messages, support tier, and owner for each
   proof.
5. Identify CI or local-only placement for each proof.
6. Write the ABI/symbol or deferral proof design artifact.

### Deliverables
- ABI/symbol/version proof design or static-first deferral proof design
- expected output and failure wording
- proof owner map
- local versus reviewed placement decision
- implementation touch-point list

### Completion Criteria
- proof checks trace directly to the selected contract
- failure output would prevent package-support drift
- local-only proofs are not mislabeled as reviewed CI support

---

## Day 10: ABI, Symbol, or Deferral Proof Implementation

**Title:** Proof Batch
**Theme:** Implement the selected proof checks and integrate them with local
package validation
**Time estimate:** 12 hours

### Tasks
1. Add ABI/symbol/version checks if shared-library support was selected, or
   static-first deferral checks if static-first remains selected.
2. Integrate the checks into existing package, install, or maintainer
   validation scripts without broadening unsupported claims.
3. Add fixtures, expected outputs, or smoke programs needed by the checks.
4. Update documentation that explains when and how the checks should run.
5. Run focused validation for the new proof mechanism.
6. Write the Day 10 proof implementation artifact.

### Deliverables
- implemented ABI/symbol/version proof or deferral proof
- updated validation script or test surface
- proof output examples
- documentation updates
- focused validation evidence

### Completion Criteria
- selected proof fails clearly when the contract drifts
- proof placement matches reviewed or local-only status
- package support wording remains evidence-bounded

---

## Day 11: Downstream CMake Consumer Proof

**Title:** CMake Consumer
**Theme:** Strengthen CMake install/export consumer proof for the selected
package contract
**Time estimate:** 12 hours

### Tasks
1. Review existing CMake install/export consumer tests against the selected
   contract.
2. Add or update CMake consumer checks for library type, imported target,
   include directories, compile definitions, and transitive link behavior.
3. Ensure the consumer proof does not rely on build-tree-only paths.
4. Update expected documentation or maintainer commands for the CMake proof.
5. Run the focused CMake install/export validation path.
6. Write the CMake downstream consumer proof artifact.

### Deliverables
- updated CMake consumer proof
- installed-package validation evidence
- imported-target and link-interface notes
- documentation or maintainer command updates
- residual CMake package queue

### Completion Criteria
- an installed CMake consumer can prove the selected contract
- build-tree leakage is detected or explicitly ruled out
- CMake package support remains aligned with public documentation

---

## Day 12: Downstream pkg-config Consumer Proof

**Title:** pkg-config Consumer
**Theme:** Strengthen pkg-config install consumer proof for the selected
package contract
**Time estimate:** 12 hours

### Tasks
1. Review existing pkg-config consumer tests and generated `.pc` fields.
2. Add or update pkg-config consumer checks for include flags, library flags,
   private dependencies, static link behavior, and selected library contract.
3. Verify that pkg-config output does not imply unsupported shared-library or
   package-manager support.
4. Update install or maintainer documentation for pkg-config validation.
5. Run focused pkg-config install validation.
6. Write the pkg-config downstream consumer proof artifact.

### Deliverables
- updated pkg-config consumer proof
- generated `.pc` field validation evidence
- static or shared link interpretation notes
- documentation or maintainer command updates
- residual pkg-config queue

### Completion Criteria
- an installed pkg-config consumer can prove the selected contract
- static/private dependency semantics are documented and tested
- unsupported package-manager or ABI claims remain fenced

---

## Day 13: Integrated Package Validation

**Title:** Package Validation
**Theme:** Run source/build, install, package, consumer, and required quality
gates for all touched Sprint 133 surfaces
**Time estimate:** 12 hours

### Tasks
1. Run source/build checks required by touched build-system, script, and C
   surfaces.
2. Run Make install and CMake install/export package validation.
3. Run pkg-config consumer validation.
4. Run ABI/symbol/version or static-first deferral proof checks.
5. Run full required quality gates when C source or header files were changed.
6. Write the integrated package validation artifact.

### Deliverables
- integrated validation log
- install/export validation evidence
- CMake and pkg-config consumer evidence
- ABI/symbol or deferral proof evidence
- unresolved validation residuals

### Completion Criteria
- every touched package surface has matching validation evidence
- required quality gates pass or blockers are explicit
- validation evidence is ready for closeout and PR review

---

## Day 14: Closeout and Package ABI Handoff

**Title:** ABI Closeout
**Theme:** Publish package/ABI support truth, residual package-manager work,
and Sprint 134 handoff notes
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 133 artifacts, implementation changes, validation logs,
   and documentation updates.
2. Publish the final selected package/ABI support truth in working notes and
   closeout artifacts.
3. Record package-manager, shared-library, ABI-versioning, platform-package,
   and downstream-consumer residuals.
4. Confirm README, INSTALL, maintainer, CMake, and pkg-config wording agree
   with the selected contract.
5. Prepare Sprint 134 handoff notes and PR review summary material.
6. Write the closeout and package ABI handoff artifact.

### Deliverables
- Sprint 133 closeout artifact
- package/ABI support truth summary
- package-manager and ABI residual queue
- Sprint 134 handoff notes
- PR review summary material

### Completion Criteria
- the selected package contract is clear to users and maintainers
- residual package/ABI work has owners, blockers, and support-tier boundaries
- Sprint 133 can close without unresolved support wording drift
