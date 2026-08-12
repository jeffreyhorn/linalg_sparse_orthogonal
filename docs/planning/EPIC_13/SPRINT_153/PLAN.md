# Sprint 153 Plan: Shared-Library ABI Product Decision

**Sprint Duration:** 14 days
**Goal:** Make a product-level shared-library ABI decision and either implement
the first supported shared surface or publish a stronger static-first deferral
with exact blockers. This sprint implements the Sprint 153 section of
`docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 153 begins from:
- Sprint 149 package parity decision available
- static-first package proof remains green
- public header and symbol surface previously inventoried at a high level
- Sprint 152 generated-report freshness handoff available
- current CMake configure path rejects `BUILD_SHARED_LIBS=ON`
- install documentation explicitly describes the maintained static-first
  package surface and defers shared-library support

The sprint must:
- audit public ABI-relevant headers, structs, macros, symbols, version
  metadata, static globals, allocator behavior, and callback contracts
- audit Linux, macOS, and Windows shared-loader requirements
- decide whether Sprint 153 implements a first supported shared surface or
  strengthens the static-first deferral
- implement the selected product decision
- add downstream proof for CMake, `pkg-config`, loader behavior, or unsupported
  artifact handling according to that decision
- update package, ABI, install, CMake, and maintainer documentation
- run install/package validation and full quality gates if code or headers
  change
- leave Sprint 154 a clean external-comparison handoff

**End State:** Sprint 153 leaves behind:
- shared-library ABI product decision record
- implemented shared support or stronger static-first deferral proof
- downstream package and consumer tests for the selected decision
- updated package/ABI documentation and non-claims
- explicit residuals and Sprint 154 comparison handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 153 project-plan estimate.

---

## Day 1: Sprint Intake And ABI Baseline

**Title:** ABI Intake
**Theme:** Establish Sprint 153 scope, artifact structure, static-first
baseline, and shared-library decision stop conditions
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 153 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review the Sprint 152 ABI/package handoff and non-claims.
3. Create Sprint 153 working notes and artifact directory structure.
4. Inventory current package/install surfaces: Make install, CMake install,
   exported targets, `pkg-config`, headers, version files, and CI package lanes.
5. Capture current shared-library deferral behavior, especially
   `BUILD_SHARED_LIBS=ON` rejection and static-first documentation.
6. Define stop conditions for accidental shared-library, ABI, loader, package,
   platform, or release claims.

### Deliverables
- Sprint 153 working-notes baseline
- artifact directory structure
- static-first package baseline
- shared-library deferral snapshot
- decision stop-condition register

### Completion Criteria
- Sprint 153 scope is tied to current repository files and Sprint 152 handoff
- current static-first package behavior is explicitly recorded
- unsupported ABI/shared claims are blocked before audit work begins

---

## Day 2: Public ABI Surface Audit

**Title:** ABI Surface
**Theme:** Inventory public symbols, headers, types, macros, version metadata,
and behavior that would become ABI-relevant under shared-library support
**Time estimate:** 12 hours

### Tasks
1. Inventory installed public headers and generated `sparse_version.h`.
2. Inventory exported public functions, public structs, enum-like macros, and
   version macros.
3. Identify public data layout, ownership, lifetime, allocator, callback, and
   error-code contracts.
4. Identify static globals, internal symbols, and accidental export risks.
5. Compare installed headers against source/internal headers.
6. Write the ABI surface audit artifact.

### Deliverables
- public header inventory
- public symbol and type inventory
- allocator/lifetime/callback contract notes
- accidental export risk list
- ABI surface audit artifact

### Completion Criteria
- every installed public header has an ABI relevance classification
- public symbols and public data layouts have owner candidates
- accidental export risks are visible before loader audit begins

---

## Day 3: Platform Loader Audit

**Title:** Loader Audit
**Theme:** Review Linux, macOS, and Windows shared-library requirements,
metadata, naming, install layout, and loader-test obligations
**Time estimate:** 12 hours

### Tasks
1. Audit Linux `.so` expectations: SONAME, RPATH/RUNPATH, symbol visibility,
   install layout, and downstream link behavior.
2. Audit macOS `.dylib` expectations: install name, framework-free packaging,
   exported symbols, and downstream link behavior.
3. Audit Windows `.dll` expectations: import library, export macros,
   `__declspec`, runtime lookup, and CMake generator behavior.
4. Identify platform-specific tests required for supported shared-library
   claims.
5. Identify toolchain constraints for GCC, Clang, Apple Clang, and MSVC.
6. Write the platform loader audit artifact.

### Deliverables
- Linux loader requirement inventory
- macOS loader requirement inventory
- Windows loader/import-library inventory
- platform-specific proof matrix
- loader audit artifact

### Completion Criteria
- each platform has explicit loader proof requirements
- Windows export/import risks are understood before product decision
- unsupported platform claims remain explicitly blocked

---

## Day 4: Product Decision Criteria

**Title:** Decision Criteria
**Theme:** Define the decision matrix for implementing shared support versus
publishing a stronger static-first deferral
**Time estimate:** 12 hours

### Tasks
1. Convert Days 2-3 findings into decision criteria.
2. Define minimum viable shared-library support requirements if implemented.
3. Define minimum viable static-first deferral requirements if deferred.
4. Score feasibility, risk, platform coverage, test cost, documentation cost,
   and user value.
5. Define rollback rules for incomplete loader, ABI, export, or downstream
   proof.
6. Write the product decision criteria artifact.

### Deliverables
- shared-support acceptance criteria
- static-deferral acceptance criteria
- risk and feasibility scorecard
- rollback rules
- decision criteria artifact

### Completion Criteria
- Sprint 153 can make a defensible product decision on Day 5
- implementation and deferral paths are both concrete enough to execute
- unsupported claims have explicit rollback triggers

---

## Day 5: Product Decision Record

**Title:** Product Decision
**Theme:** Decide whether Sprint 153 implements shared-library support or
strengthens static-first deferral with exact blockers
**Time estimate:** 12 hours

### Tasks
1. Review ABI surface audit, loader audit, and decision criteria.
2. Select the Sprint 153 product path: supported shared surface or stronger
   static-first deferral.
3. Define selected implementation scope, out-of-scope claims, and proof owners.
4. Define test, documentation, CI, and package metadata changes for the
   selected path.
5. Define exact blockers and residuals for the non-selected path.
6. Write the product decision record artifact.

### Deliverables
- shared-library ABI product decision
- selected implementation scope
- proof-owner map
- non-claim and residual register
- Day 6 implementation checklist

### Completion Criteria
- the sprint has one selected product path
- selected path is small enough to close in the remaining days
- non-selected claims are explicitly deferred with blockers

---

## Day 6: Build And Install Design

**Title:** Build Design
**Theme:** Design build-system, install, export, and metadata changes for the
selected product decision
**Time estimate:** 12 hours

### Tasks
1. Inspect Makefile, CMake, install scripts, pkg-config template, and CMake
   package export files.
2. Design selected build behavior: shared target support or stronger rejection
   diagnostics.
3. Design install behavior for library artifacts, headers, package metadata,
   and unsupported artifact absence.
4. Design CMake target/export behavior for `Sparse::sparse_lu_ortho`.
5. Design `pkg-config` behavior and static/shared wording.
6. Write the build/install design artifact.

### Deliverables
- build-system design
- install/export design
- CMake target behavior design
- `pkg-config` behavior design
- Day 7 implementation checklist

### Completion Criteria
- selected build/install changes are specified before editing
- package metadata behavior is clear for downstream consumers
- unsupported artifact handling is testable

---

## Day 7: Build And Install Implementation

**Title:** Build Batch
**Theme:** Implement selected build, install, export, and metadata changes
without widening unsupported claims
**Time estimate:** 12 hours

### Tasks
1. Implement selected Makefile and/or CMake behavior.
2. Implement install/export/package metadata changes.
3. Preserve or strengthen unsupported shared-library diagnostics if deferring.
4. Preserve static-first install behavior unless the product decision explicitly
   adds shared support.
5. Update or add focused build/install tests for changed behavior.
6. Run focused build and install validation.

### Deliverables
- selected build/install implementation
- package metadata implementation
- focused install/export tests
- unsupported artifact checks
- focused validation result

### Completion Criteria
- selected build/install behavior works locally
- unsupported artifacts are rejected or absent as designed
- focused validation passes before downstream proof design begins

---

## Day 8: Downstream Proof Design

**Title:** Consumer Design
**Theme:** Design downstream CMake, `pkg-config`, loader, and unsupported
artifact proof for the selected product decision
**Time estimate:** 12 hours

### Tasks
1. Inspect existing install tests and example downstream consumers.
2. Design downstream proof for CMake installed consumers.
3. Design downstream proof for `pkg-config` consumers where platform-feasible.
4. Design loader or unsupported-loader checks for Linux, macOS, and Windows.
5. Define exact output, diagnostics, and failure modes.
6. Write the downstream proof design artifact.

### Deliverables
- downstream CMake proof design
- downstream `pkg-config` proof design
- loader or unsupported-loader proof design
- platform feasibility matrix
- Day 9 implementation checklist

### Completion Criteria
- downstream proof matches selected product decision
- platform limitations are explicit
- proof commands are ready to implement

---

## Day 9: Downstream Proof Implementation

**Title:** Consumer Proof
**Theme:** Implement downstream consumer and loader/unsupported proof for the
selected package/ABI decision
**Time estimate:** 12 hours

### Tasks
1. Add or update CMake install/downstream proof.
2. Add or update `pkg-config` downstream proof.
3. Add or update loader checks, or unsupported shared-artifact rejection checks.
4. Ensure tests verify absence or presence of expected library artifacts.
5. Add failure diagnostics that name the product decision and remediation.
6. Run focused downstream proof validation.

### Deliverables
- downstream CMake proof
- downstream `pkg-config` proof
- loader or unsupported-artifact proof
- diagnostics for failed downstream proof
- focused validation result

### Completion Criteria
- downstream consumers prove the selected package decision
- unsupported artifacts cannot pass silently
- focused downstream proof passes

---

## Day 10: Platform Lane Review And CI Policy

**Title:** Platform Policy
**Theme:** Decide hosted CI follow-through for selected package/ABI proof while
preserving platform-specific claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Audit Linux, macOS, and Windows workflow package/ABI lanes.
2. Decide whether selected proof runs locally only, in hosted CI, or both.
3. Define expected CTest counts, install proof behavior, and platform
   exclusions if workflows change.
4. Define artifact upload, retention, and unsupported artifact policy.
5. Preserve Windows Makefile and Windows `pkg-config` non-claims unless
   explicitly closed.
6. Write the platform CI policy artifact.

### Deliverables
- platform CI policy
- local versus hosted proof matrix
- expected test-count or workflow-scope changes
- platform non-claim register
- Day 11 implementation checklist

### Completion Criteria
- CI changes are selected or explicitly deferred
- platform claims are narrower than or equal to actual proof
- expected hosted behavior is reviewable before implementation

---

## Day 11: CI And Documentation Implementation

**Title:** Docs And CI
**Theme:** Implement selected CI follow-through and align package, ABI, install,
CMake, and maintainer documentation
**Time estimate:** 12 hours

### Tasks
1. Implement selected CI workflow changes or record explicit deferral.
2. Update README, INSTALL, maintainer guide, CMake docs, package metadata
   comments, and examples as needed.
3. Update report-family metadata if proof ownership or claim boundaries change.
4. Align static-first or shared-library wording across active docs.
5. Search for stale ABI/package/shared-library wording.
6. Run focused docs, schema, and package validation.

### Deliverables
- CI implementation or deferral evidence
- updated ABI/package documentation
- updated package metadata comments
- stale wording search result
- focused validation result

### Completion Criteria
- documentation matches selected product decision
- no active doc claims unsupported shared-library or ABI behavior
- focused validation passes after docs/CI changes

---

## Day 12: Integrated Package And ABI Validation

**Title:** Integrated Proof
**Theme:** Run integrated install, export, downstream, package, ABI, and
report-index validation for the selected decision
**Time estimate:** 12 hours

### Tasks
1. Run Make install/package validation.
2. Run CMake install/export validation.
3. Run downstream CMake and `pkg-config` consumer checks where platform-feasible.
4. Run loader or unsupported shared-artifact checks according to the product
   decision.
5. Run package, CI, and runtime-backend report-index freshness checks.
6. Record integrated validation evidence.

### Deliverables
- Make install validation result
- CMake install/export validation result
- downstream consumer validation result
- loader or unsupported artifact validation result
- integrated validation artifact

### Completion Criteria
- selected package/ABI behavior passes integrated local validation
- unsupported paths fail clearly or remain absent as designed
- report-index package/CI/runtime rows preserve evidence meaning

---

## Day 13: Full Quality Gate And Residual Review

**Title:** Quality Gate
**Theme:** Run required quality gates, review residual package/ABI debt, and
prepare closeout
**Time estimate:** 12 hours

### Tasks
1. Determine whether `.c` or `.h` files changed during Sprint 153.
2. Run `make format && make lint && make test` if `.c` or `.h` files changed.
3. If only build/docs/scripts/tests changed, run focused package/report/doc
   checks and record why the C gate is not required.
4. Review residual shared-library, ABI, package, platform, and loader debt.
5. Run final whitespace, stale-reference, install, and report-index checks.
6. Record Day 13 quality-gate and residual-review artifact.

### Deliverables
- full quality-gate or focused-gate evidence
- residual ABI/package debt register
- final whitespace/stale-reference results
- Day 13 validation artifact

### Completion Criteria
- all required quality checks pass
- unresolved failures are fixed or explicitly escalated before closeout
- residual shared-library and package claims are assigned to later sprint
  candidates

---

## Day 14: Closeout And Sprint 154 Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 153 artifacts, validation status, residuals, and the
Sprint 154 external-comparison handoff
**Time estimate:** 12 hours

### Tasks
1. Finalize `WORKING_NOTES.md` with day-by-day completion notes and validation
   status.
2. Finalize all Sprint 153 artifacts and ensure links point to current paths.
3. Prepare Sprint 153 retrospective inputs: product decision, implementation,
   validation, claim changes, residuals, and follow-up risks.
4. Write the Sprint 154 external-comparison handoff.
5. Run final `git status`, whitespace, stale-reference, install/package,
   report-index, and quality checks required by changed files.
6. Record closeout summary.

### Deliverables
- finalized Sprint 153 working notes
- complete Sprint 153 artifact set
- shared-library ABI product decision summary
- Sprint 154 external-comparison handoff
- final closeout checklist

### Completion Criteria
- Sprint 153 shared-library ABI product decision is ready for retrospective
- package/ABI residuals are explicit and assigned
- branch is clean except for intentional Sprint 153 changes
- selected package/ABI evidence boundary is clear
- Sprint 154 external-comparison handoff is prepared
