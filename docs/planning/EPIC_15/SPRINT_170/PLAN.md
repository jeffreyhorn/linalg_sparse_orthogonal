# Sprint 170 Plan: Shared-Library ABI Product Decision

**Sprint Duration:** 14 days
**Goal:** Close the shared-library ABI question with an explicit product
decision and enforceable package metadata behavior. This sprint implements the
Sprint 170 section of `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Sprint 170:
Shared-Library ABI Product Decision"; the active merged Sprint 170
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the same title.

**Starting Point:** Sprint 170 begins from:

- static-first package contract and install validation;
- Make install and CMake install/export proof for the maintained static
  archive package surface;
- package metadata guards that reject unsupported shared-library, ABI,
  runtime-loader, and package-manager wording;
- Epic 15 evidence-ledger non-claims for shared-library support, dynamic ABI
  stability, runtime-loader behavior, package-manager distribution, and broad
  platform parity;
- Sprint 169 performance-methodology handoff that keeps performance evidence
  independent from package, ABI, and runtime-loader claims.

The sprint must:

- audit exported headers, structs, constants, versioning, and lifecycle
  semantics for ABI readiness;
- review Make/CMake static-only behavior, shared-library feasibility, symbol
  visibility, install metadata, and package export surfaces;
- create a product decision record choosing static-first-only continuation or
  a staged shared-library path;
- update tests or guards so unsupported shared-library and ABI claims cannot
  appear accidentally;
- align README, install/package docs, maintainer docs, and non-claim tables
  with the selected decision;
- run install/package checks, focused guard checks, docs sanity checks, and the
  full C quality gate if `.c` or `.h` files change.

**End State:** Sprint 170 leaves behind:

- a source-controlled shared-library ABI product decision record;
- audited ABI-readiness evidence for public headers and lifecycle semantics;
- build-system and package metadata feasibility notes;
- updated guard behavior for unsupported shared-library and ABI claims;
- documentation aligned with the selected product decision;
- Sprint 170 working notes, daily artifacts, and validation records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 170 project-plan estimate.

---

## Day 1: Sprint Intake And Evidence Baseline

**Title:** ABI Intake
**Theme:** Establish Sprint 170 scope from Epic 15 plan item 170.1 through
170.6 and prior sprint handoffs
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 170 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 167 evidence-ledger non-claims for shared-library, dynamic
   ABI, package-manager, runtime-loader, and platform support.
3. Review Sprint 169 handoff boundaries so performance evidence is not reused
   as ABI/package evidence.
4. Create Sprint 170 working notes and artifact directory structure.
5. Record the prompt path/source-artifact mismatch.
6. Write the Day 1 ABI-intake artifact.

### Deliverables

- Sprint 170 working-notes baseline
- artifact directory structure
- source artifact note
- prior-sprint handoff summary
- ABI/product-decision stop conditions
- Day 1 ABI-intake artifact

### Completion Criteria

- Sprint 170 scope is tied to the active Epic 15 project plan
- retained package/ABI non-claims are explicit
- no shared-library or ABI support claim is introduced by planning

---

## Day 2: Public Header ABI Surface Inventory

**Title:** Header ABI Inventory
**Theme:** Inventory exported headers, public structs, enums, constants,
macros, and version symbols
**Time estimate:** 12 hours

### Tasks

1. Enumerate installed public headers and source-controlled API reference
   owners.
2. Inventory public structs, typedefs, enums, constants, macros, and function
   declarations.
3. Identify layout-exposed objects and lifecycle-managed handles.
4. Map version macros and runtime version APIs to ABI decision needs.
5. Record obvious ABI hazards, stable candidates, and ambiguous surfaces.
6. Write the Day 2 header-ABI inventory artifact.

### Deliverables

- installed header inventory
- public symbol/type inventory
- exposed layout and handle lifecycle map
- versioning surface notes
- Day 2 header-ABI inventory artifact

### Completion Criteria

- all installed public headers are accounted for
- ABI-relevant public declarations are mapped to owner files
- unresolved ABI hazards are listed for later decision

---

## Day 3: Lifecycle And Ownership Semantics Audit

**Title:** Lifecycle Audit
**Theme:** Review allocation, initialization, destruction, ownership transfer,
and error handling for ABI readiness
**Time estimate:** 12 hours

### Tasks

1. Review public create/destroy/free/reset APIs and ownership expectations.
2. Review error-code, null-pointer, and invalid-argument behavior at public
   entry points.
3. Identify public structs whose layout or allocator assumptions would become
   ABI commitments.
4. Review callback, workspace, and backend-selection lifecycle semantics.
5. Record ABI blockers and stable lifecycle contracts.
6. Write the Day 3 lifecycle-audit artifact.

### Deliverables

- lifecycle API map
- ownership and allocator assumption inventory
- error-handling ABI notes
- lifecycle ABI blocker list
- Day 3 lifecycle-audit artifact

### Completion Criteria

- handle ownership semantics are clear enough to support a product decision
- exposed layout and allocator risks are documented
- no undocumented ABI claim is added

---

## Day 4: Symbol And Visibility Feasibility

**Title:** Symbol Visibility
**Theme:** Review exported symbol behavior, internal helper leakage, naming,
and visibility controls
**Time estimate:** 12 hours

### Tasks

1. Inspect static archive object composition and public/internal naming
   conventions.
2. Review whether internal helpers could become exported from a shared build.
3. Inventory current compiler/linker visibility controls, if any.
4. Identify symbol-versioning, prefixing, or export-list requirements for a
   future shared-library path.
5. Record static-first implications for symbol visibility.
6. Write the Day 4 symbol-visibility artifact.

### Deliverables

- public/internal symbol boundary notes
- visibility-control inventory
- shared-build leakage risk list
- symbol-governance requirements
- Day 4 symbol-visibility artifact

### Completion Criteria

- symbol exposure risk is explicit
- future shared-library requirements are separated from current static support
- static-first behavior remains accurately described

---

## Day 5: Makefile Static-Only Feasibility Review

**Title:** Make Feasibility
**Theme:** Review Makefile build, install, uninstall, package metadata, and
unsupported shared-library behavior
**Time estimate:** 12 hours

### Tasks

1. Review Makefile library build rules and static archive ownership.
2. Review install/uninstall behavior and installed artifact lists.
3. Review generated `pkg-config` metadata for static-first wording and link
   flags.
4. Identify where a shared-library build would require explicit opt-in,
   export control, install behavior, and tests.
5. Record whether Makefile behavior supports static-first-only continuation or
   staged shared-library exploration.
6. Write the Day 5 Make-feasibility artifact.

### Deliverables

- Makefile static-only inventory
- install/uninstall package metadata notes
- `pkg-config` static-first review
- shared-library Makefile feasibility risks
- Day 5 Make-feasibility artifact

### Completion Criteria

- Makefile package behavior is mapped to the decision needs
- unsupported shared-library behavior remains guarded
- feasibility notes are ready for decision synthesis

---

## Day 6: CMake Package Feasibility Review

**Title:** CMake Feasibility
**Theme:** Review CMake static target, install/export package metadata,
consumer behavior, and shared-library implications
**Time estimate:** 12 hours

### Tasks

1. Review CMake library target type and install/export rules.
2. Review generated `SparseConfig.cmake`, target exports, and version-file
   behavior.
3. Review Windows, macOS, and Linux CMake install/downstream proof boundaries.
4. Identify CMake changes required for a staged shared-library path.
5. Record where CMake package metadata must continue rejecting unsupported ABI
   claims.
6. Write the Day 6 CMake-feasibility artifact.

### Deliverables

- CMake static-target inventory
- install/export package metadata notes
- platform-specific CMake proof boundary notes
- shared-library CMake feasibility risks
- Day 6 CMake-feasibility artifact

### Completion Criteria

- CMake package behavior is mapped to the decision needs
- platform evidence boundaries are preserved
- feasibility notes are ready for decision synthesis

---

## Day 7: Package And ABI Claim Surface Audit

**Title:** Claim Surface Audit
**Theme:** Audit README, install docs, package docs, maintainer docs, tests,
and metadata guards for ABI/package wording
**Time estimate:** 12 hours

### Tasks

1. Search public documentation for shared-library, dynamic ABI,
   runtime-loader, package-manager, static-first, and install-support wording.
2. Search package metadata templates and generated checks for unsupported
   wording.
3. Review tests that enforce no shared-library artifacts are installed.
4. Identify inconsistent, stale, ambiguous, or missing non-claim wording.
5. Prepare documentation and guard-update candidates.
6. Write the Day 7 claim-surface audit artifact.

### Deliverables

- documentation claim-surface inventory
- package metadata wording inventory
- guard/test ownership notes
- candidate doc and guard updates
- Day 7 claim-surface audit artifact

### Completion Criteria

- package and ABI claim surfaces are known
- inconsistencies are ready for Day 8/Day 10 decisions
- no support claim is broadened during audit

---

## Day 8: Product Decision Synthesis

**Title:** Decision Synthesis
**Theme:** Compare static-first-only continuation against staged
shared-library work and choose the product posture
**Time estimate:** 12 hours

### Tasks

1. Synthesize Day 2 through Day 7 findings.
2. Compare user value, maintenance cost, test burden, symbol-governance cost,
   platform burden, packaging burden, and claim risk for each option.
3. Decide whether Sprint 170 should retain static-first-only support or open a
   staged shared-library path.
4. Define acceptance evidence required for the selected decision.
5. Define explicit deferred surfaces and non-claims.
6. Write the Day 8 decision-synthesis artifact.

### Deliverables

- option comparison
- selected product posture
- acceptance evidence list
- deferred surface and non-claim list
- Day 8 decision-synthesis artifact

### Completion Criteria

- one product posture is selected
- decision rationale is concrete and evidence-based
- unsupported support claims remain rejected

---

## Day 9: Product Decision Record

**Title:** Decision Record
**Theme:** Create the source-controlled shared-library ABI product decision
record
**Time estimate:** 12 hours

### Tasks

1. Draft the decision record with context, decision, rationale, consequences,
   alternatives, and follow-up gates.
2. Link the decision to ABI-surface, lifecycle, build-system, package, and
   claim-surface evidence.
3. Define what current releases may claim after the decision.
4. Define what remains explicitly unsupported or deferred.
5. Place the decision record in the Sprint 170 artifact package or a stable
   docs location if an existing ADR pattern exists.
6. Write the Day 9 decision-record artifact.

### Deliverables

- shared-library ABI product decision record
- supported claim list
- deferred/non-claim list
- evidence links
- Day 9 decision-record artifact

### Completion Criteria

- the shared-library ABI product question has a documented answer
- consequences and future gates are explicit
- documentation updates can follow the decision without ambiguity

---

## Day 10: Guard Update Design

**Title:** Guard Design
**Theme:** Design build/package/test guards that enforce the selected ABI and
shared-library decision
**Time estimate:** 12 hours

### Tasks

1. Review existing package/install tests and static-first deferral checks.
2. Decide which guards need updates after the Day 9 decision.
3. Define negative checks for unsupported shared-library, ABI,
   runtime-loader, package-manager, or platform wording.
4. Define expected generated metadata strings for Make, CMake, and
   `pkg-config` outputs.
5. Define validation commands for guard changes.
6. Write the Day 10 guard-design artifact.

### Deliverables

- guard-update design
- negative-check list
- package metadata expectation list
- validation plan
- Day 10 guard-design artifact

### Completion Criteria

- guard changes are scoped before implementation
- current package evidence remains static-first unless the decision says
  otherwise
- guard validation expectations are clear

---

## Day 11: Guard Implementation

**Title:** Guard Implementation
**Theme:** Update build/package tests and metadata checks to enforce the
selected decision
**Time estimate:** 12 hours

### Tasks

1. Implement selected guard updates in package/install checks, scripts, or
   tests.
2. Update generated package metadata wording only if required by the decision.
3. Preserve existing install/export behavior unless the decision explicitly
   changes it.
4. Add negative checks for accidental unsupported claims or artifacts.
5. Run focused guard validation.
6. Write the Day 11 guard-implementation artifact.

### Deliverables

- implemented guard updates
- package metadata check updates
- negative-check coverage
- focused validation log
- Day 11 guard-implementation artifact

### Completion Criteria

- unsupported shared-library or ABI claims fail mechanically where feasible
- package metadata matches the selected decision
- focused guard validation passes

---

## Day 12: Documentation Alignment

**Title:** Documentation Alignment
**Theme:** Align public and maintainer documentation with the selected
shared-library ABI product decision
**Time estimate:** 12 hours

### Tasks

1. Update README install/package/support wording to match the decision.
2. Update maintainer documentation and package sections with decision
   consequences.
3. Update any package, ABI, platform, runtime-loader, or package-manager
   non-claim tables.
4. Keep performance methodology evidence separate from package/ABI evidence.
5. Run targeted documentation claim scans.
6. Write the Day 12 documentation-alignment artifact.

### Deliverables

- README package/ABI wording updates
- maintainer-guide wording updates
- non-claim table updates
- claim-scan results
- Day 12 documentation-alignment artifact

### Completion Criteria

- docs match the selected decision
- unsupported claims remain explicit
- performance evidence is not cited as package/ABI evidence

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run install/package, guard, documentation, and source-quality
checks required by Sprint 170 changes
**Time estimate:** 12 hours

### Tasks

1. Run package/install checks affected by the decision.
2. Run static-first or shared-library decision guards.
3. Run focused script tests and syntax checks for changed scripts.
4. Run documentation claim scans and `git diff --check`.
5. Check whether any `.c` or `.h` files changed and run
   `make format && make lint && make test` if required.
6. Write the Day 13 integrated-validation artifact.

### Deliverables

- install/package validation log
- guard validation log
- docs claim-scan log
- C quality-gate decision
- Day 13 integrated-validation artifact

### Completion Criteria

- all required checks pass
- generated build/report/cache artifacts are not staged unintentionally
- failures stop the sprint for user input rather than weakening claims

---

## Day 14: Sprint Closeout And Sprint 171 Handoff

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 170 deliverables, final validation, and handoff to
Sprint 171 package-manager readiness work
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 170 outcomes against project-plan items 170.1 through
   170.6.
2. Confirm the decision record, guard updates, documentation alignment, and
   validation artifacts are complete.
3. Re-run final lightweight hygiene checks and any required focused checks.
4. Confirm no generated output is staged unintentionally.
5. Prepare Sprint 171 handoff notes for package-manager readiness or deferral.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 170 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 171 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- the shared-library ABI product decision is source-controlled and enforceable
- documentation and guards match the decision
- Sprint 171 can begin from a clear package/ABI boundary
