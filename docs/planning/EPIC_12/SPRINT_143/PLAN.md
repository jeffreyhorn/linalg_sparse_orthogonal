# Sprint 143 Plan: Shared-Library ABI Decision & Static-First Contract Follow-Through

**Sprint Duration:** 14 days
**Goal:** Make and implement the Epic 12 package/ABI product decision:
shared-library ABI support with proof, or stricter static-first-only contract
with no ambiguity. This sprint implements the Sprint 143 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 143 begins from:
- Sprint 137 package/ABI decision criteria
- Sprint 142 runtime/backend support boundaries and package/ABI handoff
- current static-first install, CMake, `pkg-config`, exact-version, and
  static-deferral proofs
- current README, INSTALL, maintainer guide, CMake package templates,
  `sparse.pc.in`, install scripts, CI workflows, and package report rows
- existing validation expectations for C/header, build-system, install,
  package, script, CI, and documentation changes

The sprint must:
- audit public headers, symbol surface, versioning, visibility, exports,
  `pkg-config`, CMake package metadata, loader behavior, and platform risks
- make an explicit product decision: implement shared-library ABI support with
  proof, or preserve static-first-only support with stronger guards and
  documentation
- implement the selected package path without ambiguity
- strengthen downstream consumer proof for Make, `pkg-config`, CMake, version
  constraints, loader behavior where applicable, and unsupported artifacts
- align Linux, macOS, and Windows CI/support-tier comments with the selected
  package contract
- update README, INSTALL, package metadata comments, maintainer guidance, and
  non-claim wording
- validate install/package tests, CMake install/export tests, static/shared
  guards, and full quality gates when code or build surfaces require them

**End State:** Sprint 143 leaves behind:
- explicit package/ABI product decision
- implemented selected package path
- downstream consumer proof
- CI/support-tier alignment
- updated package docs
- validation evidence for changed package/build/docs/script/code surfaces
- Sprint 144 platform promotion handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 143 project-plan estimate.

---

## Day 1: Package ABI Intake

**Title:** Package Intake
**Theme:** Establish Sprint 143 scope, package/ABI handoffs, current proof
surface, and claim boundaries
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 143 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 137 package/ABI criteria and Sprint 142 Day 13-14 handoff
   artifacts.
3. Create Sprint 143 working notes and artifact directory structure.
4. Inventory current package surfaces: public headers, `VERSION`, Make install,
   CMake install/export, `pkg-config`, static/shared guards, install tests,
   CI lanes, README, INSTALL, and maintainer docs.
5. Map Sprint 143 Items 1-7 to day-level owners.
6. Record initial claim fences for static-first support, shared-library ABI,
   dynamic loader behavior, package-manager support, platform parity, and
   runtime/backend sentinel non-claims.

### Deliverables
- Sprint 143 working-notes baseline
- artifact directory structure
- package/ABI handoff summary
- initial package surface map
- item-to-day owner map
- initial claim-boundary and stop-condition register

### Completion Criteria
- every Sprint 143 project-plan item has a day-level owner
- Sprint 142 runtime/backend sentinel evidence is kept separate from package
  proof
- shared-library ABI and static-first support boundaries are explicit before
  audit work begins

---

## Day 2: Public Header And Symbol Audit

**Title:** Header Audit
**Theme:** Audit public headers, exported API shape, symbol expectations, and
ABI-risk surfaces
**Time estimate:** 12 hours

### Tasks
1. Inspect all installed public headers and generated version headers.
2. Inventory public structs, enums, typedefs, function declarations, allocator
   contracts, ownership rules, and macro-controlled type widths.
3. Identify ABI-sensitive surfaces: struct layout, enum values, scalar/index
   widths, exported function names, visibility assumptions, and inline/static
   helpers.
4. Inspect build outputs and symbol-listing commands available locally for the
   current static archive.
5. Identify what would be required to make shared-library symbols, visibility,
   and compatibility rules reviewable.
6. Write the public header and symbol audit artifact.

### Deliverables
- public header inventory
- ABI-sensitive surface map
- symbol/export audit notes
- shared-library proof requirement list
- risk and unknowns register

### Completion Criteria
- installed public headers are accounted for
- ABI-sensitive surfaces are separated from ordinary source implementation
- shared-library proof requirements are concrete enough for Day 5 decision

---

## Day 3: Install Export Metadata Audit

**Title:** Metadata Audit
**Theme:** Audit Make install, CMake export, pkg-config, versioning,
unsupported-artifact guards, and current package report rows
**Time estimate:** 12 hours

### Tasks
1. Inspect `Makefile` install/uninstall targets, `sparse.pc.in`, CMake install
   rules, package config templates, and static/shared configure guards.
2. Inspect `tests/test_install.sh`, `tests/test_cmake_install.sh`,
   `scripts/static_package_deferral_check.sh`, and package report manifests.
3. Verify current package metadata claims around static archive, installed
   headers, exact version matching, unsupported shared libraries, and
   downstream consumer use.
4. Identify metadata changes required for either selected path: shared-library
   support or stricter static-first-only support.
5. Record package/report freshness and validation owners.
6. Write the install/export metadata audit artifact.

### Deliverables
- install/export metadata audit
- current package proof-owner map
- static/shared guard inventory
- versioning and package report row inventory
- selected-path implementation input list

### Completion Criteria
- Make, CMake, `pkg-config`, and package-report surfaces are accounted for
- unsupported shared-library behavior is documented before the decision
- Day 5 can decide from concrete metadata and proof requirements

---

## Day 4: Platform And Loader Risk Audit

**Title:** Loader Risk Audit
**Theme:** Audit dynamic-loader, CI platform, Windows/MSVC, macOS, Linux, and
support-tier risks before the product decision
**Time estimate:** 12 hours

### Tasks
1. Inspect Linux, macOS, and Windows CI workflows for package/install lanes,
   CMake consumer lanes, expected test counts, and support-tier comments.
2. Identify platform-specific blockers for shared-library proof: loader paths,
   import libraries, symbol export mechanics, RPATH/install-name behavior,
   `pkg-config` semantics, and CMake target properties.
3. Identify platform-specific blockers for stricter static-first support:
   unsupported artifact checks, CI messages, documentation wording, and
   negative proof.
4. Separate Sprint 143 package/ABI scope from Sprint 144 platform promotion.
5. Draft validation scenarios for the selected product decision.
6. Write the platform and loader risk audit artifact.

### Deliverables
- platform and loader risk audit
- CI package lane inventory
- shared-library blocker list
- static-first strengthening opportunity list
- Sprint 144 platform-separation notes

### Completion Criteria
- platform risks are visible before the product decision
- package/ABI scope does not imply platform promotion
- validation scenarios distinguish shared-library support from static-first
  deferral

---

## Day 5: Product Decision

**Title:** Product Decision
**Theme:** Decide shared-library ABI support versus stricter static-first-only
support and freeze the implementation path
**Time estimate:** 12 hours

### Tasks
1. Compare Day 2-4 audit findings against Sprint 137 package/ABI decision
   criteria and Sprint 142 handoff boundaries.
2. Score shared-library support and static-first-only strengthening by user
   value, proof burden, platform risk, implementation risk, docs burden, and
   claim risk.
3. Select the Sprint 143 implementation path.
4. Define the exact changes, tests, docs, CI updates, and non-claims required
   for the selected path.
5. Define explicit deferrals for the path not selected.
6. Write the package/ABI product decision artifact.

### Deliverables
- explicit package/ABI product decision
- selected implementation path
- required validation checklist
- non-selected path deferral ledger
- stop conditions for implementation

### Completion Criteria
- the sprint has one selected package path, not two partial paths
- the selected path can be implemented and validated inside the remaining
  sprint budget
- deferred package/ABI claims are explicit and source-owned

---

## Day 6: Implementation Design

**Title:** Implementation Design
**Theme:** Design the selected package path before touching build/install
mechanics
**Time estimate:** 12 hours

### Tasks
1. Convert the Day 5 decision into a bounded edit plan across Make, CMake,
   package metadata, scripts, tests, CI comments, docs, and report rows.
2. Define file-by-file ownership and order of operations.
3. Define compatibility behavior for existing static consumers.
4. Define negative-proof behavior for unsupported artifacts, or positive-proof
   behavior for shared libraries if selected.
5. Define focused validation commands for each touched surface.
6. Write the implementation design artifact.

### Deliverables
- selected-path implementation design
- file ownership map
- compatibility and migration notes
- focused validation plan
- implementation risk register

### Completion Criteria
- implementation edits are scoped before changes begin
- existing static consumers remain protected unless the decision explicitly
  changes them
- validation commands are mapped to touched surfaces

---

## Day 7: Package Path Implementation Batch 1

**Title:** Package Batch 1
**Theme:** Implement the first selected-path build/install/package metadata
batch
**Time estimate:** 12 hours

### Tasks
1. Implement the first Make/CMake/package metadata batch selected on Day 6.
2. Update static/shared guards or shared-library build/install/export rules
   according to the product decision.
3. Update package config templates, `pkg-config` metadata, or CMake target
   metadata as needed.
4. Preserve existing static downstream behavior unless intentionally changed.
5. Run focused configure/build/install metadata checks for touched surfaces.
6. Write the Batch 1 implementation artifact.

### Deliverables
- first selected-path package implementation batch
- updated build/install/package metadata
- focused package validation evidence
- compatibility notes

### Completion Criteria
- selected-path mechanics are partially implemented and testable
- unsupported package artifacts are either rejected or explicitly proved
- static consumer behavior remains coherent

---

## Day 8: Package Path Implementation Batch 2

**Title:** Package Batch 2
**Theme:** Complete selected-path implementation and fill remaining guards,
metadata, or export behavior
**Time estimate:** 12 hours

### Tasks
1. Complete the remaining selected-path Make/CMake/package metadata changes.
2. Add or update scripts for static/shared support checks as needed.
3. Update source-list, install-list, export-list, or generated metadata
   ownership if changed.
4. Run focused package/build checks after the full implementation batch.
5. Record any implementation-scoped repairs or deferrals.
6. Write the Batch 2 implementation artifact.

### Deliverables
- completed selected-path package implementation
- updated guard/check scripts if needed
- focused validation evidence
- repair and deferral notes

### Completion Criteria
- the selected product path is mechanically complete
- package metadata and guards match the Day 5 decision
- any remaining implementation risk has an owner and stop condition

---

## Day 9: Downstream Consumer Proof

**Title:** Consumer Proof
**Theme:** Strengthen Make, pkg-config, CMake, version, loader, and
unsupported-artifact downstream proof
**Time estimate:** 12 hours

### Tasks
1. Audit and update downstream proof scripts for the selected package path:
   `tests/test_install.sh`, `tests/test_cmake_install.sh`, examples, and
   package check scripts.
2. Add or strengthen checks for installed headers, installed libraries,
   `pkg-config --cflags`, `pkg-config --libs`, CMake `find_package`, exact
   version behavior, and unsupported artifacts.
3. Add loader/runtime checks only if shared-library support was selected.
4. Ensure skip/fail behavior is deterministic and does not convert missing
   support into pass evidence.
5. Run focused downstream consumer checks where feasible locally.
6. Write the downstream consumer proof artifact.

### Deliverables
- strengthened downstream consumer tests/scripts
- version and unsupported-artifact proof
- loader proof if applicable
- focused consumer validation evidence

### Completion Criteria
- selected package path has executable downstream proof
- unsupported artifacts are checked explicitly
- proof scripts do not overclaim platform or package-manager support

---

## Day 10: CI And Package Report Alignment

**Title:** CI Alignment
**Theme:** Align CI lanes, support-tier comments, package report rows, and
freshness semantics with the selected package contract
**Time estimate:** 12 hours

### Tasks
1. Update Linux/macOS/Windows CI workflow package comments, expected surfaces,
   and selected validation commands as needed.
2. Update package/report rows or metadata if the selected package contract
   changes row meaning.
3. Preserve Sprint 144 platform-promotion boundaries in workflow wording.
4. Run workflow syntax or targeted CI-surface checks feasible locally.
5. Run normalized package/report checks if package report metadata changed.
6. Write the CI and package-report alignment artifact.

### Deliverables
- updated CI/support-tier wording or commands
- package report row alignment if needed
- workflow/package validation evidence
- Sprint 144 platform-boundary notes

### Completion Criteria
- CI wording matches the selected package contract
- Linux/macOS/Windows comments do not imply unsupported platform parity
- package report rows preserve honest proof-owner semantics

---

## Day 11: Documentation Alignment

**Title:** Package Docs
**Theme:** Update public and maintainer package documentation for the selected
contract and preserved non-claims
**Time estimate:** 12 hours

### Tasks
1. Update README installation/package summaries for the selected package path.
2. Update INSTALL with operational setup, install verification, package
   boundaries, unsupported artifacts, and platform support-tier wording.
3. Update CMake package docs, `pkg-config` comments, maintainer guide, and any
   package report documentation affected by implementation.
4. Preserve runtime/backend sentinel non-claims and Sprint 144 platform
   separation.
5. Run focused docs path/claim/whitespace checks.
6. Write the documentation alignment artifact.

### Deliverables
- updated README and INSTALL package sections
- updated package metadata comments or maintainer docs
- preserved non-claim wording
- documentation validation evidence

### Completion Criteria
- docs match the selected package implementation
- users can distinguish supported package paths from deferrals
- no shared-library, ABI, platform, package-manager, or performance claim is
  added without proof

---

## Day 12: Focused Package Validation

**Title:** Focused Validation
**Theme:** Run focused install/export/package checks and repair scoped
failures before the full quality gate
**Time estimate:** 12 hours

### Tasks
1. Run focused selected-path install and downstream consumer tests.
2. Run CMake install/export and `pkg-config` checks.
3. Run static/shared guard checks and loader checks where applicable.
4. Run package/report freshness checks for affected families.
5. Inspect generated artifacts for ignored/untracked hygiene.
6. Write the focused package validation artifact.

### Deliverables
- focused package validation evidence
- install/export/downstream proof results
- generated-output hygiene evidence
- scoped repair notes if needed

### Completion Criteria
- focused package checks pass or scoped failures are repaired
- generated outputs remain ignored unless intentionally source-controlled
- remaining risks have owners and stop conditions

---

## Day 13: Full Quality Gate And Claim Closure

**Title:** Quality And Claims
**Theme:** Run required full quality gates and publish earned package/ABI
claims, non-claims, and residuals
**Time estimate:** 12 hours

### Tasks
1. Determine whether C/header files changed and run
   `make format && make lint && make test` if required.
2. Run required package, install, CMake, `pkg-config`, script, report, docs,
   and whitespace checks for all touched surfaces.
3. Compare final implementation against the Day 5 product decision.
4. Publish earned package/ABI claims and remaining non-claims.
5. Route remaining package, ABI, distribution, platform, or loader work to
   explicit future owners.
6. Write the quality gate and claim closure artifact.

### Deliverables
- full quality gate results
- package/ABI claim closure
- residual non-claim register
- future-owner handoff notes

### Completion Criteria
- required checks for touched surfaces pass
- earned package claims are backed by specific evidence
- unearned package/ABI/platform/distribution claims remain explicit

---

## Day 14: Closeout

**Title:** Closeout
**Theme:** Finalize Sprint 143 artifacts, validation evidence, working notes,
and Sprint 144 platform handoff
**Time estimate:** 12 hours

### Tasks
1. Re-run final report/package/docs hygiene and any required quality checks
   after Day 13 updates.
2. Review all Sprint 143 artifacts for consistency with implemented package
   behavior and selected product decision.
3. Confirm package rows and docs do not overclaim shared ABI, loader behavior,
   package-manager availability, platform parity, portable performance, or
   state-of-the-art status.
4. Update working notes with final validation, changed files, decisions,
   deferred work, and known risks.
5. Write the closeout validation summary artifact.
6. Prepare the sprint for retrospective and PR creation.

### Deliverables
- final validation evidence
- Sprint 143 closeout summary
- Sprint 144 platform promotion handoff
- updated working notes
- complete artifact package

### Completion Criteria
- Sprint 143 deliverables are present and traceable to Items 1-7
- validation evidence is current and reproducible
- remaining package/platform work is explicitly routed forward
