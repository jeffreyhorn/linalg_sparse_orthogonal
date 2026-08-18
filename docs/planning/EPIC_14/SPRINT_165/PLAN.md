# Sprint 165 Plan: Static-First Package Boundary Hardening

**Sprint Duration:** 14 days
**Goal:** Harden the static-first package boundary so shared-library, dynamic
ABI, runtime-loader, and package-manager non-claims cannot drift. This sprint
implements the Sprint 165 section of
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references the older Epic 12 project-plan
path, but the current Sprint 165 project-plan section lives in
`docs/planning/EPIC_14/PROJECT_PLAN.md`.

**Starting Point:** Sprint 165 begins from:
- Sprint 162 Windows package decision and staged parity boundary;
- Sprint 163 methodology-bound performance publication non-claims;
- Sprint 164 public-header/API cleanup and package handoff;
- existing static-first CMake package, `sparse.pc`, install, uninstall, and
  downstream proof surfaces;
- current deferral wording for shared-library, dynamic ABI, runtime-loader,
  package-manager, and broad platform package claims.

The sprint must:
- audit package metadata and docs before changing checks;
- strengthen static-first deferral guards without adding shared-library
  support;
- keep ABI language limited to explicit non-claims and version metadata;
- refresh downstream install/export proof for static archive behavior;
- align README, INSTALL, maintainer, CMake, and pkg-config wording;
- run package/install/export checks and required quality gates where affected;
- publish residuals for true shared-library and package-manager work.

**End State:** Sprint 165 leaves behind:
- hardened static-first package boundary;
- updated package metadata and ABI non-claim audit;
- refreshed downstream static package proof;
- package documentation aligned with supported behavior;
- Sprint 166 closeout handoff.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 165 project-plan estimate.

---

## Day 1: Sprint Intake And Package Surface Inventory

**Title:** Sprint Intake
**Theme:** Establish Sprint 165 scope, artifact layout, and package evidence
surfaces
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 165 section of
   `docs/planning/EPIC_14/PROJECT_PLAN.md`.
2. Review Sprint 162 package decision artifacts, Sprint 164 API-header
   handoff, and relevant Epic 14 planning notes.
3. Create Sprint 165 working notes and artifact directory structure.
4. Inventory CMake package files, `sparse.pc`, Make install/uninstall
   targets, install scripts, examples, CI package lanes, README, INSTALL, and
   maintainer package docs.
5. Record explicit non-goals for shared-library support, dynamic ABI
   stability, runtime-loader behavior, package-manager distribution, and broad
   platform parity.
6. Write the Day 1 sprint-intake artifact.

### Deliverables
- Sprint 165 working-notes baseline
- artifact directory structure
- package surface inventory
- non-goal and stop-condition register
- Day 1 sprint-intake artifact

### Completion Criteria
- Sprint 165 scope is tied to the Epic 14 project plan
- current package metadata and validation owners are identified
- unsupported package and ABI claims are separated from hardening work

---

## Day 2: Package Metadata Audit

**Title:** Metadata Audit
**Theme:** Inspect installed metadata for unsupported wording or behavior
claims
**Time estimate:** 12 hours

### Tasks
1. Inspect CMake export/config/version templates and generated installed
   package files.
2. Inspect `sparse.pc` generation, installed content, and pkg-config validation
   expectations.
3. Inspect install/uninstall targets for shared-library, dynamic-loader, or
   package-manager leakage.
4. Map CI checks that assert static archive behavior, exact-version handling,
   metadata wording, and absent shared artifacts.
5. Classify findings as supported contract, unsupported wording, stale
   historical text, validation gap, or deferred product decision.
6. Write the Day 2 package-metadata-audit artifact.

### Deliverables
- package metadata audit table
- static archive contract map
- unsupported wording register
- validation coverage map
- Day 2 package-metadata-audit artifact

### Completion Criteria
- CMake and pkg-config metadata behavior is source-backed
- unsupported claims have exact file and check owners
- audit distinguishes metadata cleanup from feature expansion

---

## Day 3: Static Deferral Guard Design

**Title:** Guard Design
**Theme:** Define stronger static-first guardrails before implementation
**Time estimate:** 12 hours

### Tasks
1. Review current `BUILD_SHARED_LIBS=ON` behavior in CMake and CI.
2. Define expected failure message, configure-time status, and documentation
   wording for shared-library deferral.
3. Define checks that reject installed shared artifacts and shared imported
   metadata.
4. Identify whether Makefile, CMake, pkg-config, or docs need guard updates.
5. Define validation commands for static guard behavior on local and CI
   surfaces.
6. Write the Day 3 static-deferral-guard-design artifact.

### Deliverables
- static deferral guard design
- expected failure and metadata absence rules
- affected file list
- validation command list
- Day 3 static-deferral-guard-design artifact

### Completion Criteria
- guard behavior is explicit before edits
- checks block drift without claiming shared-library support
- implementation scope is narrow and testable

---

## Day 4: Static Deferral Guard Implementation

**Title:** Guard Implementation
**Theme:** Strengthen configure/install checks for static-first behavior
**Time estimate:** 12 hours

### Tasks
1. Update CMake guard behavior and messages if the Day 3 design identifies a
   gap.
2. Update package metadata checks for absent shared-library imported metadata.
3. Update install validation scripts or CI snippets where shared artifact
   checks are incomplete.
4. Preserve current static archive install path and target names.
5. Run focused configure/install guard checks for changed surfaces.
6. Write the Day 4 static-deferral-guard-implementation artifact.

### Deliverables
- strengthened static-first guard implementation
- updated focused validation checks
- guard command output notes
- Day 4 static-deferral-guard-implementation artifact

### Completion Criteria
- `BUILD_SHARED_LIBS=ON` deferral remains fail-closed
- installed package metadata cannot imply shared-library support
- static archive install behavior remains unchanged

---

## Day 5: ABI Non-Claim Audit

**Title:** ABI Audit
**Theme:** Review public package and API surfaces for accidental ABI promises
**Time estimate:** 12 hours

### Tasks
1. Inspect public headers, version docs, README, INSTALL, CMake package docs,
   maintainer docs, and generated references for ABI wording.
2. Separate source API compatibility, package version metadata, and binary ABI
   support language.
3. Identify wording that could imply stable struct layout, soname policy,
   dynamic ABI compatibility, or downstream binary compatibility.
4. Map ABI-related wording to supporting evidence or deferred work.
5. Draft exact replacement language for unsupported or ambiguous claims.
6. Write the Day 5 ABI-non-claim-audit artifact.

### Deliverables
- ABI wording audit
- source API versus binary ABI distinction
- replacement wording candidates
- deferred ABI decision register
- Day 5 ABI-non-claim-audit artifact

### Completion Criteria
- accidental ABI promises are identified
- version metadata is not conflated with ABI support
- wording changes are ready for scoped implementation

---

## Day 6: ABI And Package Wording Cleanup

**Title:** Wording Cleanup
**Theme:** Apply non-claim wording to docs and package comments
**Time estimate:** 12 hours

### Tasks
1. Update README, INSTALL, package docs, maintainer docs, and package comments
   with approved static-first and ABI non-claim wording.
2. Clarify that exact-version package metadata is not a dynamic ABI guarantee.
3. Remove stale package-manager, shared-library, soname, dynamic-loader, or ABI
   implications from affected docs.
4. Preserve user-facing install instructions for supported static archive
   workflows.
5. Update cross-links so deeper package caveats have a single source of truth.
6. Write the Day 6 ABI-package-wording-cleanup artifact.

### Deliverables
- updated package and ABI wording
- static-first support boundary links
- removed stale claim notes
- Day 6 ABI-package-wording-cleanup artifact

### Completion Criteria
- docs match supported static-first package behavior
- unsupported ABI/package-manager claims are absent
- install guidance remains usable

---

## Day 7: Downstream Proof Scope Refresh

**Title:** Proof Scope
**Theme:** Refresh downstream consumer proof requirements for static archive
package behavior
**Time estimate:** 12 hours

### Tasks
1. Review downstream CMake and pkg-config examples and install validation
   scripts.
2. Define proof requirements for static archive presence, no shared artifacts,
   exact version handling, include paths, link flags, and example execution.
3. Separate Unix pkg-config proof, Windows CMake-first proof, metadata-only
   Windows pkg-config inspection, and deferred Windows Make/pkg-config parity.
4. Identify stale expectations caused by path normalization, output matching,
   or changed test counts.
5. Write the Day 7 downstream-proof-scope artifact.

### Deliverables
- downstream proof requirement table
- platform-specific proof boundaries
- stale expectation register
- Day 7 downstream-proof-scope artifact

### Completion Criteria
- downstream proof scope is explicit before edits
- Windows and Unix package expectations are not conflated
- deferred parity work remains documented

---

## Day 8: Downstream Proof Implementation

**Title:** Proof Refresh
**Theme:** Update install/export proof scripts and examples where needed
**Time estimate:** 12 hours

### Tasks
1. Update maintained install validation scripts for the Day 7 proof scope.
2. Update downstream CMake or pkg-config examples only where the proof surface
   is stale or ambiguous.
3. Ensure exact-version success and mismatch failure checks remain covered.
4. Ensure installed `sparse.pc` and CMake package metadata checks enforce
   static archive boundaries.
5. Run focused downstream proof commands where locally available.
6. Write the Day 8 downstream-proof-implementation artifact.

### Deliverables
- refreshed downstream proof scripts or examples
- focused proof command notes
- updated static metadata checks
- Day 8 downstream-proof-implementation artifact

### Completion Criteria
- downstream proof matches supported static archive behavior
- exact-version checks remain active
- unsupported Windows Make/pkg-config parity is not claimed

---

## Day 9: Package Documentation Alignment

**Title:** Docs Alignment
**Theme:** Align public and maintainer package documentation
**Time estimate:** 12 hours

### Tasks
1. Update README install/package sections for static-first behavior.
2. Update INSTALL or platform docs with the current supported and deferred
   package surfaces.
3. Update maintainer docs with validation commands and package-boundary
   ownership.
4. Update CMake and pkg-config comments or examples for consistent terminology.
5. Verify cross-links from tutorial, cookbook, solver-selection, and API docs
   do not imply package-manager or shared-library support.
6. Write the Day 9 package-documentation-alignment artifact.

### Deliverables
- aligned public package docs
- aligned maintainer package validation docs
- updated cross-link map
- Day 9 package-documentation-alignment artifact

### Completion Criteria
- users can find supported static install guidance quickly
- maintainer validation steps are repeatable
- docs do not overstate package support

---

## Day 10: Package Metadata Validation

**Title:** Metadata Validation
**Theme:** Validate installed metadata content and absence of unsupported
claims
**Time estimate:** 12 hours

### Tasks
1. Run focused package metadata generation checks.
2. Inspect installed CMake package files for static imported target metadata.
3. Inspect installed `sparse.pc` for supported description, version, cflags,
   libs, and absent unsupported terms.
4. Run static deferral guard checks for `BUILD_SHARED_LIBS=ON` if changed.
5. Record command outputs, skipped platform checks, and local environment
   constraints.
6. Write the Day 10 package-metadata-validation artifact.

### Deliverables
- package metadata validation record
- static deferral validation record
- skipped/deferred platform note
- Day 10 package-metadata-validation artifact

### Completion Criteria
- installed metadata matches static-first contract
- unsupported metadata terms are absent
- validation gaps are explicit

---

## Day 11: Install And Downstream Validation

**Title:** Downstream Validation
**Theme:** Validate install/uninstall and downstream static consumers
**Time estimate:** 12 hours

### Tasks
1. Run maintained install validation scripts on available local platform
   surfaces.
2. Build and run downstream CMake and pkg-config consumers where available.
3. Validate uninstall removes installed library, headers, and package metadata.
4. Record exact-version success and mismatch failure behavior.
5. Record any platform lanes that require hosted CI rather than local proof.
6. Write the Day 11 install-downstream-validation artifact.

### Deliverables
- install/uninstall validation record
- downstream consumer validation record
- version handling proof notes
- Day 11 install-downstream-validation artifact

### Completion Criteria
- static archive downstream consumers build and run
- uninstall behavior remains covered
- hosted-only validation requirements are documented

---

## Day 12: Full Quality Gate And Drift Scan

**Title:** Quality Gate
**Theme:** Run required quality checks and scan for package-boundary drift
**Time estimate:** 12 hours

### Tasks
1. Run `make format` if source, header, script, or docs conventions require it.
2. Run `make lint` and focused package checks if `.c` or `.h` files changed.
3. Run `make test` if `.c` or `.h` files changed.
4. Run docs/package freshness checks or link scans available in the repo.
5. Search changed files for unsupported shared-library, ABI, runtime-loader,
   package-manager, and broad platform wording.
6. Write the Day 12 quality-gate-drift-scan artifact.

### Deliverables
- quality gate record
- drift scan results
- unresolved validation issue list
- Day 12 quality-gate-drift-scan artifact

### Completion Criteria
- required checks pass or blockers are documented
- changed files do not introduce unsupported claims
- validation evidence is sufficient for closeout

---

## Day 13: Evidence Review And Residual Register

**Title:** Evidence Review
**Theme:** Review package hardening evidence and document residual product
work
**Time estimate:** 11 hours

### Tasks
1. Review Day 1-12 artifacts against Sprint 165 deliverables.
2. Confirm package metadata, static guard, ABI wording, downstream proof, and
   docs alignment each have evidence.
3. Build a residual register for true shared-library support, dynamic ABI
   policy, package-manager distribution, runtime-loader behavior, and Windows
   Make/pkg-config parity.
4. Identify Sprint 166 final validation and closeout handoff items.
5. Write the Day 13 evidence-review artifact.

### Deliverables
- Sprint 165 evidence checklist
- package residual register
- Sprint 166 handoff items
- Day 13 evidence-review artifact

### Completion Criteria
- every Sprint 165 deliverable has supporting evidence
- residuals are product decisions, not hidden implementation gaps
- Sprint 166 receives a clear closeout handoff

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 165 package boundary hardening and hand off to
Sprint 166
**Time estimate:** 11 hours

### Tasks
1. Update working notes with final changed files, validation commands, and
   known residuals.
2. Prepare the sprint closeout artifact summarizing package metadata,
   static-first guard, ABI non-claim, downstream proof, and documentation
   outcomes.
3. Verify no Sprint 165 artifact points to stale Epic 12 locations without a
   source-note explanation.
4. Confirm the plan, artifacts, and working notes are internally consistent.
5. Record PR description bullets and review-risk notes for package-boundary
   changes.
6. Write the Day 14 closeout artifact.

### Deliverables
- final Sprint 165 working-notes update
- Day 14 closeout artifact
- PR description outline
- Sprint 166 validation and closeout handoff

### Completion Criteria
- Sprint 165 deliverables are closed or explicitly deferred
- package, ABI, shared-library, and package-manager claims remain bounded
- branch is ready for review with validation evidence
