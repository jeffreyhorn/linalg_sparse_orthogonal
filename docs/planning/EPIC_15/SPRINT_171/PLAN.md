# Sprint 171 Plan: Package-Manager Readiness First Provider

**Sprint Duration:** 14 days
**Goal:** Close one package-manager readiness path or formally document and
enforce package-manager deferral. This sprint implements the Sprint 171
section of `docs/planning/EPIC_15/PROJECT_PLAN.md`.

**Source Artifact Note:** The prompt references
`docs/planning/EPIC_12/PROJECT_PLAN.md` and the title "Sprint 171:
Package-Manager Readiness First Provider"; the active merged Sprint 171
project-plan section lives in `docs/planning/EPIC_15/PROJECT_PLAN.md` and has
the same title.

**Starting Point:** Sprint 171 begins from:

- Sprint 170 shared-library ABI product decision record;
- static-first-only package posture and explicit shared-library/dynamic ABI
  non-claims;
- validated Make install/`pkg-config` and CMake install/export behavior for
  the maintained static archive package surface;
- Windows CMake-first install/downstream validation with metadata-only
  `sparse.pc` inspection;
- static package deferral guard coverage for unsupported shared-library,
  dynamic ABI, runtime-loader, package-manager, and Windows parity claims.

The sprint must:

- select one package-manager path, such as vcpkg, Homebrew, or explicit
  deferral;
- add the selected recipe/proof artifact or formal unsupported-provider
  decision;
- add local validation for install, compile, version query, and cleanup where
  provider support is selected;
- update claim guards so source install, CMake/`pkg-config` install, and
  package-manager support remain distinct;
- update user-facing package-manager guidance or explicit non-claim wording;
- run install validation and provider proof or deferral checks.

**End State:** Sprint 171 leaves behind:

- one explicit package-manager readiness decision;
- provider proof artifact or formal deferral artifact;
- updated package documentation and claim guards;
- validation records for the selected provider path or deferral;
- Sprint 171 working notes, daily artifacts, and closeout records.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 171 project-plan estimate.

---

## Day 1: Sprint Intake And Package Boundary Baseline

**Title:** Package Intake
**Theme:** Establish Sprint 171 scope from Epic 15 plan items 171.1 through
171.6 and Sprint 170 handoff boundaries
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 171 section of
   `docs/planning/EPIC_15/PROJECT_PLAN.md`.
2. Review Sprint 170 decision record, closeout, and retrospective.
3. Create Sprint 171 working notes and artifact directory structure.
4. Record the prompt path/source-artifact mismatch.
5. Define retained package-manager, shared-library, ABI, Windows parity, and
   platform non-claims.
6. Write the Day 1 package-intake artifact.

### Deliverables

- Sprint 171 working-notes baseline
- artifact directory structure
- source artifact note
- Sprint 170 handoff summary
- package-manager stop conditions
- Day 1 package-intake artifact

### Completion Criteria

- Sprint 171 scope is tied to the active Epic 15 project plan
- source install and package-manager support are clearly separated
- no package-manager support claim is introduced by planning alone

---

## Day 2: Provider Candidate Inventory

**Title:** Provider Inventory
**Theme:** Inventory candidate package-manager paths and classify readiness,
risk, and proof cost
**Time estimate:** 12 hours

### Tasks

1. Inventory candidate providers: vcpkg, Homebrew, pkgsrc, Conan, system
   packages, and explicit deferral.
2. Map each candidate to expected manifest/formula/recipe inputs.
3. Identify platform support, dependency, compiler, install-prefix, and
   downstream-consumer proof requirements.
4. Identify licensing, versioning, source archive, and checksum requirements.
5. Compare provider proof cost against Sprint 171 time budget.
6. Write the Day 2 provider-inventory artifact.

### Deliverables

- provider candidate matrix
- proof-cost estimate
- platform/dependency requirement notes
- provider risk list
- Day 2 provider-inventory artifact

### Completion Criteria

- viable and non-viable provider paths are visible
- deferral remains available if no provider can be proven safely
- provider support is not inferred from source install support

---

## Day 3: Provider Selection Decision

**Title:** Provider Selection
**Theme:** Select one package-manager readiness path or choose formal deferral
with explicit rationale
**Time estimate:** 12 hours

### Tasks

1. Compare provider candidates against user value, proof feasibility, platform
   risk, and maintenance cost.
2. Select one first-provider path or select formal package-manager deferral.
3. Define support claims allowed by the selected path.
4. Define unsupported claims that remain out of scope.
5. Identify implementation and validation artifacts needed for the selected
   path.
6. Write the Day 3 provider-selection decision artifact.

### Deliverables

- selected provider or deferral decision
- supported-claim list
- unsupported-claim list
- implementation artifact list
- Day 3 provider-selection artifact

### Completion Criteria

- exactly one package-manager readiness path is selected
- claim boundaries are explicit before implementation
- unsupported providers remain non-claims

---

## Day 4: Recipe Or Deferral Artifact Design

**Title:** Artifact Design
**Theme:** Design the selected recipe/proof artifact or formal deferral record
before implementation
**Time estimate:** 12 hours

### Tasks

1. Define the source-controlled artifact shape for the selected path.
2. If provider support is selected, define manifest/formula/recipe fields,
   source inputs, version handling, and install outputs.
3. If deferral is selected, define the unsupported-provider decision record
   and minimum evidence needed to revisit it.
4. Define how static-first package metadata maps to the selected artifact.
5. Define expected failure modes and rollback criteria.
6. Write the Day 4 artifact-design record.

### Deliverables

- recipe or deferral artifact design
- version/source metadata requirements
- static-first mapping notes
- failure-mode list
- Day 4 artifact-design artifact

### Completion Criteria

- implementation shape is clear before files are changed
- selected path preserves Sprint 170 package/ABI boundaries
- deferral criteria are explicit if no provider is selected

---

## Day 5: Recipe Or Deferral Artifact Implementation

**Title:** Artifact Implementation
**Theme:** Add the selected package-manager recipe/proof artifact or formal
deferral record
**Time estimate:** 12 hours

### Tasks

1. Implement the selected manifest/formula/recipe or deferral record.
2. Keep the artifact static-first unless the selected decision explicitly says
   otherwise.
3. Avoid adding generated archives, build outputs, or local install prefixes.
4. Add source comments or notes only where they clarify maintenance ownership.
5. Run focused syntax or schema checks available for the artifact type.
6. Write the Day 5 artifact-implementation record.

### Deliverables

- provider recipe/proof artifact or deferral record
- focused artifact validation notes
- generated-output hygiene check
- Day 5 artifact-implementation artifact

### Completion Criteria

- selected artifact exists in source control
- unsupported package-manager claims remain out of scope
- no generated package/build outputs are added accidentally

---

## Day 6: Local Proof Script Design

**Title:** Proof Script Design
**Theme:** Design local validation for provider install, compile, version
query, cleanup, or deferral enforcement
**Time estimate:** 12 hours

### Tasks

1. Review existing install proof scripts and static package guards.
2. Define provider-specific install, query, compile/link/run, and cleanup
   checks where provider support is selected.
3. Define deferral validation checks where formal deferral is selected.
4. Define skip behavior for unavailable local provider tooling.
5. Define expected output and failure messages.
6. Write the Day 6 proof-script design artifact.

### Deliverables

- local proof script design
- provider-tool availability policy
- expected pass/fail messages
- cleanup and generated-output policy
- Day 6 proof-script design artifact

### Completion Criteria

- local proof behavior is scoped before implementation
- provider-tool absence behavior is explicit
- validation cannot silently broaden package-manager claims

---

## Day 7: Local Proof Script Implementation

**Title:** Proof Script
**Theme:** Implement the local provider proof or deferral-enforcement script
**Time estimate:** 12 hours

### Tasks

1. Add or update the local proof script for the selected path.
2. Include install, compile/link/run, version query, and cleanup checks where
   provider support is selected.
3. Include explicit deferral checks where package-manager support is deferred.
4. Add clear failure output for unsupported provider claims.
5. Run shell syntax checks and focused local validation.
6. Write the Day 7 proof-script implementation artifact.

### Deliverables

- local proof or deferral script
- focused validation output
- cleanup behavior
- Day 7 proof-script artifact

### Completion Criteria

- provider proof or deferral enforcement is executable
- failure output identifies unsupported claims clearly
- local validation passes or stops for user input

---

## Day 8: Package Claim Guard Design

**Title:** Claim Guard Design
**Theme:** Design guard updates that distinguish source install,
CMake/`pkg-config` install, and package-manager support
**Time estimate:** 12 hours

### Tasks

1. Review current package and ABI guard scripts.
2. Identify public docs, package metadata, workflow text, and provider artifact
   surfaces that need package-manager claim checks.
3. Define positive checks for the selected provider or deferral decision.
4. Define negative checks for unsupported package-manager, shared-library, ABI,
   runtime-loader, and platform claims.
5. Define validation commands for guard changes.
6. Write the Day 8 claim-guard design artifact.

### Deliverables

- package claim guard design
- positive and negative check lists
- validation command list
- Day 8 claim-guard design artifact

### Completion Criteria

- guard changes are scoped before implementation
- package-manager support cannot be inferred from static source install
- unsupported shared-library/ABI claims remain protected

---

## Day 9: Package Claim Guard Implementation

**Title:** Claim Guard Implementation
**Theme:** Implement claim guards for the selected provider or deferral path
**Time estimate:** 12 hours

### Tasks

1. Update guard scripts or tests to enforce the selected package-manager
   decision.
2. Add positive checks for the selected provider artifact or deferral record.
3. Add negative checks for unsupported package-manager wording in package
   metadata and public docs.
4. Preserve Sprint 170 static-first/shared-library ABI guard behavior.
5. Run focused guard validation.
6. Write the Day 9 claim-guard implementation artifact.

### Deliverables

- implemented package claim guards
- selected provider or deferral checks
- focused guard validation log
- Day 9 claim-guard artifact

### Completion Criteria

- unsupported package-manager claims fail mechanically where feasible
- existing static-first package guards still pass
- package-manager docs can be updated from a guarded boundary

---

## Day 10: User Documentation Design

**Title:** Documentation Design
**Theme:** Design concise package-manager guidance or explicit non-claim
wording for users
**Time estimate:** 12 hours

### Tasks

1. Review README, INSTALL, maintainer guide, cookbook, and package sections
   for package-manager wording.
2. Define user-facing guidance for the selected provider path or deferral.
3. Define where to place quick-start versus maintainer detail.
4. Preserve separation between source install, CMake/`pkg-config` install, and
   package-manager support.
5. Define targeted documentation claim scans.
6. Write the Day 10 documentation-design artifact.

### Deliverables

- documentation update plan
- quick-start and maintainer-detail split
- documentation claim-scan plan
- Day 10 documentation-design artifact

### Completion Criteria

- documentation changes are scoped before editing
- user-facing wording cannot imply unsupported providers
- performance, ABI, and runtime-loader claims remain separate

---

## Day 11: User Documentation Implementation

**Title:** Documentation Update
**Theme:** Update user and maintainer documentation for the selected
package-manager decision
**Time estimate:** 12 hours

### Tasks

1. Update README package-manager summary.
2. Update INSTALL package-manager guidance or explicit deferral section.
3. Update maintainer guide ownership and claim boundaries.
4. Update any package docs or non-claim tables affected by the selected path.
5. Run targeted documentation claim scans.
6. Write the Day 11 documentation-update artifact.

### Deliverables

- README package-manager wording
- INSTALL package-manager guidance
- maintainer-guide ownership update
- claim-scan results
- Day 11 documentation-update artifact

### Completion Criteria

- docs match the selected provider or deferral decision
- unsupported package-manager claims remain explicit
- static-first install claims are not broadened

---

## Day 12: Provider Or Deferral Validation

**Title:** Provider Validation
**Theme:** Run package/provider proof or deferral checks with install
validation
**Time estimate:** 12 hours

### Tasks

1. Run the selected local provider proof or deferral-enforcement script.
2. Run static-first package deferral guard validation.
3. Run Make install/`pkg-config` proof if package metadata or install docs
   changed.
4. Run CMake install/export proof if CMake package expectations changed.
5. Check whether `.c` or `.h` files changed and run the full C quality gate if
   required.
6. Write the Day 12 validation artifact.

### Deliverables

- provider/deferral validation log
- static package guard validation log
- install proof decision and results
- C quality-gate decision
- Day 12 validation artifact

### Completion Criteria

- selected package-manager path or deferral validates locally
- install/package proof remains green where relevant
- failures stop the sprint for user input

---

## Day 13: Integrated Claim Review

**Title:** Claim Review
**Theme:** Reconcile recipe/proof, guards, docs, and validation into one
package-manager claim boundary
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 171 artifacts and working notes.
2. Reconcile selected package-manager decision against README, INSTALL,
   maintainer guide, guards, and validation results.
3. Run targeted package-manager, shared-library, ABI, runtime-loader, and
   platform claim scans.
4. Confirm no generated package artifacts are listed for staging.
5. Identify residuals and Sprint 172 handoff needs.
6. Write the Day 13 integrated-claim-review artifact.

### Deliverables

- integrated claim review
- claim-scan results
- generated-output staging check
- residual and handoff list
- Day 13 claim-review artifact

### Completion Criteria

- package-manager claim boundary is internally coherent
- source install, CMake/`pkg-config`, and provider support remain distinct
- no generated artifacts are staged unintentionally

---

## Day 14: Sprint Closeout And Sprint 172 Handoff

**Title:** Sprint Closeout
**Theme:** Reconcile Sprint 171 deliverables, final validation, and handoff to
Sprint 172 public-header coherence work
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 171 outcomes against project-plan items 171.1 through
   171.6.
2. Confirm the provider decision, recipe/proof or deferral artifact, guard
   updates, documentation alignment, and validation artifacts are complete.
3. Re-run final lightweight hygiene checks and any required focused checks.
4. Confirm no generated output is staged unintentionally.
5. Prepare Sprint 172 handoff notes for public-header coherence work.
6. Write the Day 14 sprint-closeout artifact.

### Deliverables

- final Sprint 171 validation record
- project-plan item reconciliation
- generated-output staging check
- Sprint 172 handoff
- Day 14 sprint-closeout artifact

### Completion Criteria

- one package-manager readiness decision is source-controlled and enforceable
- documentation and guards match the selected decision
- Sprint 172 can begin from a clear package/adoption boundary
