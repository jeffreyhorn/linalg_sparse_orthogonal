# Sprint 180 Plan: Package-Manager Provider Decision

**Sprint Duration:** 14 days
**Goal:** Prove one package-manager provider path or close the provider
question with a stronger formal deferral and guard update. This sprint
implements the Sprint 180 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_180/PLAN.md` and implements the Sprint 180
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 180 begins from:

- the static-first package contract maintained by prior Epic 16 work;
- the Sprint 171 package-manager deferral guard;
- the Sprint 177 acceptance gate for selecting a preferred provider candidate
  or renewed deferral criteria;
- current package metadata, install/export proof, downstream compile proof,
  version-query behavior, and cleanup checks;
- current README, INSTALL, maintainer guide, API reference, and package
  support-tier wording;
- Epic 16 emphasis on evidence-backed provider decisions rather than
  aspirational package-manager claims.

The sprint must:

- compare vcpkg, Homebrew, Conan, and pkgsrc for static-first fit, CI
  feasibility, recipe complexity, maintenance cost, user value, and claim risk;
- select one provider proof path or write a stronger formal deferral with exact
  blockers and revisit criteria;
- add source-controlled provider prototype material or a stronger provider
  deferral artifact;
- add a local proof path for install, downstream compile, version query,
  cleanup, and claim-safe failure behavior where feasible;
- update package-manager guards, README, INSTALL, maintainer guide, package
  metadata, and non-claim wording;
- run package/install checks, provider proof or deferral checks, relevant docs
  checks, and whitespace review.

**End State:** Sprint 180 leaves behind:

- one package-manager provider product decision;
- a provider proof artifact or stronger formal deferral;
- proof-script behavior for the selected path or deferral;
- updated package docs, metadata wording, and guard behavior;
- validation records for package, install, provider, docs, and whitespace
  checks;
- Sprint 180 working notes, daily artifacts, and retrospective inputs.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 180 project-plan estimate.

---

## Day 1: Provider Decision Intake

**Title:** Decision Intake
**Theme:** Establish Sprint 180 scope, inherited evidence, artifact layout,
and provider decision criteria
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 180 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review the Sprint 171 package-manager deferral guard and related package
   non-claims.
3. Review the Sprint 177 acceptance gate and preferred-provider or renewed
   deferral criteria.
4. Create Sprint 180 working notes and artifact directory structure.
5. Define provider evaluation criteria for static-first fit, CI feasibility,
   recipe complexity, maintenance cost, user value, and claim risk.
6. Write the Day 1 provider-decision-intake artifact.

### Deliverables

- Sprint 180 working-notes baseline
- artifact directory structure
- inherited guard and acceptance-gate notes
- provider evaluation criteria
- Day 1 provider-decision-intake artifact

### Completion Criteria

- Sprint 180 scope is tied to the Epic 16 project plan
- inherited deferral and acceptance-gate requirements are explicit
- provider comparison work starts from shared decision criteria

---

## Day 2: Current Package Surface Audit

**Title:** Package Surface Audit
**Theme:** Inventory current package metadata, install/export proof,
downstream proof, and package-manager claim surfaces
**Time estimate:** 12 hours

### Tasks

1. Inspect package metadata, CMake install/export files, `pkg-config` files,
   and static-first package contract owners.
2. Inspect current install, downstream compile, version query, and cleanup
   scripts or checks.
3. Inventory README, INSTALL, maintainer guide, API reference, and support-tier
   package-manager wording.
4. Identify package-manager claims, non-claims, warning text, and guard
   behavior that may change during the sprint.
5. Record current failure behavior for unsupported package-manager paths.
6. Write the Day 2 package-surface-audit artifact.

### Deliverables

- package metadata inventory
- install/export and downstream proof inventory
- package-manager claim and non-claim inventory
- current guard and failure-behavior notes
- Day 2 package-surface-audit artifact

### Completion Criteria

- every relevant package surface is accounted for before provider evaluation
- current supported and unsupported package-manager wording is explicit
- implementation candidates are tied to existing proof owners

---

## Day 3: vcpkg Feasibility Audit

**Title:** vcpkg Audit
**Theme:** Evaluate vcpkg as a static-first provider proof candidate
**Time estimate:** 12 hours

### Tasks

1. Review vcpkg recipe requirements against the project install/export layout.
2. Assess static-only behavior, feature flags, dependency declaration, license
   metadata, and versioning expectations.
3. Identify local and CI proof requirements for a vcpkg prototype.
4. Estimate recipe complexity, maintenance burden, and user value.
5. Identify claim risks and failure modes specific to vcpkg.
6. Write the Day 3 vcpkg-feasibility artifact.

### Deliverables

- vcpkg fit assessment
- vcpkg recipe complexity notes
- vcpkg local and CI proof requirements
- vcpkg claim-risk notes
- Day 3 vcpkg-feasibility artifact

### Completion Criteria

- vcpkg is evaluated against the shared provider criteria
- prototype blockers and required proof are concrete
- vcpkg remains eligible or is rejected with evidence

---

## Day 4: Homebrew Feasibility Audit

**Title:** Homebrew Audit
**Theme:** Evaluate Homebrew as a static-first provider proof candidate
**Time estimate:** 12 hours

### Tasks

1. Review Homebrew formula requirements against the project build and install
   layout.
2. Assess static-library fit, platform scope, bottle expectations, test block
   feasibility, and versioning behavior.
3. Identify local and CI proof requirements for a Homebrew prototype.
4. Estimate formula complexity, maintenance burden, and user value.
5. Identify claim risks and failure modes specific to Homebrew.
6. Write the Day 4 homebrew-feasibility artifact.

### Deliverables

- Homebrew fit assessment
- Homebrew formula complexity notes
- Homebrew local and CI proof requirements
- Homebrew claim-risk notes
- Day 4 homebrew-feasibility artifact

### Completion Criteria

- Homebrew is evaluated against the shared provider criteria
- macOS-specific and cross-platform claim boundaries are explicit
- Homebrew remains eligible or is rejected with evidence

---

## Day 5: Conan Feasibility Audit

**Title:** Conan Audit
**Theme:** Evaluate Conan as a static-first provider proof candidate
**Time estimate:** 12 hours

### Tasks

1. Review Conan recipe requirements against the project CMake package and
   static-first contract.
2. Assess package ID behavior, options, generators, dependency metadata,
   versioning, and profile requirements.
3. Identify local and CI proof requirements for a Conan prototype.
4. Estimate recipe complexity, maintenance burden, and user value.
5. Identify claim risks and failure modes specific to Conan.
6. Write the Day 5 conan-feasibility artifact.

### Deliverables

- Conan fit assessment
- Conan recipe complexity notes
- Conan local and CI proof requirements
- Conan claim-risk notes
- Day 5 conan-feasibility artifact

### Completion Criteria

- Conan is evaluated against the shared provider criteria
- CMake package integration requirements are explicit
- Conan remains eligible or is rejected with evidence

---

## Day 6: pkgsrc Feasibility Audit

**Title:** pkgsrc Audit
**Theme:** Evaluate pkgsrc as a static-first provider proof candidate and
complete the provider comparison set
**Time estimate:** 12 hours

### Tasks

1. Review pkgsrc package requirements against the project build and install
   layout.
2. Assess static-library fit, platform assumptions, metadata requirements,
   patch requirements, and test feasibility.
3. Identify local and CI proof requirements for a pkgsrc prototype.
4. Estimate package complexity, maintenance burden, and user value.
5. Identify claim risks and failure modes specific to pkgsrc.
6. Write the Day 6 pkgsrc-feasibility artifact.

### Deliverables

- pkgsrc fit assessment
- pkgsrc package complexity notes
- pkgsrc local and CI proof requirements
- pkgsrc claim-risk notes
- Day 6 pkgsrc-feasibility artifact

### Completion Criteria

- pkgsrc is evaluated against the shared provider criteria
- the four-provider feasibility audit is complete
- pkgsrc remains eligible or is rejected with evidence

---

## Day 7: Provider Decision Matrix

**Title:** Decision Matrix
**Theme:** Compare provider candidates and select proof path or renewed
deferral candidate
**Time estimate:** 12 hours

### Tasks

1. Build a comparison matrix for vcpkg, Homebrew, Conan, and pkgsrc using the
   Day 1 criteria.
2. Compare static-first fit, CI feasibility, recipe complexity, maintenance
   cost, user value, and claim risk.
3. Identify the strongest provider proof candidate and the strongest renewed
   deferral case.
4. Record rejected-provider rationale with evidence from Days 3-6.
5. Define the decision recommendation and unresolved questions.
6. Write the Day 7 provider-decision-matrix artifact.

### Deliverables

- four-provider comparison matrix
- recommended provider proof or deferral candidate
- rejected-provider rationale
- unresolved question list
- Day 7 provider-decision-matrix artifact

### Completion Criteria

- every provider candidate is compared on the same criteria
- recommendation is evidence-backed rather than preference-based
- open questions are narrow enough for final decision work

---

## Day 8: Product Decision Record

**Title:** Product Decision Record
**Theme:** Select one provider proof path or a renewed formal deferral with
exact blockers and revisit criteria
**Time estimate:** 12 hours

### Tasks

1. Write the provider product decision record.
2. State the selected proof path or renewed deferral decision.
3. Record accepted evidence, rejected options, exact blockers, and revisit
   criteria.
4. Define support-tier wording allowed after the decision.
5. Define implementation boundaries, stop conditions, and validation gates.
6. Review the decision against static-first package contract requirements.

### Deliverables

- provider product decision record
- selected proof path or formal deferral
- blocker and revisit-criteria list
- support-tier wording boundaries
- implementation and validation gate list

### Completion Criteria

- exactly one provider decision is recorded
- rejected options have concrete rationale
- downstream implementation work has clear boundaries

---

## Day 9: Artifact Design

**Title:** Artifact Design
**Theme:** Design the provider prototype material or stronger deferral
artifact before implementation
**Time estimate:** 12 hours

### Tasks

1. Identify the files required for the selected provider prototype or deferral
   artifact.
2. Define artifact ownership, update frequency, and relationship to package
   metadata.
3. Design expected success and failure behavior for local proof commands.
4. Define how the artifact avoids unsupported package-manager claims.
5. Identify docs and guard text that must reference the artifact.
6. Write the Day 9 artifact-design artifact.

### Deliverables

- provider prototype or deferral artifact design
- expected success and failure behavior
- artifact ownership notes
- docs and guard reference checklist
- Day 9 artifact-design artifact

### Completion Criteria

- source-controlled artifact work is designed before implementation
- claim boundaries are explicit for both success and failure paths
- proof-script requirements are ready for implementation

---

## Day 10: Artifact Implementation

**Title:** Artifact Implementation
**Theme:** Add source-controlled provider prototype material or stronger
provider deferral artifact
**Time estimate:** 12 hours

### Tasks

1. Add the selected provider prototype material or formal deferral artifact.
2. Connect the artifact to existing package metadata where appropriate.
3. Add or update explanatory notes that preserve static-first support
   boundaries.
4. Add focused checks for artifact presence, freshness, or claim-safe wording
   where practical.
5. Run focused validation for the new artifact.
6. Record implementation notes and residual risks.

### Deliverables

- source-controlled provider prototype or deferral artifact
- static-first boundary notes
- focused artifact checks
- focused validation summary
- Day 10 implementation notes

### Completion Criteria

- selected artifact exists in source control
- artifact wording does not promote unsupported provider status
- focused validation passes or failures are recorded with blockers

---

## Day 11: Proof Script Design

**Title:** Proof Script Design
**Theme:** Design install, downstream compile, version query, cleanup, and
claim-safe failure proof behavior
**Time estimate:** 12 hours

### Tasks

1. Inspect existing package, install, downstream, version, cleanup, and guard
   scripts.
2. Define proof-script command flow for the selected provider path or deferral
   case.
3. Define temporary directory handling, cleanup behavior, logging, and failure
   messages.
4. Define locally feasible checks and CI-feasible checks separately.
5. Define how proof failures preserve package-manager non-claims.
6. Write the Day 11 proof-script-design artifact.

### Deliverables

- proof-script command-flow design
- cleanup and temporary-directory rules
- local-versus-CI proof split
- failure-message and non-claim requirements
- Day 11 proof-script-design artifact

### Completion Criteria

- proof-script behavior is specified before implementation
- cleanup and failure behavior are explicit
- local proof does not depend on unavailable provider infrastructure

---

## Day 12: Proof Script Implementation

**Title:** Proof Script Implementation
**Theme:** Add local proof for install, downstream compile, version query,
cleanup, and claim-safe failure behavior where feasible
**Time estimate:** 12 hours

### Tasks

1. Implement or update the provider proof or deferral proof script.
2. Wire the proof to install/export, downstream compile, version query, and
   cleanup checks where feasible.
3. Add claim-safe failure messages for unavailable tools or unsupported
   provider states.
4. Add focused tests or shell checks for success, failure, and cleanup paths
   where practical.
5. Run the focused proof command and capture output.
6. Record implementation notes and unsupported proof gaps.

### Deliverables

- provider proof or deferral proof script
- install/downstream/version/cleanup proof wiring
- claim-safe failure messages
- focused proof validation summary
- Day 12 implementation notes

### Completion Criteria

- proof script exists and is locally runnable where feasible
- unavailable-provider behavior fails safely without implying support
- cleanup behavior is validated or the remaining gap is documented

---

## Day 13: Guard And Docs Update

**Title:** Guard And Docs Update
**Theme:** Update package-manager guards, package metadata, and user-facing
non-claim documentation
**Time estimate:** 12 hours

### Tasks

1. Update package-manager guards to reflect the product decision.
2. Update README and INSTALL package-manager wording.
3. Update maintainer guide instructions for the provider proof or deferral
   artifact.
4. Update package metadata non-claims and support-tier wording as needed.
5. Add or tighten docs checks for stale or unsupported package-manager claims.
6. Run focused docs and guard checks.

### Deliverables

- updated package-manager guard behavior
- updated README and INSTALL wording
- updated maintainer guide wording
- updated package metadata and support-tier non-claims
- focused docs and guard check summary

### Completion Criteria

- documentation matches the provider decision exactly
- guards reject unsupported package-manager claims
- package metadata does not imply unearned provider support

---

## Day 14: Integrated Validation And Closeout

**Title:** Validation And Closeout
**Theme:** Run package, install, provider, docs, and whitespace checks; record
Sprint 180 closeout evidence
**Time estimate:** 12 hours

### Tasks

1. Run package and install checks relevant to the selected provider or
   deferral path.
2. Run the provider proof or deferral proof checks.
3. Run relevant docs, guard, metadata, and claim-surface checks.
4. Run `git diff --check` and any focused formatting checks required by
   touched files.
5. Record validation output, residual risks, and Sprint 181 handoff notes.
6. Prepare Sprint 180 retrospective inputs.

### Deliverables

- package and install validation summary
- provider proof or deferral validation summary
- docs, guard, metadata, and whitespace validation summary
- residual risk and handoff notes
- Sprint 180 retrospective inputs

### Completion Criteria

- Sprint 180 deliverables are validated or blockers are explicitly recorded
- product decision, artifact, proof behavior, guards, and docs are consistent
- residual risks are ready for Sprint 181 planning
