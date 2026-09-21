# Sprint 207 Plan: Package Distribution Support Decision

**Sprint Duration:** 14 days
**Goal:** Promote one exact package distribution path, or close the package
distribution residual with stronger deferral guards and no user-facing package
claim.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 207 estimate in the Epic 19
project plan.

**Primary scope:** Decide whether the project can earn a bounded package
distribution claim, audit the existing Homebrew proof and metadata, implement
the selected proof or deferral path, align package guards and docs with the
selected support tier, and validate that user-facing package claims match
available evidence.

**Non-goals:** Homebrew/core acceptance unless explicitly selected and proven,
bottle support, Linuxbrew support, broad package-manager distribution, ABI or
shared-library support, binary release support, portable install guarantees,
platform parity, performance claims, or state-of-the-art claims.

---

## Day 1: Package Intake

**Title:** Package Intake
**Theme:** Establish Sprint 207 scope, inherited package evidence, and package
claim boundaries before changing provider-facing files.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 207 Epic 19 project-plan section and map items 207.1
   through 207.6 to artifacts, docs, scripts, formulas, guards, and validation
   commands.
2. Inventory Sprint 198 package metadata, Homebrew proof, PR review follow-up,
   and residual decisions that affect current package distribution status.
3. Review `packaging/homebrew/`, package scripts, root license metadata,
   README, INSTALL, maintainer guide, support matrix, and package guards.
4. Create `WORKING_NOTES.md` with an item checklist, decision log, evidence
   map, risk register, validation matrix, and open questions.
5. Record explicit non-goals for Homebrew/core, bottles, Linuxbrew, broad
   package-manager support, ABI, releases, and platform parity.

### Deliverables

- Sprint 207 working-notes scaffold.
- Item-to-evidence traceability map.
- Current package surface inventory.
- Initial package decision risk register and validation matrix.

### Completion Criteria

- Every Sprint 207 item has an initial evidence path or artifact category.
- Existing Homebrew and package metadata evidence is identified before edits.
- Unsupported package, ABI, binary, platform, and release claims remain out of
  scope.

---

## Day 2: Provider Scope Options

**Title:** Provider Options
**Theme:** Compare exact package provider paths and define the evidence needed
for each possible decision.
**Time estimate:** 12 hours

### Tasks

1. Define the public Homebrew tap/source-formula option, including formula
   ownership, install proof, metadata, and maintenance implications.
2. Define the Homebrew/core readiness option, including license, URL,
   checksum, stable versioning, audit, test, and upstream submission needs.
3. Define the continued-deferral option, including stronger guards, public
   wording, residual queue entries, and validation requirements.
4. Score each option by user value, evidence cost, maintenance burden,
   platform risk, claim risk, and review surface.
5. Write the Day 2 provider-scope option artifact.

### Deliverables

- Package provider option matrix.
- Evidence and maintenance scoring table.
- Preliminary selected-path candidates and rejected-path risks.
- Day 2 provider-scope artifact.

### Completion Criteria

- Item 207.1 has concrete provider options before a decision is made.
- Every option has explicit validation requirements.
- Unsupported package claims cannot be inferred from undecided options.

---

## Day 3: Formula And Metadata Baseline

**Title:** Formula Baseline
**Theme:** Audit package formula inputs, metadata, archive behavior, and
installed static surface against the provider options.
**Time estimate:** 12 hours

### Tasks

1. Inspect `packaging/homebrew/` formulas, templates, helper scripts, and
   local proof scripts for source archive, checksum, license, dependency, and
   install behavior.
2. Audit root metadata files, MIT license wording, package docs, install docs,
   and formula metadata for provider compatibility.
3. Verify static install surfaces: headers, archive library, examples,
   pkg-config or CMake metadata if present, and installed documentation.
4. Identify stale metadata, missing reproducibility fields, hard-coded local
   paths, cleanup hazards, or unguarded package claims.
5. Write the Day 3 formula and metadata baseline artifact.

### Deliverables

- Formula and metadata audit.
- Installed static surface inventory.
- Source archive and checksum behavior notes.
- Day 3 baseline artifact.

### Completion Criteria

- Item 207.2 has current evidence for formula and metadata state.
- Static install contents are understood before proof changes.
- Any missing metadata or reproducibility issue is recorded with an owner.

---

## Day 4: Environment And Proof Baseline

**Title:** Proof Baseline
**Theme:** Reproduce or block the existing package proof and record exact
environment constraints.
**Time estimate:** 12 hours

### Tasks

1. Run or attempt the existing local Homebrew proof command and record exact
   command output, environment variables, Homebrew config, and blockers.
2. Capture host platform, compiler, CLT or Xcode state, Homebrew support tier,
   CMake state, and proof cleanup behavior.
3. Verify whether package proof scripts leave taps, temp files, Cellar entries,
   logs, or untracked repository changes.
4. Identify which failures are project defects, provider limitations,
   environment blockers, or intentionally unsupported platform states.
5. Write the Day 4 proof-baseline artifact.

### Deliverables

- Package proof command record.
- Environment support-tier inventory.
- Cleanup and side-effect audit.
- Day 4 proof-baseline artifact.

### Completion Criteria

- Item 207.2 has reproducible package proof baseline evidence or a documented
  environment blocker.
- Environment limitations are separated from project implementation defects.
- Cleanup requirements are known before proof path implementation.

---

## Day 5: Provider Decision

**Title:** Provider Decision
**Theme:** Select exactly one package distribution path or close the residual
with explicit continued deferral.
**Time estimate:** 12 hours

### Tasks

1. Apply the Day 2 option criteria to the Day 3 and Day 4 evidence.
2. Select one path: public Homebrew tap/source formula, Homebrew/core
   readiness, or explicit continued deferral with stronger guards.
3. Document rejected paths with exact evidence gaps, maintenance costs, and
   residual follow-up requirements.
4. Define the exact files, scripts, docs, guards, tests, and artifacts allowed
   to change for the selected path.
5. Write the Day 5 provider decision artifact and update `WORKING_NOTES.md`.

### Deliverables

- Explicit provider-scope decision.
- Rejected-option residual ledger.
- Selected implementation boundary.
- Day 5 decision artifact.

### Completion Criteria

- Item 207.1 is complete with one selected decision path.
- Unselected package paths remain non-claims with documented rationale.
- Implementation can proceed without ambiguity about provider scope.

---

## Day 6: Proof Design

**Title:** Proof Design
**Theme:** Design the selected proof or deferral implementation before editing
package scripts and guards.
**Time estimate:** 12 hours

### Tasks

1. Map the selected provider decision to required changes in formula files,
   scripts, checksums, archive generation, cleanup, docs, and validation.
2. Define environment gates for unsupported Homebrew tiers, missing tooling,
   stale CLT or Xcode states, and provider-specific blockers.
3. Specify formula audit behavior, install proof behavior, uninstall cleanup,
   and failure diagnostics for the selected path.
4. Map each proof or deferral change to a regression test, guard check, or
   documented validation command.
5. Write the Day 6 proof-path design artifact.

### Deliverables

- Proof-path implementation design.
- Environment-gate and cleanup design.
- Validation and regression mapping.
- Day 6 design artifact.

### Completion Criteria

- Item 207.3 has an implementation-ready design.
- Selected proof commands have clear pass, fail, and blocked states.
- Deferral or promotion cannot accidentally imply broader package support.

---

## Day 7: Proof Implementation Batch One

**Title:** Proof Implementation One
**Theme:** Implement the selected formula, script, archive, or deferral
changes with narrowly scoped package behavior.
**Time estimate:** 12 hours

### Tasks

1. Update package proof scripts, formula templates, metadata checks, or
   deferral guards according to the Day 6 design.
2. Add diagnostics for unsupported provider scope, missing dependencies,
   stale toolchains, unsupported Homebrew tiers, and cleanup failures.
3. Preserve existing static source proof behavior unless the selected decision
   explicitly replaces it.
4. Record changed files, command examples, and package claim rationale in
   `WORKING_NOTES.md`.
5. Run focused script syntax or dry-run checks available without external
   package-provider side effects.

### Deliverables

- First package proof or deferral implementation batch.
- Improved package diagnostics.
- Focused syntax or dry-run validation notes.
- Day 7 implementation record.

### Completion Criteria

- Item 207.3 has concrete implementation progress.
- Existing package proof behavior is either preserved or intentionally
  replaced with documented rationale.
- Unsupported provider states fail clearly.

---

## Day 8: Proof Implementation Batch Two

**Title:** Proof Implementation Two
**Theme:** Complete selected package proof behavior and cleanup handling.
**Time estimate:** 12 hours

### Tasks

1. Finish formula, checksum, archive, install, uninstall, and cleanup changes
   needed by the selected decision.
2. Add or update regression fixtures for formula metadata, generated archive
   behavior, proof cleanup, and environment-gate diagnostics.
3. Verify the selected proof command either passes in the available
   environment or reports a claim-safe blocker.
4. Ensure temporary taps, Cellar entries, build logs, and source archives are
   handled according to the selected policy.
5. Write the Day 8 proof implementation artifact.

### Deliverables

- Completed proof or deferral implementation.
- Regression fixtures for package proof behavior.
- Cleanup and blocker evidence.
- Day 8 implementation artifact.

### Completion Criteria

- Item 207.3 is implementation-complete for the selected path.
- Proof cleanup behavior is guarded or documented.
- Any environment blocker prevents support promotion rather than weakening the
  decision boundary.

---

## Day 9: Package Guard Alignment

**Title:** Guard Alignment
**Theme:** Align package-manager and static-package guards with the selected
support tier and retained non-claims.
**Time estimate:** 12 hours

### Tasks

1. Review existing package-manager, static-package, support-matrix, README,
   INSTALL, and maintainer claim guards.
2. Update guards to enforce the selected provider scope, support tier,
   package proof status, and retained non-claims.
3. Add regression coverage for forbidden broad package-manager claims,
   Homebrew/core overclaims, bottle claims, Linuxbrew claims, ABI claims, and
   release artifact claims.
4. Ensure guard diagnostics point maintainers to the selected provider
   decision and validation commands.
5. Write the Day 9 guard-alignment artifact.

### Deliverables

- Updated package and static support guards.
- Regression tests for retained package non-claims.
- Guard diagnostic update notes.
- Day 9 alignment artifact.

### Completion Criteria

- Item 207.4 is covered by enforceable guard behavior.
- Unselected package paths remain blocked by validation.
- Guard failures explain how to update evidence before changing claims.

---

## Day 10: User Documentation

**Title:** User Package Docs
**Theme:** Update public package, install, support, and adoption documentation
with only the earned provider status.
**Time estimate:** 12 hours

### Tasks

1. Update README and INSTALL package wording to match the selected provider
   decision and proof evidence.
2. Update support matrix, adoption quick reference, and package sections so
   users can distinguish source builds, local proofs, provider support, and
   retained non-claims.
3. Add exact commands only when they are supported by the selected proof or
   documented as local evidence with environment caveats.
4. Remove or qualify stale package caveats that conflict with the selected
   decision.
5. Record public documentation changes and claim rationale in
   `WORKING_NOTES.md`.

### Deliverables

- Updated README and INSTALL package guidance.
- Updated support/adoption package routing.
- Public claim-boundary notes.
- Day 10 documentation record.

### Completion Criteria

- Item 207.5 has user-facing documentation implementation.
- Users can tell what package path is supported, deferred, or local-only.
- Public docs do not imply Homebrew/core, bottles, Linuxbrew, ABI, release, or
  broad package-manager support unless explicitly proven.

---

## Day 11: Maintainer Documentation

**Title:** Maintainer Package Docs
**Theme:** Update maintainer and planning-adjacent documentation with package
ownership, validation, and residual decisions.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer guide package validation, claim-boundary, release, and
   support/readiness sections.
2. Update packaging docs, residual queue notes, and Sprint 207 planning
   artifacts to route maintainers to the selected proof or deferral policy.
3. Document when maintainers may change package claims and which commands or
   hosted evidence must pass first.
4. Preserve historical Sprint 198 evidence while identifying what Sprint 207
   supersedes, narrows, or retains.
5. Write the Day 11 maintainer documentation artifact.

### Deliverables

- Updated maintainer package guidance.
- Package validation ownership notes.
- Residual and supersession notes.
- Day 11 maintainer-doc artifact.

### Completion Criteria

- Item 207.5 has maintainer-facing documentation implementation.
- Maintainers have exact evidence requirements for future package claim
  changes.
- Historical package evidence is not mistaken for broader current support.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run package proof, guard, documentation, and C quality checks
required by the final changed surface.
**Time estimate:** 12 hours

### Tasks

1. Run the selected package proof command or record a claim-safe environment
   blocker with exact output.
2. Run package-manager, static-package, support/readiness, documentation, and
   claim-boundary guards affected by the sprint.
3. Run script or Python regression tests added or changed during Sprint 207.
4. Run `make format && make lint && make test` if any `.c` or `.h` files were
   modified.
5. Write the Day 12 integrated validation artifact with commands, status,
   blockers, and follow-up items.

### Deliverables

- Integrated package validation command log.
- Guard and regression-test results.
- Full C gate result or explicit documentation-only rationale.
- Day 12 validation artifact.

### Completion Criteria

- Item 207.6 has integrated validation evidence.
- All required checks pass before closeout, or the sprint stops with a clear
  blocker.
- No support claim is promoted on failed, skipped, or ambiguous evidence.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Audit the package decision, guards, docs, and validation evidence
for overclaim risk before closeout.
**Time estimate:** 12 hours

### Tasks

1. Review every changed package script, formula, metadata file, guard, test,
   and documentation surface for unnecessary breadth.
2. Verify that selected provider wording is consistent across README,
   INSTALL, support matrix, maintainer guide, packaging docs, and planning
   artifacts.
3. Check guard fixtures against likely review concerns: broad package-manager
   support, Homebrew/core readiness, bottles, Linuxbrew, ABI, releases, and
   platform parity.
4. Fix stale evidence links, contradictory status wording, and unclear
   residual entries discovered during review.
5. Write the Day 13 review-hardening artifact.

### Deliverables

- Review-surface audit.
- Claim consistency corrections.
- Guard and fixture hardening notes.
- Day 13 review-hardening artifact.

### Completion Criteria

- Package support wording is consistent across all changed surfaces.
- Retained non-claims are protected by docs and guards.
- The branch is ready for final closeout validation.

---

## Day 14: Closeout And Retrospective Prep

**Title:** Package Closeout
**Theme:** Finalize Sprint 207 evidence, residuals, project-plan status, and
retrospective inputs.
**Time estimate:** 10 hours

### Tasks

1. Re-run focused validation needed after Day 13 changes and record final
   status.
2. Update Epic 19 project-plan status, residual notes, and Sprint 207 working
   notes with final package decision evidence.
3. Record completed items 207.1 through 207.6, partial items, blockers, and
   residual package work.
4. Prepare retrospective source material: accomplishments, validation,
   residuals, risks, changed surfaces, and claim-boundary summary.
5. Perform final `git status`, changed-file inventory, and documentation-only
   or C-gate rationale.

### Deliverables

- Final Sprint 207 status ledger.
- Residual package distribution queue.
- Retrospective input notes.
- Day 14 closeout artifact.

### Completion Criteria

- Sprint 207 has evidence-backed status for items 207.1 through 207.6.
- Package support is either promoted only to the selected proven tier or
  explicitly deferred with stronger guards.
- Retrospective creation can proceed without unresolved validation ambiguity.
