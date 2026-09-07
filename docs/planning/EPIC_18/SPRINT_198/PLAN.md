# Sprint 198 Plan: Homebrew License Metadata and Formula Proof Closure

**Sprint Duration:** 14 days
**Goal:** Close the package-manager/Homebrew blocker by adding approved
license metadata, proving the selected local formula workflow, and promoting
only the earned support claim.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 198 estimate in the Epic 18
project plan.

**Primary scope:** Record the approved standalone license metadata decision,
implement exact root and formula license metadata, prove the selected local
Homebrew formula path through archive/checksum, render, install, `brew test`,
uninstall, and cleanup, update package-manager guards, and recalibrate public
and maintainer documentation to the exact support level earned by evidence.

**Non-goals:** Homebrew/core readiness, bottle support, Linuxbrew support,
public tap maintenance, binary package distribution, other package managers,
shared-library package support, dynamic ABI guarantees, broad package-manager
support, or solver/API behavior changes.

---

## Day 1: Sprint Intake and License Evidence Map

**Title:** Package Metadata Intake
**Theme:** Establish the Sprint 198 scope, current blocker state, and evidence
owners before changing metadata.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 198 section of the Epic 18 project plan and map items
   198.1 through 198.6 to expected artifacts.
2. Inventory current package-manager owners: root license files, Homebrew
   template, proof script, package guards, package docs, install docs, and
   maintainer guidance.
3. Review Sprint 188 Homebrew proof artifacts and Epic 18 residual records for
   prior decisions, blockers, and retained non-claims.
4. Run package proof and guard commands in observation mode when local tools
   are available, and record current failure mode.
5. Create `WORKING_NOTES.md` with sprint checklist, evidence ledger, validation
   matrix, risk register, and open questions.

### Deliverables

- Sprint 198 working-notes scaffold.
- Package metadata and proof-owner inventory.
- Current Homebrew blocker record.
- Item-to-artifact traceability map.

### Completion Criteria

- Every Sprint 198 item has an identified owner artifact.
- Current Homebrew proof state is known before metadata edits begin.
- Unsupported package-manager claims are recorded as non-goals.

---

## Day 2: Approved License Metadata Decision

**Title:** License Decision
**Theme:** Record the approved standalone root license metadata and exact
Homebrew formula identifier.
**Time estimate:** 12 hours

### Tasks

1. Inspect existing repository license references, package metadata, formula
   placeholders, README wording, and maintainer guidance.
2. Confirm the approved standalone root license metadata file and its expected
   source archive path.
3. Select the exact Homebrew formula license identifier and document why it
   matches the approved metadata.
4. Define invalid metadata values that guards must reject, including
   placeholders and missing root files.
5. Produce the Day 2 decision artifact for item 198.1.

### Deliverables

- License metadata decision record.
- Selected root metadata file path.
- Selected Homebrew formula license identifier.
- Invalid metadata and placeholder list.

### Completion Criteria

- The license decision is explicit enough to implement without guesswork.
- The formula license identifier has a documented source.
- Placeholder and missing-license states remain blocker evidence.

---

## Day 3: Root Metadata Implementation

**Title:** Root License Metadata
**Theme:** Add or update standalone root license metadata according to the Day
2 decision.
**Time estimate:** 12 hours

### Tasks

1. Add or update the approved root license metadata file.
2. Verify the file is included by the local source archive inputs used by the
   Homebrew proof.
3. Remove stale placeholder references that treated missing metadata as a
   durable expected state.
4. Update working notes with changed files, validation impact, and any
   compatibility risk.
5. Run focused checks that can validate metadata presence without requiring
   Homebrew.

### Deliverables

- Implemented standalone root license metadata.
- Archive inclusion expectation.
- Focused metadata validation notes.
- Updated working-notes evidence ledger.

### Completion Criteria

- The repository has the approved standalone metadata file.
- Package scripts can find the metadata path deterministically.
- Missing-license blocker wording no longer contradicts implemented metadata.

---

## Day 4: Formula Metadata Wiring

**Title:** Formula License Wiring
**Theme:** Wire exact license metadata into the Homebrew formula template and
render path.
**Time estimate:** 12 hours

### Tasks

1. Update the Homebrew formula template or render inputs with the exact license
   identifier from Day 2.
2. Ensure formula rendering fails if the license value is missing, unresolved,
   or placeholder-like.
3. Verify homepage, version, source archive URL, SHA-256, and static package
   metadata remain coherent.
4. Confirm formula comments and README language still describe a temporary
   local proof formula.
5. Record template and render-path evidence for item 198.2.

### Deliverables

- Formula metadata implementation.
- Render-path placeholder rejection.
- Formula-template coherence notes.
- Item 198.2 evidence update.

### Completion Criteria

- Rendered formulas carry the exact approved license identifier.
- Unresolved or placeholder license metadata fails before install.
- The template remains scoped to local static source proof.

---

## Day 5: Archive and Checksum Proof Hardening

**Title:** Archive Proof
**Theme:** Harden deterministic source archive and checksum proof behavior.
**Time estimate:** 12 hours

### Tasks

1. Review source archive creation for included files, excluded generated
   outputs, temp-root handling, and cleanup.
2. Ensure the approved license metadata is included in the archive used by the
   formula proof.
3. Harden SHA-256 calculation and formula injection diagnostics.
4. Add or update checks that fail clearly on missing required archive entries.
5. Record expected pass/fail output in working notes.

### Deliverables

- Hardened archive/checksum proof path.
- Required archive entry checklist.
- Clear diagnostics for archive metadata failures.
- Day 5 validation notes.

### Completion Criteria

- Archive creation proves the license metadata is packaged.
- Checksum injection is deterministic and logged.
- Generated proof outputs remain outside version control.

---

## Day 6: Formula Render Validation

**Title:** Render Proof
**Theme:** Validate the rendered local Homebrew formula before install
execution.
**Time estimate:** 12 hours

### Tasks

1. Run or dry-run the formula render path with the local archive and checksum
   inputs.
2. Verify there are no unresolved placeholders in the rendered formula.
3. Validate the rendered formula includes exact version, homepage, license,
   archive URL, SHA-256, and static build expectations.
4. Confirm render diagnostics separate missing Homebrew from metadata or
   template failures.
5. Save the Day 6 render validation artifact.

### Deliverables

- Rendered-formula validation record.
- Placeholder-free formula evidence.
- Tool-availability and failure-mode notes.
- Updated proof checklist.

### Completion Criteria

- Formula rendering is independently validated before install.
- Metadata failures are distinguishable from local environment residuals.
- Render evidence does not imply Homebrew/core or bottle readiness.

---

## Day 7: Install Surface Proof

**Title:** Install Surface
**Theme:** Prove the installed static package surface from the rendered local
formula.
**Time estimate:** 12 hours

### Tasks

1. Run the local formula install path where Homebrew is available.
2. Verify installed headers, static archive, CMake package files, and
   `sparse.pc` metadata.
3. Reject unsupported shared-library artifacts, static/shared selectors, export
   macros, runtime-loader wording, and dynamic ABI claims.
4. Harden cleanup for partial installs, temporary formula paths, caches, and
   prefixes.
5. Record install proof output and residuals.

### Deliverables

- Installed static package proof notes.
- Installed artifact checklist.
- Shared/ABI rejection evidence.
- Cleanup and retry notes.

### Completion Criteria

- The installed package surface is complete for the maintained static path.
- Unsupported shared or ABI surfaces cannot count as proof.
- Failed attempts leave clear diagnostics and cleanup behavior.

---

## Day 8: Downstream Formula Test Proof

**Title:** Formula Test
**Theme:** Prove `brew test` exercises the installed CMake consumer path.
**Time estimate:** 12 hours

### Tasks

1. Review and update the formula `test do` block for exact-version
   `find_package(Sparse ...)`.
2. Ensure the test links `Sparse::sparse_lu_ortho` and builds a minimal
   executable against the installed package.
3. Run `brew test` where Homebrew is available and capture diagnostics.
4. Confirm the downstream test does not rely on source-tree include or build
   paths.
5. Record item 198.3 test proof evidence.

### Deliverables

- Downstream CMake consumer proof.
- `brew test` command evidence or environment residual.
- Formula test hardening notes.
- Item 198.3 evidence update.

### Completion Criteria

- `brew test` validates the installed static package surface.
- Source-tree leakage is rejected or absent.
- Test success is not broadened into public package-manager support.

---

## Day 9: End-to-End Local Formula Proof

**Title:** Full Formula Proof
**Theme:** Run and stabilize the complete local Homebrew proof sequence.
**Time estimate:** 12 hours

### Tasks

1. Run the full proof sequence: archive, checksum, render, install,
   `brew test`, uninstall, and cleanup.
2. Capture proof exit status, key output lines, environment details, and
   generated artifact locations.
3. Fix deterministic failures found in the end-to-end path.
4. Confirm unavailable local Homebrew environments remain recorded as
   residuals rather than successes.
5. Produce the Day 9 proof execution artifact.

### Deliverables

- End-to-end local Homebrew proof log.
- Failure and residual classification.
- Cleanup verification.
- Item 198.3 completion evidence.

### Completion Criteria

- A successful local proof exits `0` with all required stages completed, or an
  unavailable environment is explicitly residualized.
- Proof failures block support promotion.
- Temporary proof artifacts are not committed.

---

## Day 10: Package Guard Promotion

**Title:** Guard Promotion
**Theme:** Update package-manager and static-package guards to enforce earned
support wording and metadata.
**Time estimate:** 12 hours

### Tasks

1. Update `scripts/package_manager_deferral_check.sh` to recognize the exact
   approved metadata and proven local Homebrew formula result.
2. Update static package guard expectations so package claims remain scoped to
   maintained static install evidence.
3. Preserve rejection checks for unselected package-manager providers and
   unsupported generated recipe artifacts.
4. Add guard assertions for the exact promoted wording and retained non-claim
   boundaries.
5. Run focused guard checks and record results.

### Deliverables

- Updated package-manager guard.
- Updated static-package guard expectations.
- Exact support wording assertions.
- Focused guard validation log.

### Completion Criteria

- Guards fail on missing metadata, placeholder license values, and overbroad
  package claims.
- Guards pass for the exact locally proven Homebrew formula wording.
- Unselected package-manager providers remain rejected.

---

## Day 11: Public Documentation Promotion

**Title:** Public Package Docs
**Theme:** Update public documentation to reflect the exact earned support
tier and retained non-claims.
**Time estimate:** 12 hours

### Tasks

1. Update `README.md` package-manager wording with the proof result and exact
   support boundary.
2. Update `INSTALL.md` so users can distinguish source install support, local
   Homebrew formula proof, and unclaimed distribution channels.
3. Update `packaging/homebrew/README.md` with command flow, prerequisites,
   evidence interpretation, cleanup behavior, and non-claims.
4. Keep Homebrew/core, bottle, Linuxbrew, tap, other provider, shared-library,
   and dynamic ABI support clearly unclaimed.
5. Record public documentation changes in working notes.

### Deliverables

- Updated README package claim surface.
- Updated INSTALL package-manager guidance.
- Updated Homebrew proof README.
- Public-doc claim evidence log.

### Completion Criteria

- Public docs match the proof and guard vocabulary.
- Users are not told to use an unmaintained distribution channel.
- Stronger claims are backed by exact Day 9 evidence.

---

## Day 12: Maintainer Guidance and Planning Alignment

**Title:** Maintainer Alignment
**Theme:** Align maintainer guidance, planning artifacts, and residual queues
with the final package proof status.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` with exact guard ownership, proof command
   expectations, metadata requirements, and claim boundaries.
2. Update Sprint 198 working notes with status for items 198.1 through 198.6.
3. Record any residual package-manager gaps that remain after local proof.
4. Check Epic 18 planning references for stale missing-license blocker text.
5. Draft retrospective inputs while evidence is fresh.

### Deliverables

- Maintainer guidance updates.
- Item status ledger.
- Residual package gap list.
- Retrospective draft inputs.

### Completion Criteria

- Maintainers have one coherent package proof runbook.
- Planning status does not overstate support beyond evidence.
- Residuals are explicit and do not conflict with public docs.

---

## Day 13: Integrated Validation

**Title:** Validation Gate
**Theme:** Run the required package, docs, install, and code quality gates.
**Time estimate:** 12 hours

### Tasks

1. Run the Homebrew local formula proof when the local environment supports it.
2. Run package-manager guard, static-package guard, install checks, docs
   checks, and any focused tests changed by the sprint.
3. Run `make format && make lint && make test` if any `.c` or `.h` files were
   modified.
4. Re-run focused checks after any validation fix.
5. Save a concise Day 13 validation log with command results and residuals.

### Deliverables

- Integrated validation log.
- Passing package and documentation guards.
- Full C quality gate record if code changed.
- Residual environment note for unavailable Homebrew proof, if applicable.

### Completion Criteria

- Required local checks pass before closeout.
- Any unavailable environment is recorded as a residual, not success.
- Changed files have validation coverage appropriate to their risk.

---

## Day 14: Closeout Review and Retrospective Inputs

**Title:** Closeout Review
**Theme:** Finalize evidence, claim boundaries, and retrospective-ready
closure notes.
**Time estimate:** 12 hours

### Tasks

1. Review all Sprint 198 artifacts against items 198.1 through 198.6.
2. Confirm public and maintainer docs agree with package guards and proof
   evidence.
3. Verify no generated Homebrew formula outputs, archives, caches, or Python
   bytecode are staged.
4. Draft `RETROSPECTIVE.md` source notes covering completed work, residuals,
   validation, and follow-up candidates.
5. Run final documentation-only or code-aware status checks before commit.

### Deliverables

- Sprint 198 closeout artifact.
- Retrospective source notes.
- Final claim-boundary checklist.
- Clean generated-artifact review.

### Completion Criteria

- Sprint 198 is ready for retrospective and PR creation.
- Completed items are supported by evidence links.
- Residuals and non-claims are visible and not contradicted by docs.
