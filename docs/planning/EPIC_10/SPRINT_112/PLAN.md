# Sprint 112 Plan: Packaging, ABI & Cross-Platform Validation Expansion

**Sprint Duration:** 14 days
**Goal:** Strengthen packaging and platform support truth, including an
explicit decision on static-first versus shared-library/ABI proof. This sprint
implements the Sprint 112 section of
`docs/planning/EPIC_10/PROJECT_PLAN.md`.

**Starting Point:** Sprint 112 begins from:
- the Sprint 100 platform and package evidence template
- the Sprint 110 no-public-header-drift package/platform baseline
- the Sprint 111 user-facing docs and adoption surfaces
- existing Make, CMake, pkg-config, install, example, and downstream consumer
  validation surfaces
- an explicit need to distinguish earned package/platform support from
  unstated ABI, shared-library, Windows, or macOS claims

The sprint must:
- audit every packaging and consumer-facing surface before changing support
  wording
- decide whether Epic 10 will prove shared-library/ABI support or preserve a
  static-first support tier
- strengthen install/export/downstream consumer checks according to that
  support decision
- define Linux, macOS, and Windows support tiers with reviewed checks,
  staged exclusions, and non-claims
- perform practical Windows/macOS parity follow-through without overclaiming
  coverage
- update package, install, CMake, pkg-config, README, and maintainer docs to
  match actual validation
- close with validation evidence and a residual package/platform handoff

**End State:** Sprint 112 leaves behind:
- a package-surface audit and support-tier decision
- explicit static-first or shared-library/ABI support truth
- stronger install/export and downstream consumer proof
- documented platform tiers for Linux, macOS, and Windows
- package and CI docs aligned with actual reviewed validation
- Sprint 112 artifacts, working notes, and closeout residuals

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 112 project-plan estimate.

---

## Day 1: Sprint 112 Scope & Evidence Baseline

**Title:** Package Baseline
**Theme:** Establish the package, ABI, and platform support evidence map
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 112 section of
   `docs/planning/EPIC_10/PROJECT_PLAN.md`.
2. Re-read Sprint 100 platform/package evidence, Sprint 110 no-public-header
   drift evidence, and Sprint 111 adoption docs.
3. Inventory package and platform surfaces:
   - Make install
   - CMake install/export
   - pkg-config
   - examples
   - downstream consumer samples
   - README/install docs
   - maintainer and CI docs
4. Create Sprint 112 working notes and artifact directory.
5. Define validation expectations for documentation-only, build-system,
   example, install/export, and public-header changes.
6. Write the Day 1 evidence-baseline artifact.

### Deliverables
- package/platform surface inventory
- support-claim baseline
- Sprint 112 working-notes baseline
- validation expectations by touched surface
- Day 1 evidence-baseline artifact

### Completion Criteria
- every Sprint 112 item has an initial owner and evidence source
- no support-tier or ABI claim is made before the decision artifact
- downstream days can proceed without rediscovering package surfaces

---

## Day 2: Package Surface Audit

**Title:** Package Audit
**Theme:** Audit install, export, versioning, and consumer entry points
**Time estimate:** 12 hours

### Tasks
1. Audit Make install behavior, install paths, generated headers, libraries,
   pkg-config output, and uninstall or cleanup expectations.
2. Audit CMake install/export behavior, target names, include directories,
   package config files, and downstream `find_package` paths.
3. Audit examples and consumer documentation for package assumptions.
4. Audit versioning and exact-package behavior for current truth and gaps.
5. Identify stale, unsupported, ambiguous, or overbroad package claims.
6. Write the package-surface audit artifact.

### Deliverables
- Make install audit
- CMake install/export audit
- pkg-config and versioning audit
- downstream consumer entry-point inventory
- stale or unsupported package-claim list

### Completion Criteria
- package behavior is tied to concrete files and commands
- unclear support claims are identified before docs change
- install/export proof gaps are ordered for later days

---

## Day 3: ABI Support Options

**Title:** ABI Options
**Theme:** Compare static-first and shared-library/ABI proof paths
**Time estimate:** 12 hours

### Tasks
1. Inventory public headers, installed headers, exported symbols, library
   naming, and build products.
2. Identify what evidence would be required to claim shared-library or ABI
   compatibility support.
3. Identify what evidence supports a static-first package support tier.
4. Review Sprint 110 no-public-header-drift evidence and define its limits.
5. Compare risk, cost, validation burden, and public-doc impact for both
   support paths.
6. Write the ABI options artifact.

### Deliverables
- static-first support evidence summary
- shared-library/ABI proof requirements
- Sprint 110 evidence-use boundaries
- risk and validation comparison
- Day 3 ABI options artifact

### Completion Criteria
- Sprint 110 public-header stability is not overstated as ABI stability
- shared-library/ABI support requirements are explicit
- Day 4 can make a support-tier decision from evidence

---

## Day 4: ABI Support Decision

**Title:** Support Decision
**Theme:** Decide the package support tier and freeze public claims
**Time estimate:** 12 hours

### Tasks
1. Review Day 3 ABI options with package and platform goals.
2. Decide whether Sprint 112 proves shared-library/ABI support or preserves
   static-first support as the explicit Epic 10 tier.
3. Record non-claims for unsupported package or ABI surfaces.
4. Identify install/export and consumer checks required by the decision.
5. Identify docs that must change after validation lands.
6. Write the ABI support decision artifact.

### Deliverables
- static-first or shared-library/ABI support decision
- explicit package and ABI non-claims
- validation requirements for the selected tier
- docs update queue

### Completion Criteria
- the support-tier decision is evidence-backed and reviewable
- unsupported ABI/package claims are fenced before implementation
- install/consumer proof work is scoped to the chosen tier

---

## Day 5: Install/Consumer Proof Design

**Title:** Proof Design
**Theme:** Design install/export and downstream consumer validation
**Time estimate:** 12 hours

### Tasks
1. Define the Make install validation command set for the selected support
   tier.
2. Define the CMake install/export validation command set.
3. Define pkg-config validation and downstream compile/link checks.
4. Select example consumers that prove package behavior without broadening
   public support claims.
5. Identify temporary directories, environment variables, and cleanup rules for
   repeatable validation.
6. Write the install/consumer proof design artifact.

### Deliverables
- Make install validation plan
- CMake install/export validation plan
- pkg-config validation plan
- downstream consumer proof matrix
- cleanup and reproducibility notes

### Completion Criteria
- validation commands are concrete enough to automate or run directly
- proof design matches the Day 4 support decision
- downstream checks avoid private headers and planning-only scaffolding

---

## Day 6: Make Install Proof

**Title:** Make Install Proof
**Theme:** Strengthen Makefile install and package validation
**Time estimate:** 12 hours

### Tasks
1. Run or update Make install validation in an isolated staging prefix.
2. Verify installed headers, library artifacts, pkg-config files, and include
   paths match the selected support tier.
3. Compile at least one public example or consumer against the staged Make
   install.
4. Document any unsupported paths or staged exclusions.
5. Update build scripts or docs only if the proof exposes a real gap.
6. Record Make install validation evidence.

### Deliverables
- staged Make install validation evidence
- installed artifact inventory
- public consumer compile/link result
- Make install residuals or fixes

### Completion Criteria
- Make install truth is validated or explicitly deferred
- consumer proof uses installed public surfaces only
- no support claim exceeds the validated Make install behavior

---

## Day 7: CMake Export and pkg-config Proof

**Title:** CMake Package Proof
**Theme:** Strengthen CMake install/export and pkg-config validation
**Time estimate:** 12 hours

### Tasks
1. Run or update CMake configure, build, install, and export validation in an
   isolated staging prefix.
2. Verify exported targets, include directories, library paths, and package
   config behavior.
3. Validate pkg-config compile/link flags against the staged install.
4. Build a downstream CMake consumer using installed package metadata.
5. Document unsupported or staged CMake/package behavior.
6. Record CMake and pkg-config validation evidence.

### Deliverables
- CMake install/export validation evidence
- pkg-config validation evidence
- downstream CMake consumer proof
- CMake/package residuals or fixes

### Completion Criteria
- CMake and pkg-config package truth is validated
- downstream consumers do not depend on source-tree-only paths
- CMake support wording can be updated from evidence

---

## Day 8: Downstream Consumer Proof Batch

**Title:** Consumer Proof
**Theme:** Exercise user-facing install and package consumption paths
**Time estimate:** 12 hours

### Tasks
1. Build a small downstream consumer against the selected package surface.
2. Exercise public headers used by common workflows:
   - matrix creation
   - compressed input
   - Matrix Market load/save when supported by package artifacts
   - one direct solver
3. Verify compile/link flags are sufficient without private headers.
4. Check example and install docs against the observed consumer workflow.
5. Capture consumer proof output and failures.
6. Write the downstream consumer proof artifact.

### Deliverables
- downstream consumer source or command record
- compile/link evidence
- public-header coverage notes
- docs gaps discovered by the consumer proof

### Completion Criteria
- at least one downstream consumer proves the selected package tier
- private implementation headers are not required by consumers
- docs update needs are concrete and evidence-backed

---

## Day 9: Platform Tier Contract

**Title:** Platform Tiers
**Theme:** Define Linux, macOS, and Windows support truth
**Time estimate:** 12 hours

### Tasks
1. Inventory current reviewed checks for Linux, macOS, and Windows.
2. Separate CI-enforced, reviewed, staged, local-only, and unsupported lanes.
3. Define support-tier wording for each platform.
4. Define platform non-claims for Makefile parity, CMake parity, install
   validation, shared-library behavior, thread/runtime behavior, and excluded
   tests.
5. Identify platform docs and CI comments that need updates.
6. Write the platform-tier contract artifact.

### Deliverables
- Linux support-tier definition
- macOS support-tier definition
- Windows support-tier definition
- reviewed checks and staged exclusions list
- platform non-claims list

### Completion Criteria
- platform wording distinguishes reviewed evidence from staged exclusions
- Windows and macOS coverage is not inferred from unrelated header stability
- docs update work is ready for later days

---

## Day 10: Windows Follow-Through

**Title:** Windows Scope
**Theme:** Move practical Windows exclusions into reviewed truth or document why they remain staged
**Time estimate:** 12 hours

### Tasks
1. Review current Windows CI, CMake consumer proof, CTest registration, and
   staged exclusions.
2. Decide whether any practical Windows exclusion can move into reviewed
   parity this sprint.
3. Update validation or docs only when the evidence supports it.
4. Preserve explicit non-claims for unreviewed Makefile parity, install
   validation, or unsupported tests.
5. Record Windows reviewed scope and residual exclusions.
6. Write the Windows follow-through artifact.

### Deliverables
- Windows reviewed-scope artifact
- staged-exclusion decision list
- Windows docs or validation updates if justified
- Windows non-claims for unreviewed surfaces

### Completion Criteria
- Windows coverage is explicit and evidence-bound
- staged exclusions remain visible, not silently converted into support
- no Windows support claim exceeds reviewed checks

---

## Day 11: macOS Follow-Through

**Title:** macOS Scope
**Theme:** Confirm macOS package/platform truth and local validation boundaries
**Time estimate:** 12 hours

### Tasks
1. Review macOS local validation, compiler/backend expectations, install paths,
   and package behavior.
2. Decide whether any macOS package or platform claim needs tightening.
3. Validate practical macOS package commands available in the local
   environment.
4. Document backend, SDK, OpenMP, install, and package non-claims where needed.
5. Update docs or validation scripts only if evidence requires it.
6. Write the macOS follow-through artifact.

### Deliverables
- macOS package/platform evidence summary
- macOS local-validation boundaries
- macOS non-claims list
- docs or validation updates if justified

### Completion Criteria
- macOS support wording matches actual local and reviewed validation
- backend/runtime details are not overclaimed
- macOS residuals are ready for closeout

---

## Day 12: Packaging Documentation Alignment

**Title:** Package Docs
**Theme:** Update public and maintainer docs to match validated support truth
**Time estimate:** 12 hours

### Tasks
1. Update install docs to match package, platform, and support-tier truth.
2. Update README package/platform wording as needed.
3. Update CMake and pkg-config documentation or comments as needed.
4. Update maintainer docs with reviewed package/platform checks and exclusions.
5. Keep user docs concise and move proof details to maintainer-facing docs.
6. Record documentation changes and validation needs.

### Deliverables
- install documentation updates
- README package/platform updates if needed
- CMake/pkg-config docs updates if needed
- maintainer package/platform evidence updates
- Day 12 documentation alignment artifact

### Completion Criteria
- package docs match actual validation evidence
- support tiers and non-claims are consistent across public docs
- maintainer proof detail does not become the first adoption path

---

## Day 13: Integrated Package and Platform Validation

**Title:** Integrated Validation
**Theme:** Run the full package, consumer, platform, and docs validation set
**Time estimate:** 12 hours

### Tasks
1. Run required package, install/export, consumer, and docs checks from Days
   5-12.
2. Run code quality checks required by any touched `.c` or `.h` files.
3. Verify no public API, install-header, helper-target, or reviewed CTest drift
   was introduced unless explicitly intended and documented.
4. Verify README, install docs, maintainer docs, package metadata, and platform
   wording agree.
5. Capture command output summaries and residual failures if any.
6. Write the integrated validation artifact.

### Deliverables
- integrated validation command list
- package and consumer validation results
- public API/install-header drift check
- docs consistency checklist
- residual validation queue if needed

### Completion Criteria
- all required checks pass before closeout
- support-tier docs match the validated package/platform behavior
- any residuals are explicit and non-blocking

---

## Day 14: Sprint 112 Closeout and Handoff

**Title:** Closeout Handoff
**Theme:** Close Sprint 112 with package/platform truth and residual queue
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 112 artifacts, working notes, docs, scripts, and
   validation output.
2. Confirm all seven Sprint 112 project-plan items are complete or explicitly
   deferred.
3. Summarize the final package support tier and platform support tiers.
4. Record residual package, ABI, Windows, macOS, install/export, and consumer
   work for Sprint 113 or later.
5. Run final applicable documentation and hygiene checks.
6. Write the closeout and handoff artifact.

### Deliverables
- completed Sprint 112 item checklist
- final package and platform support summary
- residual deferred-debt queue
- validation summary
- Day 14 closeout and handoff artifact

### Completion Criteria
- all Sprint 112 items are closed or explicitly deferred
- residuals are dependency-ordered and non-duplicative
- final checks pass
- Sprint 113 and final Epic 10 closeout have clear package/platform handoff
