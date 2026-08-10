# Sprint 148 Plan: Windows Staged Test Portability Closure

**Sprint Duration:** 14 days
**Goal:** Promote or replace the staged Windows-excluded test surfaces with
reviewed Windows-compatible coverage. This sprint implements the Sprint 148
section of `docs/planning/EPIC_13/PROJECT_PLAN.md`.

**Starting Point:** Sprint 148 begins from:
- Sprint 147 Windows evidence gate and closeout handoff merged into `master`
- current reviewed Windows CMake subset in `.github/workflows/windows-ci.yml`
- current enforced Windows CTest count of `EXPECTED_WINDOWS_CTEST_COUNT=56`
- staged Windows exclusions for `test_threads`, `test_sprint4_integration`,
  and `test_fuzz`
- Sprint 147 quality surface map and public claim freeze audit

The sprint must:
- audit the staged pthread/POSIX and CMake blockers before editing
- choose direct ports, Windows-native equivalents, proof splits, retained
  staged status, or explicit replacement per test surface
- preserve Linux/macOS/POSIX proof while adding Windows-compatible coverage
- update CMake registration and Windows expected-count policy only when backed
  by concrete evidence
- align workflow comments, report rows, public docs, and non-claims
- run local feasible checks and the full C gate for any `.c` or `.h` changes
- leave Sprint 149 a clear Windows install-validation parity handoff

**End State:** Sprint 148 leaves behind:
- Windows staged-test portability closure decision and implementation
- updated CMake and Windows CI registration policy
- updated support-tier wording and report rows
- validation evidence and residual list
- Sprint 149 install-parity handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 148 project-plan estimate.

---

## Day 1: Sprint Intake And Windows Baseline Refresh

**Title:** Windows Intake
**Theme:** Re-establish Sprint 148 scope, artifact structure, current Windows
CMake baseline, and staged-test closure rules
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 148 section of
   `docs/planning/EPIC_13/PROJECT_PLAN.md`.
2. Review Sprint 147 Day 7 and Day 14 Windows handoff artifacts.
3. Create Sprint 148 working notes and artifact directory structure.
4. Capture the current Windows workflow, CMake gates, expected CTest count, and
   staged exclusions from the repo.
5. Define audit fields for `test_threads`, `test_sprint4_integration`, and
   `test_fuzz`.
6. Record stop conditions for unsupported Windows parity, unclear hosted proof,
   and expected-count drift.

### Deliverables
- Sprint 148 working-notes baseline
- artifact directory structure
- Windows CMake baseline snapshot
- staged-test audit template
- stop-condition register

### Completion Criteria
- Sprint 148 scope is tied to current files and Sprint 147 handoff artifacts
- current Windows reviewed, supplemental, staged, deferred, and unsupported
  tiers are recorded
- each staged test has an audit owner and evidence format

---

## Day 2: Staged Test Source Audit

**Title:** Source Audit
**Theme:** Audit pthread, POSIX temp-file, platform API, and test-behavior
blockers in the staged Windows-excluded tests
**Time estimate:** 12 hours

### Tasks
1. Inspect `tests/test_threads.c` for pthread API usage, synchronization
   assumptions, timing assumptions, and assertions.
2. Inspect `tests/test_sprint4_integration.c` for pthread usage and integration
   behavior that must remain covered.
3. Inspect `tests/test_fuzz.c` for POSIX temp-file assumptions, file cleanup,
   deterministic seeds, and bounded property behavior.
4. Map shared helper opportunities and platform-specific helper risks.
5. Identify any Linux/macOS proof that must not be weakened by a Windows port.
6. Write the staged source audit artifact.

### Deliverables
- per-test blocker inventory
- pthread/POSIX API usage table
- behavior-preservation map
- helper extraction candidate list
- Linux/macOS preservation warnings

### Completion Criteria
- every staged test blocker is tied to exact source behavior
- behavior to preserve is separated from platform mechanics
- no implementation direction is chosen before the blocker audit is complete

---

## Day 3: CMake, CI, And Expected-Count Audit

**Title:** Registration Audit
**Theme:** Audit CMake registration, Windows workflow enforcement, CTest count
policy, and support-tier report ownership
**Time estimate:** 12 hours

### Tasks
1. Inspect CMake test registration gates for `test_threads`,
   `test_sprint4_integration`, and `test_fuzz`.
2. Inspect `.github/workflows/windows-ci.yml` expected-count enforcement,
   workflow comments, and staged-exclusion wording.
3. Capture current Linux/macOS registration of the same tests.
4. Identify report-family rows and docs that must change if a test is promoted
   or remains staged.
5. Define the before/after CTest enumeration evidence format.
6. Write the registration audit artifact.

### Deliverables
- CMake gate inventory
- Windows workflow expected-count audit
- cross-platform registration table
- report/docs update candidate list
- CTest before/after evidence template

### Completion Criteria
- all registration owners are known before CMake edits
- expected-count changes require documented before/after evidence
- Windows reviewed and supplemental claims remain separate

---

## Day 4: Portability Decision Matrix

**Title:** Portability Decision
**Theme:** Decide per-test promotion, replacement, split-proof, or retained
staged strategy before implementation
**Time estimate:** 12 hours

### Tasks
1. Define decision options for each staged test: direct port, Windows-native
   equivalent, split proof owner, retained staged status, or rejected
   promotion.
2. Score each option by implementation risk, behavior preservation, hosted
   evidence need, and support-claim impact.
3. Choose the Sprint 148 implementation targets.
4. Define rollback rules if a selected port breaks local or hosted validation.
5. Map chosen decisions to planned code, CMake, CI, docs, and report changes.
6. Write the portability decision artifact.

### Deliverables
- per-test decision matrix
- selected implementation target list
- rollback criteria
- support-claim impact map
- implementation sequence for Days 5-10

### Completion Criteria
- every staged test has a selected Sprint 148 disposition
- selected work can be completed within the remaining sprint budget
- retained or rejected paths stay explicit non-claims

---

## Day 5: Thread Test Port Design

**Title:** Thread Design
**Theme:** Design Windows-compatible thread lifecycle coverage without weakening
existing pthread/POSIX proof
**Time estimate:** 12 hours

### Tasks
1. Translate the Day 4 `test_threads` decision into a concrete design.
2. Define any portable thread helper, Windows-native path, or split test file
   needed for reviewed Windows coverage.
3. Define assertions, timeouts, cleanup, and failure diagnostics for the
   Windows-compatible lane.
4. Define how Linux/macOS pthread coverage remains registered and executable.
5. Define CMake registration and expected-count implications.
6. Write the thread test port design artifact.

### Deliverables
- thread portability design
- helper/API boundary sketch
- assertion and timeout policy
- cross-platform registration plan
- validation checklist for the thread lane

### Completion Criteria
- implementation can proceed without changing the selected behavior
- Windows-compatible coverage is bounded to the promoted thread lifecycle
- POSIX pthread coverage remains preserved or explicitly split

---

## Day 6: Thread Test Port Implementation

**Title:** Thread Port
**Theme:** Implement the selected Windows-compatible thread lifecycle coverage
and local proof path
**Time estimate:** 12 hours

### Tasks
1. Implement the selected `test_threads` portability design.
2. Update local build/test registration needed for the implemented path.
3. Preserve or split existing POSIX coverage according to the Day 5 design.
4. Add focused diagnostics for unsupported or skipped platform paths.
5. Run focused local build/test checks for the touched thread test surface.
6. Write the thread implementation artifact and validation notes.

### Deliverables
- implemented thread portability changes
- focused thread validation notes
- CMake registration update draft or applied change
- residuals for any unpromoted thread behavior

### Completion Criteria
- thread test changes build locally on feasible platforms
- implementation does not remove existing POSIX behavior without replacement
- full C gate scope is recorded for later validation

---

## Day 7: Sprint 4 Integration Port Design

**Title:** Integration Design
**Theme:** Design Windows-compatible coverage for the Sprint 4 pthread-backed
integration behavior
**Time estimate:** 12 hours

### Tasks
1. Translate the Day 4 `test_sprint4_integration` decision into a concrete
   design.
2. Identify the integration behavior that must be preserved independently from
   pthread mechanics.
3. Define direct port, Windows-native equivalent, or split proof owner.
4. Define local and hosted validation requirements.
5. Define CMake registration and expected-count implications.
6. Write the Sprint 4 integration port design artifact.

### Deliverables
- integration portability design
- behavior-preservation checklist
- platform split or helper plan
- validation checklist for the integration lane
- expected-count impact note

### Completion Criteria
- the integration behavior to promote is precise
- Windows wording cannot imply broader platform parity
- implementation risks and rollback rules are recorded

---

## Day 8: Sprint 4 Integration Port Implementation

**Title:** Integration Port
**Theme:** Implement the selected Windows-compatible Sprint 4 integration proof
path
**Time estimate:** 12 hours

### Tasks
1. Implement the selected `test_sprint4_integration` portability design.
2. Update build/test registration needed for the implemented path.
3. Preserve existing POSIX integration proof or split it explicitly.
4. Add focused diagnostics for unsupported or platform-specific paths.
5. Run focused local build/test checks for the touched integration surface.
6. Write the integration implementation artifact and validation notes.

### Deliverables
- implemented Sprint 4 integration portability changes
- focused integration validation notes
- CMake registration update draft or applied change
- residuals for any unpromoted integration behavior

### Completion Criteria
- integration changes build locally on feasible platforms
- POSIX proof is preserved, replaced, or explicitly retained as staged
- full C gate scope is recorded for later validation

---

## Day 9: Fuzz And Property Port Design

**Title:** Fuzz Design
**Theme:** Design portable temp-file and bounded property coverage for Windows
without weakening deterministic fuzz behavior
**Time estimate:** 12 hours

### Tasks
1. Translate the Day 4 `test_fuzz` decision into a concrete design.
2. Identify POSIX temp-file assumptions and cleanup behavior to replace or
   split.
3. Define portable temp-file helper, Windows-specific helper, or bounded
   Windows property lane.
4. Define deterministic seed, artifact cleanup, timeout, and failure-diagnostic
   policy.
5. Define CMake registration and expected-count implications.
6. Write the fuzz/property port design artifact.

### Deliverables
- fuzz portability design
- temp-file helper or split-lane plan
- deterministic seed and cleanup policy
- validation checklist for the fuzz/property lane
- expected-count impact note

### Completion Criteria
- property behavior is separated from POSIX temp-file mechanics
- Windows path cannot create nondeterministic temporary file residue
- implementation and rollback rules are clear

---

## Day 10: Fuzz And Property Port Implementation

**Title:** Fuzz Port
**Theme:** Implement portable or split Windows-compatible fuzz/property
coverage
**Time estimate:** 12 hours

### Tasks
1. Implement the selected `test_fuzz` portability design.
2. Update build/test registration needed for the implemented path.
3. Preserve existing POSIX fuzz proof or split it explicitly.
4. Add cleanup checks and diagnostics for temporary file behavior.
5. Run focused local build/test checks for the touched fuzz/property surface.
6. Write the fuzz implementation artifact and validation notes.

### Deliverables
- implemented fuzz/property portability changes
- focused fuzz validation notes
- CMake registration update draft or applied change
- residuals for any unpromoted fuzz behavior

### Completion Criteria
- fuzz/property changes build locally on feasible platforms
- cleanup and deterministic behavior are validated locally where possible
- full C gate scope is recorded for later validation

---

## Day 11: CMake And Windows CI Promotion Batch

**Title:** CMake Promotion
**Theme:** Align CMake registration, Windows expected-count policy, and workflow
comments with the promoted or retained staged-test outcomes
**Time estimate:** 12 hours

### Tasks
1. Reconcile Day 6, Day 8, and Day 10 implementation outcomes.
2. Update CMake gates for promoted, split, retained staged, or rejected test
   paths.
3. Update Windows workflow expected-count policy and comments only when backed
   by local enumeration and planned hosted proof.
4. Update report-family rows or support metadata affected by promoted Windows
   coverage.
5. Run local CMake configure/build/CTest enumeration where feasible.
6. Write the CMake/CI promotion artifact.

### Deliverables
- CMake registration updates
- Windows expected-count policy update or no-change rationale
- workflow comment updates or no-change rationale
- report/support metadata updates or no-change rationale
- local CTest enumeration log

### Completion Criteria
- CMake and workflow policy match the implemented test dispositions
- expected-count changes are explained, not mechanical
- hosted Windows proof requirements remain explicit

---

## Day 12: Documentation And Support-Tier Alignment

**Title:** Docs Alignment
**Theme:** Align README, INSTALL, maintainer guide, report rows, and residual
wording with the actual Windows staged-test outcome
**Time estimate:** 12 hours

### Tasks
1. Audit README, INSTALL, maintainer guide, solver-selection, benchmark docs,
   and relevant sprint artifacts for Windows support wording.
2. Update public/support wording for promoted or retained staged Windows
   coverage.
3. Preserve non-claims for Windows Makefile parity, Windows `pkg-config`
   parity, install-validation parity, shared libraries, dynamic ABI, runtime
   loader, package-manager support, and broad Windows parity.
4. Update residual and Sprint 149 handoff wording.
5. Run documentation-focused validation.
6. Write the documentation alignment artifact.

### Deliverables
- updated Windows support-tier docs or no-change rationale
- residual/non-claim wording updates
- Sprint 149 install-parity handoff updates
- documentation validation notes

### Completion Criteria
- public docs match the implemented Windows staged-test outcome
- no unsupported Windows parity claim is introduced
- Sprint 149 install-validation decision remains separate

---

## Day 13: Integrated Validation And Hosted Evidence Intake

**Title:** Validation Intake
**Theme:** Run local feasible validation, prepare hosted Windows evidence
requirements, and reconcile any CI observations
**Time estimate:** 12 hours

### Tasks
1. Run focused tests for every touched staged-test surface.
2. Run local CMake configure/build/CTest enumeration and relevant CTest subset
   where feasible.
3. Run `make format && make lint && make test` if any `.c` or `.h` file was
   modified.
4. Inspect available hosted Windows CI results when a branch/PR run exists.
5. Record failures, skips, red hosted checks, unavailable hosted proof, and
   residuals.
6. Write the integrated validation artifact.

### Deliverables
- focused local validation log
- CMake enumeration and subset test log
- full C gate log or docs-only rationale
- hosted Windows evidence intake or unavailable-proof residual
- validation residual list

### Completion Criteria
- every touched surface has a validation result or explicit blocker
- full C gate runs and passes if C/header files changed
- hosted Windows proof is recorded or clearly marked as pending PR evidence

---

## Day 14: Sprint Closeout And Install-Parity Handoff

**Title:** Closeout Handoff
**Theme:** Close Sprint 148, publish final Windows staged-test outcome, and
prepare Sprint 149 install-validation parity decision input
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 148 artifacts and working notes for consistency.
2. Confirm each Sprint 148 project-plan item is complete or explicitly
   residualized.
3. Publish the final Windows staged-test closure outcome.
4. Publish the Sprint 149 Windows install-validation parity handoff.
5. Run lightweight documentation validation over Sprint 148 artifacts.
6. Prepare Sprint 148 retrospective input notes.

### Deliverables
- Sprint 148 closeout artifact
- final Windows staged-test closure summary
- validation summary and residual list
- Sprint 149 install-parity handoff
- retrospective input notes

### Completion Criteria
- Windows staged-test outcome is complete, rejected, or explicitly residualized
- Sprint 149 can begin without reopening staged-test baseline decisions
- validation evidence and unsupported non-claims remain explicit
- documentation validation passes
