# Sprint 144 Plan: Platform Promotion Lane Closure

**Sprint Duration:** 14 days
**Goal:** Fully promote one high-value platform support lane, or explicitly
reject promotion with source-level blockers and proof requirements. This sprint
implements the Sprint 144 section of
`docs/planning/EPIC_12/PROJECT_PLAN.md`.

**Starting Point:** Sprint 144 begins from:
- Sprint 143 static-first package/ABI decision, install/export proof, and
  package metadata guards
- current Linux, macOS, and Windows CI tier comments and staged exclusions
- current Make, CMake, `pkg-config`, install, downstream consumer, and report
  validation surfaces
- current platform portability blockers for shell scripts, PowerShell checks,
  CMake/CTest registration, path handling, POSIX APIs, pthread APIs, and
  package evidence
- existing README, INSTALL, maintainer guide, support-tier wording, and report
  rows that describe platform claims and non-claims

The sprint must:
- select one high-value platform lane for complete closure
- fix source, script, CI, CMake, PowerShell, path, pthread/POSIX, or test
  blockers for the selected lane
- update workflow jobs, expected counts, failure messages, support-tier
  comments, and artifact proof for the selected lane
- connect platform promotion evidence to package/report/freshness artifacts
  where applicable
- update README, INSTALL, maintainer guidance, and platform support wording
- run locally feasible checks, workflow syntax checks, package/report checks,
  and full quality gates if `.c` or `.h` files change
- publish platform promotion evidence, residual non-claims, and the Sprint 145
  adoption handoff

**End State:** Sprint 144 leaves behind:
- selected platform lane closed or explicitly rejected with source-level proof
- source/script/CI updates for the selected lane
- support-tier documentation aligned with earned evidence
- package/report integration evidence
- residual platform blocker and non-claim ledger
- Sprint 145 adoption handoff

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 144 project-plan estimate.

---

## Day 1: Platform Promotion Intake

**Title:** Promotion Intake
**Theme:** Establish Sprint 144 scope, inherited package/ABI evidence, platform
lanes, and closure criteria
**Time estimate:** 12 hours

### Tasks
1. Re-read the Sprint 144 section of
   `docs/planning/EPIC_12/PROJECT_PLAN.md`.
2. Review Sprint 143 Day 13-14 artifacts, retrospective, CI notes, and package
   support-tier handoff.
3. Create Sprint 144 working notes and artifact directory structure.
4. Inventory candidate platform lanes: macOS reviewed install/export parity,
   Windows reviewed install/downstream parity, Windows staged test portability,
   and Linux source-of-truth strengthening.
5. Map Sprint 144 Items 1-7 to day-level owners.
6. Record initial platform promotion criteria, evidence requirements,
   non-claims, and stop conditions.

### Deliverables
- Sprint 144 working-notes baseline
- artifact directory structure
- candidate platform lane inventory
- item-to-day owner map
- initial promotion criteria and non-claim register

### Completion Criteria
- every Sprint 144 project-plan item has a day-level owner
- Sprint 143 package/ABI proof is treated as prerequisite evidence, not a new
  platform claim by itself
- platform promotion requires explicit proof, not CI wording alone

---

## Day 2: Platform Lane Selection

**Title:** Lane Selection
**Theme:** Select the single platform lane that can be closed completely within
the sprint budget
**Time estimate:** 12 hours

### Tasks
1. Score each candidate lane by user value, current evidence, blocker severity,
   implementation cost, CI cost, portability risk, and documentation impact.
2. Compare the score against Sprint 143 static-first package contract and
   current Linux/macOS/Windows workflow expectations.
3. Identify which lane can be fully promoted or explicitly rejected with
   source-level evidence inside 14 days.
4. Select the primary lane and one backup lane.
5. Define the exact promotion or rejection criteria for the selected lane.
6. Write the platform lane selection artifact.

### Deliverables
- lane scoring table
- selected platform lane
- backup lane and defer reasons
- promotion or rejection criteria
- validation and evidence checklist

### Completion Criteria
- exactly one platform lane is selected for complete closure
- non-selected lanes have explicit defer reasons
- selected-lane proof requirements are concrete enough for design work

---

## Day 3: Blocker Reproduction And Evidence Baseline

**Title:** Blocker Baseline
**Theme:** Reproduce or document selected-lane blockers and establish the
before-change evidence record
**Time estimate:** 12 hours

### Tasks
1. Inspect source, test, script, CMake, Make, PowerShell, workflow, and
   documentation surfaces touched by the selected lane.
2. Reproduce locally feasible failures or confirm why reproduction requires CI.
3. Capture current CTest registration counts, install/export behavior,
   downstream consumer behavior, package/report rows, and workflow messages for
   the selected lane.
4. Separate source-level blockers from CI-only configuration drift.
5. Define expected post-fix outputs and failure messages.
6. Write the blocker baseline artifact.

### Deliverables
- selected-lane blocker baseline
- before-change command output summary
- source-level blocker list
- CI-only drift list
- expected post-fix evidence matrix

### Completion Criteria
- implementation starts from concrete blockers or proof gaps
- source blockers and workflow bookkeeping issues are separated
- expected evidence is specific enough to avoid broad platform overclaims

---

## Day 4: Portability Design

**Title:** Portability Design
**Theme:** Design selected-lane source, script, path, shell, CMake, or
PowerShell fixes before implementation
**Time estimate:** 12 hours

### Tasks
1. Design the smallest source/script/build changes needed to close selected-lane
   blockers.
2. Define path normalization, newline handling, shell quoting, CMake generator,
   PowerShell, POSIX, pthread, or temporary-file portability rules as needed.
3. Identify tests that should be promoted, added, skipped, or kept staged.
4. Define implementation order and rollback/stop conditions.
5. Review compatibility with static-first package metadata and existing report
   freshness gates.
6. Write the portability design artifact.

### Deliverables
- selected-lane portability design
- source/script/build change checklist
- promoted/staged test decision table
- portability rule list
- implementation stop conditions

### Completion Criteria
- portability changes are scoped to the selected lane
- test promotion does not rely on stale expected counts or vague exclusions
- design preserves Sprint 143 static-first package boundaries

---

## Day 5: Source And Script Fix Batch

**Title:** Portability Fixes
**Theme:** Implement the selected-lane source, test, script, and build-system
portability fixes
**Time estimate:** 12 hours

### Tasks
1. Apply selected source, test, script, CMake, Make, or PowerShell fixes.
2. Normalize path, newline, quoting, temporary-file, shell, pthread/POSIX, or
   generator behavior where required by the selected lane.
3. Add focused assertions that capture the intended portability behavior.
4. Update local helper scripts only when they are direct proof owners.
5. Run focused syntax and smoke checks for touched scripts/build files.
6. Record implementation notes and changed proof owners.

### Deliverables
- selected-lane portability fixes
- focused assertions or script checks
- changed proof-owner list
- local syntax/smoke check summary
- implementation notes

### Completion Criteria
- selected source/script blockers are fixed or explicitly rejected with proof
- touched scripts/build files pass focused syntax checks
- no unrelated platform lanes are promoted accidentally

---

## Day 6: CI Promotion Design

**Title:** CI Design
**Theme:** Design workflow updates, expected counts, artifact evidence, and
failure messages for selected-lane promotion
**Time estimate:** 12 hours

### Tasks
1. Inspect affected Linux, macOS, and Windows workflow jobs and support-tier
   comments.
2. Define exact CI changes for the selected lane: jobs, matrix entries,
   expected CTest counts, install/export proof, downstream checks, report
   checks, and artifact upload rules.
3. Replace stale hard-coded assumptions with source-owned counts or explicit
   update instructions where feasible.
4. Draft failure messages that identify blockers and staged exclusions clearly.
5. Define workflow syntax and local validation commands.
6. Write the CI promotion design artifact.

### Deliverables
- CI promotion design
- expected count and staged-exclusion policy
- workflow failure-message draft
- artifact proof checklist
- workflow validation checklist

### Completion Criteria
- workflow changes are tied to selected-lane evidence
- expected counts and staged exclusions have clear ownership
- CI messages distinguish promoted support from remaining non-claims

---

## Day 7: CI Promotion Implementation

**Title:** CI Implementation
**Theme:** Implement selected-lane workflow, expected-count, support-tier, and
artifact-proof updates
**Time estimate:** 12 hours

### Tasks
1. Update affected workflow files for selected-lane promotion or explicit
   rejection.
2. Update expected test counts, CTest inspection steps, package/install proof
   steps, downstream consumer checks, and artifact rules as designed.
3. Update support-tier comments and failure messages.
4. Run workflow syntax checks available locally.
5. Run focused local commands that mirror the changed CI steps when feasible.
6. Record CI implementation evidence.

### Deliverables
- updated workflow files
- updated expected-count and exclusion comments
- selected-lane CI proof steps
- local workflow syntax check summary
- CI implementation artifact

### Completion Criteria
- selected-lane CI path reflects the intended support tier
- failure messages explain remaining staged blockers without implying support
- workflow syntax and locally feasible mirrored checks pass

---

## Day 8: Package And Report Integration

**Title:** Report Integration
**Theme:** Connect selected-lane promotion evidence to package, report, and
freshness artifacts where applicable
**Time estimate:** 12 hours

### Tasks
1. Inspect package report rows, report indexes, freshness scripts, install
   reports, and selected-lane evidence artifacts.
2. Add or update selected-lane package/report rows only where they are
   source-owned and validated.
3. Ensure report wording distinguishes promoted support, supplemental evidence,
   advisory evidence, and non-claims.
4. Run report normalization and freshness checks for affected report families.
5. Update artifact references in working notes.
6. Write the package/report integration artifact.

### Deliverables
- package/report row updates as needed
- freshness and normalization check summary
- selected-lane evidence references
- report wording audit notes
- package/report integration artifact

### Completion Criteria
- selected-lane evidence is discoverable from report artifacts
- report rows do not claim unsupported platform parity
- affected report normalization and freshness checks pass

---

## Day 9: Documentation Support-Tier Alignment

**Title:** Docs Alignment
**Theme:** Align README, INSTALL, maintainer guide, and platform support docs
with the earned selected-lane status
**Time estimate:** 12 hours

### Tasks
1. Inspect README, INSTALL, maintainer guide, CI/support-tier comments, and
   platform-specific documentation for selected-lane wording.
2. Update support-tier language to reflect promoted or rejected status.
3. Remove stale references to blockers that were closed.
4. Keep remaining blockers, exclusions, and non-claims explicit and
   source-level.
5. Add links or references to selected-lane evidence artifacts.
6. Write the documentation alignment artifact.

### Deliverables
- updated support-tier documentation
- stale wording cleanup
- selected-lane evidence references
- residual blocker and non-claim wording
- documentation alignment artifact

### Completion Criteria
- public docs match the evidence earned this sprint
- remaining platform limitations are concrete and discoverable
- docs do not imply package-manager, shared-library, or platform parity claims
  beyond proof

---

## Day 10: Selected-Lane Validation Pass

**Title:** Lane Validation
**Theme:** Run focused validation for the selected platform lane and close
remaining proof gaps
**Time estimate:** 12 hours

### Tasks
1. Run all locally feasible selected-lane checks.
2. Run affected install/export, downstream consumer, CMake, Make,
   `pkg-config`, report, or workflow-syntax checks.
3. Inspect failures and classify them as implementation defects, environment
   constraints, or explicit non-claims.
4. Fix implementation defects found by focused validation.
5. Update working notes with command outputs, skipped checks, and rationale.
6. Write the selected-lane validation artifact.

### Deliverables
- focused validation results
- fixed validation defects
- skipped-check rationale
- selected-lane evidence summary
- validation artifact

### Completion Criteria
- selected-lane proof passes locally where feasible
- unresolved checks are tied to explicit environment or source blockers
- working notes contain enough evidence for closeout and review

---

## Day 11: Cross-Platform Non-Regression Review

**Title:** Non-Regression Review
**Theme:** Confirm selected-lane changes did not weaken other platform,
package, report, or support-tier contracts
**Time estimate:** 12 hours

### Tasks
1. Review diffs across workflows, scripts, CMake, Make, docs, and reports.
2. Compare Linux, macOS, and Windows support-tier wording for consistency.
3. Re-run focused package/report checks affected by selected-lane changes.
4. Confirm static-first package guards from Sprint 143 remain intact.
5. Identify any non-selected platform lane whose status changed and correct
   wording or tests if needed.
6. Write the non-regression review artifact.

### Deliverables
- cross-platform support-tier review
- static-first package guard confirmation
- package/report non-regression check summary
- corrected wording or test drift
- non-regression artifact

### Completion Criteria
- non-selected lanes remain staged or unsupported unless explicitly promoted
- Sprint 143 static-first package contract remains intact
- selected-lane changes do not create undocumented platform claims

---

## Day 12: Quality Gate Execution

**Title:** Quality Gates
**Theme:** Run required quality gates for touched surfaces and resolve all
blocking failures
**Time estimate:** 12 hours

### Tasks
1. Run required formatting, linting, tests, install checks, CMake checks,
   report checks, and script syntax checks based on changed files.
2. If `.c` or `.h` files changed, run `make format`, `make lint`, and
   `make test`.
3. If only docs/scripts/workflows changed, run the focused checks that own
   those surfaces.
4. Fix any blocking failures and re-run the failing checks.
5. Record exact commands, pass/fail status, and any environment constraints.
6. Write the quality gate artifact.

### Deliverables
- quality gate command log
- fixed quality failures
- environment constraint notes
- final validation status
- quality gate artifact

### Completion Criteria
- all required checks for changed surfaces pass
- any unrun checks have explicit environment constraints
- no quality failure remains unresolved at the end of the day

---

## Day 13: Promotion Evidence And Residual Non-Claims

**Title:** Evidence Closure
**Theme:** Finalize platform promotion evidence, blocker disposition, and
residual non-claims before sprint closeout
**Time estimate:** 12 hours

### Tasks
1. Consolidate selected-lane evidence from design, implementation, validation,
   CI, report, and documentation artifacts.
2. Mark the selected lane as promoted or explicitly rejected according to Day 2
   criteria.
3. Document remaining source-level blockers for non-selected or rejected lanes.
4. Verify support-tier wording, report rows, workflow comments, and working
   notes agree.
5. Draft Sprint 145 adoption handoff with platform implications.
6. Write the promotion evidence and residual non-claims artifact.

### Deliverables
- final selected-lane promotion or rejection decision
- evidence index
- residual blocker ledger
- support-tier consistency check
- Sprint 145 adoption handoff draft

### Completion Criteria
- selected-lane status is backed by concrete evidence
- residual platform non-claims are explicit and source-owned
- Sprint 145 handoff identifies adoption-facing platform constraints

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Complete Sprint 144 documentation, validation summary, and Sprint 145
handoff
**Time estimate:** 12 hours

### Tasks
1. Review all Sprint 144 artifacts and working notes for consistency.
2. Update the final validation summary with commands, touched surfaces, and
   selected-lane status.
3. Confirm deliverables from the Sprint 144 project-plan section are satisfied
   or explicitly rejected with proof.
4. Finalize Sprint 145 adoption handoff.
5. Run final lightweight repository checks such as status, diff review, and
   whitespace checks.
6. Prepare closeout notes for the retrospective.

### Deliverables
- Sprint 144 closeout validation summary
- completed working notes
- Sprint 145 adoption handoff
- final deliverable checklist
- retrospective input notes

### Completion Criteria
- selected platform lane is closed or rejected with source-level proof
- support-tier docs, CI comments, report rows, and artifacts agree
- Sprint 145 can start from a clear adoption-facing platform contract
