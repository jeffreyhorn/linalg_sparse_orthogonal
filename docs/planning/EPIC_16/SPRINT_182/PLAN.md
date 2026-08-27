# Sprint 182 Plan: Windows Report Freshness Decision

**Sprint Duration:** 14 days
**Goal:** Promote one Windows-safe generated report freshness path or close
Windows report freshness as an explicit product deferral with guard coverage.
This sprint implements the Sprint 182 section of
`docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Source Artifact Note:** This plan lives under
`docs/planning/EPIC_16/SPRINT_182/PLAN.md` and implements the Sprint 182
section of `docs/planning/EPIC_16/PROJECT_PLAN.md`.

**Starting Point:** Sprint 182 begins from:

- the Sprint 181 selected report target manifest at
  `tests/corpus/manifests/selected_report_targets.tsv`;
- current Windows CMake-first CI and install/downstream validation;
- current selected Linux oracle/comparison and benchmark hosted report lanes;
- current macOS selected comparison hosted report lane;
- current workflow guards that reject accidental Windows selected report
  freshness;
- README, INSTALL, maintainer-guide, benchmark, corpus, and report-index docs
  that preserve Windows report freshness as a non-claim.

The sprint must:

- audit report commands for Windows shell, path, newline, executable, Python,
  dependency, generator, artifact, and runtime assumptions;
- choose exactly one Windows-safe report freshness promotion path or a formal
  deferral closure with exact blockers;
- implement either the Windows hosted lane or the deferral artifact and guard;
- align `selected_report_targets.tsv`, support-tier docs, workflow comments,
  README, INSTALL, maintainer guide, benchmark docs, and report-index wording;
- validate feasible local guards, report checks, CMake/PowerShell syntax, and
  claim-boundary guards.

**End State:** Sprint 182 leaves behind:

- a Windows report freshness promotion or formal deferral decision;
- manifest and documentation aligned with the chosen Windows support tier;
- guard coverage against accidental Windows report freshness claims;
- clear validation records, residual risks, and Sprint 183 handoff notes.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 182 project-plan estimate.

---

## Day 1: Windows Freshness Scope Intake

**Title:** Scope Intake
**Theme:** Establish Sprint 182 scope, inherited selected-target authority,
Windows CI boundaries, and decision criteria
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 182 section of
   `docs/planning/EPIC_16/PROJECT_PLAN.md`.
2. Review Sprint 181 closeout, retrospective, selected target manifest,
   workflow guard tests, and Windows handoff notes.
3. Review current Windows workflow jobs, CMake install/downstream validation,
   and explicit Windows report freshness non-claims.
4. Create Sprint 182 working notes and artifact directory structure.
5. Define candidate evaluation fields for shell compatibility, path
   semantics, newline behavior, executable availability, Python dependency
   availability, runtime cost, artifact scope, support tier, claim boundary,
   and guardability.
6. Write the Day 1 windows-freshness-scope-intake artifact.

### Deliverables

- Sprint 182 working-notes baseline
- artifact directory structure
- inherited Sprint 181 selected-target authority notes
- Windows freshness decision criteria
- Day 1 windows-freshness-scope-intake artifact

### Completion Criteria

- Sprint 182 scope is tied to the Epic 16 project plan
- inherited Windows non-claims and selected-target authority are explicit
- audit work starts from concrete compatibility and claim-boundary criteria

---

## Day 2: Windows Workflow And Toolchain Audit

**Title:** Workflow Audit
**Theme:** Audit Windows workflow jobs, shells, tools, paths, generated output
locations, and report-lane constraints
**Time estimate:** 12 hours

### Tasks

1. Inspect `.github/workflows/windows-ci.yml` job layout, shell choices,
   runner image assumptions, environment variables, and artifact behavior.
2. Inspect Linux/macOS selected report freshness jobs for structural patterns
   that may or may not transfer to Windows.
3. Inventory available Windows-side tools implied by existing CI:
   CMake, PowerShell, Python, MSVC, `ctest`, and installed artifacts.
4. Identify unavailable or unproven Windows assumptions such as Makefile
   execution, POSIX shell scripts, Unix paths, executable suffixes, and
   optional external dependencies.
5. Record exact workflow guard boundaries that must remain fail-closed during
   any promotion or deferral.
6. Write the Day 2 windows-workflow-and-toolchain-audit artifact.

### Deliverables

- Windows workflow job inventory
- Windows toolchain and shell assumption table
- cross-platform lane comparison notes
- report-lane constraint list
- Day 2 windows-workflow-and-toolchain-audit artifact

### Completion Criteria

- Windows workflow capabilities and limitations are explicit
- candidate freshness paths can be evaluated against actual Windows CI state
- current non-claim guard boundaries are documented before selection

---

## Day 3: Report Command Compatibility Audit

**Title:** Command Audit
**Theme:** Audit selected oracle, comparison, benchmark, and advisory report
commands for Windows execution risk
**Time estimate:** 12 hours

### Tasks

1. Audit `make report-index-oracle-freshness` and underlying
   `scripts/run_corpus_oracle.py` behavior for Windows shell, path,
   executable, compiler, and artifact assumptions.
2. Audit `make report-index-comparison-freshness` and
   `scripts/run_external_comparison.py` for Windows Python, executable,
   dense-reference, optional dependency, and path assumptions.
3. Audit `make bench-canonical-report-freshness`,
   `scripts/bench_canonical_report.sh`, and
   `scripts/check_bench_canonical_freshness.py` for shell and runner
   assumptions.
4. Audit report-index, coverage, dead-code, sentinel, guardrail, package, CI,
   and documentation rows for Windows relevance and promotion risk.
5. Classify commands as promotable candidate, possible with refactor,
   deferral candidate, or out-of-scope.
6. Write the Day 3 report-command-compatibility-audit artifact.

### Deliverables

- selected report command compatibility matrix
- shell/path/executable/dependency risk notes
- out-of-scope advisory surface notes
- preliminary candidate/deferral classification
- Day 3 report-command-compatibility-audit artifact

### Completion Criteria

- every current selected report freshness command has a Windows risk
  assessment
- advisory or unselected report families remain separate from promotion
  candidates
- at least one promotion or deferral path is ready for deeper evaluation

---

## Day 4: Artifact And Data Semantics Audit

**Title:** Artifact Semantics
**Theme:** Audit generated files, row counts, newline behavior, path casing,
and normalized-index semantics for Windows safety
**Time estimate:** 12 hours

### Tasks

1. Inspect selected report artifact patterns and required files in
   `selected_report_targets.tsv`.
2. Audit generated TSV/CSV writing code for newline, delimiter, encoding,
   path separator, and executable suffix assumptions.
3. Audit normalizer path matching, freshness checks, source commit checks, and
   support-tier interpretation for Windows-generated rows.
4. Identify artifact upload scope differences needed for a Windows hosted
   lane versus a formal deferral.
5. Record artifact naming and retention rules that must be exact if Windows is
   promoted.
6. Write the Day 4 artifact-and-data-semantics-audit artifact.

### Deliverables

- artifact/newline/path compatibility notes
- selected manifest artifact risk table
- normalized-index Windows semantics notes
- Windows upload scope requirements or blockers
- Day 4 artifact-and-data-semantics-audit artifact

### Completion Criteria

- generated file semantics are evaluated independently from command runtime
- Windows promotion cannot rely on broad artifact globs
- deferral blockers distinguish data-format issues from workflow issues

---

## Day 5: Candidate Decision Matrix

**Title:** Decision Matrix
**Theme:** Compare promotion and deferral options against feasibility, user
value, maintenance cost, runtime cost, and claim risk
**Time estimate:** 12 hours

### Tasks

1. Define candidate options for selected oracle freshness, selected
   comparison freshness, selected benchmark freshness, and formal deferral.
2. Score each option against Windows CI feasibility, shell portability,
   artifact stability, dependency requirements, runtime cost, maintenance
   cost, user value, and claim risk.
3. Identify the smallest claim-safe promoted lane if promotion is feasible.
4. Identify exact blockers if formal deferral is the better product decision.
5. Choose the Day 6 decision target: one promotion path or one deferral path.
6. Write the Day 5 candidate-decision-matrix artifact.

### Deliverables

- Windows report freshness decision matrix
- candidate scoring table
- selected promotion or deferral target
- blocker list for rejected options
- Day 5 candidate-decision-matrix artifact

### Completion Criteria

- Sprint 182 no longer carries multiple parallel implementation paths
- chosen path is justified by evidence rather than preference
- rejected paths have concrete blockers or risk reasons

---

## Day 6: Decision Record Design

**Title:** Decision Record
**Theme:** Design the exact promotion or deferral contract before
implementation
**Time estimate:** 12 hours

### Tasks

1. Draft the Windows report freshness decision record structure.
2. If promoting a lane, define exact workflow job ID, command, artifact upload
   name, required files, support tier, freshness policy, claim scope, and
   non-claims.
3. If deferring, define exact blocker taxonomy, unsupported command surface,
   guard behavior, support-tier wording, and revisit criteria.
4. Map required manifest changes for the selected path.
5. Map required documentation and workflow guard changes for the selected
   path.
6. Write the Day 6 decision-record-design artifact.

### Deliverables

- promotion or deferral decision-record design
- manifest change plan
- workflow guard change plan
- docs change plan
- Day 6 decision-record-design artifact

### Completion Criteria

- implementation work has one precise contract to follow
- support-tier and freshness-policy wording are known before code changes
- claim boundaries are explicit before workflow/docs edits

---

## Day 7: Implementation Batch 1

**Title:** Implementation Batch 1
**Theme:** Implement the first half of the chosen promotion or deferral path
with focused guard coverage
**Time estimate:** 12 hours

### Tasks

1. Implement initial manifest or decision-record changes for the chosen path.
2. Update workflow guard tests for the selected promotion or deferral behavior.
3. If promoting, add exact Windows job/upload expectations without broad YAML
   scans.
4. If deferring, add or refine guard checks that reject accidental Windows
   selected report commands or selected artifact uploads.
5. Add focused tests for missing job, wrong command, wrong artifact, or
   missing blocker diagnostics as applicable.
6. Write the Day 7 implementation-batch-1 artifact.

### Deliverables

- first implementation slice
- workflow guard updates
- focused promotion/deferral tests
- Day 7 implementation-batch-1 artifact

### Completion Criteria

- selected path has executable guard coverage
- workflow assertions remain scoped to exact jobs and upload blocks
- unsupported Windows interpretations fail clearly

---

## Day 8: Implementation Batch 2

**Title:** Implementation Batch 2
**Theme:** Complete workflow, script, manifest, and diagnostics changes for
the chosen path
**Time estimate:** 12 hours

### Tasks

1. Complete remaining workflow, manifest, script, or deferral artifact changes.
2. If promoting, ensure Windows-generated rows produce or validate exact
   expected artifacts and freshness diagnostics.
3. If deferring, ensure deferral diagnostics name exact blockers and rejected
   claim surfaces.
4. Update selected-target manifest tests for Windows support-tier,
   workflow-platform, artifact, expected-row, and freshness-policy behavior.
5. Preserve Linux/macOS selected report behavior while adding Windows
   promotion or deferral logic.
6. Write the Day 8 implementation-batch-2 artifact.

### Deliverables

- complete implementation of selected path
- manifest and diagnostics regression tests
- preserved Linux/macOS behavior notes
- Day 8 implementation-batch-2 artifact

### Completion Criteria

- implementation no longer has known incomplete branches
- tests cover expected success or deferral failure behavior
- existing selected report lanes remain unchanged unless explicitly justified

---

## Day 9: Manifest And Support-Tier Alignment

**Title:** Manifest Alignment
**Theme:** Align selected target manifest, report-family metadata, and support
tier docs with the Windows decision
**Time estimate:** 12 hours

### Tasks

1. Update `selected_report_targets.tsv` if the Windows decision adds,
   modifies, or formalizes Windows workflow metadata.
2. Update `report_families.tsv` only if the broad family policy requires a
   changed support or freshness interpretation.
3. Update corpus and report-index docs for Windows support-tier and
   freshness-policy semantics.
4. Add regression tests for unsupported support tiers, missing Windows
   metadata, or accidental promotion depending on the selected decision.
5. Check that selected target validation still rejects inconsistent workflow
   artifact/platform cardinality.
6. Write the Day 9 manifest-and-support-tier-alignment artifact.

### Deliverables

- manifest/support-tier updates
- corpus/report-index docs updates
- selected-target schema regressions
- Day 9 manifest-and-support-tier-alignment artifact

### Completion Criteria

- manifest state matches the chosen Windows decision
- support-tier wording is consistent across schema, docs, and tests
- no accidental Windows selected row can pass silently

---

## Day 10: Documentation Alignment

**Title:** Documentation Alignment
**Theme:** Update public and maintainer docs to reflect the Windows report
freshness promotion or formal deferral
**Time estimate:** 12 hours

### Tasks

1. Update README report freshness and cross-platform CI sections.
2. Update INSTALL only if package/install or Windows support wording is
   affected by the decision.
3. Update maintainer guide report-index and Windows CI guidance.
4. Update benchmark docs if selected performance freshness is promoted or
   formally deferred on Windows.
5. Update workflow comments so hosted CI wording matches the selected
   decision.
6. Write the Day 10 documentation-alignment artifact.

### Deliverables

- README updates
- maintainer guide updates
- INSTALL updates if needed
- benchmark docs/workflow comment updates
- Day 10 documentation-alignment artifact

### Completion Criteria

- users and maintainers can tell whether Windows report freshness is promoted
  or deferred
- docs do not imply unsupported package, platform, performance, or
  external-library claims
- workflow comments and docs describe the same support boundary

---

## Day 11: Guard And Failure Diagnostics Hardening

**Title:** Diagnostics Hardening
**Theme:** Harden failure messages for missing Windows jobs, wrong commands,
wrong artifacts, stale rows, and unsupported claims
**Time estimate:** 12 hours

### Tasks

1. Add or refine tests for missing Windows workflow job or missing deferral
   record.
2. Add or refine tests for wrong Windows report command or accidental selected
   command promotion.
3. Add or refine tests for wrong upload artifact, broad upload paths, missing
   required files, and wrong `if-no-files-found` behavior.
4. Add or refine tests for stale generated rows, missing generated rows, or
   exact blocker diagnostics as applicable.
5. Ensure diagnostics name the workflow block, manifest row, or deferral
   artifact that must be fixed.
6. Write the Day 11 guard-and-failure-diagnostics-hardening artifact.

### Deliverables

- workflow drift diagnostics tests
- manifest/deferral diagnostics tests
- stale/missing generated-row or blocker diagnostics
- Day 11 guard-and-failure-diagnostics-hardening artifact

### Completion Criteria

- failure modes direct maintainers to the exact file or row to fix
- accidental Windows support claims fail in guards
- stale data and missing data are distinguishable when promotion is selected

---

## Day 12: Validation Sweep

**Title:** Validation Sweep
**Theme:** Run feasible local report, workflow, manifest, CMake, PowerShell,
Python, docs, and guard checks for the chosen decision
**Time estimate:** 12 hours

### Tasks

1. Run selected-target schema and manifest regression tests.
2. Run workflow guard tests for Linux, macOS, and Windows report freshness
   behavior.
3. Run report normalizer tests and selected freshness checks affected by the
   decision.
4. Run feasible Windows syntax checks locally, such as PowerShell parsing or
   workflow text guards, where available.
5. Run package/static deferral guards if support-tier or package wording was
   touched.
6. Run Python compile checks and `git diff --check`.

### Deliverables

- validation command output summary
- fixed validation failures if any
- residual risk notes
- Day 12 validation-sweep artifact

### Completion Criteria

- all feasible local checks pass or blockers are explicitly recorded
- validation covers the selected promotion or deferral path
- unsupported platform/package/performance claims remain guarded

---

## Day 13: Decision Reconciliation

**Title:** Decision Reconciliation
**Theme:** Reconcile implementation, docs, manifests, guards, validation
records, and residual Windows risks
**Time estimate:** 11 hours

### Tasks

1. Reconcile Sprint 182 project-plan items against produced artifacts and
   changed files.
2. Confirm manifest state, workflow behavior, documentation wording, and
   validation records all describe the same Windows decision.
3. Record residual risks, intentionally retained non-claims, and unsupported
   report/platform/package/performance interpretations.
4. Prepare retrospective inputs.
5. Prepare Sprint 183 handoff notes.
6. Write the Day 13 decision-reconciliation artifact.

### Deliverables

- project-plan reconciliation
- residual risk list
- non-claim and support-boundary list
- retrospective inputs
- Day 13 decision-reconciliation artifact

### Completion Criteria

- Windows report freshness decision is internally consistent
- docs, guards, and manifests agree on support tier and claim scope
- remaining risks are explicit enough for Sprint 183 planning

---

## Day 14: Closeout And Handoff

**Title:** Closeout
**Theme:** Finalize Sprint 182 closeout records, handoff notes, and PR-ready
validation summary
**Time estimate:** 11 hours

### Tasks

1. Re-read all Sprint 182 artifacts and working notes for consistency.
2. Finalize decision summary, validation summary, and handoff notes.
3. Confirm no generated build/report artifacts are staged.
4. Confirm final changed-file surface and C/header change status.
5. Record final validation commands and any known sequential-only checks.
6. Write the Day 14 closeout-and-handoff artifact.

### Deliverables

- Sprint 182 closeout artifact
- final validation and changed-surface summary
- Sprint 183 handoff notes
- PR-ready risk and claim-boundary summary

### Completion Criteria

- Sprint 182 deliverables are complete and internally consistent
- working notes and artifacts are ready for retrospective creation
- branch is ready for final validation, retrospective, commit, push, and PR
