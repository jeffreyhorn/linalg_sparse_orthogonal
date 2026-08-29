# Sprint 187 Plan: Epic 17 Baseline, Gap Ledger & Acceptance Gates

**Sprint Duration:** 14 days
**Goal:** Freeze the post-Epic-16 baseline, convert the Codex review into a
source-controlled gap ledger, and select the exact Epic 17 closures.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 187 estimate in the Epic 17
project plan.

**Primary scope:** Convert the Epic 17 review and Epic 16 residual handoff into
an actionable baseline, deduplicated gap ledger, closure selection, acceptance
gates, quality surface map, and handoff package for package and Windows work.

**Non-goals:** Implementing package, Windows, comparison, performance,
maintainability, adoption, or reliability fixes directly; changing source
behavior; expanding public API contracts; or claiming state-of-the-art,
platform, package, ABI, performance, or external-parity support without later
evidence.

---

## Day 1: Sprint Intake and Source Map

**Title:** Baseline Intake
**Theme:** Establish the Sprint 187 scope and source artifacts.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 187 section of the Epic 17 project plan and capture the
   acceptance boundaries for items 187.1 through 187.6.
2. Inventory the Codex review, Codex todo, Epic 16 retrospective, Epic 16
   residual queue, and current user-facing support documents.
3. Create the Sprint 187 working-notes scaffold with sprint goal, source
   artifacts, planned outputs, risks, and open questions.
4. Identify the owner surfaces needed for package, Windows, comparison,
   performance, maintainability, adoption, and reliability planning.
5. Draft the Day 2 gap-ledger schema.

### Deliverables

- Sprint 187 working-notes scaffold.
- Source-artifact inventory.
- Initial gap-ledger schema.

### Completion Criteria

- Sprint scope is traceable to items 187.1 through 187.6.
- All baseline source artifacts are identified.
- Day 2 can start from a concrete ledger schema.

---

## Day 2: Review Finding Extraction

**Title:** Review Intake Matrix
**Theme:** Convert the Codex review into structured findings.
**Time estimate:** 12 hours

### Tasks

1. Extract every material finding from
   `reviews/review-codex-2026-08-28.md`.
2. Classify each finding by area: efficiency, maintainability, usability,
   documentation, coherence, test coverage, packaging, platform, performance,
   comparison, or state-of-the-art readiness.
3. Record owner files, current evidence, claim risk, user value, and likely
   closure horizon for each finding.
4. Separate complete-gap candidates from long-horizon or intentionally
   retained non-claims.
5. Write the first gap-ledger artifact.

### Deliverables

- Review finding extraction artifact.
- Initial Epic 17 gap ledger.
- Long-horizon non-goal list.

### Completion Criteria

- Review findings are represented as structured ledger rows.
- Each row has owner surfaces and a claim-risk classification.
- Non-goals are separated from candidate Epic 17 work.

---

## Day 3: Epic 16 Residual Reconciliation

**Title:** Residual Reconciliation
**Theme:** Merge Epic 16 residuals into the Epic 17 ledger.
**Time estimate:** 12 hours

### Tasks

1. Reconcile `EPIC_16_RESIDUAL_QUEUE.md` with the Day 2 review ledger.
2. Deduplicate package/license, PowerShell validation, Windows report
   freshness, hosted API, comparison breadth, and review-surface items.
3. Preserve original residual IDs where they remain useful for traceability.
4. Identify any residuals that should stay long-horizon rather than become
   Sprint 187-196 closure targets.
5. Update working notes with reconciliation decisions.

### Deliverables

- Deduplicated residual ledger.
- Mapping from Epic 16 residual IDs to Epic 17 candidate closures.
- Long-horizon residual list.

### Completion Criteria

- Epic 16 residuals are not duplicated under new names.
- Every residual has a disposition for Epic 17 planning.
- Closure candidates and retained deferrals are visibly distinct.

---

## Day 4: Owner Surface and Evidence Inventory

**Title:** Owner Surface Inventory
**Theme:** Tie each candidate gap to concrete source, docs, tests, and CI.
**Time estimate:** 12 hours

### Tasks

1. Inventory owner files for each candidate closure.
2. Map related tests, scripts, CI jobs, report manifests, benchmark drivers,
   docs, and planning artifacts.
3. Identify current validation commands and missing validation commands.
4. Note environment dependencies such as Homebrew, `pwsh`, hosted Windows, or
   optional comparison dependencies.
5. Add evidence-owner fields to the ledger.

### Deliverables

- Owner surface inventory.
- Validation command inventory.
- Environment dependency notes.

### Completion Criteria

- Each candidate gap has concrete owner files.
- Existing and missing validation commands are visible.
- Environment-dependent work is clearly flagged before selection.

---

## Day 5: Gap Ranking and Feasibility

**Title:** Closure Feasibility Ranking
**Theme:** Rank gaps by user value, proof value, risk, and closure feasibility.
**Time estimate:** 12 hours

### Tasks

1. Score each candidate gap for user value, state-of-the-art relevance,
   implementation risk, validation difficulty, and support-tier impact.
2. Identify dependencies between package, Windows, comparison, performance,
   maintainability, adoption, and reliability work.
3. Mark gaps that cannot be completely closed inside Epic 17.
4. Rank candidate closures for Sprint 188 through Sprint 195.
5. Record selection risks and fallback decisions.

### Deliverables

- Ranked gap ledger.
- Feasibility and dependency notes.
- Candidate closure shortlist.

### Completion Criteria

- Gap ranking is evidence-backed.
- Infeasible or too-broad work is not selected by default.
- Sprint 188-195 closure candidates are ready for Day 6 selection.

---

## Day 6: Closure Target Selection

**Title:** Epic Closure Selection
**Theme:** Select the complete gaps Epic 17 will target.
**Time estimate:** 12 hours

### Tasks

1. Select the exact closure targets for Sprints 188 through 195.
2. Confirm each selected target has a plausible complete definition of done.
3. Record explicit non-goals for broad state-of-the-art, ABI, shared-library,
   platform, package-manager, external-parity, and portable-performance claims.
4. Write the closure-selection artifact.
5. Update working notes with selected and rejected targets.

### Deliverables

- Epic 17 closure-selection artifact.
- Selected closure target list.
- Explicit non-goal register.

### Completion Criteria

- Every selected target can be completed within its planned sprint.
- Rejected or deferred candidates have rationale.
- Non-goals are ready to feed acceptance gate wording.

---

## Day 7: Package Acceptance Gates

**Title:** Package Gate Definition
**Theme:** Define package/license and Homebrew proof acceptance criteria.
**Time estimate:** 12 hours

### Tasks

1. Define the exact closure criteria for the Homebrew license metadata blocker.
2. Map required changes to root license metadata, Homebrew formula material,
   proof scripts, package guards, and install docs.
3. Define required validation commands for proof success and claim safety.
4. Record retained non-claims for Homebrew/core, bottles, Linuxbrew, public
   taps, binary packages, and broad provider support.
5. Create the package acceptance-gate artifact.

### Deliverables

- Package and Homebrew acceptance gates.
- Required validation command list.
- Package non-claim wording inputs.

### Completion Criteria

- Sprint 188 has exact package proof criteria.
- Support promotion requires passing validation.
- Unsupported package surfaces remain explicitly out of scope.

---

## Day 8: Windows Acceptance Gates

**Title:** Windows Gate Definition
**Theme:** Define PowerShell validation and Windows report freshness gates.
**Time estimate:** 12 hours

### Tasks

1. Define the acceptance criteria for PowerShell validation ownership.
2. Define decision criteria for Windows report freshness promotion or renewed
   deferral.
3. Map required owner surfaces in Windows CI, report manifests, report
   scripts, workflow guards, and support-tier docs.
4. Define artifact upload scope and stale-output protections for any selected
   Windows freshness lane.
5. Create the Windows acceptance-gate artifact.

### Deliverables

- PowerShell validation acceptance gates.
- Windows report freshness decision gates.
- Windows support-tier non-claim inputs.

### Completion Criteria

- Sprints 189 and 190 have exact acceptance criteria.
- Windows promotion and deferral paths are both reviewable.
- Broad Windows parity remains protected as a non-claim.

---

## Day 9: Comparison and Performance Gates

**Title:** Evidence Lane Gates
**Theme:** Define bounded comparison and performance evidence criteria.
**Time estimate:** 12 hours

### Tasks

1. Define acceptance criteria for one bounded external comparison family.
2. Define acceptance criteria for one methodology-bound hosted performance
   evidence lane.
3. Specify required fixture, metric, tolerance, dependency, manifest, report,
   freshness, and artifact metadata fields.
4. Identify claim wording that distinguishes local/hosted evidence from broad
   external parity or portable performance.
5. Create the comparison/performance gate artifact.

### Deliverables

- Bounded comparison acceptance gates.
- Methodology-bound performance acceptance gates.
- Report and manifest metadata requirements.

### Completion Criteria

- Sprints 191 and 192 have exact evidence requirements.
- Required report fields and freshness checks are explicit.
- Broad ecosystem and performance-superiority claims remain out of scope.

---

## Day 10: Maintainability and Reliability Gates

**Title:** Code-Quality Gate Definition
**Theme:** Define selected review-surface and failure-path proof criteria.
**Time estimate:** 12 hours

### Tasks

1. Define ranking criteria for the selected large review-surface reduction.
2. Define no-behavior-change invariants and required focused tests for the
   selected maintainability cluster.
3. Define selection criteria for one reliability or allocation-failure owner.
4. Map required focused gates, source-list checks, and full C quality gates.
5. Create the maintainability/reliability gate artifact.

### Deliverables

- Review-surface reduction acceptance gates.
- Reliability/failure-path proof acceptance gates.
- Required C quality gate map.

### Completion Criteria

- Sprints 193 and 195 have measurable definitions of done.
- C/header changes have explicit validation requirements.
- Maintainability work cannot silently expand behavior or API claims.

---

## Day 11: Adoption and Documentation Gates

**Title:** Adoption Gate Definition
**Theme:** Define user-facing simplification and documentation coherence gates.
**Time estimate:** 12 hours

### Tasks

1. Define the acceptance criteria for README, INSTALL, tutorial, cookbook,
   solver-selection, API-reference, and example simplification.
2. Specify the production-readiness/support matrix content and owner docs.
3. Define diagnostics wording requirements across direct, iterative, QR/SVD,
   and eigensolver workflows.
4. Define local link, Doxygen, example-build, install, and claim-guard checks.
5. Create the adoption/documentation gate artifact.

### Deliverables

- Adoption and API coherence acceptance gates.
- Support/readiness matrix requirements.
- Documentation validation command map.

### Completion Criteria

- Sprint 194 has exact documentation and usability targets.
- User-facing truth is separated from historical planning evidence.
- Docs validation requirements are explicit.

---

## Day 12: Quality Surface Map

**Title:** Quality Surface Map
**Theme:** Map required validation by changed surface.
**Time estimate:** 12 hours

### Tasks

1. Build a quality surface map for docs-only, planning-only, script, workflow,
   package, report, benchmark, public-header, and C implementation changes.
2. Identify minimum required validation and stronger optional validation for
   each surface.
3. Define when `make format && make lint && make test` is mandatory.
4. Record hosted validation dependencies and local skip rules.
5. Update working notes with the final quality policy for Sprint 187 handoff.

### Deliverables

- Epic 17 quality surface map.
- Required validation matrix.
- Hosted/local skip-rule notes.

### Completion Criteria

- Every selected closure has a mapped validation requirement.
- C/header validation policy is unambiguous.
- Future sprint handoffs can reuse the map directly.

---

## Day 13: Sprint Handoff Packages

**Title:** Implementation Handoffs
**Theme:** Package the selected closures for Sprints 188 through 195.
**Time estimate:** 12 hours

### Tasks

1. Create handoff notes for Sprint 188 Homebrew proof completion.
2. Create handoff notes for Sprint 189 PowerShell validation and Sprint 190
   Windows report freshness.
3. Create handoff notes for Sprint 191 comparison and Sprint 192 performance
   evidence work.
4. Create handoff notes for Sprint 193 maintainability, Sprint 194 adoption,
   and Sprint 195 reliability work.
5. Link each handoff to ledger rows, acceptance gates, owner files, and
   validation commands.

### Deliverables

- Sprint 188-195 handoff package.
- Cross-links from ledger rows to acceptance gates.
- Final pre-closeout open question list.

### Completion Criteria

- Each future sprint has a ready starting package.
- Handoffs include owner files and validation commands.
- Remaining open questions are explicit before closeout.

---

## Day 14: Baseline Closeout and Review Prep

**Title:** Sprint 187 Closeout
**Theme:** Finalize baseline artifacts and prepare the sprint for review.
**Time estimate:** 10 hours

### Tasks

1. Review all Sprint 187 artifacts against project-plan items 187.1 through
   187.6.
2. Confirm the gap ledger, residual reconciliation, closure selection,
   acceptance gates, quality surface map, and handoffs are internally
   consistent.
3. Check for stale TODOs, unresolved open questions, broken links, generated
   artifacts, and accidental scope expansion.
4. Update `WORKING_NOTES.md` with final closeout results and retrospective
   inputs.
5. Produce review-ready notes summarizing selected gaps, non-goals, gates,
   validation expectations, and next sprint readiness.

### Deliverables

- Review-ready Sprint 187 working notes.
- Final Sprint 187 closeout summary.
- PR-ready baseline, gate, and handoff notes.

### Completion Criteria

- All Sprint 187 project-plan items have a planned evidence owner.
- Future sprint handoffs are consistent and reviewable.
- The branch is ready for retrospective creation and PR preparation.

