# Sprint 208 Plan: Selected Windows Cholesky Freshness Promotion

**Sprint Duration:** 14 days
**Goal:** Fully promote or deliberately re-defer the selected Windows Cholesky
freshness lane using hosted evidence, manifest metadata, documentation, and
guards.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 208 estimate in the Epic 19
project plan.

**Primary scope:** Inspect hosted Windows `cholesky-spd-tridiag-5` evidence,
decide whether selected Windows freshness can be promoted, update manifest or
absence guards according to that decision, harden normalizer coverage for
Windows artifact paths and row filtering, calibrate public and maintainer docs,
and run the selected validation set.

**Non-goals:** Broad Windows parity, Windows package distribution, Homebrew or
other package-manager support, Linuxbrew, bottles, ABI or shared-library
support, portable performance claims, release claims, new solver behavior,
unselected benchmark freshness, or state-of-the-art claims.

---

## Day 1: Windows Cholesky Intake

**Title:** Windows Cholesky Intake
**Theme:** Establish Sprint 208 scope, inherited evidence, and claim boundaries
before changing Windows selected-freshness surfaces.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 208 Epic 19 project-plan section and map items 208.1
   through 208.6 to artifacts, docs, scripts, manifests, workflows, and
   validation commands.
2. Inventory Sprint 199 and Sprint 206 records that mention the selected
   Windows Cholesky lane, re-deferral evidence, and residual blockers.
3. Inspect selected-target manifest rows, report-index schemas, Windows
   workflow files, PowerShell guards, normalizer scripts, and maintainer docs.
4. Create `WORKING_NOTES.md` with item checklist, evidence map, risk register,
   validation matrix, decision log, and open questions.
5. Record explicit non-goals for broad Windows parity, package support, ABI,
   performance, release, and state-of-the-art claims.

### Deliverables

- Sprint 208 working-notes scaffold.
- Item-to-evidence traceability map.
- Windows Cholesky selected-freshness surface inventory.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 208 item has an initial evidence path or artifact category.
- Existing Windows Cholesky promotion and re-deferral evidence is identified
  before edits.
- Unsupported Windows, package, ABI, performance, release, and state-of-the-art
  claims remain explicitly out of scope.

---

## Day 2: Hosted Artifact Inventory

**Title:** Hosted Artifact Inventory
**Theme:** Locate and inspect the hosted Windows evidence needed to decide
whether selected freshness can be promoted.
**Time estimate:** 12 hours

### Tasks

1. Identify the latest hosted Windows workflow runs relevant to
   `cholesky-spd-tridiag-5`.
2. Record run identifiers, workflow names, job names, artifact names, commit
   SHAs, timestamps, and retention constraints.
3. Inspect artifact membership for selected report rows, index files,
   manifests, logs, and expected Windows-style paths.
4. Compare hosted artifact contents against local generated evidence and
   selected manifest expectations.
5. Write the Day 2 hosted artifact inventory artifact.

### Deliverables

- Hosted Windows artifact ledger.
- Artifact membership and path inventory.
- Initial hosted-versus-local comparison notes.
- Day 2 artifact inventory record.

### Completion Criteria

- Item 208.1 has exact hosted evidence identifiers or documented blockers.
- Artifact contents are mapped to selected target expectations.
- Missing, stale, or ambiguous evidence is recorded before promotion decisions.

---

## Day 3: Row And Path Traceability

**Title:** Row Traceability
**Theme:** Trace selected Cholesky row IDs, artifact paths, and generated
reports across Windows evidence and local tooling.
**Time estimate:** 12 hours

### Tasks

1. Enumerate selected Windows Cholesky row IDs, target keys, report families,
   and expected artifact patterns.
2. Audit normalizer handling for Windows separators, drive-like prefixes,
   artifact roots, generated rows, and selected target filtering.
3. Compare manifest target metadata with actual hosted artifact paths and
   generated report-index rows.
4. Identify any mismatch between workflow artifact naming, manifest metadata,
   and freshness checker assumptions.
5. Write the Day 3 row and path traceability artifact.

### Deliverables

- Selected row and path traceability table.
- Windows path-normalization gap list.
- Manifest-to-artifact consistency notes.
- Day 3 traceability artifact.

### Completion Criteria

- Item 208.1 has row-level traceability for the selected Cholesky target.
- Path normalization risks are separated from evidence availability risks.
- Promotion cannot proceed with unidentified selected row or artifact mapping.

---

## Day 4: Promotion Criteria

**Title:** Promotion Criteria
**Theme:** Define exact evidence thresholds for promotion versus deliberate
re-deferral.
**Time estimate:** 12 hours

### Tasks

1. Define the minimum hosted evidence required to promote selected Windows
   Cholesky freshness.
2. Define the exact re-deferral conditions for stale rows, missing artifacts,
   uninspected hosted runs, path mismatches, or incomplete workflow metadata.
3. Map each decision outcome to manifest, workflow, documentation, guard, and
   residual-queue changes.
4. Decide which validation commands must pass for each outcome.
5. Write the Day 4 promotion-criteria artifact.

### Deliverables

- Promotion versus re-deferral criteria.
- Decision-to-change mapping.
- Validation requirements for each decision path.
- Day 4 criteria artifact.

### Completion Criteria

- Item 208.2 has objective decision rules before implementation.
- The selected lane cannot be promoted without hosted evidence and matching
  metadata.
- A re-deferral path preserves claim safety without ambiguity.

---

## Day 5: Promotion Decision

**Title:** Promotion Decision
**Theme:** Apply hosted evidence to the criteria and select promotion or
continued re-deferral.
**Time estimate:** 12 hours

### Tasks

1. Evaluate Day 2 and Day 3 evidence against Day 4 promotion criteria.
2. Decide whether the selected Windows Cholesky lane earns manifest freshness
   metadata or remains re-deferred.
3. Record evidence gaps, residual blockers, and rejected stronger claims for
   the unselected path.
4. Define the exact implementation boundary for manifest, workflow, guard,
   docs, and tests.
5. Write the Day 5 promotion decision artifact and update `WORKING_NOTES.md`.

### Deliverables

- Explicit promotion or re-deferral decision.
- Evidence-backed decision rationale.
- Residual blocker ledger for rejected paths.
- Day 5 decision artifact.

### Completion Criteria

- Item 208.2 is complete with one selected decision path.
- Stronger Windows support claims are either evidence-backed or retained as
  non-claims.
- Implementation can proceed without unresolved decision ambiguity.

---

## Day 6: Manifest Metadata Design

**Title:** Manifest Design
**Theme:** Design selected-target manifest and workflow metadata changes for
the chosen decision.
**Time estimate:** 12 hours

### Tasks

1. Map the Day 5 decision to exact `selected_report_targets.tsv` field changes
   or absence-guard expectations.
2. Define claim scope, workflow files, workflow jobs, artifact names,
   platforms, evidence labels, and non-claims for the selected Cholesky row.
3. Identify schema, manifest-contract, and guard tests that must protect the
   chosen metadata state.
4. Plan documentation wording needed to explain promoted or re-deferred
   Windows selected freshness.
5. Write the Day 6 manifest metadata design artifact.

### Deliverables

- Manifest metadata implementation design.
- Workflow metadata and artifact naming map.
- Contract-test and guard coverage plan.
- Day 6 design artifact.

### Completion Criteria

- Item 208.3 has an implementation-ready manifest design.
- Metadata fields and non-claims match the selected decision.
- Future manifest drift has a planned regression or guard owner.

---

## Day 7: Manifest And Guard Implementation

**Title:** Manifest Implementation
**Theme:** Implement manifest, workflow metadata, or absence guards according
to the selected decision.
**Time estimate:** 12 hours

### Tasks

1. Update selected-target manifest rows or retain absence state with stronger
   guard coverage.
2. Update workflow metadata references only when hosted evidence supports the
   selected freshness claim.
3. Update manifest contract tests for exact row identity, workflow files,
   platforms, artifact names, claim scope, and non-claims.
4. Run focused manifest and schema validation commands.
5. Record implementation details in `WORKING_NOTES.md`.

### Deliverables

- Manifest or absence-guard implementation.
- Updated manifest contract tests.
- Focused manifest validation results.
- Day 7 implementation notes.

### Completion Criteria

- Item 208.3 has concrete manifest or guard changes.
- The selected target row cannot silently drift into unsupported Windows
  claims.
- Focused manifest validation passes before broader normalizer work begins.

---

## Day 8: Workflow And PowerShell Guard Alignment

**Title:** Workflow Guard Alignment
**Theme:** Align Windows workflow and PowerShell guards with the selected
promotion or re-deferral state.
**Time estimate:** 12 hours

### Tasks

1. Audit Windows workflow guard scripts for selected Cholesky metadata,
   artifact naming, generated report paths, and non-claim boundaries.
2. Update PowerShell validation or absence checks according to the Day 5
   decision.
3. Add regressions for stale workflow metadata, missing selected artifact
   names, unsupported platform entries, and reintroduced broad Windows claims.
4. Verify guard failures are specific enough for maintainers to fix.
5. Write the Day 8 workflow guard alignment artifact.

### Deliverables

- Updated Windows workflow or PowerShell guard coverage.
- Regression tests for selected Cholesky workflow metadata.
- Guard diagnostic evidence.
- Day 8 alignment artifact.

### Completion Criteria

- Item 208.3 covers workflow and PowerShell guard surfaces.
- Promoted or re-deferred workflow state is enforced by tests.
- Guard diagnostics identify the exact stale or unsupported field.

---

## Day 9: Normalizer Regression Design

**Title:** Normalizer Design
**Theme:** Design regression coverage for Windows path normalization and
selected target freshness filtering.
**Time estimate:** 12 hours

### Tasks

1. Define normalizer fixtures for Windows separators, artifact-root prefixes,
   generated rows, missing selected rows, stale rows, and artifact mismatch.
2. Map each fixture to a specific failure mode in selected Windows Cholesky
   freshness validation.
3. Decide expected diagnostics for row-filtering and path-normalization
   failures.
4. Identify any test helpers that should be reused instead of duplicated.
5. Write the Day 9 normalizer regression design artifact.

### Deliverables

- Normalizer regression fixture plan.
- Expected diagnostic matrix.
- Reuse map for existing normalizer tests.
- Day 9 design artifact.

### Completion Criteria

- Item 208.4 has an implementation-ready regression plan.
- Every planned fixture ties to a selected Windows Cholesky risk.
- Expected failure messages are defined before tests are written.

---

## Day 10: Normalizer Regression Implementation

**Title:** Normalizer Tests
**Theme:** Add selected Windows Cholesky normalizer and freshness regression
coverage.
**Time estimate:** 12 hours

### Tasks

1. Implement tests for Windows path normalization in report-index and artifact
   freshness flows.
2. Add selected target filtering tests for missing rows, stale rows, extra
   rows, wrong artifact names, and generated-row mismatches.
3. Verify diagnostics include selected target ID, artifact path, row ID, and
   freshness reason where applicable.
4. Run focused normalizer, manifest, and selected freshness tests.
5. Record changed surfaces and results in `WORKING_NOTES.md`.

### Deliverables

- Normalizer and freshness regression tests.
- Focused validation output.
- Diagnostic coverage notes.
- Day 10 implementation record.

### Completion Criteria

- Item 208.4 has executable regression coverage.
- Windows-style selected Cholesky path and row failures are covered.
- Focused normalizer validation passes.

---

## Day 11: Public Documentation Calibration

**Title:** Public Docs
**Theme:** Update user-facing support and freshness wording to match the
selected Windows Cholesky decision.
**Time estimate:** 12 hours

### Tasks

1. Update README support, selected evidence, and benchmark/freshness wording
   affected by the selected decision.
2. Update INSTALL support/readiness language for Windows selected freshness
   without implying broad Windows parity or package support.
3. Keep package, ABI, performance, release, and state-of-the-art non-claims
   intact.
4. Add or update user-facing links to authoritative selected evidence and
   validation commands.
5. Write the Day 11 public documentation artifact.

### Deliverables

- Claim-calibrated README and INSTALL updates.
- User-facing evidence-link notes.
- Updated public non-claim wording where needed.
- Day 11 documentation artifact.

### Completion Criteria

- Item 208.5 has public documentation aligned with evidence.
- Users can identify what the selected Windows Cholesky lane does and does not
  prove.
- Public docs do not imply broad Windows, package, ABI, performance, or release
  support.

---

## Day 12: Maintainer And Corpus Documentation

**Title:** Maintainer Docs
**Theme:** Align maintainer, corpus, schema, and planning docs with the final
promotion or re-deferral state.
**Time estimate:** 12 hours

### Tasks

1. Update maintainer guide sections for selected Windows Cholesky freshness,
   workflow metadata, validation commands, and guard ownership.
2. Update corpus or schema documentation affected by selected target metadata
   and report-index expectations.
3. Update Epic 19 project-plan status or residual references only where the
   branch has evidence.
4. Add or update claim-boundary guard expectations for broad Windows and
   package non-claims.
5. Write the Day 12 maintainer documentation artifact.

### Deliverables

- Maintainer and corpus documentation updates.
- Current-status or residual wording updates where earned.
- Guard ownership notes.
- Day 12 documentation artifact.

### Completion Criteria

- Item 208.5 has maintainer-facing documentation aligned with implementation.
- Maintainers know which commands protect the selected Cholesky lane.
- Current-status wording does not overstate Sprint 208 outcomes.

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the selected validation matrix and fix any doc, guard, manifest,
or test failures before closeout.
**Time estimate:** 12 hours

### Tasks

1. Run selected manifest, schema, workflow, PowerShell, normalizer, freshness,
   documentation, and claim-boundary validation commands.
2. Run C quality gates only if Sprint 208 modified `.c` or `.h` files.
3. Inspect generated or temporary proof artifacts for cleanup requirements and
   untracked file leakage.
4. Fix validation failures within the selected Sprint 208 scope.
5. Write the Day 13 integrated validation artifact.

### Deliverables

- Integrated validation command log.
- Failure and fix ledger, if any.
- Cleanup and untracked-file audit.
- Day 13 validation artifact.

### Completion Criteria

- Item 208.6 has current validation evidence.
- Required commands pass or blockers are documented with exact failing output.
- No generated proof artifacts or unsupported claim changes remain untracked.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Finalize Sprint 208 evidence, status, residuals, and handoff notes.
**Time estimate:** 10 hours

### Tasks

1. Review all Sprint 208 artifacts, working notes, implementation changes, and
   validation results for consistency.
2. Update item status for 208.1 through 208.6 with evidence links and final
   promoted, re-deferred, residual, or blocked dispositions.
3. Reconcile public, maintainer, manifest, workflow, and planning wording so
   support claims match the selected decision.
4. Record residual work for any unearned Windows, package, ABI, performance,
   release, or state-of-the-art claim.
5. Write the Day 14 closeout review artifact and prepare retrospective inputs.

### Deliverables

- Final Sprint 208 closeout artifact.
- Item status and evidence ledger.
- Residual queue inputs.
- Retrospective-ready validation summary.

### Completion Criteria

- Item 208.6 has closeout evidence for the final branch state.
- Sprint 208 outcomes are traceable to artifacts, tests, and docs.
- Any stronger claims not earned by the branch remain explicitly residual or
  non-claims.

