# Sprint 209 Plan: Windows QR Incompatible Promotion Decision

**Sprint Duration:** 14 days
**Goal:** Add hosted Windows/MSVC proof for `qr-incompatible-ls` and promote
selected metadata only if the exact evidence supports it.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 209 estimate in the Epic 19
project plan.

**Primary scope:** Design and inspect hosted Windows/MSVC evidence for the
selected `qr-incompatible-ls` comparison target, add or adjust workflow and
artifact proof paths, harden Windows-style artifact parsing and manifest
contracts, make an evidence-backed promotion or re-deferral decision, calibrate
documentation and claim guards, and complete validation.

**Non-goals:** Broad Windows parity, Windows Makefile parity, Windows
`pkg-config` execution parity, package-manager support, Homebrew/core readiness,
bottles, Linuxbrew, public taps, shared-library or dynamic ABI support, portable
performance claims, release claims, broad QR parity, broad least-squares parity,
external-library parity, or state-of-the-art claims.

---

## Day 1: QR Promotion Intake

**Title:** QR Promotion Intake
**Theme:** Establish the selected Windows QR incompatible scope, inherited
evidence, and claim boundaries before workflow or manifest edits.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 209 Epic 19 project-plan section and map items 209.1
   through 209.6 to planned artifacts, scripts, workflows, manifests, docs, and
   validation commands.
2. Inventory Sprint 203, Sprint 206, and Sprint 208 records that mention
   Windows QR incompatible proof, re-deferral, selected Cholesky promotion
   boundaries, and residual blockers.
3. Inspect `selected_report_targets.tsv`, report-index schemas, Windows
   workflow files, PowerShell guards, normalizer tests, QR comparison
   generator paths, and maintainer docs.
4. Create `WORKING_NOTES.md` with an item checklist, evidence map, risk
   register, validation matrix, decision log, and open questions.
5. Record explicit non-goals for broad Windows parity, package support, ABI,
   performance, release, external-library parity, and state-of-the-art claims.

### Deliverables

- Sprint 209 working-notes scaffold.
- Item-to-evidence traceability map.
- Selected Windows QR incompatible surface inventory.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 209 item has an initial evidence path or artifact category.
- Existing QR incompatible promotion and re-deferral evidence is identified
  before edits.
- Unsupported Windows, package, ABI, performance, release, external-library,
  and state-of-the-art claims remain explicitly out of scope.

---

## Day 2: MSVC Probe Design

**Title:** MSVC Probe Design
**Theme:** Define the hosted Windows/MSVC QR incompatible proof command,
artifact layout, and expected generated rows.
**Time estimate:** 12 hours

### Tasks

1. Identify the exact generator command needed for
   `qr-incompatible-ls` on Windows/MSVC, including CMake generator, architecture,
   configuration, and library path assumptions.
2. Define expected output directories, Windows-style artifact paths, row IDs,
   manifest files, logs, and stale-output cleanup expectations.
3. Compare the QR incompatible target shape with the already guarded Windows
   Cholesky path and document shared versus target-specific metadata.
4. Define failure modes for missing MSVC evidence, stale generated rows, partial
   artifacts, path separator drift, and dependency-status mismatches.
5. Write the Day 2 MSVC probe design artifact.

### Deliverables

- Hosted MSVC QR incompatible probe command design.
- Expected artifact layout and row-ID inventory.
- Target-specific risk list.
- Day 2 probe design artifact.

### Completion Criteria

- Item 209.1 has an exact proposed proof command and artifact contract.
- The design identifies all files required to support promotion.
- Promotion criteria cannot rely on unlisted or ambiguous generated evidence.

---

## Day 3: Hosted Evidence Inventory

**Title:** Hosted Evidence Inventory
**Theme:** Locate and inspect hosted Windows evidence needed to decide whether
selected QR incompatible freshness can be promoted.
**Time estimate:** 12 hours

### Tasks

1. Identify latest hosted Windows workflow runs relevant to
   `qr-incompatible-ls` and record run IDs, commit SHAs, job names, timestamps,
   and artifact retention constraints.
2. Inspect artifact membership for expected QR incompatible rows, index files,
   manifests, logs, dependency-status rows, and generated comparison output.
3. Compare hosted artifact contents against local generated evidence and
   selected manifest expectations.
4. Record whether hosted proof is complete enough for promotion or whether
   evidence remains missing, stale, inaccessible, or locally advisory only.
5. Write the Day 3 hosted evidence inventory artifact.

### Deliverables

- Hosted Windows evidence ledger.
- Artifact membership and path inventory.
- Hosted-versus-local comparison notes.
- Day 3 evidence inventory artifact.

### Completion Criteria

- Item 209.1 has exact hosted evidence identifiers or documented blockers.
- Artifact contents are mapped to selected target expectations.
- Missing, stale, or ambiguous evidence is recorded before implementation.

---

## Day 4: Workflow Implementation Design

**Title:** Workflow Design
**Theme:** Design hosted Windows QR proof workflow updates without broad Windows
promotion or unrelated selected-target changes.
**Time estimate:** 12 hours

### Tasks

1. Map the Day 2 proof command into `.github/workflows/windows-ci.yml` job and
   step boundaries.
2. Define artifact upload names, exact upload paths, fail-closed behavior,
   timeout limits, shell ownership, and generated-output cleanup.
3. Identify PowerShell validator updates needed to recognize owned QR proof
   snippets while continuing to reject unowned selected freshness surfaces.
4. Define workflow guard regressions for wrong target, broad artifact paths,
   missing uploads, stale names, missing fail-closed settings, and unowned
   PowerShell snippets.
5. Write the Day 4 workflow implementation design artifact.

### Deliverables

- Workflow implementation design.
- Artifact upload and ownership contract.
- PowerShell guard update list.
- Day 4 workflow design artifact.

### Completion Criteria

- Item 209.2 has a precise workflow change plan.
- QR proof steps are bounded to the selected target and do not imply broad
  Windows report freshness.
- Guard requirements are known before workflow implementation.

---

## Day 5: Workflow Implementation

**Title:** Workflow Implementation
**Theme:** Implement the selected Windows QR proof path or record why hosted
workflow implementation remains blocked.
**Time estimate:** 12 hours

### Tasks

1. Add or update hosted Windows workflow steps for the
   `qr-incompatible-ls` selected proof path according to the Day 4 design.
2. Add or update PowerShell workflow guard logic for QR-specific owned commands,
   artifact names, upload paths, and fail-closed behavior.
3. Preserve existing selected Cholesky and non-claim guard behavior.
4. Add regression tests for target drift, artifact drift, path breadth, and
   unowned Windows selected freshness commands.
5. Update `WORKING_NOTES.md` with changed files, guard intent, and known
   environment limits.

### Deliverables

- Workflow or explicit implementation-blocker update.
- PowerShell guard changes and regression tests.
- Working-notes implementation record.

### Completion Criteria

- Item 209.2 is either implemented with bounded QR ownership or explicitly
  re-deferred with evidence.
- Existing Windows Cholesky guard behavior remains protected.
- Workflow changes cannot upload broad or unrelated generated artifacts.

---

## Day 6: Artifact Inspection Tests

**Title:** Artifact Tests
**Theme:** Harden tests for Windows-style QR artifact paths, row filtering,
generated rows, and stale or missing artifacts.
**Time estimate:** 12 hours

### Tasks

1. Add or update normalizer tests for Windows separators, artifact roots,
   generated comparison rows, selected-target filtering, and QR incompatible
   row IDs.
2. Add missing-artifact and stale-artifact regressions for the exact QR
   incompatible required-file set.
3. Verify path normalization does not broaden selected evidence to unrelated QR,
   Cholesky, oracle, benchmark, package, or CI rows.
4. Record local generator and freshness commands needed to reproduce selected QR
   incompatible rows.
5. Write the Day 6 artifact inspection test artifact.

### Deliverables

- Windows-style artifact path regression tests.
- Missing and stale artifact diagnostics.
- Selected-target row-filtering evidence.
- Day 6 artifact test artifact.

### Completion Criteria

- Item 209.3 has test coverage for Windows-style QR artifact handling.
- Generated QR incompatible rows are checked by exact target and required-file
  contract.
- Negative cases fail clearly without broadening selected freshness claims.

---

## Day 7: Manifest Decision Criteria

**Title:** Manifest Criteria
**Theme:** Define exact manifest promotion versus re-deferral criteria for the
selected QR incompatible Windows target.
**Time estimate:** 12 hours

### Tasks

1. Define minimum hosted evidence required to add Windows workflow metadata,
   hosted support tier, artifact names, platform metadata, and claim-scope
   wording to the QR incompatible selected manifest row.
2. Define exact re-deferral conditions for missing hosted MSVC proof, stale
   rows, incomplete artifacts, inaccessible run logs, or inconsistent workflow
   metadata.
3. Map each decision outcome to manifest, schema, workflow, guard, docs, and
   residual-queue changes.
4. Define non-claim wording for broad QR parity, least-squares parity,
   external-library parity, package, ABI, performance, release, and
   state-of-the-art surfaces.
5. Write the Day 7 manifest decision criteria artifact.

### Deliverables

- Promotion versus re-deferral criteria.
- Manifest field contract for each decision path.
- Non-claim wording inventory.
- Day 7 criteria artifact.

### Completion Criteria

- Item 209.4 has objective decision rules before manifest edits.
- The selected target cannot be promoted without matching hosted evidence.
- A re-deferral path remains claim-safe and testable.

---

## Day 8: Manifest Decision

**Title:** Manifest Decision
**Theme:** Apply hosted evidence to the criteria and either promote or
re-defer selected Windows QR incompatible metadata.
**Time estimate:** 12 hours

### Tasks

1. Evaluate Day 3 and Day 6 evidence against Day 7 criteria.
2. Decide whether the QR incompatible row earns Windows selected freshness
   metadata or remains re-deferred.
3. Implement selected manifest metadata changes or absence-guard expectations
   for the chosen path.
4. Add or update manifest contract tests for exact identity, row count,
   required files, workflow metadata, support tier, claim scope, and non-claims.
5. Write the Day 8 manifest decision artifact and update `WORKING_NOTES.md`.

### Deliverables

- Explicit promotion or re-deferral decision.
- Manifest updates or absence guards.
- Manifest contract regression tests.
- Day 8 decision artifact.

### Completion Criteria

- Item 209.4 is complete with one evidence-backed decision path.
- Manifest metadata matches the chosen decision exactly.
- Stronger Windows QR claims are either evidence-backed or retained as
  non-claims.

---

## Day 9: Guard Integration

**Title:** Guard Integration
**Theme:** Integrate workflow, manifest, PowerShell, schema, and normalizer
guards around the selected QR decision.
**Time estimate:** 12 hours

### Tasks

1. Add or update shell and Python guards that enforce the Day 8 QR decision.
2. Ensure non-QR selected rows cannot inherit QR Windows workflow or artifact
   metadata accidentally.
3. Ensure QR incompatible metadata cannot drift to a different target key,
   subfamily, artifact pattern, generator command, row count, or required-file
   set.
4. Add regressions for mismatched workflow tuple lengths, reordered or missing
   platforms, stale artifact names, and missing non-claims.
5. Write the Day 9 guard integration artifact.

### Deliverables

- Integrated guard updates.
- Positive and negative regression coverage.
- Day 9 guard integration record.

### Completion Criteria

- Workflow and manifest guards enforce the selected QR decision.
- Guard failures name the drifting field or unsupported claim clearly.
- Existing selected Cholesky and non-Windows contracts remain intact.

---

## Day 10: Public Docs Calibration

**Title:** Public Docs
**Theme:** Update README and INSTALL wording to reflect the QR decision without
overstating Windows support.
**Time estimate:** 12 hours

### Tasks

1. Update README support, evidence, and selected report freshness sections for
   the chosen QR incompatible decision.
2. Update INSTALL support/readiness tables and Windows sections with exact
   promoted or re-deferred wording.
3. Preserve non-claims for broad Windows parity, package-manager support,
   shared-library and ABI support, external-library parity, performance,
   release, and state-of-the-art status.
4. Add or update claim-boundary scans for public docs.
5. Write the Day 10 public docs calibration artifact.

### Deliverables

- README updates.
- INSTALL updates.
- Public-doc claim-boundary guard updates.
- Day 10 docs artifact.

### Completion Criteria

- Item 209.5 public docs match the manifest decision.
- Users can distinguish selected QR evidence from broad Windows or QR support.
- Unsupported claims remain absent and guarded.

---

## Day 11: Maintainer And Corpus Docs

**Title:** Maintainer Docs
**Theme:** Update maintainer, corpus, schema, and planning docs so future
reviewers see the exact QR evidence boundary.
**Time estimate:** 12 hours

### Tasks

1. Update `docs/maintainer_guide.md` with the selected QR decision, evidence
   commands, residuals, and interpretation boundaries.
2. Update corpus README and report-index schema wording for QR incompatible
   Windows metadata or continued re-deferral.
3. Update Epic 19 planning status surfaces and any residual queue entries
   affected by Sprint 209.
4. Add documentation guard coverage for stale, contradictory, or overbroad QR
   Windows wording.
5. Write the Day 11 maintainer and corpus docs artifact.

### Deliverables

- Maintainer guide updates.
- Corpus and schema documentation updates.
- Project-plan or residual-status updates.
- Day 11 maintainer docs artifact.

### Completion Criteria

- Item 209.5 maintainer-facing docs match public docs and manifest state.
- Planning status does not contradict the selected QR decision.
- Future maintainers can find the validation commands and residual boundaries.

---

## Day 12: Focused Validation

**Title:** Focused Validation
**Theme:** Run focused QR generator, freshness, normalizer, manifest, workflow,
PowerShell, and documentation checks.
**Time estimate:** 12 hours

### Tasks

1. Run the selected QR incompatible generator or freshness command required by
   the chosen decision path.
2. Run normalizer, selected manifest, workflow, PowerShell, corpus schema, and
   documentation guard tests affected by the sprint.
3. Run formatting or shell/Python syntax checks for modified guard scripts.
4. Record skipped or unavailable hosted-only evidence separately from pass
   evidence.
5. Write the Day 12 focused validation artifact.

### Deliverables

- Focused validation transcript summary.
- Pass/skip/unavailable evidence table.
- Residual validation blocker list.
- Day 12 validation artifact.

### Completion Criteria

- Item 209.6 has focused validation evidence for every modified non-C surface.
- Hosted-only gaps are not counted as local pass evidence.
- Any failure is fixed or explicitly blocks closeout.

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run broader integration checks and complete review hardening before
closeout.
**Time estimate:** 12 hours

### Tasks

1. Run broader repository validation appropriate to the changed files, including
   full C gates if any `.c` or `.h` files changed.
2. Run package, Windows, selected report, docs, and static deferral guards that
   protect adjacent claim boundaries.
3. Review changed files for stale target names, contradictory Sprint 209 status,
   broken artifact links, and inconsistent claim vocabulary.
4. Update evidence tables, working notes, and artifacts with exact command
   results.
5. Write the Day 13 integrated validation artifact.

### Deliverables

- Integrated validation result table.
- Review hardening notes.
- Updated working-notes evidence ledger.
- Day 13 validation artifact.

### Completion Criteria

- All required validation passes before closeout, or a blocker is recorded.
- Changed surfaces are internally consistent.
- No broad Windows, package, ABI, performance, release, external-library, or
  state-of-the-art claim is introduced accidentally.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Package Sprint 209 evidence, final decision state, residuals, and
handoff notes for PR review.
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 209 items 209.1 through 209.6 against artifacts,
   implementation changes, docs, tests, and validation results.
2. Update `WORKING_NOTES.md` with final changed surfaces, validation commands,
   residuals, and PR-ready summary.
3. Verify `PLAN.md`, daily artifacts, docs, guards, and manifest state use
   consistent terminology for promotion or re-deferral.
4. Prepare closeout notes for the eventual retrospective and PR description.
5. Write the Day 14 closeout review artifact.

### Deliverables

- Complete Sprint 209 closeout artifact.
- Final working-notes checklist and validation ledger.
- PR-ready summary and residual list.
- Retrospective input notes.

### Completion Criteria

- Every Sprint 209 item has a final disposition and evidence reference.
- Promotion or re-deferral status is consistent across manifests, docs, guards,
  and planning artifacts.
- The branch is ready for retrospective creation, final commit, and PR review.
