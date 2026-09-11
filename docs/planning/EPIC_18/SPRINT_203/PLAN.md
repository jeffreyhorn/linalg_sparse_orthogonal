# Sprint 203 Plan: Windows QR Incompatible Comparison Promotion

**Sprint Duration:** 14 days
**Goal:** Promote the QR incompatible comparison target on Windows only if
MSVC/CMake generation, artifacts, manifest metadata, and docs support it.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 203 estimate in the Epic 18
project plan.

**Primary scope:** Reproduce or run the selected QR incompatible comparison
under the reviewed Windows/MSVC path, fix only selected-target generator,
path, or CMake issues, promote manifest metadata only with hosted evidence,
harden normalizer/workflow tests for Windows artifact behavior, calibrate docs
to exact selected QR boundaries, and run the focused validation set.

**Non-goals:** Broad Windows report freshness, broad QR comparison freshness,
new comparison target families, package-manager or ABI support, performance
claims, Linux/macOS behavior changes outside compatibility preservation,
unselected report-index promotion, or state-of-the-art claims.

---

## Day 1: Windows QR Intake

**Title:** Windows QR Intake
**Theme:** Establish Sprint 203 scope, inherited comparison evidence, and
current Windows promotion constraints.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 203 Epic 18 project-plan section and map items 203.1
   through 203.6 to expected artifacts and validation commands.
2. Review Sprint 191 QR incompatible comparison artifacts, target metadata,
   generator behavior, and local comparison outputs.
3. Review Sprint 190 and Sprint 199 Windows selected comparison promotion
   decisions for reusable guard, manifest, and documentation patterns.
4. Create `WORKING_NOTES.md` with an item checklist, risk register,
   evidence map, validation matrix, and open questions.
5. Record explicit non-goals so the sprint cannot be interpreted as broad
   Windows or QR comparison support.

### Deliverables

- Sprint 203 working-notes scaffold.
- Item-to-artifact traceability map.
- Current QR incompatible comparison inventory.
- Initial Windows promotion risk register.

### Completion Criteria

- Every Sprint 203 item has an initial evidence path or artifact category.
- Prior Windows promotion lessons are captured before implementation starts.
- Broad Windows and broad QR comparison claims remain explicitly out of scope.

---

## Day 2: MSVC Probe Design

**Title:** Probe Design
**Theme:** Define the exact Windows/MSVC/CMake probe needed for the selected
QR incompatible comparison target.
**Time estimate:** 12 hours

### Tasks

1. Identify the selected target key, artifact directory, required files,
   expected row ids, and dependency-status rows for `qr-incompatible-ls`.
2. Map the current generator command to MSVC/CMake inputs, include paths,
   library path expectations, and temporary project layout.
3. Define how Windows path separators, drive letters, generator names,
   configuration names, and architecture flags must be represented.
4. Determine whether hosted CI, local simulation, or both can supply the
   evidence needed for item 203.1.
5. Record the probe command, expected success outputs, and failure diagnostics.

### Deliverables

- MSVC/CMake probe design artifact.
- Selected target file and row inventory.
- Windows path and build-configuration checklist.
- Item 203.1 probe command record.

### Completion Criteria

- The probe is scoped to exactly the QR incompatible comparison target.
- Expected artifacts and diagnostics are defined before running or changing
  generator code.
- Windows-specific path and configuration assumptions are explicit.

---

## Day 3: Probe Execution And Failure Record

**Title:** Probe Execution
**Theme:** Run or faithfully reproduce the selected Windows QR incompatible
comparison and record proof output or blocking failures.
**Time estimate:** 12 hours

### Tasks

1. Run the selected comparison through the available Windows/MSVC path, or
   document the closest reproducible simulation if hosted evidence is pending.
2. Capture generated project files, CMake configure/build output, dependency
   status rows, comparison rows, and manifest files.
3. Classify failures as generator, path normalization, CMake configuration,
   artifact packaging, dependency status, or environment residual.
4. Record whether the evidence is sufficient for promotion or requires a
   re-deferral decision.
5. Update the validation matrix with the exact commands and outputs observed.

### Deliverables

- Day 3 probe execution artifact.
- Failure or proof-output classification ledger.
- Updated validation matrix.
- Initial promotion or re-deferral recommendation.

### Completion Criteria

- Item 203.1 has concrete evidence rather than an assumed Windows outcome.
- Any failure has a scoped owner and next action.
- No manifest promotion occurs without adequate evidence.

---

## Day 4: Generator And CMake Fix Design

**Title:** Fix Design
**Theme:** Design the minimal generator, path, or CMake changes required for
the selected Windows QR incompatible comparison.
**Time estimate:** 12 hours

### Tasks

1. Review the probe failures and identify the smallest affected generator,
   CMake template, path normalization, or artifact-writing surfaces.
2. Define compatibility requirements for existing Linux/macOS selected
   comparison behavior.
3. Specify how dependency status should distinguish unavailable external tools
   from project build failures on Windows.
4. Map each planned fix to a test or validation command before editing.
5. Record unchanged surfaces to prevent accidental broadening of comparison
   families or workflow uploads.

### Deliverables

- Generator/CMake fix design artifact.
- Compatibility preservation checklist.
- Test mapping for each planned fix.
- Item 203.2 implementation boundary.

### Completion Criteria

- Item 203.2 has a narrow implementation plan.
- Existing selected comparison behavior has explicit preservation checks.
- No new target family or broad comparison workflow is introduced by design.

---

## Day 5: Selected Generator Fixes

**Title:** Generator Fixes
**Theme:** Implement selected Windows generator, path, or CMake repairs without
broadening the comparison surface.
**Time estimate:** 12 hours

### Tasks

1. Apply the minimal generator or CMake template changes identified on Day 4.
2. Normalize Windows paths only where selected artifact or temporary project
   behavior requires it.
3. Preserve current target-key filtering, row generation, and dependency
   status semantics for non-Windows platforms.
4. Add focused regression coverage for the repaired path or CMake behavior.
5. Run the focused generator tests or local simulation commands after editing.

### Deliverables

- Selected generator/CMake fixes.
- Focused path or project-generation regression tests.
- Local validation output.
- Updated working-notes implementation record.

### Completion Criteria

- The selected QR incompatible comparison can generate or fail diagnostically
  on the Windows path.
- Existing comparison targets are not promoted or broadened.
- Focused tests cover the repaired behavior.

---

## Day 6: Artifact Path And Row Filtering Tests

**Title:** Artifact Tests
**Theme:** Harden tests for Windows QR artifact paths, selected row filtering,
and stale or missing output diagnostics.
**Time estimate:** 12 hours

### Tasks

1. Add or update tests for backslash, forward-slash, absolute, and
   repo-relative artifact path matching.
2. Verify selected-target filtering keeps only QR incompatible rows when
   Windows metadata is supplied.
3. Add stale-output, missing-file, duplicate-row, and wrong-target fixtures
   for the selected QR incompatible comparison.
4. Confirm diagnostics name the selected target and artifact path clearly.
5. Record all new fixture cases in the Sprint 203 evidence ledger.

### Deliverables

- Windows artifact path regression tests.
- Selected row filtering fixtures.
- Stale and missing output diagnostics.
- Item 203.4 test evidence.

### Completion Criteria

- Windows path separator differences cannot silently drop selected rows.
- Stale or missing QR incompatible artifacts fail clearly.
- Tests remain selected-target scoped.

---

## Day 7: Manifest Promotion Decision

**Title:** Promotion Decision
**Theme:** Decide whether hosted Windows evidence supports selected QR
incompatible manifest promotion or requires explicit re-deferral.
**Time estimate:** 12 hours

### Tasks

1. Compare Day 3 and Day 5 evidence against manifest promotion requirements.
2. Verify expected rows, required files, workflow file, workflow job,
   workflow artifact, and platform metadata for the selected target.
3. Identify retained non-claims for QR correctness, broad parity, package
   proof, ABI proof, performance superiority, and broad Windows freshness.
4. Choose promotion or re-deferral and document the evidence basis.
5. Update the risk register with any hosted evidence still pending.

### Deliverables

- Manifest promotion or re-deferral decision artifact.
- Selected metadata checklist.
- Non-claim inventory.
- Hosted evidence residual list, if needed.

### Completion Criteria

- Item 203.3 has a recorded decision before manifest edits.
- Promotion requires exact selected-target evidence.
- Deferral remains explicit if evidence is insufficient.

---

## Day 8: Manifest And Workflow Metadata

**Title:** Metadata Update
**Theme:** Update selected manifest and workflow metadata only if the Day 7
promotion decision supports it.
**Time estimate:** 12 hours

### Tasks

1. If promoted, add Windows workflow metadata only to the QR incompatible
   selected target row.
2. If re-deferred, update the deferral evidence and keep Windows out of the
   selected target metadata for this row.
3. Add manifest contract tests for expected rows, required files, workflow
   alignment, artifact naming, and retained non-claims.
4. Ensure Windows Cholesky metadata from previous sprints remains unchanged.
5. Run manifest and selected report target validation tests.

### Deliverables

- Promoted or re-deferred manifest state.
- Manifest contract regression tests.
- Workflow metadata alignment evidence.
- Updated selected target source of truth.

### Completion Criteria

- Manifest metadata matches the Day 7 decision.
- Windows QR promotion cannot reuse the wrong artifact or row count.
- Existing selected targets retain their prior support boundaries.

---

## Day 9: Workflow Guard Integration

**Title:** Workflow Guard
**Theme:** Extend workflow guards for the selected Windows QR incompatible
comparison path and artifact set.
**Time estimate:** 12 hours

### Tasks

1. Update Windows workflow guard expectations for the selected QR incompatible
   job only if promotion proceeds.
2. Add negative fixtures for wrong target key, wrong generator command,
   missing artifact upload, extra artifact upload, and wrong runner metadata.
3. Verify selected workflow upload paths match the manifest-required artifact
   set exactly.
4. Preserve existing Windows Cholesky and non-Windows selected comparison
   workflow guard behavior.
5. Record workflow guard coverage in working notes.

### Deliverables

- Selected Windows QR workflow guard changes.
- Exact upload-path regression tests.
- Negative workflow drift fixtures.
- Workflow guard evidence ledger.

### Completion Criteria

- Workflow metadata and artifact uploads cannot drift silently.
- The guard remains selected-target and selected-platform scoped.
- Existing selected comparison workflow tests continue to pass.

---

## Day 10: Normalizer And Freshness Diagnostics

**Title:** Freshness Diagnostics
**Theme:** Validate normalizer and freshness diagnostics for Windows QR
incompatible selected artifacts.
**Time estimate:** 12 hours

### Tasks

1. Exercise the normalizer against selected QR incompatible generated rows
   with Windows-style artifact paths.
2. Verify selected-target freshness checks report missing, stale, wrong-row,
   duplicate-row, and dependency-status failures accurately.
3. Add regression fixtures for any uncovered diagnostic paths.
4. Confirm diagnostics distinguish selected QR promotion from broad Windows
   report-index freshness.
5. Record the normalizer and freshness validation commands.

### Deliverables

- Normalizer/freshness diagnostic regression coverage.
- Windows QR selected freshness validation artifact.
- Diagnostic vocabulary update, if needed.
- Item 203.4 completion evidence.

### Completion Criteria

- Selected Windows QR freshness diagnostics are deterministic and clear.
- Broad report-index freshness remains unpromoted.
- Existing selected comparison freshness tests pass.

---

## Day 11: Documentation Calibration

**Title:** Docs Calibration
**Theme:** Update user, corpus, install, and maintainer docs with exact Windows
QR incompatible comparison boundaries.
**Time estimate:** 12 hours

### Tasks

1. Update README selected target wording to reflect promotion or re-deferral.
2. Update INSTALL and maintainer guide claim boundaries for the selected QR
   Windows path.
3. Update corpus docs and manifest notes so target lists and non-claims point
   back to the manifest as source of truth.
4. Preserve non-claims for broad QR parity, broad Windows freshness, package
   proof, ABI proof, performance, and state-of-the-art claims.
5. Add or update documentation guard markers for the new wording.

### Deliverables

- Claim-safe README, INSTALL, corpus, and maintainer guide updates.
- Documentation guard test changes.
- Updated Sprint 203 evidence notes.
- Item 203.5 documentation evidence.

### Completion Criteria

- Docs state exactly what was promoted or deferred.
- No documentation implies broad Windows, package, ABI, performance, or
  state-of-the-art support.
- Documentation guard tests enforce the selected boundary.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run focused validation across generator, manifest, workflow,
normalizer, docs, QR tests, and Windows guards.
**Time estimate:** 12 hours

### Tasks

1. Run selected comparison generator and freshness checks for the QR
   incompatible target.
2. Run manifest, workflow, normalizer, documentation, and Windows guard tests.
3. Run focused QR comparison or solver tests that validate the selected target
   still behaves as expected.
4. Run `make format && make lint && make test` if any `.c` or `.h` files were
   modified.
5. Record commands, results, environment caveats, and any residuals.

### Deliverables

- Integrated validation artifact.
- Command/result table.
- Residual or hosted-evidence checklist.
- Item 203.6 validation evidence.

### Completion Criteria

- All required focused validation passes.
- Full C gate is run and passes if C/header files changed.
- Any missing hosted evidence is explicit and blocks promotion if required.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Audit the selected Windows QR changes for review-surface size,
claim consistency, and guard completeness.
**Time estimate:** 12 hours

### Tasks

1. Review the full diff for unrelated churn, stale wording, and accidental
   broadening of target metadata or workflows.
2. Cross-check manifest, docs, workflow, and test vocabulary for consistent
   selected QR Windows wording.
3. Verify every changed validation file is listed in Sprint 203 evidence
   inventories.
4. Add final negative tests for any uncovered guard or freshness drift path.
5. Prepare reviewer notes that separate evidence, residuals, and non-claims.

### Deliverables

- Review hardening artifact.
- Diff-scope and changed-file inventory.
- Final guard coverage checklist.
- Reviewer-facing evidence summary.

### Completion Criteria

- The review surface is intentionally scoped and internally consistent.
- Evidence inventories match actual changed files.
- Remaining residuals are explicit and do not overstate promotion.

---

## Day 14: Closeout And Retrospective Inputs

**Title:** Closeout
**Theme:** Finalize Sprint 203 artifacts, validation records, and retrospective
inputs.
**Time estimate:** 12 hours

### Tasks

1. Re-run the required final focused validation set after review hardening.
2. Finalize working notes, daily artifacts, residuals, and project-plan
   disposition language for Sprint 203.
3. Summarize whether Windows QR incompatible comparison was promoted or
   explicitly re-deferred.
4. Record final deliverables, non-claims, risks closed, and risks carried
   forward.
5. Prepare retrospective inputs and PR description material.

### Deliverables

- Day 14 closeout artifact.
- Final validation summary.
- Promotion or re-deferral disposition.
- Retrospective input package.

### Completion Criteria

- Sprint 203 has complete day-by-day evidence and closeout notes.
- The final disposition is supported by validation evidence.
- No broad Windows, package, ABI, performance, or state-of-the-art claim is
  introduced.
