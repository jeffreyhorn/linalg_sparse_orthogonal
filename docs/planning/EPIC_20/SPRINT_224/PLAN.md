# Sprint 224 Plan: Selected Windows QR Incompatible Freshness

**Sprint Duration:** 14 days
**Goal:** Promote selected Windows freshness for `qr-incompatible-ls` only when an exact-commit hosted MSVC run produces an inspected six-file artifact and every selected-report contract surface agrees.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours; Epic 20 permits at most 168 hours per sprint. Hosted CI elapsed time is not counted as engineering effort.

**Prerequisites:** Sprint 223's [selected Windows promotion and Sprint 224 handoff](../SPRINT_223/RETROSPECTIVE.md), Sprint 209's [QR collection lane and re-deferral](../../EPIC_19/SPRINT_209/RETROSPECTIVE.md), and Sprint 208's [promotion criteria](../../EPIC_19/SPRINT_208/artifacts/day4-promotion-criteria.md). The target is `qr-incompatible-ls`, selected manifest ID `SRT-COMP-QR-INCOMPATIBLE-LS`, Windows job `Windows selected QR incompatible comparison freshness (MSVC)`, and artifact `sprint209-windows-selected-comparison-qr-incompatible`.

**Evidence rule:** An available workflow, local run, old artifact, or ZIP URL alone cannot establish selected Windows freshness. Download the artifact from an exact-commit successful hosted job, inspect its bytes and six-file membership, reconcile observations with the six study rows, and rerun selected-target freshness. After any source change, repeat the hosted check against the new head. If required evidence is missing, stale, contradictory, or inaccessible, retain Linux/macOS-only selection and record a re-deferral.

**Claim boundary:** Any promotion applies only to the named QR incompatible least-squares comparison rows against the selected source-controlled dense reference helper. It does not establish broad Windows report freshness, QR or least-squares parity, raw QR basis identity, a global rank policy, package or ABI support, performance superiority, or state-of-the-art status. Keep generated API documentation local-only.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 224.1 Hosted QR Collection | 20 h | Days 1-2: 10 + 10 |
| 224.2 Selected Row Audit | 30 h | Days 3-5: 10 + 10 + 10 |
| 224.3 Atomic Metadata Update | 26 h | Days 6-8: 10 + 10 + 6 |
| 224.4 Failure Fixtures | 24 h | Days 8-10: 4 + 10 + 10 |
| 224.5 Hosted Recheck | 22 h | Days 11-13: 10 + 10 + 2 |
| 224.6 Exact-Scope Review | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Freeze the QR Selection Contract

**Title:** Freeze the QR Selection Contract

**Theme:** Define the exact target and evidence required before promotion.

**Time estimate:** 10 hours (224.1: 10 h)

### Tasks

1. Read the Sprint 208 criteria, Sprint 209 re-deferral, Sprint 223 promotion pattern, and Epic 20 item 224 requirements; record which conditions must be independently satisfied for QR.
2. Inventory the current manifest row, six expected `comparison_qr_overdetermined_incompatible_4x2_*` IDs, six filenames, workflow job, artifact name, generator command, and Windows path forms.
3. Confirm the current Windows MSVC job builds the selected library, runs `qr-incompatible-ls`, checks selected freshness, and uploads only the six required files.
4. Start `WORKING_NOTES.md` with the source commit, evidence checklist, owners, commands, and an explicit blocked state until an artifact is inspected.

### Deliverables

- Exact QR target and six-file contract inventory.
- Initial evidence ledger and promotion-blocker checklist.

### Completion Criteria

- The QR target is separated from the already reviewed Cholesky target and other QR cases.
- No Windows QR manifest or public-claim promotion rests on workflow presence alone.

---

## Day 2: Collect the Hosted MSVC QR Artifact

**Title:** Collect the Hosted MSVC QR Artifact

**Theme:** Obtain an inspectable run, job, and six-file ZIP.

**Time estimate:** 10 hours (224.1: 10 h)

### Tasks

1. Locate or trigger the QR job on the exact branch commit; record run ID, event, head SHA, branch, timestamps, job ID, and conclusion.
2. Inspect the job log for the CMake/MSVC build, QR generator command, selected normalizer result, and artifact upload.
3. Record artifact ID, URL, size, API digest, creation and expiry; download the ZIP to a temporary audit directory.
4. List archive members and require exactly `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md`, and `manifest.tsv`; record why any missing or inaccessible artifact blocks promotion.

### Deliverables

- Exact-commit hosted run/job/artifact intake ledger.
- Downloaded six-file inventory or a precise missing-evidence record.

### Completion Criteria

- Run, job, artifact and commit can be traced to one hosted MSVC execution.
- Item 224.1 closes with an inspected input or an explicit blocker, never a guessed result.

---

## Day 3: Verify Archive Bytes and Study Rows

**Title:** Verify Archive Bytes and Study Rows

**Theme:** Prove that the uploaded study is the selected QR study.

**Time estimate:** 10 hours (224.2: 10 h)

### Tasks

1. Compare the downloaded ZIP SHA-256 with the artifact API digest; hash all six extracted files without rewriting their contents.
2. Require the six unique study row IDs in the exact order recorded by `SRT-COMP-QR-INCOMPATIBLE-LS`; reject missing, duplicate, extra, or wrong-target rows.
3. Inspect every study status and reason, source revision, branch, worktree state, artifact path, and generated timestamp.
4. Build a row-to-file traceability table for the study, project and baseline observations; record every discrepancy.

### Deliverables

- ZIP and per-file hash ledger.
- Six-row identity and status audit with observation traceability.

### Completion Criteria

- ZIP membership, file hashes, row identities and row statuses agree or promotion remains blocked.
- No partial row set is treated as completed selected evidence.

---

## Day 4: Reconcile Numerical and Generator Provenance

**Title:** Reconcile Numerical and Generator Provenance

**Theme:** Check the incompatible least-squares result against its inputs and environment.

**Time estimate:** 10 hours (224.2: 10 h)

### Tasks

1. Match each study project and baseline value to the corresponding observation row; recompute reported residual, solution-norm, solution-value and maximum-delta checks against the stated tolerances.
2. Verify the deliberately nonzero incompatible residual and selected solution against the fixture contract; distinguish comparison freshness from broad QR correctness.
3. Reconcile `manifest.tsv`, `summary.md` and six study rows on source SHA, branch, clean worktree, target, baseline identity and command, Python runtime, Windows platform, compiler and CMake configuration.
4. Check required dependencies pass and optional package rows remain deferred; compare support tier, claim scope and non-claims with generator and manifest definitions.

### Deliverables

- Observation-to-study numerical audit and tolerance worksheet.
- Field-by-field hosted provenance and claim-boundary matrix.

### Completion Criteria

- Numerical values and provenance have a coherent six-file explanation.
- A conflict in baseline, platform, status or claim metadata is recorded as a promotion blocker.

---

## Day 5: Decide Whether QR Promotion Is Eligible

**Title:** Decide Whether QR Promotion Is Eligible

**Theme:** Apply the hosted evidence gates before making an atomic change.

**Time estimate:** 10 hours (224.2: 10 h)

### Tasks

1. Apply Sprint 208's promotion criteria to the Day 2-4 run, ZIP, row, provenance and numerical evidence; record pass, fail or unknown with a specific source.
2. Run the selected-target normalizer on the downloaded six-file bundle and compare its output with the source-controlled row contract.
3. Identify the exact manifest, generator, workflow guard, schema, user-doc and maintainer-doc fields that would need a synchronized QR-only update.
4. Authorize a candidate only if all mandatory hosted conditions pass; otherwise preserve Linux/macOS-only selection and write a re-deferral decision.

### Deliverables

- Evidence-backed promotion or re-deferral decision ledger.
- QR-only atomic change map and exact proposed claim sentence if eligible.

### Completion Criteria

- Item 224.2's 30-hour audit has an explicit outcome.
- No candidate uses Cholesky evidence or an artifact from a different head.

---

## Day 6: Align the Selected Manifest and Generator

**Title:** Align the Selected Manifest and Generator

**Theme:** Make QR Windows metadata truthful for one selected target.

**Time estimate:** 10 hours (224.3: 10 h)

### Tasks

1. If Day 5 permits promotion, update only the `SRT-COMP-QR-INCOMPATIBLE-LS` row with the reviewed Windows workflow, job, artifact, platform and hosted support tier.
2. Align generated QR support-tier, summary and claim/non-claim wording with the observed hosted lane while preserving the unrelated QR, package, ABI and performance exclusions.
3. Check exact six-file paths, row ordering and selected normalizer compatibility; leave other target selections unchanged.
4. Record before/after values and stop the candidate if any manifest or generator field contradicts the artifact.

### Deliverables

- QR-only manifest/generator candidate or documented re-deferral.
- Field-level contract diff and unchanged-target check.

### Completion Criteria

- Selected QR metadata is internally consistent and restricted to the named Windows lane.
- A failed evidence gate leaves Windows QR unpromoted.

---

## Day 7: Align Docs and Workflow Ownership

**Title:** Align Docs and Workflow Ownership

**Theme:** Carry the bounded QR freshness claim across maintained surfaces.

**Time estimate:** 10 hours (224.3: 10 h)

### Tasks

1. If promoted, update README, INSTALL, maintainer, corpus and schema wording to the same named QR Windows freshness sentence and explicit non-claims.
2. Review Windows workflow ownership and PowerShell upload guards to ensure only the QR job's six paths are accepted; preserve the Cholesky lane and Sprint 220 local-only API-docs rule.
3. Check the candidate for accidental claims about all Windows QR solves, rank policy, least-squares parity, external packages, ABI or timing.
4. Record doc-to-manifest and workflow-to-artifact correspondence, including any wording that must remain deferred.

### Deliverables

- Synchronized QR documentation and workflow-guard candidate.
- Public/maintainer claim and non-claim matrix.

### Completion Criteria

- Every maintained surface describes the same narrow QR target.
- No generated API publication or unrelated target promotion is introduced.

---

## Day 8: Finish Atomic Alignment and Specify Rejections

**Title:** Finish Atomic Alignment and Specify Rejections

**Theme:** Close promotion edits and freeze adversarial cases.

**Time estimate:** 10 hours (224.3: 6 h; 224.4: 4 h)

### Tasks

1. Review manifest, generator, workflow/PowerShell guard, normalized rows and docs as one QR-only diff; complete item 224.3's 26-hour allocation.
2. Run focused manifest, generator, normalizer, workflow and support-doc checks; return the candidate to blocked status if a contract surface disagrees.
3. Design negative fixtures for missing, renamed and extra files; stale source SHA; wrong or duplicate rows; and wrong Windows path, compiler or baseline identity.
4. Specify expected diagnostics for contradictory hosted/local-only support, claim text, workflow job and upload paths.

### Deliverables

- Coherent atomic candidate or explicit re-deferral.
- Negative-fixture matrix tied to each contract rule.

### Completion Criteria

- Item 224.3 is complete without partial metadata promotion.
- Each failure fixture has a clear observable rejection condition.

---

## Day 9: Add Artifact and Row Failure Fixtures

**Title:** Add Artifact and Row Failure Fixtures

**Theme:** Reject missing, stale or misidentified QR evidence.

**Time estimate:** 10 hours (224.4: 10 h)

### Tasks

1. Add missing/renamed/extra sidecar and upload-path fixtures for the exact six-file QR contract.
2. Add stale commit, dirty worktree, missing/duplicate/extra/wrong-target row and out-of-order row cases.
3. Exercise Windows absolute and backslash paths, mixed separators and near-match paths in the normalizer and workflow guards.
4. Run the affected suites and record the diagnostic that each negative fixture actually produced.

### Deliverables

- Artifact, row and path rejection regressions.
- Negative-case results and retained positive QR fixture.

### Completion Criteria

- Malformed evidence fails for its intended reason; the valid QR bundle still passes.
- Cholesky selected checks remain independent.

---

## Day 10: Add Provenance and Claim Failure Fixtures

**Title:** Add Provenance and Claim Failure Fixtures

**Theme:** Prevent contradictory Windows QR promotion.

**Time estimate:** 10 hours (224.4: 10 h)

### Tasks

1. Add wrong-platform, compiler, generator, baseline runtime/helper and cross-file provenance mutations.
2. Reject hosted study rows paired with `local_only`, `no hosted CI proof`, `no Windows report freshness`, or a mismatched manifest/summary.
3. Exercise workflow job and artifact-name contradictions plus unintended broad Windows QR claims.
4. Run normalizer, generator, selected manifest, comparison workflow and Windows PowerShell guard suites; document pass and fail outcomes.

### Deliverables

- Complete metadata/claim rejection suite and item 224.4 evidence ledger.
- Positive exact-target result and unaffected-target check.

### Completion Criteria

- Item 224.4's 24-hour scope is complete and intended negative cases fail closed.
- A valid named QR fixture remains accepted without claiming broad support.

---

## Day 11: Run the Hosted QR Candidate

**Title:** Run the Hosted QR Candidate

**Theme:** Recheck the proposed change on hosted Windows/MSVC.

**Time estimate:** 10 hours (224.5: 10 h)

### Tasks

1. Run or locate the selected QR job for the exact candidate commit; record the run, job, event, head SHA and conclusion.
2. Inspect CMake/MSVC and generator logs, selected normalizer output and six-file upload.
3. Download the candidate artifact and repeat the ZIP digest, file hash, row, observation and provenance audit.
4. Correct any source failure and retain failed-run IDs; treat a missing or unavailable artifact as a blocker rather than reusing Day 2 evidence.

### Deliverables

- Hosted candidate run/job/artifact record and inspectable URLs.
- Rechecked six-file audit or precise failure log.

### Completion Criteria

- Candidate evidence is tied to its exact source revision.
- Failed or absent evidence cannot promote the selected claim.

---

## Day 12: Recheck the Final Proposed Commit

**Title:** Recheck the Final Proposed Commit

**Theme:** Remove post-evidence source drift.

**Time estimate:** 10 hours (224.5: 10 h)

### Tasks

1. Freeze the proposed final commit and rerun the named Windows selected QR job for that SHA.
2. Require successful job, downloadable six-file ZIP, matching API digest, exact selected rows and passing selected-target normalizer.
3. Record run/job/artifact IDs, URLs, timestamps, expiry, ZIP and file hashes, and manifest/study/summary provenance.
4. Mark this evidence stale after any later source edit and schedule another exact-head run before sign-off.

### Deliverables

- Final proposed-head hosted evidence ledger.
- Explicit stale-evidence and recheck rule.

### Completion Criteria

- The proposed claim is backed by the commit that produced the retained artifact.
- Missing, stale or contradictory evidence blocks approval.

---

## Day 13: Close Hosted and Local Validation

**Title:** Close Hosted and Local Validation

**Theme:** Finish the hosted recheck and begin exact-scope sign-off.

**Time estimate:** 10 hours (224.5: 2 h; 224.6: 8 h)

### Tasks

1. Confirm final artifact accessibility and link its six file hashes and row IDs to the final commit; close item 224.5's 22-hour scope.
2. Run selected manifest, generator, workflow, normalizer, Windows PowerShell, support-doc and QR comparison freshness checks.
3. Run `make docs-check` and `git diff --check`; if any `.c` or `.h` file changed, also run `make format && make lint && make test`.
4. Review the exact QR claim and all non-claims, recording each check's command, outcome and blocker status.

### Deliverables

- Hosted recheck closeout and local quality-gate transcript.
- Draft named-target approval or explicit blocked disposition.

### Completion Criteria

- Every affected local check passes and hosted evidence still matches the proposed final head.
- Local checks alone do not authorize Windows freshness promotion.

---

## Day 14: Sign Off or Re-Defer QR Freshness

**Title:** Sign Off or Re-Defer QR Freshness

**Theme:** Close the sprint with a traceable narrow decision.

**Time estimate:** 10 hours (224.6: 10 h)

### Tasks

1. Reapply the Sprint 208 promotion criteria to the final run/artifact, six files, numerical rows, selected manifest, generator, guards and docs.
2. Approve only the named Windows QR incompatible selected-freshness sentence if every gate passes; otherwise retain Linux/macOS-only selection and state the exact blocker.
3. Verify that Cholesky proof, other QR targets, package/ABI claims, performance rankings and broad Windows parity are not inferred from this artifact.
4. Finalize the planned 140-hour scope ledger, reviewer-visible evidence links, failure log, residual queue and Sprint 225 handoff.

### Deliverables

- Final QR promotion or re-deferral decision with evidence links.
- Complete quality/negative-test record and next-sprint handoff.

### Completion Criteria

- Windows freshness is promoted only for the selected QR target with exact-head hosted proof and consistent contract surfaces, or is explicitly blocked without contradictory metadata.
- Item 224.6's 18-hour scope and all 14 days are accounted for; no broader Windows solver claim is made.
