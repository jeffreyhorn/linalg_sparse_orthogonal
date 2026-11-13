# Sprint 223 Plan: Selected Windows Cholesky Freshness

**Sprint Duration:** 14 days
**Goal:** Promote freshness for the one selected Windows Cholesky comparison target only after current, inspectable MSVC evidence and all selected-report contract surfaces agree.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours; Epic 20 permits at most 168 hours per sprint. Hosted CI elapsed time is not counted as engineering effort.

**Prerequisites:** Sprint 208's [promotion criteria](../../EPIC_19/SPRINT_208/artifacts/day4-promotion-criteria.md) and six-file artifact contract; Sprint 216's [RQ-04 residual](../../EPIC_19/SPRINT_216/RESIDUAL_QUEUE.md); Sprint 220's structured local-only workflow guard; Sprint 222's [handoff](../SPRINT_222/artifacts/day14-sprint223-handoff.md). The selected target is `cholesky-spd-tridiag-5`, manifest ID `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5`, and hosted job `Windows selected Cholesky comparison freshness (MSVC)`.

**Evidence rule:** A successful local run, an old hosted artifact, or an artifact link without its exact commit, job, six files, row identities and generator metadata does not establish current selected Windows freshness. Recheck the proposed final commit on the hosted lane. If its artifact is absent, expired, wrong, or stale, keep Windows out of the selected manifest and record a re-deferral. The Sprint 222 local CHOLMOD pilot supplies no Windows evidence.

**Claim boundary:** The only possible new claim is freshness of the named Windows Cholesky comparison rows against the selected source-controlled dense reference helper. Do not infer broad Windows report freshness, external-library parity, solver correctness coverage, package/ABI support, performance superiority, release readiness or state-of-the-art status. Keep generated API documentation local-only.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 223.1 Hosted Run Intake | 20 h | Days 1-2: 10 + 10 |
| 223.2 Row And Artifact Audit | 30 h | Days 3-5: 10 + 10 + 10 |
| 223.3 Atomic Promotion | 26 h | Days 6-8: 10 + 10 + 6 |
| 223.4 Negative Guards | 24 h | Days 8-10: 4 + 10 + 10 |
| 223.5 Hosted Recheck | 22 h | Days 11-13: 10 + 10 + 2 |
| 223.6 Scope Sign-Off | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Freeze the Exact Windows Target

**Title:** Freeze the Exact Windows Target

**Theme:** Establish the selected target and hosted evidence requirements.

**Time estimate:** 10 hours (223.1: 10 h)

### Tasks

1. Read Sprint 208's thirteen promotion criteria, Sprint 216 RQ-04, the Sprint 220 guard and the Sprint 222 handoff; list every condition that must agree before promotion.
2. Record the current `master` and branch commit IDs, selected manifest row, six required file names, six expected row IDs, workflow/job/artifact names and expected Windows generator fields.
3. Inspect the current Windows workflow for the exact MSVC build, selected comparison command, normalizer gate and six-file upload; identify the run path needed for this branch's exact commit.
4. Open `WORKING_NOTES.md` with the evidence ledger, commands, owners and an explicit blocked state until hosted artifacts are inspected.

### Deliverables

- Exact target and contract inventory with source revision.
- Initial working notes and hosted run intake checklist.

### Completion Criteria

- The selected Cholesky scope is fixed without conflating it with QR, benchmark or CHOLMOD evidence.
- No Windows manifest or claim change is made from historical evidence alone.

---

## Day 2: Inspect the Hosted MSVC Run

**Title:** Inspect the Hosted MSVC Run

**Theme:** Obtain current, downloadable six-file evidence.

**Time estimate:** 10 hours (223.1: 10 h)

### Tasks

1. Trigger or locate the exact-commit hosted Windows run and record run ID, URL, event, head SHA, branch, timestamps and conclusion.
2. Verify the named selected Cholesky job succeeded and capture its job ID, commands and relevant log excerpts.
3. Record artifact ID, URL, size, upload digest, expiry and download status; download it into a temporary audit directory.
4. Compare archive membership with the exact six-file list. If the run, job or artifact is unavailable, log the blocker and retain the unpromoted state.

### Deliverables

- Hosted run/job/artifact intake ledger and local audit copy or precise missing-evidence report.
- Six-file membership inventory with individual byte hashes.

### Completion Criteria

- Run, job, artifact and commit are linked to the same current hosted execution.
- Missing or wrong membership is recorded as a promotion blocker.

---

## Day 3: Audit Files and Target Rows

**Title:** Audit Files and Target Rows

**Theme:** Prove that the uploaded files contain the selected result.

**Time estimate:** 10 hours (223.2: 10 h)

### Tasks

1. Hash and inspect `project_observations.tsv`, `baseline_observations.tsv`, `dependency_status.tsv`, `study.tsv`, `summary.md` and `manifest.tsv` without rewriting the downloaded bytes.
2. Require exactly the six manifest-owned `comparison_cholesky_spd_tridiag_5_*` row IDs in `study.tsv`; reject missing, duplicate, extra and wrong-target rows.
3. Check every row's `status=pass`, status reason, source revision, branch and clean worktree fields against the hosted run.
4. Trace each selected row to its input/output file and preserve a row-to-artifact evidence table.

### Deliverables

- Per-file hash and exact row-set audit.
- Row-to-artifact traceability table and discrepancy log.

### Completion Criteria

- File membership, row IDs, statuses and provenance all agree or are marked blocked.
- No result is promoted on a subset of the required rows.

---

## Day 4: Audit Generator and Platform Metadata

**Title:** Audit Generator and Platform Metadata

**Theme:** Match the hosted Windows environment to generated report semantics.

**Time estimate:** 10 hours (223.2: 10 h)

### Tasks

1. Check `platform=windows-amd64`, compiler `cmake-probe:Visual Studio 17 2022:Release`, CMake configuration and source commit in the uploaded rows and manifests.
2. Reconcile study, summary, manifest and dependency fields with the workflow commands and selected comparison generator.
3. Identify support-tier, claim-scope or non-claim wording that still says `local_only`, `no hosted CI proof` or `no Windows report freshness` for the exact candidate lane.
4. Record a field-by-field correction map; distinguish hosted observations from source-controlled desired metadata.

### Deliverables

- Generator/platform consistency audit.
- Support-tier and wording correction map.

### Completion Criteria

- Windows provenance is tied to the same six files and target rows.
- Contradictory metadata is identified before any promotion edit.

---

## Day 5: Promotion Decision From Current Evidence

**Title:** Promotion Decision From Current Evidence

**Theme:** Apply the thirteen Sprint 208 gates to the current artifact.

**Time estimate:** 10 hours (223.2: 10 h)

### Tasks

1. Evaluate each Sprint 208 criterion against the Day 2-4 evidence; record pass, fail or unknown with its source.
2. Check selected manifest alignment, row/path handling, generated support tier and claim wording as one contract.
3. Decide whether a promotion candidate can be prepared. If a mandatory hosted condition fails, retain Windows exclusion and write a re-deferral record.
4. Freeze the exact target, artifact IDs and proposed sentence for the subsequent atomic edit and final-commit recheck.

### Deliverables

- Thirteen-gate promotion decision ledger.
- Candidate change list or explicit re-deferral with blocker ownership.

### Completion Criteria

- Item 223.2's 30-hour audit scope has an evidence-backed outcome.
- A candidate is prepared only when the hosted artifact passes all applicable intake and row gates.

---

## Day 6: Align Selected Manifest and Generator

**Title:** Align Selected Manifest and Generator

**Theme:** Make selected Windows metadata truthful for one target.

**Time estimate:** 10 hours (223.3: 10 h)

### Tasks

1. Update only the `SRT-COMP-CHOLESKY-SPD-TRIDIAG-5` row with platform-aligned Windows workflow file, job, artifact and platform entries if Day 5 authorized a candidate.
2. Reconcile the generator's support tier, claim scope and generated summary/non-claim vocabulary for this exact lane; retain all broader exclusions.
3. Check selected row ordering, six-file contract and normalizer compatibility after the edit.
4. Record the before/after contract and any generator field that cannot be reconciled; a contradiction returns the candidate to blocked status.

### Deliverables

- Proposed atomic manifest/generator change or documented re-deferral.
- Field-level contract diff.

### Completion Criteria

- Manifest and generated evidence use consistent, narrow selected-Windows semantics.
- No broad Windows claim is introduced by a support-tier or non-claim edit.

---

## Day 7: Align Documentation and Workflow Guard

**Title:** Align Documentation and Workflow Guard

**Theme:** Carry the exact claim across user and maintainer surfaces.

**Time estimate:** 10 hours (223.3: 10 h)

### Tasks

1. Update README, INSTALL, corpus/schema docs, maintainer guide and planning status to the same exact selected Cholesky freshness wording, conditional on Day 5's evidence.
2. Keep Windows QR, oracle, benchmark, general package support, timing and external-library claims excluded.
3. Check that the Windows workflow still uploads only the six named files and that Sprint 220's generated-API local-only guard remains effective.
4. Review the entire proposed diff for accidental changes to unrelated selected targets or publication destinations.

### Deliverables

- Cohesive candidate documentation and workflow/guard alignment.
- Explicit claim/non-claim matrix.

### Completion Criteria

- All public and maintainer surfaces describe the same bounded target.
- Generated API artifacts remain local-only.

---

## Day 8: Finish Atomic Change and Design Rejection Cases

**Title:** Finish Atomic Change and Design Rejection Cases

**Theme:** Close metadata alignment and freeze adversarial cases.

**Time estimate:** 10 hours (223.3: 6 h; 223.4: 4 h)

### Tasks

1. Complete the 26-hour atomic-promotion scope with manifest, generator, normalized rows, workflow guard and docs reviewed as one diff.
2. Run focused manifest, workflow, generator and normalizer checks; revert or hold promotion if any surface disagrees.
3. Specify negative fixtures for stale commit, missing/renamed file, wrong platform/generator, contradictory support/non-claims and mismatched artifact paths.
4. Map every negative fixture to the contract rule and expected diagnostic.

### Deliverables

- Coherent candidate diff or re-deferral record.
- Negative-test matrix with expected rejection reasons.

### Completion Criteria

- Item 223.3's 26-hour scope is complete without partial claim promotion.
- Each planned failure has an observable rejection condition.

---

## Day 9: Implement Artifact and Row Guards

**Title:** Implement Artifact and Row Guards

**Theme:** Reject stale or structurally wrong hosted evidence.

**Time estimate:** 10 hours (223.4: 10 h)

### Tasks

1. Add or strengthen tests for absent, renamed or extra uploaded files and wrong artifact identity.
2. Add stale source-commit, dirty worktree, missing/duplicate/extra/wrong-target row fixtures.
3. Exercise Windows backslash, mixed-separator and absolute-path matching plus near-match rejection.
4. Run the affected normalizer and workflow suites; keep exact diagnostics in the evidence ledger.

### Deliverables

- Artifact, row and path negative regressions.
- Test results with failure messages for each mutation.

### Completion Criteria

- Every malformed artifact or row fixture is rejected for the intended reason.
- A valid exact six-file fixture remains accepted.

---

## Day 10: Implement Metadata and Claim Guards

**Title:** Implement Metadata and Claim Guards

**Theme:** Prevent wrong-platform and contradictory promotion.

**Time estimate:** 10 hours (223.4: 10 h)

### Tasks

1. Add wrong-platform, wrong-generator and mismatched workflow/job/artifact metadata fixtures.
2. Reject contradictory support-tier, `local_only`, hosted-proof and non-claim combinations for the promoted target.
3. Run selected manifest, comparison workflow, normalizer, generator and Windows PowerShell guard tests.
4. Confirm the negative suite leaves all other selected targets and the API-docs local-only boundary intact.

### Deliverables

- Metadata/claim rejection regressions and full negative-case ledger.
- Focused test transcript for item 223.4.

### Completion Criteria

- Item 223.4's 24-hour scope is complete; all intended negative fixtures fail closed.
- The positive selected target still passes its focused checks.

---

## Day 11: First Hosted Candidate Recheck

**Title:** First Hosted Candidate Recheck

**Theme:** Test the proposed change on the hosted MSVC lane.

**Time estimate:** 10 hours (223.5: 10 h)

### Tasks

1. Push or otherwise run the exact candidate commit through the selected Windows Cholesky job; record run and head SHA.
2. Inspect job conclusion, selected comparison logs, six-file artifact upload and artifact expiry/download status.
3. Download and rerun the Day 3-4 hash, row, platform and generator audit on this new artifact.
4. Fix any candidate failure and preserve failed-run IDs; treat an unavailable artifact as a hard blocker.

### Deliverables

- Hosted candidate run/job/artifact record with reviewer-visible URLs.
- Rechecked six-file/row audit or a precise failed-run report.

### Completion Criteria

- The candidate's hosted evidence is tied to its exact commit.
- No old artifact is substituted for a failed candidate run.

---

## Day 12: Recheck the Final Proposed Commit

**Title:** Recheck the Final Proposed Commit

**Theme:** Eliminate post-evidence drift.

**Time estimate:** 10 hours (223.5: 10 h)

### Tasks

1. Freeze the proposed final source revision, then run the selected hosted MSVC lane again for that exact commit.
2. Confirm successful job, downloadable six-file artifact, row freshness, platform/compiler identity and normalized selected-target result.
3. Record run/job/artifact URLs, IDs, head SHA, timestamps, digest and file hashes in reviewer-visible notes.
4. If any source edit follows this recheck, mark the evidence stale and schedule a new exact-commit run before sign-off.

### Deliverables

- Final-commit hosted evidence ledger and artifact inventory.
- Explicit stale-evidence rule for any subsequent edit.

### Completion Criteria

- The proposed claim points to the exact commit that produced the retained hosted artifact.
- Missing, stale or contradictory evidence blocks promotion.

---

## Day 13: Close Hosted Evidence and Local Tests

**Title:** Close Hosted Evidence and Local Tests

**Theme:** Finish recheck and begin bounded sign-off.

**Time estimate:** 10 hours (223.5: 2 h; 223.6: 8 h)

### Tasks

1. Finish the 22-hour hosted-recheck scope by confirming final artifact accessibility and linking its six hashes to the final commit.
2. Run manifest, workflow, normalizer, external comparison, Windows PowerShell guard and generated-report freshness checks for the selected target.
3. Run `make docs-check` and `git diff --check`; if any `.c` or `.h` changed, also run `make format && make lint && make test`.
4. Audit the final claim/non-claim matrix and log every check with its exact command and outcome.

### Deliverables

- Hosted recheck closeout and local quality-gate transcript.
- Draft exact selected-freshness sign-off sentence or explicit blocked disposition.

### Completion Criteria

- All affected local checks pass and hosted proof remains current.
- No claim is approved merely because the local checks pass.

---

## Day 14: Sign Off or Re-Defer the Exact Claim

**Title:** Sign Off or Re-Defer the Exact Claim

**Theme:** Close the sprint with a traceable narrow decision.

**Time estimate:** 10 hours (223.6: 10 h)

### Tasks

1. Reapply all thirteen promotion criteria to the final commit, hosted run/artifact, selected manifest, generated rows, guards and public docs.
2. Approve only the named Windows Cholesky selected-freshness sentence if every gate passes; otherwise keep Windows absent from selected metadata and document the blocker.
3. Verify no broad Windows, performance, package, ABI, CHOLMOD or state-of-the-art claim leaked into docs or generated reports.
4. Finalize the 140-hour planned-scope ledger, reviewer-visible evidence links, failure log, residual queue and Sprint 224 handoff.

### Deliverables

- Final promotion or re-deferral decision with exact evidence links.
- Complete quality/negative-test record and handoff notes.

### Completion Criteria

- The named Windows Cholesky row is fresh and every contract surface agrees, or promotion is explicitly blocked with no contradictory metadata left behind.
- Item 223.6's 18-hour scope and the 14-day, 140-hour plan are accounted for; no broader Windows solver claim is made.
