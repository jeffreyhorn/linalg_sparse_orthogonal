# Sprint 226 Plan: Static Candidate And Installed First-Use Closeout

**Sprint Duration:** 14 days

**Goal:** Validate one exact static source-release candidate, prove the installed first-use route for each claimed platform, and close Epic 20 with evidence-calibrated claims.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds 12 hours. Hosted queue and archive transfer time are separate from these estimates.

**Prerequisites:** Reconcile Sprints 217–225 against their evidence and blockers; use the [Sprint 214 static release checklist](../../EPIC_19/SPRINT_214/RETROSPECTIVE.md) and [Sprint 219 installed index-width contract](../SPRINT_219/RETROSPECTIVE.md). Identify the release owner and proposed candidate commit before claiming readiness.

**Evidence rule:** Tie the archive, installed artifacts, quality gates, hosted runs, first-use audit, and approvals to one exact source commit. A source change after candidate freeze requires a new candidate identifier and rerun of affected proof. Keep raw commands, logs, hashes, platform/compiler details, and decision records. Record unavailable hosted evidence as a blocker.

**Existing blocker:** [Sprint 225](../SPRINT_225/RETROSPECTIVE.md) ended without an accepted timing baseline or measured candidate result. Carry that state into the candidate gate. Establish an accepted baseline and rerun the candidate timing job under its reviewed policy if the gate is required for readiness; otherwise record an explicit owner-approved scope decision that leaves the timing claim blocked. Never present a threshold-free freshness result as a timing pass.

**Claim boundary:** This sprint covers a static source candidate and named installed-consumer platforms. A readiness decision does not itself create a tag or public release, shared ABI guarantee, provider package, portable speed claim, or state-of-the-art ranking.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 226.1 Candidate Freeze | 20 h | Days 1–2: 10 + 10 |
| 226.2 Reproducible Source Archive | 28 h | Days 3–5: 10 + 10 + 8 |
| 226.3 Candidate Platform Proof | 26 h | Days 5–8: 2 + 10 + 10 + 4 |
| 226.4 First-Use Audit | 24 h | Days 8–10: 6 + 10 + 8 |
| 226.5 Readiness Decision | 22 h | Days 10–12: 2 + 10 + 10 |
| 226.6 Epic Closeout | 20 h | Days 13–14: 10 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Inventory Candidate Preconditions

**Title:** Inventory Candidate Preconditions

**Theme:** Establish what can be frozen and what remains blocked.

**Time estimate:** 10 hours (226.1: 10 h)

### Tasks

1. Reconcile Sprint 214 static release boundaries, Sprint 219 width-aware install behavior, and Sprints 217–225 completed versus blocked outcomes; record evidence links.
2. Inventory the current `VERSION`, changelog, `LICENSE`, source files, install manifest, platform tiers, quality gates, and release checklist.
3. Identify the candidate selection authority, release owner, rollback owner, and required reviewers; mark unconfirmed assignments open.
4. Start `WORKING_NOTES.md` and a candidate evidence ledger with source SHA, platform names, job IDs, artifact hashes, gate states, and unresolved prerequisites.

### Deliverables

- Candidate precondition and ownership inventory.
- Initial evidence ledger with explicit Sprint 225 timing status.

### Completion Criteria

- Every prerequisite has a source of evidence or a named blocker; no owner or approval is inferred.
- The static-only and timing-claim boundaries are recorded before candidate selection.

---

## Day 2: Freeze the Static Candidate Definition

**Title:** Freeze the Static Candidate Definition

**Theme:** Define one reviewable candidate and a refreeze rule.

**Time estimate:** 10 hours (226.1: 10 h)

### Tasks

1. Select a clean source commit and record its full SHA, tree state, version, changelog entry, and the source/license manifest.
2. Name the claimed platform and compiler tiers, index-width variants, installation path, quality/hosted gates, and exclusions.
3. Record the release and rollback owners, review checkpoints, and explicit decision for any inherited blocker; leave missing approvals pending.
4. Specify how any later source or packaging edit invalidates the candidate identifier and which archive, install, and hosted proofs must be repeated.

### Deliverables

- Candidate freeze record and source/license inventory.
- Platform and gate matrix with owners and refreeze policy.

### Completion Criteria

- One exact commit and version are identified, or candidate selection is recorded as blocked with a specific reason.
- The matrix has no unnamed platform claim or implied release approval.

---

## Day 3: Build the First Source Archive

**Title:** Build the First Source Archive

**Theme:** Produce a traceable static source artifact.

**Time estimate:** 10 hours (226.2: 10 h)

### Tasks

1. Define the deterministic archive command, file ordering, timestamps, ownership metadata, exclusions, and expected archive name from the frozen commit.
2. Build the first archive from a clean checkout; retain command, tool versions, stdout/stderr, exit code, and SHA-256.
3. Inspect archive members for source, `LICENSE`, `VERSION`, changelog, build files, examples, and install documentation; flag unexpected or missing files.
4. Extract into a clean directory and smoke-test that the source tree can configure without hidden checkout files.

### Deliverables

- First candidate archive, manifest, hash, and build log.
- Extracted-tree inspection report.

### Completion Criteria

- The archive is tied to the frozen SHA and has a complete, reviewable member list.
- Any non-reproducible metadata or missing release source is logged as a blocking defect.

---

## Day 4: Independently Rebuild and Compare

**Title:** Independently Rebuild and Compare

**Theme:** Test source-archive reproducibility.

**Time estimate:** 10 hours (226.2: 10 h)

### Tasks

1. Create a second clean checkout or independent build directory at the same commit and rebuild the archive without reusing the first work tree's outputs.
2. Compare SHA-256 values and, on mismatch, compare member names, ordering, metadata, and bytes to identify the source.
3. Verify extracted source-file hashes against the candidate tree and verify that untracked/generated files did not enter either archive.
4. Retain both archive build logs, commands, manifests, and comparison results in the candidate evidence ledger.

### Deliverables

- Second independent archive and reproducibility comparison.
- Root-cause record for any difference.

### Completion Criteria

- Both archives are byte-identical, or reproducibility is explicitly blocked and not presented as passing.
- The comparison can be repeated from the recorded commit and commands.

---

## Day 5: Map Archive to Install and Prepare Platform Runs

**Title:** Map Archive to Install and Prepare Platform Runs

**Theme:** Close archive provenance and lock the test matrix.

**Time estimate:** 10 hours (226.2: 8 h; 226.3: 2 h)

### Tasks

1. Build and install from the extracted archive into an empty prefix; map installed headers, static library, CMake files, and pkg-config metadata back to archive inputs.
2. Validate source/license manifest coverage and record hashes of installed deliverables and the exact build configuration.
3. Resolve archive or install-map defects; if source changes, refreeze the candidate and invalidate prior archive proof.
4. Finalize the platform proof matrix: exact candidate SHA, claimed platforms, width variants, hosted job selectors, required quality checks, and artifact-retention route.

### Deliverables

- Archive-to-installed-file map and clean-prefix receipt.
- Platform test matrix ready for exact-candidate execution.

### Completion Criteria

- Every required installed file has a documented archive source and build step.
- No platform proof is scheduled against a different candidate commit without an explicit refreeze.

---

## Day 6: Exercise Local Static Installed Consumers

**Title:** Exercise Local Static Installed Consumers

**Theme:** Verify the candidate through a clean installed package.

**Time estimate:** 10 hours (226.3: 10 h)

### Tasks

1. Build the extracted candidate in clean directories for supported index-width configurations and install each into a separate prefix.
2. Compile and run downstream CMake and pkg-config consumers against the installed static library where those routes are supported; verify width headers and metadata.
3. Run the applicable package-contract, install/uninstall, and negative width-mismatch checks.
4. Capture complete commands, compiler/OS identity, output, hashes, and failing diagnostics; distinguish local evidence from hosted platform claims.

### Deliverables

- Local static installed-consumer and width-contract receipts.
- Defect and rerun ledger for the frozen candidate.

### Completion Criteria

- Required local consumers run from a clean prefix with the candidate archive, or each failure is a blocker.
- No local result is relabeled as proof for another platform.

---

## Day 7: Run Required Quality and Hosted Gates

**Title:** Run Required Quality and Hosted Gates

**Theme:** Collect platform-specific exact-candidate evidence.

**Time estimate:** 10 hours (226.3: 10 h)

### Tasks

1. Run required format, lint, test, CMake parity, install, and package-contract gates for the candidate as defined by the release checklist.
2. Trigger or locate Linux, macOS, and Windows hosted jobs for every claimed platform tier, including the appropriate static installed-consumer path; record run URLs, attempts, source SHAs, and toolchains.
3. Download retained reports and verify artifact hashes and test counts against job logs; record any unavailable job as missing evidence.
4. Assess the Sprint 225 timing gate separately: accepted baseline, exact-candidate run, decision, or continued blocker.

### Deliverables

- Exact-candidate gate and hosted-run evidence matrix.
- Separate timing-gate status with source-linked proof.

### Completion Criteria

- Each claimed platform has an exact-SHA job and installed-consumer result, or is explicitly unclaimed/blocked.
- A missing hosted result or timing baseline is never described as a pass.

---

## Day 8: Audit Platform Proof and Start First Use

**Title:** Audit Platform Proof and Start First Use

**Theme:** Check result provenance and follow the user-facing entry path.

**Time estimate:** 10 hours (226.3: 4 h; 226.4: 6 h)

### Tasks

1. Reconcile each hosted result with candidate SHA, archive identity, compiler, width, install prefix, logs, and expected test inventory; resolve rerun or stale-artifact ambiguity.
2. Mark the platform proof result per platform as pass, fail, or missing; preserve the separate timing-gate verdict.
3. Walk `README.md` to `INSTALL.md`, `examples/README.md`, installed headers, and API docs as a new external user would.
4. Inventory broken links, duplicated historical instructions, undocumented installed headers, stale commands, and claims that exceed the candidate evidence.

### Deliverables

- Final platform-proof matrix with provenance links.
- First-use route and documentation defect inventory.

### Completion Criteria

- Every platform result names a source run and artifact; unresolved rows remain blocked.
- The first-use inventory starts from the public entry point and reaches an installed example.

---

## Day 9: Test the Clean-Prefix First-Use Path

**Title:** Test the Clean-Prefix First-Use Path

**Theme:** Reproduce installation and first solve from published instructions.

**Time estimate:** 10 hours (226.4: 10 h)

### Tasks

1. Follow the README and INSTALL commands from a fresh extracted archive and clean prefix, without relying on developer build outputs.
2. Build and run the maintained basic example as an external consumer; check compile flags, link flags, index-width selection, and runtime output.
3. Check links to examples/API pages and compare public-header coverage with the installed header manifest.
4. Repair inaccurate or duplicated first-use text and rerun affected commands and links; refreeze the candidate if source or packaged content changes.

### Deliverables

- First-use command transcript and clean-prefix result.
- Focused documentation corrections and link/header coverage report.

### Completion Criteria

- A new user can get from the public entry point to a successful installed static solve using the written commands.
- Corrections are verified against the candidate being evaluated, with no stale archive proof retained after a refreeze.

---

## Day 10: Finish Documentation Audit and Assemble the Decision Packet

**Title:** Finish Documentation Audit and Assemble the Decision Packet

**Theme:** Make the candidate evidence reviewable by owners.

**Time estimate:** 10 hours (226.4: 8 h; 226.5: 2 h)

### Tasks

1. Complete README, INSTALL, example, API-link, and installed-header checks for the claimed first-use route; remove only duplicated or stale prose that affects that route.
2. Recheck version, static-only wording, index-width behavior, platform scope, and absence of unsupported ABI/provider/performance claims.
3. Gather candidate freeze, archive hashes, install mapping, platform results, first-use transcript, checklist, owners, rollback notes, and timing status into one decision packet.
4. Identify missing approvals or evidence and assign each an owner and next action.

### Deliverables

- Completed first-use audit and verified documentation changes.
- Draft candidate readiness packet with blocker list.

### Completion Criteria

- All first-use defects are fixed and retested or listed as blockers.
- The packet distinguishes candidate facts, approvals, and unresolved claims.

---

## Day 11: Review Candidate Gates and Rollback

**Title:** Review Candidate Gates and Rollback

**Theme:** Apply the release checklist to exact-candidate evidence.

**Time estimate:** 10 hours (226.5: 10 h)

### Tasks

1. Review reproducibility, source/license contents, install contract, platform-specific gates, first-use proof, and artifact provenance against the Sprint 214 checklist.
2. Validate rollback owner, rollback trigger, and recovery steps for an incorrect candidate or installed-package regression.
3. Resolve any stale SHA or missing artifact by rerunning the affected candidate proof; do not carry forward an earlier commit's pass.
4. Request and record required maintainer/release-owner reviews using the packet; keep unreceived approvals pending.

### Deliverables

- Checklist-by-evidence review and rollback plan.
- Approval and outstanding-gate register.

### Completion Criteria

- Each readiness gate has a specific artifact and reviewer or a blocking status.
- Rollback ownership and steps are explicit; no approval is assumed from silence.

---

## Day 12: Record the Static Candidate Readiness Decision

**Title:** Record the Static Candidate Readiness Decision

**Theme:** State the supported candidate outcome and its limits.

**Time estimate:** 10 hours (226.5: 10 h)

### Tasks

1. Reconcile final candidate SHA and archive hashes with all platform jobs, installed-consumer receipts, first-use output, and owner approvals.
2. Decide ready or blocked using the frozen checklist; require all candidate gates and approvals before the exact candidate is called release-ready.
3. State the Sprint 225 timing outcome separately and document whether its missing baseline/candidate proof blocks readiness under the agreed scope.
4. Publish the decision record with platform-specific evidence, outstanding actions, rollback reference, and explicit static-only/release/provider/ABI claim boundaries.

### Deliverables

- Signed or clearly pending candidate-readiness decision.
- Exact-candidate evidence index and residual blocker list.

### Completion Criteria

- The readiness label follows the gate results and recorded owner approvals, not an inferred CI state.
- No tag, public release, shared ABI, provider package, or broad performance claim is asserted by the decision alone.

---

## Day 13: Reconcile Epic 20 Work and Claims

**Title:** Reconcile Epic 20 Work and Claims

**Theme:** Account for completed and blocked outcomes across ten sprints.

**Time estimate:** 10 hours (226.6: 10 h)

### Tasks

1. Compare the Epic 20 plan and Sprint 217–226 artifacts with actual merged work and test evidence; classify each planned gap as closed, blocked, or deferred.
2. Trace every user-facing capability, platform, numerical, timing, and release claim to its controlling evidence and constraints.
3. Record unresolved timing baseline, hosted platform, external evidence, and release approval work without converting it into a success claim.
4. Draft the Epic 20 validation and claim ledger with artifact paths, SHA or run ID, owner, and next action for residual work.

### Deliverables

- Epic 20 completion/blocker reconciliation.
- Draft validation and claim ledger.

### Completion Criteria

- Every Epic 20 gap has an evidence-backed status and no blocked item is counted as fully closed.
- The ledger identifies the exact provenance needed to support each retained claim.

---

## Day 14: Publish Closeout and Handoff

**Title:** Publish Closeout and Handoff

**Theme:** Finish the sprint and hand residual work to the next epic.

**Time estimate:** 10 hours (226.6: 10 h)

### Tasks

1. Review the final candidate decision, archive/provenance, platform matrix, first-use audit, and Epic 20 ledger for internal consistency and working links.
2. Publish the Epic 20 closeout and next-epic residual handoff, with owners, acceptance gates, and specific missing evidence for every blocked item.
3. Update `WORKING_NOTES.md` with day-by-day results, commands, test outcomes, candidate SHA/refreeze history, and final claim wording.
4. Prepare the sprint retrospective input: planned versus achieved outcomes, lessons, unresolved risks, and the exact next action for any blocked release or timing gate.

### Deliverables

- Final Epic 20 validation/claim ledger and residual handoff.
- Sprint 226 working notes and closeout evidence index.

### Completion Criteria

- The exact static candidate is called release-ready only if every required gate and owner approval passed; otherwise the record says blocked.
- The existing static source first-use path remains usable, and any unproved timing, platform, ABI, provider, or release claim stays out of the closeout.
