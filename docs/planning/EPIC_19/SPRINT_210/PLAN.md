# Sprint 210 Plan: Additional Allocation-Failure Owner Proof

**Sprint Duration:** 14 days
**Goal:** Add deterministic allocation-failure proof for one new high-value
owner outside the already closed selected symbolic LU path.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 210 estimate in the Epic 19
project plan.

**Primary scope:** Select one high-value allocation owner, document its
lifecycle invariants, extend deterministic allocation-failure hooks and focused
fixtures, add cleanup and retry regressions, wire a focused validation gate,
and calibrate reliability documentation without broadening claims beyond the
selected owner.

**Non-goals:** Broad allocation-failure proof for every owner, package-manager
support, Homebrew/core readiness, bottles, Linuxbrew, public taps,
shared-library or dynamic ABI support, broad platform parity, portable
performance claims, release claims, external-library parity, or
state-of-the-art claims.

---

## Day 1: Allocation Proof Intake

**Title:** Allocation Proof Intake
**Theme:** Establish the selected-owner proof scope, inherited evidence, and
candidate surfaces before implementation.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 210 Epic 19 project-plan section and map items 210.1
   through 210.6 to planned artifacts, tests, scripts, docs, and validation
   commands.
2. Review Sprint 200 symbolic LU allocation proof records, harness behavior,
   cleanup expectations, and follow-up notes.
3. Inventory candidate owner surfaces across matrix import/export, QR
   workspace, LDLT/Cholesky, eigensolver, and SVD workspace paths.
4. Create `WORKING_NOTES.md` with item checklist, owner-candidate ledger, risk
   register, validation matrix, and decision log.
5. Record explicit non-goals for broad allocation coverage, package support,
   ABI, performance, release, external-library parity, and state-of-the-art
   claims.

### Deliverables

- Sprint 210 working-notes scaffold.
- Candidate owner inventory.
- Item-to-evidence traceability map.
- Initial validation matrix and risk register.

### Completion Criteria

- Every Sprint 210 item has an initial evidence path or artifact category.
- Sprint 200 reusable allocation-proof assumptions are identified.
- Unsupported broad reliability, package, ABI, platform, performance, release,
  external-library, and state-of-the-art claims remain out of scope.

---

## Day 2: Owner Ranking

**Title:** Owner Ranking
**Theme:** Rank candidate owners by failure risk, user value, harness
feasibility, and claim safety.
**Time estimate:** 12 hours

### Tasks

1. Define ranking criteria for user-facing impact, allocation density,
   ownership clarity, cleanup complexity, stale-output risk, retry behavior,
   and fixture cost.
2. Inspect candidate paths and existing tests to identify observable outputs,
   caller-owned inputs, partial publications, and cleanup obligations.
3. Compare candidate feasibility against available allocation hook
   infrastructure and current Make/CTest gates.
4. Select one primary owner and one fallback owner, with explicit rationale and
   scope boundaries.
5. Write the Day 2 owner-ranking artifact.

### Deliverables

- Owner ranking matrix.
- Selected primary and fallback owner decision.
- Initial selected-owner boundary statement.
- Day 2 ranking artifact.

### Completion Criteria

- Item 210.1 has an evidence-backed owner selection.
- The selected owner has observable allocation-failure behavior suitable for
  deterministic proof.
- Fallback conditions are documented before implementation begins.

---

## Day 3: Lifecycle Baseline

**Title:** Lifecycle Baseline
**Theme:** Document selected-owner lifecycle invariants before changing tests or
hooks.
**Time estimate:** 12 hours

### Tasks

1. Trace selected-owner entry points, allocation sites, output publication
   points, cleanup paths, and status returns.
2. Identify caller inputs that must be preserved across allocation failure.
3. Identify output buffers, handles, matrices, workspaces, or metadata that
   must remain stale-safe or explicitly cleared.
4. Define retry expectations after each injected failure point.
5. Write the Day 3 lifecycle invariant artifact.

### Deliverables

- Selected-owner lifecycle map.
- Cleanup and stale-output invariant list.
- Caller-input preservation invariant list.
- Retry expectation record.

### Completion Criteria

- Item 210.2 has a concrete invariant baseline.
- Every selected-owner output has a planned allocation-failure expectation.
- Retry and cleanup claims are bounded to the selected owner only.

---

## Day 4: Harness Design

**Title:** Harness Design
**Theme:** Design deterministic allocation-failure fixtures and hook coverage
for the selected owner.
**Time estimate:** 12 hours

### Tasks

1. Inspect existing allocation hook helpers, failure counters, reset behavior,
   and Sprint 200 focused fixtures.
2. Define the selected-owner fixture data, expected successful baseline, and
   failure-injection sweep boundaries.
3. Design helper APIs for status checks, cleanup verification, stale-output
   assertions, preservation checks, and retry checks.
4. Identify registration or source-list changes needed for focused owner tests.
5. Write the Day 4 harness-design artifact.

### Deliverables

- Allocation harness extension design.
- Fixture and helper API plan.
- Failure sweep boundary definition.
- Day 4 harness-design artifact.

### Completion Criteria

- Item 210.3 has a precise implementation design.
- The planned harness can prove cleanup, preservation, stale-output, and retry
  behavior without nondeterministic allocation outcomes.
- Required test registration changes are known before code edits.

---

## Day 5: Harness Implementation

**Title:** Harness Implementation
**Theme:** Implement selected-owner deterministic allocation-failure fixtures
and reusable assertions.
**Time estimate:** 12 hours

### Tasks

1. Add or extend focused test fixtures for the selected owner.
2. Implement allocation hook setup, reset, sweep, and teardown helpers.
3. Implement selected-owner success-baseline assertions.
4. Add cleanup and stale-output assertion helpers.
5. Record changed files, helper boundaries, and any implementation deviations in
   `WORKING_NOTES.md`.

### Deliverables

- Selected-owner allocation-failure fixture implementation.
- Deterministic hook setup and teardown helpers.
- Initial success-baseline assertions.
- Working-notes implementation record.

### Completion Criteria

- Item 210.3 has concrete harness code.
- The selected owner can be exercised under deterministic allocation injection.
- Fixture helpers remain focused and do not imply broad allocation coverage.

---

## Day 6: Failure Sweep Tests

**Title:** Failure Sweep Tests
**Theme:** Add failed-allocation regressions for every selected-owner
allocation site in the proof boundary.
**Time estimate:** 12 hours

### Tasks

1. Add deterministic failure-sweep tests over the selected-owner allocation
   range.
2. Assert expected error/status results for each injected failure point.
3. Verify no successful output is published after failed allocation unless the
   selected-owner contract explicitly permits it.
4. Ensure allocation hook state resets between iterations and does not leak
   into later tests.
5. Write the Day 6 failure-sweep artifact.

### Deliverables

- Failed-allocation sweep tests.
- Status and publication assertions.
- Hook reset evidence.
- Day 6 failure-sweep artifact.

### Completion Criteria

- Item 210.4 has deterministic failed-allocation coverage.
- Every selected-owner failure point has an asserted status and publication
  outcome.
- Test order and repeated execution do not affect results.

---

## Day 7: Cleanup Proof

**Title:** Cleanup Proof
**Theme:** Prove selected-owner cleanup behavior after partial allocation and
early failure.
**Time estimate:** 12 hours

### Tasks

1. Add cleanup assertions for all owned allocations created before each injected
   failure point.
2. Verify partially initialized handles, matrices, workspaces, or metadata do
   not leak ownership to the caller on failure.
3. Add teardown checks that remain valid after both failure and success paths.
4. Record any intentionally caller-owned values that must not be freed by the
   selected owner.
5. Write the Day 7 cleanup-proof artifact.

### Deliverables

- Cleanup-focused regression tests.
- Ownership and teardown evidence.
- Caller-owned preservation notes.
- Day 7 cleanup-proof artifact.

### Completion Criteria

- Item 210.4 includes cleanup proof for the selected owner.
- Partial allocations are cleaned or retained according to documented ownership.
- No cleanup assertion broadens the claim beyond the selected owner.

---

## Day 8: Stale Output And Preservation

**Title:** Stale Output Proof
**Theme:** Prove selected-owner failed allocations do not corrupt caller inputs
or publish stale outputs.
**Time estimate:** 12 hours

### Tasks

1. Add sentinels for caller inputs and output slots before injected failures.
2. Verify caller inputs remain unchanged after allocation failure.
3. Verify output slots are either unchanged, null-cleared, or status-gated
   according to the Day 3 invariants.
4. Add regressions for partial-publication and stale-output edge cases.
5. Write the Day 8 stale-output and preservation artifact.

### Deliverables

- Caller-input preservation tests.
- Stale-output and partial-publication regressions.
- Sentinel fixture evidence.
- Day 8 stale-output artifact.

### Completion Criteria

- Item 210.4 covers preservation and stale-output behavior.
- Failed allocation cannot leave ambiguous selected-owner outputs in tests.
- Preservation claims are evidence-backed and selected-owner scoped.

---

## Day 9: Retry Proof

**Title:** Retry Proof
**Theme:** Prove the selected owner can be retried successfully after injected
allocation failures.
**Time estimate:** 12 hours

### Tasks

1. Add retry-after-failure tests for representative and boundary failure
   points.
2. Verify allocation hook reset restores normal successful behavior.
3. Compare post-retry outputs with the success baseline.
4. Verify cleanup from the failed attempt does not poison the successful retry.
5. Write the Day 9 retry-proof artifact.

### Deliverables

- Retry-after-failure regression tests.
- Success-baseline comparison evidence.
- Hook reset and cleanup interaction notes.
- Day 9 retry-proof artifact.

### Completion Criteria

- Item 210.4 covers retry behavior for the selected owner.
- A failed allocation attempt does not prevent later selected-owner success.
- Retry evidence remains deterministic across repeated test runs.

---

## Day 10: Focused Gate Wiring

**Title:** Focused Gate
**Theme:** Wire a focused Make/CTest validation path for the selected-owner
allocation proof.
**Time estimate:** 12 hours

### Tasks

1. Add or update focused Make targets and CTest registration for the selected
   allocation-failure proof.
2. Add registration/source-list guards if the selected-owner tests introduce
   new files or helper ownership boundaries.
3. Ensure the focused gate composes with existing allocation and reliability
   gates without broadening the tested claim.
4. Run the focused gate locally and record commands and outcomes.
5. Write the Day 10 gate-wiring artifact.

### Deliverables

- Focused allocation-proof gate.
- Registration or source-list guard updates if needed.
- Local focused-gate validation record.
- Day 10 gate artifact.

### Completion Criteria

- Item 210.5 has a repeatable focused validation command.
- New tests are discoverable by the intended local and CI paths.
- Gate names and documentation do not imply broad allocation-proof coverage.

---

## Day 11: Documentation Calibration

**Title:** Documentation Calibration
**Theme:** Update user and maintainer reliability wording to reflect exactly
the selected-owner allocation proof.
**Time estimate:** 12 hours

### Tasks

1. Update reliability, support, or maintainer documentation to describe the new
   selected-owner allocation-failure proof.
2. Preserve non-claims for broad allocation coverage, package support, ABI,
   platform parity, performance, release, external-library parity, and
   state-of-the-art status.
3. Add claim-boundary wording for selected-owner scope, failure modes tested,
   and validation command names.
4. Update planning artifacts and `WORKING_NOTES.md` with documentation changes.
5. Write the Day 11 documentation artifact.

### Deliverables

- Claim-safe reliability documentation updates.
- Maintainer validation notes.
- Updated working-notes documentation ledger.
- Day 11 documentation artifact.

### Completion Criteria

- Item 210.5 includes documentation for the selected proof.
- Documentation names the selected owner and avoids broad allocation proof
  wording.
- Unsupported package, ABI, platform, performance, release, external-library,
  and state-of-the-art claims remain excluded.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run focused, family, source-list, documentation, and C quality checks
for the selected-owner proof.
**Time estimate:** 12 hours

### Tasks

1. Run the selected-owner focused allocation-proof gate.
2. Run relevant family tests for the selected owner and nearby code paths.
3. Run source-list, formatting, lint, docs, and registration guards as
   applicable to changed files.
4. Run the full C quality gate required by code or header changes.
5. Write the Day 12 integrated-validation artifact with exact commands and
   outcomes.

### Deliverables

- Integrated validation command log.
- Pass/fail ledger and remediation notes.
- Day 12 validation artifact.

### Completion Criteria

- Item 210.6 has validation evidence for focused and relevant family coverage.
- Full C quality checks pass when `.c` or `.h` files are changed.
- Any unavailable environment is recorded as non-pass evidence, not promoted.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Audit the selected-owner proof for overclaims, missing cleanup
assertions, brittle fixtures, and guard gaps.
**Time estimate:** 12 hours

### Tasks

1. Review the selected-owner failure-sweep, cleanup, stale-output,
   preservation, retry, and gate evidence against the Day 3 invariants.
2. Add regressions for any discovered missing edge cases or guard weaknesses.
3. Check documentation and planning artifacts for stale owner names,
   over-broad reliability wording, or unsupported package/platform claims.
4. Re-run focused validation after hardening changes.
5. Write the Day 13 review-hardening artifact.

### Deliverables

- Review-hardening findings and fixes.
- Additional regressions or guard updates if needed.
- Focused revalidation record.
- Day 13 hardening artifact.

### Completion Criteria

- Selected-owner proof is internally consistent across tests, docs, and
  artifacts.
- Known cleanup, stale-output, preservation, retry, and claim-boundary gaps are
  either fixed or explicitly recorded.
- Focused validation passes after hardening.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Finalize Sprint 210 evidence, validation, residuals, and readiness
for retrospective and PR review.
**Time estimate:** 12 hours

### Tasks

1. Reconcile Sprint 210 item status against items 210.1 through 210.6.
2. Verify all daily artifacts, `WORKING_NOTES.md`, tests, gates, and
   documentation references agree on the selected owner and claim boundary.
3. Run final focused validation and any required full C quality checks.
4. Record residual risks, deferred owner candidates, and follow-up
   recommendations.
5. Write the Day 14 closeout artifact.

### Deliverables

- Day 14 closeout review artifact.
- Final validation and changed-surface ledger.
- Residual queue entries for unselected owners.
- Retrospective-ready evidence summary.

### Completion Criteria

- Item 210.6 has final validation and closeout evidence.
- Sprint artifacts are complete enough to support a Sprint 210 retrospective.
- The completed proof claims exactly one selected allocation-failure owner and
  does not imply broader reliability, package, ABI, platform, performance,
  release, external-library, or state-of-the-art support.

