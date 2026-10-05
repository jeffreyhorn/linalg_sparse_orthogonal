# Sprint 212 Plan: Benchmark Methodology And Threshold Policy

**Sprint Duration:** 14 days
**Goal:** Decide and implement one selected benchmark methodology policy:
either a thresholded gate for one selected benchmark row or a stronger
threshold-free deferral proof.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 212 estimate in the Epic 19
project plan.

**Primary scope:** Inventory selected benchmark evidence, decide whether one
bounded threshold gate is supportable, implement the chosen threshold or
threshold-free methodology policy, add freshness/manifest/docs guard coverage,
and close with validation that preserves non-portable performance wording and
avoids broad performance claims.

**Non-goals:** Portable performance claims, broad benchmark leadership claims,
state-of-the-art claims, release claims, package-manager claims, ABI or shared
library claims, solver behavior changes, benchmark-result retuning to pass a
gate, and any threshold policy that lacks runner/compiler/repeat/variance
metadata.

---

## Day 1: Benchmark Evidence Intake

**Title:** Benchmark Evidence Intake
**Theme:** Establish the selected benchmark evidence surface and decision
inputs before changing tooling or documentation.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 212 Epic 19 project-plan section and map items 212.1
   through 212.6 to expected artifacts, guard updates, docs changes, and
   validation commands.
2. Inventory current selected benchmark freshness scripts, hosted evidence
   rows, manifest fields, report-index fields, and support/readiness wording.
3. Identify benchmark artifacts and docs that currently mention thresholds,
   performance freshness, selected target scope, runner metadata, or
   non-portable performance limitations.
4. Create `WORKING_NOTES.md` with an item checklist, evidence ledger,
   decision log, risk register, validation matrix, and changed-surface tracker.
5. Record sprint non-goals and claim-boundary rules before evaluating
   threshold options.

### Deliverables

- Sprint 212 working-notes scaffold.
- Benchmark evidence inventory.
- Item-to-evidence traceability map.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 212 item has a planned evidence path or artifact category.
- Current benchmark freshness, selected manifest, and documentation surfaces
  are identified.
- Unsupported portable performance, release, package, ABI, and
  state-of-the-art claims remain out of scope.

---

## Day 2: Current Benchmark Baseline

**Title:** Current Benchmark Baseline
**Theme:** Capture the current selected benchmark workflow, metadata, and guard
behavior before making a policy decision.
**Time estimate:** 12 hours

### Tasks

1. Run or inspect the existing selected benchmark freshness and performance
   documentation checks that can execute locally.
2. Record current selected target rows, artifact paths, report metadata,
   runner labels, compiler labels, repeat counts, and known non-claims.
3. Identify stale, missing, ambiguous, or branch-local benchmark metadata that
   would block a thresholded gate.
4. Record current documentation wording in README, INSTALL, maintainer guide,
   benchmark README, and planning artifacts.
5. Write the Day 2 current-benchmark-baseline artifact.

### Deliverables

- Baseline command and inspection record.
- Current selected benchmark metadata table.
- Gap list for threshold eligibility.
- Day 2 baseline artifact.

### Completion Criteria

- Item 212.1 has evidence-backed current-state data.
- Baseline results distinguish hosted evidence from local-only inspection.
- Threshold-blocking metadata gaps are documented before decision analysis.

---

## Day 3: Runner And Variance Inventory

**Title:** Runner And Variance Inventory
**Theme:** Determine whether existing benchmark evidence is stable enough to
support a thresholded policy.
**Time estimate:** 12 hours

### Tasks

1. Inventory runner class, operating system, compiler, CPU disclosure,
   benchmark command, warmup policy, repeat count, and artifact retention for
   selected benchmark lanes.
2. Inspect available historical benchmark artifacts for variance, outliers,
   missing samples, and comparability limits.
3. Define the minimum metadata needed for a thresholded gate and compare it
   with current evidence.
4. Identify the narrowest possible threshold candidate row, if one exists.
5. Write the Day 3 runner-and-variance artifact.

### Deliverables

- Runner/compiler/repeat/warmup metadata matrix.
- Variance and comparability gap list.
- Candidate threshold row shortlist or blocker list.
- Day 3 methodology inventory artifact.

### Completion Criteria

- The sprint can explain why a threshold candidate is eligible or blocked.
- Metadata requirements are explicit and testable.
- The inventory does not generalize one runner into a portable claim.

---

## Day 4: Threshold Decision Criteria

**Title:** Threshold Decision Criteria
**Theme:** Define the acceptance gate for choosing thresholded enforcement or
threshold-free deferral.
**Time estimate:** 12 hours

### Tasks

1. Draft decision criteria for thresholded enforcement, including runner class,
   compiler, repeat count, warmup, variance rule, allowed regression threshold,
   artifact freshness, and selected target key.
2. Draft decision criteria for threshold-free deferral, including stronger
   non-claim wording, metadata completeness guards, and stale threshold
   rejection.
3. Define evidence that would force a stop instead of an implementation choice.
4. Map each decision criterion to a test, script, manifest field, or
   documentation assertion.
5. Write the Day 4 decision-criteria artifact.

### Deliverables

- Thresholded-gate acceptance criteria.
- Threshold-free deferral acceptance criteria.
- Stop-condition checklist.
- Day 4 decision artifact.

### Completion Criteria

- Item 212.2 has explicit decision criteria before the decision is made.
- Both policy branches have testable outcomes.
- The criteria prevent accidental portable performance overclaims.

---

## Day 5: Product Policy Decision

**Title:** Product Policy Decision
**Theme:** Select the Sprint 212 benchmark methodology policy based on the
inventory and decision criteria.
**Time estimate:** 12 hours

### Tasks

1. Evaluate the Day 2-4 evidence against the thresholded and threshold-free
   criteria.
2. Decide whether Sprint 212 will implement one selected threshold gate or
   close threshold deferral with stronger methodology guards.
3. Record rationale, rejected alternatives, required non-claims, and the exact
   selected scope.
4. Define the implementation plan for scripts, tests, manifest fields, and
   documentation based on the selected policy.
5. Write the Day 5 product-policy-decision artifact.

### Deliverables

- Sprint 212 benchmark policy decision.
- Selected target or deferral scope statement.
- Rejected-options table.
- Day 5 decision artifact.

### Completion Criteria

- Item 212.2 is complete with a documented decision.
- The selected policy has a bounded implementation surface.
- Unsupported performance claims and unreviewed threshold expansion are
  explicitly rejected.

---

## Day 6: Methodology Schema Design

**Title:** Methodology Schema Design
**Theme:** Design the fields, checks, and ownership rules for the selected
benchmark methodology policy.
**Time estimate:** 12 hours

### Tasks

1. Design threshold metadata fields or threshold-free deferral metadata fields
   needed by the selected policy.
2. Map field ownership across selected target manifest, report index schema,
   benchmark scripts, and documentation.
3. Define allowed values, exact text requirements, and forbidden stale wording
   for non-claims.
4. Plan fixtures for missing metadata, stale metadata, unsupported threshold
   expansion, and docs overclaim regressions.
5. Write the Day 6 methodology-schema-design artifact.

### Deliverables

- Methodology field design.
- Ownership and source-of-truth map.
- Guard fixture plan.
- Day 6 schema artifact.

### Completion Criteria

- Item 212.3 has an implementation-ready design.
- Every new or strengthened field has a validation owner.
- Documentation and manifest wording remain aligned by design.

---

## Day 7: Tooling Implementation Batch One

**Title:** Tooling Implementation Batch One
**Theme:** Implement the first tooling changes for the selected benchmark
methodology policy.
**Time estimate:** 12 hours

### Tasks

1. Update the selected benchmark freshness or methodology script for the
   selected policy's required metadata checks.
2. Add or update fixtures for missing runner class, compiler, repeat, warmup,
   variance, threshold, or deferral metadata as applicable.
3. Preserve existing selected benchmark freshness behavior outside the chosen
   policy scope.
4. Run the smallest relevant script or test after the first implementation
   batch.
5. Record changed files, guard behavior, and any deviations in
   `WORKING_NOTES.md`.

### Deliverables

- First benchmark methodology tooling update.
- Initial negative fixtures.
- Focused validation output.
- Working-notes implementation record.

### Completion Criteria

- Item 212.3 has initial code or guard implementation.
- Existing unrelated benchmark lanes remain unchanged.
- The first batch has focused validation evidence.

---

## Day 8: Tooling Implementation Batch Two

**Title:** Tooling Implementation Batch Two
**Theme:** Complete the selected tooling path and ensure metadata enforcement
matches the Day 5 policy decision.
**Time estimate:** 12 hours

### Tasks

1. Finish thresholded-gate or threshold-free-deferral script logic.
2. Add positive fixtures that prove the selected policy passes with exact
   expected metadata.
3. Add negative fixtures for scope expansion, stale wording, missing fields,
   unsupported threshold rows, and incompatible artifacts.
4. Ensure validation output explains failures clearly enough for maintainers
   to repair evidence.
5. Write the Day 8 tooling-implementation artifact.

### Deliverables

- Complete selected tooling implementation.
- Positive and negative methodology fixtures.
- Failure-diagnostic examples.
- Day 8 implementation artifact.

### Completion Criteria

- Item 212.3 implementation is functionally complete.
- Tooling rejects unsupported policy expansion.
- Failure messages identify missing or inconsistent methodology metadata.

---

## Day 9: Manifest And Report Guards

**Title:** Manifest And Report Guards
**Theme:** Bind benchmark methodology policy to selected manifest and report
metadata.
**Time estimate:** 12 hours

### Tasks

1. Update selected manifest rows or report-index fixtures for the chosen
   threshold or threshold-free methodology fields.
2. Add tests that require exact selected target key, artifact pattern, runner
   metadata, compiler metadata, freshness source, and claim scope.
3. Add tests that reject missing non-claims, unsupported threshold carryover,
   and unreviewed performance scope expansion.
4. Confirm report schema documentation describes the authoritative fields and
   non-claim boundary.
5. Write the Day 9 manifest-and-report-guards artifact.

### Deliverables

- Manifest/report metadata updates.
- Exact-field and non-claim tests.
- Schema documentation updates, if required.
- Day 9 guard artifact.

### Completion Criteria

- Item 212.4 covers manifest and report metadata drift.
- Selected benchmark evidence cannot broaden claim scope silently.
- Threshold or deferral policy is tied to authoritative fields.

---

## Day 10: Documentation Calibration Batch One

**Title:** Documentation Calibration Batch One
**Theme:** Update user-facing benchmark methodology and support wording.
**Time estimate:** 12 hours

### Tasks

1. Update benchmark README or benchmark-specific docs with the selected
   methodology decision, exact scope, required metadata, and non-claims.
2. Update README and INSTALL wording for selected benchmark freshness and
   performance support boundaries.
3. Remove or revise stale wording that implies portable performance,
   broad benchmark support, or threshold support beyond the selected policy.
4. Add docs guard tests or fixtures for forbidden performance overclaims.
5. Write the Day 10 documentation-calibration artifact.

### Deliverables

- User-facing benchmark methodology documentation.
- README and INSTALL claim-boundary updates.
- Forbidden-wording guard coverage.
- Day 10 documentation artifact.

### Completion Criteria

- Item 212.5 has user-facing documentation coverage.
- Documentation matches the Day 5 policy decision.
- Portable performance and broad threshold claims remain explicitly unclaimed.

---

## Day 11: Documentation Calibration Batch Two

**Title:** Documentation Calibration Batch Two
**Theme:** Update maintainer and planning documentation so the policy is
operable after the sprint.
**Time estimate:** 12 hours

### Tasks

1. Update the maintainer guide with benchmark methodology ownership,
   validation commands, freshness expectations, and repair workflow.
2. Update Epic 19 project-plan status wording for Sprint 212 once evidence is
   implemented.
3. Update working notes with final changed-surface, line-count, command, and
   policy-decision snapshots.
4. Add docs tests or review checks for stale threshold wording in maintainer
   surfaces.
5. Write the Day 11 maintainer-docs artifact.

### Deliverables

- Maintainer guide benchmark methodology section.
- Project-plan status update draft or implementation.
- Current evidence snapshots.
- Day 11 maintainer documentation artifact.

### Completion Criteria

- Maintainers have exact validation and repair instructions.
- Planning status does not contradict implemented Sprint 212 evidence.
- Documentation guards cover both user-facing and maintainer-facing claim
  boundaries.

---

## Day 12: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the selected benchmark methodology validation suite and required
repository checks for changed surfaces.
**Time estimate:** 12 hours

### Tasks

1. Run benchmark freshness, selected performance docs, manifest/report tests,
   and docs checks relevant to Sprint 212.
2. Run `make format && make lint && make test` if any `.c` or `.h` files were
   modified during implementation.
3. Run focused Python or shell regression suites for benchmark methodology and
   claim-boundary guards.
4. Record pass/fail output, skipped checks, environment limitations, and
   rerun commands in working notes.
5. Write the Day 12 integrated-validation artifact.

### Deliverables

- Integrated validation record.
- Required quality-check evidence.
- Known limitations and rerun commands.
- Day 12 validation artifact.

### Completion Criteria

- Item 212.6 has concrete validation evidence.
- Required full quality checks are complete if C/header files changed.
- Any unavailable hosted or platform evidence is documented as a residual
  risk, not a claim.

---

## Day 13: Review Hardening

**Title:** Review Hardening
**Theme:** Review the full Sprint 212 diff as a reviewer and close concrete
guard or documentation gaps before closeout.
**Time estimate:** 11 hours

### Tasks

1. Review changed scripts, tests, manifests, and docs for raw-text bypasses,
   stale wording, missing exact-field checks, and claim-boundary gaps.
2. Add focused regressions for any discovered guard weakness.
3. Verify line counts, changed-surface tables, and status rows match the actual
   diff.
4. Rerun focused validation impacted by hardening changes.
5. Write the Day 13 review-hardening artifact.

### Deliverables

- Review-hardening findings and fixes.
- Additional regression tests, if needed.
- Updated evidence snapshots.
- Day 13 hardening artifact.

### Completion Criteria

- The branch has been reviewed for common guard and documentation bypasses.
- Any new hardening has passing focused validation.
- Planning evidence matches the implemented branch state.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Reconcile Sprint 212 items, evidence, validation, residual risks,
and non-claims.
**Time estimate:** 11 hours

### Tasks

1. Reconcile items 212.1 through 212.6 against implemented evidence,
   artifacts, tests, docs, and validation results.
2. Record final benchmark methodology policy status and whether thresholded
   enforcement or threshold-free deferral was implemented.
3. Update final changed-surface and validation matrices in `WORKING_NOTES.md`.
4. List residual risks, unsupported claims, and future Epic 19 handoff items.
5. Write the Day 14 closeout-review artifact.

### Deliverables

- Final Sprint 212 item status table.
- Final validation and changed-surface summaries.
- Residual-risk and non-claim list.
- Day 14 closeout artifact.

### Completion Criteria

- Item 212.6 is closed with validation evidence or an explicit blocker.
- Sprint 212 has a coherent final policy decision and documented evidence.
- Remaining benchmark methodology or threshold work is captured as residual
  work without overstating current support.

