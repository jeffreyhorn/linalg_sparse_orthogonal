# Sprint 192 Plan: Methodology-Bound Performance Evidence Lane

**Sprint Duration:** 14 days
**Goal:** Promote one performance lane from local threshold-free context to a
methodology-bound hosted evidence lane with explicit limits.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `168` hours, matching the Sprint 192 estimate in the Epic 17
project plan.

**Primary scope:** Select one benchmark family and promote it into a reviewed
hosted performance evidence lane with explicit methodology metadata, stable
artifact publication, bounded runtime, conservative regression policy, report
index integration, documentation, and validation.

**Non-goals:** Broad performance leadership claims, state-of-the-art claims,
unbounded benchmark suites, multiple new performance lanes, package-manager
proofs, cross-platform performance parity, microarchitecture generalization,
release readiness, or performance claims without methodology context.

---

## Day 1: Sprint Intake and Performance Baseline

**Title:** Performance Lane Intake
**Theme:** Establish Sprint 192 scope, current benchmark/report surfaces, and
the acceptance boundary for methodology-bound evidence.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 192 section of the Epic 17 project plan and map items
   192.1 through 192.6 to owner files, scripts, tests, workflows, and docs.
2. Review Sprint 187 performance target selection and acceptance-gate
   artifacts for the intended evidence standard.
3. Inventory existing benchmark generators, report freshness targets,
   canonical benchmark fixtures, CI lanes, artifact upload scopes, and report
   index metadata.
4. Identify current threshold-free benchmark rows, methodology gaps, and
   claim-boundary risks in public and maintainer documentation.
5. Create `WORKING_NOTES.md` with baseline findings, candidate lane list,
   selection criteria, risks, and Day 2 audit questions.

### Deliverables

- Sprint 192 working-notes scaffold.
- Benchmark/report infrastructure inventory.
- Candidate performance lane list.
- Initial methodology and claim-boundary risk register.

### Completion Criteria

- Sprint scope is traceable to items 192.1 through 192.6.
- Current benchmark freshness and artifact ownership are understood before
  implementation begins.
- Candidate selection is constrained to one hosted methodology-bound lane.

---

## Day 2: Candidate Benchmark Lane Audit

**Title:** Lane Selection Audit
**Theme:** Select one benchmark family, fixture, platform, runtime budget, and
acceptance meaning.
**Time estimate:** 12 hours

### Tasks

1. Compare candidate benchmark families by evidence value, determinism,
   runtime cost, fixture stability, implementation risk, and documentation
   clarity.
2. Inspect each candidate's current report rows, generated artifacts,
   benchmark command, input fixture, and host assumptions.
3. Decide the target platform, compiler/backend policy, repeat count, warmup
   behavior, and runtime budget for the selected lane.
4. Rank candidates and record why rejected lanes are out of scope for this
   sprint.
5. Record the selected lane with exact target key, report family, artifact
   paths, and initial claim boundary.

### Deliverables

- Candidate benchmark lane audit artifact.
- Selected performance lane decision.
- Rejected-lane rationale.
- Initial hosted runtime and artifact-size estimate.

### Completion Criteria

- Exactly one benchmark lane is selected for Sprint 192.
- The selected lane has an exact fixture, command, platform, and artifact
  boundary.
- Selection rationale prevents adding multiple partial performance lanes.

---

## Day 3: Methodology Contract

**Title:** Methodology Contract
**Theme:** Define the measurement method, metadata fields, and interpretation
rules before changing generators.
**Time estimate:** 12 hours

### Tasks

1. Define compiler, flags, CPU, thread count, warmup, repeats, timeout,
   timestamp, branch, commit, matrix metadata, and environment metadata
   required for the selected lane.
2. Decide which metadata fields are required, advisory, generated, or
   source-controlled.
3. Define how variance, min/median/max, sample count, and unavailable states
   are represented in report rows.
4. Define what the lane can and cannot claim when metadata is present.
5. Document the methodology contract in working notes and a Day 3 artifact.

### Deliverables

- Methodology metadata contract.
- Required and advisory field table.
- Report-row interpretation rules.
- Claim and non-claim wording draft.

### Completion Criteria

- Every planned performance row has a documented measurement meaning.
- Missing or partial metadata has a defined diagnostic behavior.
- The methodology supports bounded evidence without implying portability.

---

## Day 4: Existing Generator and Fixture Alignment

**Title:** Generator Alignment
**Theme:** Align the selected benchmark generator and fixture with the
methodology contract.
**Time estimate:** 12 hours

### Tasks

1. Inspect the selected benchmark generator, fixture setup, report writer, and
   freshness normalizer paths.
2. Identify where methodology metadata should be captured without duplicating
   authoritative fixture data.
3. Decide whether any benchmark output schema changes are required.
4. Add or update fixture coherence checks if the selected lane lacks stable
   matrix metadata.
5. Record implementation risks and schema migration notes.

### Deliverables

- Generator alignment artifact.
- Fixture metadata ownership decision.
- Output schema change list.
- Implementation checklist for Day 5.

### Completion Criteria

- The selected generator can emit the required methodology fields.
- Fixture metadata has one source of truth.
- Any report schema changes are small, explicit, and testable.

---

## Day 5: Methodology Metadata Implementation

**Title:** Metadata Implementation
**Theme:** Add compiler, runtime, commit, matrix, repeat, and environment
metadata to the selected performance output.
**Time estimate:** 12 hours

### Tasks

1. Extend the selected benchmark/report generator to emit the approved
   methodology metadata fields.
2. Normalize timestamps, branch names, commit IDs, compiler strings, and
   machine/environment fields into stable report values.
3. Add tests for complete metadata, missing metadata, malformed metadata, and
   deterministic fixture metadata.
4. Ensure generated local artifacts remain ignored unless existing corpus
   policy requires committed rows.
5. Update working notes with changed files and validation commands.

### Deliverables

- Implemented methodology metadata output.
- Metadata parser/normalizer tests.
- Updated generated artifact inventory.
- Day 5 implementation artifact.

### Completion Criteria

- Selected benchmark output includes the required methodology metadata.
- Tests fail clearly when required metadata is missing or malformed.
- No generated scratch artifacts are accidentally committed.

---

## Day 6: Report Index Normalization

**Title:** Report Index Integration
**Theme:** Teach report-index normalization and schema validation to recognize
the methodology-bound performance lane.
**Time estimate:** 12 hours

### Tasks

1. Update report index normalization for the selected performance artifact,
   row IDs, metadata fields, support tier, and source-commit freshness.
2. Add schema or manifest entries that identify the selected performance lane
   as methodology-bound hosted evidence.
3. Add tests for missing artifacts, stale commits, row-count mismatch,
   duplicate rows, unsupported states, and malformed metadata.
4. Verify diagnostics include the exact remediation command for the selected
   lane.
5. Record report-index behavior and row semantics in the Day 6 artifact.

### Deliverables

- Normalizer support for the selected performance lane.
- Manifest/schema metadata updates.
- Freshness and failure diagnostics tests.
- Report-index integration artifact.

### Completion Criteria

- Freshness checks can validate the selected performance lane without
  requiring unrelated report families.
- Stale, missing, malformed, duplicate, and incomplete evidence fails clearly.
- Row metadata and manifest metadata agree.

---

## Day 7: Hosted Lane Design

**Title:** Hosted Lane Design
**Theme:** Design the CI job, timeout, artifact publication, and runtime limits
for the selected performance lane.
**Time estimate:** 12 hours

### Tasks

1. Select the hosted workflow, runner, build prerequisites, benchmark command,
   freshness command, timeout, and artifact name.
2. Define exact artifact upload paths for the selected lane only.
3. Ensure the hosted lane does not broaden benchmark or performance upload
   scope beyond the selected artifacts.
4. Add workflow guard tests for job name, command, timeout, artifact name,
   exact paths, and forbidden broad upload patterns.
5. Record hosted-lane design and review risks.

### Deliverables

- Hosted performance lane design artifact.
- CI job contract.
- Exact artifact upload list.
- Workflow guard test plan.

### Completion Criteria

- The hosted lane has a bounded runtime and exact artifact scope.
- Workflow design does not imply broad performance support.
- Guard tests can detect accidental scope expansion.

---

## Day 8: Hosted Lane Implementation

**Title:** Hosted Lane Implementation
**Theme:** Add the reviewed CI lane and artifact upload scope for the selected
methodology-bound performance evidence.
**Time estimate:** 12 hours

### Tasks

1. Implement the selected hosted workflow job or workflow step with the
   approved build, benchmark, freshness, and artifact-upload commands.
2. Add timeout and failure behavior consistent with the Day 7 design.
3. Upload only exact selected performance artifacts with
   `if-no-files-found: error` or the repository's established equivalent.
4. Update workflow tests and selected manifest metadata to match the hosted
   lane.
5. Run focused workflow, manifest, schema, and normalizer checks.

### Deliverables

- Implemented hosted performance evidence lane.
- Exact artifact upload configuration.
- Updated workflow/manifest tests.
- Day 8 implementation artifact.

### Completion Criteria

- The hosted lane is source-controlled and bounded.
- Artifact paths are exact and reviewable.
- Local tests verify workflow structure and selected metadata.

---

## Day 9: Regression Policy Decision

**Title:** Regression Policy
**Theme:** Define whether the selected performance lane enforces a conservative
sentinel or remains threshold-free with reviewed rationale.
**Time estimate:** 12 hours

### Tasks

1. Review historical benchmark variability, fixture size, runner variability,
   and CI runtime behavior for the selected lane.
2. Decide whether to add one conservative regression sentinel or retain
   threshold-free rows with methodology-only freshness.
3. If a sentinel is selected, define baseline source, tolerance, comparison
   statistic, and failure remediation.
4. If rows remain threshold-free, document why freshness plus methodology
   metadata is the safer sprint outcome.
5. Add tests for the selected policy, including unacceptable threshold
   broadening and unsupported pass claims.

### Deliverables

- Regression policy decision artifact.
- Sentinel or threshold-free rationale.
- Policy tests.
- Updated claim-boundary notes.

### Completion Criteria

- The selected policy is conservative and reviewable.
- The policy cannot be interpreted as broad performance superiority.
- Diagnostics distinguish methodology freshness from performance pass claims.

---

## Day 10: Performance Docs and Claim Calibration

**Title:** Claim Calibration
**Theme:** Update public, maintainer, and report docs with methodology-bound
performance wording and non-claims.
**Time estimate:** 12 hours

### Tasks

1. Update benchmark docs, maintainer guide, README/INSTALL if needed, report
   schema docs, and corpus docs with the selected methodology-bound lane.
2. State the exact benchmark family, fixture, platform, repeats, metadata, and
   artifact boundary.
3. Add explicit non-claims for portable performance, speed leadership,
   state-of-the-art status, package-manager proof, ABI support, and unrelated
   benchmark families.
4. Add or update docs guard checks for required performance claim markers and
   forbidden overclaims.
5. Record docs changes and claim-boundary review in working notes.

### Deliverables

- Updated performance documentation.
- Claim-boundary guard updates.
- Non-claim checklist.
- Day 10 claim calibration artifact.

### Completion Criteria

- Documentation explains how to interpret the lane without overclaiming.
- Required non-claims are present in active docs.
- Guard checks catch missing methodology context or broad performance claims.

---

## Day 11: Failure and Drift Coverage

**Title:** Failure Coverage
**Theme:** Harden tests for generator, normalizer, workflow, docs, and
regression-policy failure modes.
**Time estimate:** 12 hours

### Tasks

1. Add focused generator tests for missing metadata, malformed samples,
   unsupported fixture data, and unavailable benchmark prerequisites.
2. Add normalizer tests for stale artifacts, missing methodology fields,
   duplicate rows, wrong target rows, and unsupported policy states.
3. Add workflow tests for timeout drift, artifact scope drift, runner drift,
   and command drift.
4. Add docs guard tests for missing non-claims and forbidden performance
   overclaims.
5. Run the focused test set and record failures fixed.

### Deliverables

- Failure-mode regression tests.
- Drift guard updates.
- Updated working notes.
- Day 11 failure coverage artifact.

### Completion Criteria

- Common implementation mistakes fail before CI artifacts can be misread.
- Tests cover both positive evidence and negative claim-boundary behavior.
- Failure diagnostics are specific enough for future review comments.

---

## Day 12: Integrated Local Validation

**Title:** Integrated Validation
**Theme:** Run the selected benchmark lane, freshness checks, docs guards, and
focused tests as one local validation set.
**Time estimate:** 12 hours

### Tasks

1. Regenerate the selected performance artifacts locally with the final
   benchmark command.
2. Run target-specific freshness, aggregate relevant report freshness, schema
   validation, manifest tests, workflow tests, docs guards, and performance
   policy tests.
3. Inspect generated artifacts for methodology metadata, sample count,
   variance fields, source branch, source commit, runner/platform fields, and
   artifact paths.
4. Confirm generated artifacts remain ignored and no Python caches or scratch
   outputs are left in the worktree.
5. Record validation commands, results, artifact observations, and any
   residuals in working notes.

### Deliverables

- Integrated validation artifact.
- Final generated artifact inspection notes.
- Validation command list and results.
- Residual queue draft.

### Completion Criteria

- Local validation supports the implemented claim surface.
- Generated performance artifacts include the required methodology metadata.
- No generated or cache artifacts are staged accidentally.

---

## Day 13: Review Surface and Residual Audit

**Title:** Review Surface Audit
**Theme:** Reduce review risk by auditing changed files, claim scope,
generated artifacts, tests, and residuals.
**Time estimate:** 12 hours

### Tasks

1. Review the full diff for unnecessary churn, broad performance language,
   accidental generated output, brittle tests, and duplicated authority.
2. Verify selected lane identity is consistent across generator, Makefile,
   manifests, normalizer, workflows, docs, tests, and planning artifacts.
3. Confirm hosted workflow scope matches the selected artifact list and does
   not upload broad benchmark or report directories.
4. Reconcile residuals for unselected benchmark families, platforms,
   package-backed evidence, and portable performance claims.
5. Record final review findings and Day 14 closeout checklist.

### Deliverables

- Review surface audit artifact.
- Final claim-boundary checklist.
- Residual queue.
- Day 14 closeout checklist.

### Completion Criteria

- The branch reads as one coherent methodology-bound performance lane.
- Remaining gaps are documented as residuals, not implied support.
- Reviewers can trace every claim to a specific artifact and validation result.

---

## Day 14: Sprint Closeout and Handoff

**Title:** Sprint Closeout
**Theme:** Finalize validation evidence, close the sprint, and prepare the
branch for retrospective and PR review.
**Time estimate:** 12 hours

### Tasks

1. Rerun final focused validation from Day 12 and any additional guards added
   during Day 13.
2. Verify generated report rows, methodology metadata, manifest metadata,
   workflow artifact paths, docs, and tests agree on the selected performance
   lane contract.
3. Update `WORKING_NOTES.md` with final outcomes, validation results, changed
   files, artifact inspection notes, and residual queue entries.
4. Draft retrospective inputs covering completed scope, retained non-claims,
   accepted risks, and follow-up candidates.
5. Prepare the sprint branch for commit, PR description, and review.

### Deliverables

- Day 14 closeout and handoff artifact.
- Final validation evidence summary.
- Retrospective input notes.
- PR-ready change summary.

### Completion Criteria

- Sprint 192 delivers exactly one methodology-bound hosted performance
  evidence lane.
- Validation evidence supports the implemented claim surface.
- Remaining performance gaps are documented as residuals, not implied support.
