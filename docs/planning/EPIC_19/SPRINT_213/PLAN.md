# Sprint 213 Plan: Generated API Publication Decision

**Sprint Duration:** 14 days
**Goal:** Decide whether generated API HTML remains local-only or is published,
then implement the selected policy with matching automation, routing guards,
documentation, and validation evidence.

**Time budget:** Each day is capped at 12 hours as requested. This day-by-day
plan totals `166` hours, matching the Sprint 213 estimate in the Epic 19
project plan.

**Primary scope:** Compare generated API publication options, make one bounded
policy decision, implement the selected workflow or stronger local-only
automation, add routing and staging guard tests, update user and maintainer
documentation, and close with validation that preserves the generated-output
claim boundary.

**Non-goals:** Broad API stability claims, package-manager distribution claims,
release artifact publication unless explicitly selected, committed generated
HTML unless explicitly selected, ABI/shared-library claims, source API
redesign, Doxygen theme redesign, and any hosted documentation claim that lacks
matching routing, retention, and freshness automation.

---

## Day 1: Generated API Evidence Intake

**Title:** Generated API Evidence Intake
**Theme:** Establish the current generated API evidence surface before
evaluating publication options.
**Time estimate:** 12 hours

### Tasks

1. Re-read the Sprint 213 Epic 19 project-plan section and map items 213.1
   through 213.6 to expected artifacts, code surfaces, documentation, and
   validation commands.
2. Inventory current generated API policy artifacts from Sprint 204, including
   freshness checks, local-only guards, routing checks, and maintainer wording.
3. Identify all generated API references in README, INSTALL, API reference
   docs, maintainer guide, planning files, workflow files, and test fixtures.
4. Create `WORKING_NOTES.md` with an item checklist, evidence ledger, decision
   log, risk register, validation matrix, and changed-surface tracker.
5. Record sprint non-goals and claim-boundary rules before comparing policy
   options.

### Deliverables

- Sprint 213 working-notes scaffold.
- Generated API evidence inventory.
- Item-to-evidence traceability map.
- Initial risk register and validation matrix.

### Completion Criteria

- Every Sprint 213 item has a planned evidence path or artifact category.
- Existing freshness, routing, workflow, and local-only guard surfaces are
  identified.
- Unsupported hosted, release, retained-artifact, package, ABI, and broad API
  claims remain out of scope until explicitly selected.

---

## Day 2: Current Local-Only Baseline

**Title:** Current Local-Only Baseline
**Theme:** Capture the current generated API local-only behavior before
changing policy or automation.
**Time estimate:** 12 hours

### Tasks

1. Run or inspect current API documentation checks that can execute locally,
   including `docs-check`, `api-docs-freshness`, routing, and local-only
   guard behavior.
2. Record generated output paths, ignored artifacts, public header coverage,
   Doxygen inputs, and source-controlled route expectations.
3. Identify current workflow, archive, staging, and link patterns that are
   rejected by the local-only policy.
4. Record existing user and maintainer wording around generated HTML,
   retained artifacts, hosted links, source API routes, and generated-output
   exclusions.
5. Write the Day 2 current-local-only-baseline artifact.

### Deliverables

- Baseline command and inspection record.
- Current generated API local-only behavior table.
- Existing guard coverage and known-gap list.
- Day 2 baseline artifact.

### Completion Criteria

- Item 213.1 has evidence-backed current-state data.
- Baseline results distinguish source-controlled API routes from generated
  Doxygen output.
- Any policy change can be compared against the current local-only contract.

---

## Day 3: Publication Option Inventory

**Title:** Publication Option Inventory
**Theme:** Compare local-only, hosted Pages, retained artifact, and committed
generated HTML options.
**Time estimate:** 12 hours

### Tasks

1. Define each candidate publication policy: keep local-only, publish hosted
   Pages, retain uploaded artifacts, or commit generated HTML.
2. For each option, identify workflow changes, retention implications,
   routing requirements, freshness requirements, and documentation changes.
3. Inventory risks for stale generated output, unsupported API claims,
   source/generated route confusion, artifact retention, and review noise.
4. Compare options against current repository constraints, ignored generated
   paths, and existing Sprint 204 guard behavior.
5. Write the Day 3 publication-option-inventory artifact.

### Deliverables

- Option comparison matrix.
- Workflow and routing impact map.
- Retention and staleness risk list.
- Day 3 publication inventory artifact.

### Completion Criteria

- Item 213.1 covers all four project-plan policy options.
- Each option has explicit automation and claim-boundary implications.
- No option is selected before its validation and maintenance costs are
  documented.

---

## Day 4: Decision Criteria And Stop Conditions

**Title:** Decision Criteria And Stop Conditions
**Theme:** Define the acceptance gate for selecting or rejecting generated API
publication.
**Time estimate:** 12 hours

### Tasks

1. Draft decision criteria for keeping generated API docs local-only,
   including stronger guard behavior and non-claim documentation.
2. Draft decision criteria for hosted publication, including freshness,
   routing, retention, deployment, and stale-output rollback requirements.
3. Draft decision criteria for retained artifacts and committed generated HTML,
   including retention duration, review noise, repository size, and staleness
   controls.
4. Define conditions that force the sprint to stop and ask for user input
   instead of selecting an implementation path.
5. Write the Day 4 decision-criteria artifact.

### Deliverables

- Local-only acceptance criteria.
- Hosted publication acceptance criteria.
- Retained artifact and committed HTML acceptance criteria.
- Stop-condition checklist and Day 4 artifact.

### Completion Criteria

- Item 213.2 has explicit decision criteria before the decision is made.
- Every candidate policy has testable success and failure conditions.
- The criteria prevent accidental generated-output publication or stale hosted
  API claims.

---

## Day 5: Product Policy Decision

**Title:** Product Policy Decision
**Theme:** Select the Sprint 213 generated API policy from the documented
options and criteria.
**Time estimate:** 12 hours

### Tasks

1. Evaluate Day 2-4 evidence against each candidate policy.
2. Select one policy: stronger local-only behavior, hosted publication,
   retained artifact publication, or committed generated HTML.
3. Record rationale, rejected alternatives, support implications, retention
   implications, route implications, and claim boundaries.
4. Define the implementation plan for workflows, scripts, tests, and
   documentation based on the selected policy.
5. Write the Day 5 product-policy-decision artifact.

### Deliverables

- Sprint 213 generated API policy decision.
- Selected implementation scope statement.
- Rejected-options table.
- Day 5 decision artifact.

### Completion Criteria

- Item 213.2 is complete with a documented decision.
- The selected policy has a bounded automation and documentation surface.
- Unsupported generated API publication claims remain explicitly rejected.

---

## Day 6: Automation Design

**Title:** Automation Design
**Theme:** Design the workflow, freshness, routing, staging, and guard changes
needed by the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Map selected-policy requirements to Makefile targets, shell or Python
   guard scripts, workflow files, Doxygen generation steps, and tests.
2. Define ownership for generated output paths, source-controlled API routes,
   hosted or retained destinations, and ignored local artifacts.
3. Design fixtures for stale generated pages, forbidden publication links,
   broad workflow paths, archive staging, source route coverage, and
   selected-policy exceptions.
4. Identify serialization requirements so validation runs after generated
   output is produced.
5. Write the Day 6 automation-design artifact.

### Deliverables

- Automation design map.
- Guard ownership and path policy table.
- Fixture and regression plan.
- Day 6 design artifact.

### Completion Criteria

- Item 213.3 has an implementation design before code changes begin.
- Selected-policy automation can be tested locally.
- Workflow, routing, freshness, and staging responsibilities are not
  ambiguous.

---

## Day 7: Automation Implementation Batch One

**Title:** Automation Implementation Batch One
**Theme:** Implement the first selected-policy automation changes and focused
regressions.
**Time estimate:** 12 hours

### Tasks

1. Update the primary freshness, local-only, routing, workflow, or publication
   automation required by the selected policy.
2. Add or update focused regression fixtures for the highest-risk path class.
3. Preserve existing source-controlled API route behavior unless the selected
   policy explicitly changes it.
4. Run the narrow tests for touched scripts or workflow validators.
5. Write the Day 7 implementation artifact.

### Deliverables

- First automation implementation batch.
- Focused regression fixtures.
- Local validation transcript.
- Day 7 implementation artifact.

### Completion Criteria

- Item 213.3 has concrete branch-local implementation progress.
- The highest-risk selected-policy path is covered by a failing-then-passing
  regression.
- Existing local-only or publication semantics remain coherent.

---

## Day 8: Automation Implementation Batch Two

**Title:** Automation Implementation Batch Two
**Theme:** Complete the selected-policy automation surface across workflows,
routes, and staging checks.
**Time estimate:** 12 hours

### Tasks

1. Finish remaining automation updates for workflow publication, artifact
   retention, generated output staging, routing, or strengthened local-only
   checks.
2. Add regressions for bypass shapes not covered on Day 7, including broad
   paths, archive commands, reference links, HTML anchors, and generated-output
   route variants as applicable.
3. Update Makefile wiring so validation order matches the selected policy.
4. Run relevant script tests and target-level checks.
5. Write the Day 8 implementation artifact.

### Deliverables

- Completed selected-policy automation changes.
- Additional guard and routing regressions.
- Makefile or workflow wiring updates.
- Day 8 implementation artifact.

### Completion Criteria

- Item 213.3 is functionally complete for the selected automation path.
- Item 213.4 has regression coverage for the main bypass categories.
- Validation targets execute in a deterministic order.

---

## Day 9: Routing And Link Validation

**Title:** Routing And Link Validation
**Theme:** Ensure documentation routes distinguish source-controlled API pages
from generated API publication targets.
**Time estimate:** 12 hours

### Tasks

1. Audit Markdown, HTML, and reference-style API links in user docs,
   maintainer docs, planning files, and fixtures.
2. Add or update routing tests for source-controlled routes, forbidden
   generated output routes, selected-policy publication routes, and unrelated
   external documentation links.
3. Validate link normalization for fragments, encoded paths, case variants,
   code fences, comments, inline code, and HTML anchors.
4. Confirm routing behavior matches the Day 5 policy decision.
5. Write the Day 9 routing-and-link-validation artifact.

### Deliverables

- Routing validation updates.
- Link normalization regression set.
- Source vs generated route evidence.
- Day 9 routing artifact.

### Completion Criteria

- Item 213.4 covers generated-output links and source-controlled API routes.
- Valid unrelated external documentation links are not rejected.
- Generated API publication links are allowed only when selected and guarded.

---

## Day 10: Workflow And Staging Validation

**Title:** Workflow And Staging Validation
**Theme:** Validate workflow paths, artifact staging, archive commands, and
publication semantics for the selected policy.
**Time estimate:** 12 hours

### Tasks

1. Audit GitHub workflow files, shell snippets, upload steps, archive commands,
   and deployment commands for generated API exposure.
2. Add or update workflow guard fixtures for broad docs paths, generated API
   literals, local actions, command publishers, dynamic paths, and archive
   staging as applicable.
3. If publication is selected, verify uploads or deployments are narrow,
   fresh, retained as intended, and documented.
4. If local-only is selected, verify publication, retained artifact, and
   committed generated HTML bypasses fail closed.
5. Write the Day 10 workflow-and-staging artifact.

### Deliverables

- Workflow guard updates.
- Staging and archive regression coverage.
- Publication or local-only workflow evidence.
- Day 10 workflow artifact.

### Completion Criteria

- Item 213.4 covers workflow references and generated-output staging.
- Workflow behavior matches the selected Day 5 policy.
- Broad or ambiguous generated API publication paths are rejected unless they
  are explicitly selected and guarded.

---

## Day 11: User Documentation Update

**Title:** User Documentation Update
**Theme:** Update user-facing API documentation to match the selected
generated API policy.
**Time estimate:** 12 hours

### Tasks

1. Update README API documentation wording for generated HTML, source API
   routes, publication status, and retained-artifact status.
2. Update INSTALL and API reference documentation with selected-policy usage,
   local generation, hosted access, or non-publication wording.
3. Ensure docs avoid unsupported claims about API stability, hosted freshness,
   retention, package distribution, ABI, or release artifacts.
4. Add or update documentation guard markers for selected-policy claims and
   non-claims.
5. Write the Day 11 user-documentation artifact.

### Deliverables

- README, INSTALL, and API reference updates.
- User-facing selected-policy wording.
- Documentation guard marker updates.
- Day 11 documentation artifact.

### Completion Criteria

- Item 213.5 is complete for user-facing documentation.
- Users can tell where generated API docs live and what is not claimed.
- Documentation wording matches automation behavior.

---

## Day 12: Maintainer Documentation And Residuals

**Title:** Maintainer Documentation And Residuals
**Theme:** Update maintainer guidance, residual status, and repair workflow
for the selected generated API policy.
**Time estimate:** 12 hours

### Tasks

1. Update the maintainer guide with selected-policy validation commands,
   repair workflow, failure diagnostics, and expected artifacts.
2. Update Epic 19 project-plan status or residual notes affected by the
   generated API decision.
3. Record any remaining unclaimed generated API publication, retention,
   committed-output, or hosted-route options.
4. Ensure planning metadata is consistent with Sprint 213 artifacts and
   selected-policy scope.
5. Write the Day 12 maintainer-documentation artifact.

### Deliverables

- Maintainer guide updates.
- Residual status updates.
- Repair and diagnostic workflow.
- Day 12 maintainer artifact.

### Completion Criteria

- Item 213.5 is complete for maintainer and planning documentation.
- Residual status names only genuinely remaining future work.
- Maintainers have exact commands for diagnosing selected-policy failures.

---

## Day 13: Integrated Validation

**Title:** Integrated Validation
**Theme:** Run the selected generated API validation chain and resolve
integration failures.
**Time estimate:** 11 hours

### Tasks

1. Run docs generation and checking commands required by the selected policy,
   including `make docs-check` and `make api-docs-freshness` where applicable.
2. Run routing, local-only, workflow, staging, and documentation guard tests
   touched by the sprint.
3. Run full C quality gates only if public headers or C sources changed.
4. Record validation commands, results, skipped checks, and reasons.
5. Write the Day 13 integrated-validation artifact.

### Deliverables

- Integrated validation transcript.
- Failure-resolution notes if needed.
- Final changed-surface summary.
- Day 13 validation artifact.

### Completion Criteria

- Item 213.6 has command-backed validation evidence.
- All required selected-policy checks pass.
- Any skipped full C gate is justified by no `*.c` or `*.h` changes.

---

## Day 14: Closeout Review

**Title:** Closeout Review
**Theme:** Confirm Sprint 213 closes the selected generated API policy scope
without overclaiming publication or support.
**Time estimate:** 11 hours

### Tasks

1. Review all Sprint 213 artifacts, working notes, changed files, and
   validation records for consistency.
2. Confirm items 213.1 through 213.6 map to deliverables and completion
   evidence.
3. Check README, INSTALL, maintainer guide, API docs, workflows, Makefile
   targets, scripts, and tests for contradictory generated API wording.
4. Prepare closeout notes with final policy status, residual options, and
   follow-up candidates for later sprints.
5. Write the Day 14 closeout-review artifact.

### Deliverables

- Closeout review artifact.
- Final Sprint 213 item status table.
- Residual and follow-up list.
- Ready-to-retrospective notes.

### Completion Criteria

- All Sprint 213 deliverables are present or explicitly deferred with
  rationale.
- The selected generated API publication policy is implemented and documented.
- The branch is ready for retrospective, commit, and PR creation.

