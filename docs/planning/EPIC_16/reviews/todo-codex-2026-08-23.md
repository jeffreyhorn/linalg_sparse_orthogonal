# Epic 16 Gap Closure Todo - 2026-08-23

## Goal

Close the highest-value residual gaps after Epic 15 without spreading work
thin. Epic 16 should improve product maturity, failure-path confidence,
distribution readiness, generated API discoverability, report governance, and
public API coherence more than it expands algorithm count.

## Step 1: Establish the Epic 16 Evidence Baseline

Tasks:

- Re-read the Epic 13, Epic 14, and Epic 15 retrospectives.
- Extract the residual queue into selected Epic 16 targets, explicit
  non-goals, and long-horizon deferrals.
- Inventory current source, header, test, script, benchmark, docs, package,
  generated API, generated report, and workflow surfaces.
- Create an Epic 16 evidence/status matrix that identifies support tier,
  owner, validation command, hosted/local status, and claim boundary for each
  selected surface.
- Define acceptance gates for allocation-failure proof, generated API
  publication, package-provider readiness, report-target metadata, and public
  header cleanup.

Completion criteria:

- A single baseline artifact identifies the exact gaps Epic 16 will close.
- Unsupported claims and retained non-claims are explicit.
- Every Epic 16 sprint maps to a closeable gap and a validation command.

## Step 2: Broaden Allocation-Failure Evidence to One More Subsystem

Tasks:

- Select one allocation-heavy subsystem beyond iterative repeated-run handles.
- Document ownership and cleanup invariants before implementation.
- Extend the internal allocation-failure hook or add a subsystem-local harness
  that can fail deterministically at selected allocation counts.
- Add regression tests for partial-construction cleanup, no stale state
  publication, and successful retry after failure.
- Add a focused Make/CTest gate and documentation wording that keeps the claim
  subsystem-local.

Completion criteria:

- One additional subsystem has deterministic allocation-failure proof.
- Cleanup invariants are documented.
- Public docs do not imply broad allocation-failure coverage.

## Step 3: Decide Generated API HTML Publication

Tasks:

- Audit the current local-only Doxygen policy, generated output paths, ignored
  file behavior, and freshness checks.
- Decide whether generated API HTML becomes hosted, artifact-retained, or
  remains local-only.
- If hosted, add a workflow publication path, artifact retention policy,
  freshness gate, and navigation docs.
- If local-only remains the product decision, strengthen the local-only guard
  and residual wording.
- Keep public headers and Markdown docs as the source of truth.

Completion criteria:

- Generated API HTML has one explicit product status.
- Users can find the supported API reference path.
- CI/local checks enforce the selected status.

## Step 4: Close One Package-Manager Provider Decision

Tasks:

- Choose exactly one provider candidate, likely vcpkg or Homebrew, or renew
  formal provider deferral with precise blockers.
- If a provider is selected, add a source-controlled prototype recipe/formula
  and provenance notes.
- Add a local proof script for install, downstream compile, version query, and
  cleanup where feasible.
- Update package docs so source install, CMake/pkg-config install, and
  package-manager status remain distinct.
- Replace or update package-manager deferral guards to match the decision.

Completion criteria:

- One provider path is proven or explicitly deferred with stronger evidence.
- Documentation does not imply unearned provider availability.
- Package guards match the selected support tier.

## Step 5: Centralize Selected Report Target Metadata

Tasks:

- Inventory selected oracle, comparison, and performance target lists across
  workflows, tests, scripts, docs, and report manifests.
- Add one source-controlled manifest for selected report targets, including
  family, directory, required files, expected row counts, support tier, and
  hosted/local status.
- Update workflow guards and script tests to read the manifest.
- Keep workflow YAML validation scoped to the exact job and artifact upload
  blocks.
- Preserve fail-closed behavior for selected hosted artifacts.

Completion criteria:

- One manifest owns selected report target metadata.
- Workflow/report guards detect duplicate, missing, stale, or mis-scoped
  target rows.
- Adding a selected family no longer requires uncoordinated edits in multiple
  places.

## Step 6: Promote or Close Windows Report Freshness

Tasks:

- Audit generated report commands for Windows shell, path, newline, Python,
  executable, and dependency assumptions.
- Select one Windows-safe report freshness candidate or record a stronger
  product deferral.
- If promoted, add a Windows CMake-first hosted freshness lane with bounded
  runtime and artifact summary.
- If deferred, add a deferral artifact and guard that prevents accidental
  Windows report freshness claims.
- Update README, maintainer guide, workflow comments, and report metadata.

Completion criteria:

- Windows report freshness is either proven for one selected family or
  formally closed as deferred with exact blockers.
- Documentation and workflows agree on Windows support tier.
- No broad Windows report parity claim is introduced.

## Step 7: Add One Bounded External Comparison Family

Tasks:

- Select the highest-value next comparison family that can be fully closed in
  one sprint.
- Define fixtures, external reference, metrics, tolerances, expected rows, and
  non-parity wording.
- Extend the comparison runner and source-controlled metadata.
- Generate, index, and freshness-check the comparison report.
- Update the selected target manifest from Step 5.

Completion criteria:

- One additional comparison family is complete.
- Its generated rows are indexed and freshness-checked.
- Claims remain fixture-local and comparator-local.

## Step 8: Clean One More Public Header Family

Tasks:

- Select the highest-risk remaining public header family, likely QR/SVD or
  LDLT.
- Normalize lifecycle, ownership, error-code, options/result, tolerance, and
  workflow comments.
- Preserve declarations and ABI-relevant public structure layout.
- Add or extend a lightweight docs/declaration guard.
- Update examples, API reference, tutorial, solver-selection, or cookbook
  references as needed.

Completion criteria:

- The selected header family has a clearer public contract.
- Declaration drift is guarded or recorded.
- Full C quality gates pass if headers or C files changed.

## Step 9: Reduce One Large Test or Solver Review Surface

Tasks:

- Choose one large test/source cluster with recurring review cost.
- Extract helpers, fixtures, or proof-owner files without changing behavior.
- Update Make/CMake/source-list metadata and drift checks.
- Add a short internal maintenance note for the selected cluster.
- Run the full quality gate for code changes.

Completion criteria:

- One large review surface is smaller or easier to navigate.
- Build/test registration remains synchronized.
- Behavior is preserved by existing and focused tests.

## Step 10: Final Claim Recalibration and Epic Closeout

Tasks:

- Re-run required quality gates and focused validation commands.
- Reconcile README, INSTALL, maintainer guide, report/index docs, generated
  API docs, package docs, project plan, and sprint retrospectives.
- Produce the Epic 16 retrospective with earned claims, non-claims, validation
  evidence, and residual queue.
- Keep state-of-the-art language tied only to evidence that exists.

Completion criteria:

- Epic 16 deliverables have evidence or explicit deferral records.
- Unsupported claims remain explicit non-claims.
- The next epic residual queue is prioritized and bounded.

