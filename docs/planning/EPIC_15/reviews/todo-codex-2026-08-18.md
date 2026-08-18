# Epic 15 Gap Closure Todo - 2026-08-18

## Goal

Close the highest-value residual gaps identified after Epic 14 without spreading work too thin. Epic 15 should improve the library's evidence, adoption, packaging, and claim credibility more than it expands algorithm count.

## Step 1: Establish the Epic 15 Evidence Baseline

Tasks:

- Re-read the Epic 13 and Epic 14 retrospectives and extract all unresolved residuals.
- Confirm the current master branch after PR #184.
- Inventory current source, header, test, script, report, package, and CI surfaces.
- Create an Epic 15 evidence ledger listing every claim that is supported, partially supported, or unsupported.
- Map each candidate gap to the exact file, command, report, or CI lane that would close it.

Completion criteria:

- A single baseline artifact identifies the gaps Epic 15 will close.
- Unsupported claims are explicitly named.
- Every planned Epic 15 sprint maps to at least one closeable gap.

## Step 2: Promote Hosted Performance Evidence

Tasks:

- Select one solver family and one benchmark/report path suitable for hosted CI.
- Define benchmark methodology: compiler, flags, CPU metadata, thread settings, repeat count, warmup policy, and variance reporting.
- Add or harden the report generator so it emits claim-safe metadata.
- Add freshness checks for the generated performance report.
- Wire the selected lane into CI without claiming broad performance superiority.

Completion criteria:

- A hosted job proves one selected performance report is fresh.
- The report includes methodology and scope.
- Documentation states exactly what the performance result supports and what it does not support.

## Step 3: Close the Shared-Library ABI Product Decision

Tasks:

- Audit current static-first package guards.
- Evaluate whether the current API and build surfaces are ready for a shared-library ABI.
- Decide between continued static-first-only support and a staged shared-library track.
- If static-first is retained, strengthen docs and tests so shared-library and ABI non-claims cannot drift.
- If shared support is selected, create only a minimal gated prototype with explicit non-claims for unsupported platforms.

Completion criteria:

- The repository has one clear product decision.
- Package metadata and README claims align with that decision.
- CI guards prevent accidental shared-library claims.

## Step 4: Close One Package-Manager Readiness Path

Tasks:

- Choose one package-manager proof path or formally defer package-manager support.
- If a provider is chosen, add a source-controlled manifest/formula/recipe and a local validation script.
- Prove install, downstream compile, version query, and uninstall/cleanup behavior for the selected path.
- If deferred, document exact blockers and strengthen non-claims.

Completion criteria:

- One package-manager readiness decision is complete.
- Documentation distinguishes source install, CMake/pkg-config install, and package-manager support.
- Claims are backed by an executable proof or explicit deferral.

## Step 5: Continue Public Header Coherence Cleanup

Tasks:

- Select the next highest-impact public header family.
- Normalize declaration grouping, lifecycle wording, ownership rules, error-code semantics, and examples.
- Add a lightweight declaration or documentation coverage check where practical.
- Update generated API input comments and README links.

Completion criteria:

- The selected header family has a coherent public contract.
- At least one mechanical or reviewable guard reduces future drift.
- Examples and docs point to the cleaned-up entry points.

## Step 6: Publish Generated API HTML or Lock Local-Only Status

Tasks:

- Audit the generated API HTML path and freshness checks.
- Decide whether generated API HTML is a hosted artifact, committed artifact, or local-only developer artifact.
- Implement the selected publication path or update docs and CI to enforce local-only wording.
- Add freshness checks for whichever status is selected.

Completion criteria:

- Users can find the supported API reference path.
- Generated API claims are aligned with CI evidence.
- Local-only artifacts are not described as hosted or published.

## Step 7: Add One More Bounded External Comparison Family

Tasks:

- Select the next comparison family based on risk and usefulness.
- Define matrix fixtures, tolerance policy, expected outputs, and comparator scope.
- Add report generation and index integration.
- Add claim-safe documentation that avoids broad parity language.

Completion criteria:

- One additional comparison family is complete.
- Its report is generated, indexed, and freshness-checked.
- Claims are bounded by solver family, fixture family, and comparator.

## Step 8: Promote Cross-Platform Report Freshness Where Feasible

Tasks:

- Identify which generated report families are Linux-only, macOS-ready, or Windows-ready.
- Promote one selected report family beyond Linux if feasible.
- Where not feasible, document platform blockers and keep CI wording explicit.
- Update tier statements in README and planning docs.

Completion criteria:

- One cross-platform report freshness gap is closed or formally deferred.
- CI and documentation agree on platform scope.
- No broad platform-parity claim is introduced.

## Step 9: Add Allocation-Failure Evidence for One Subsystem

Tasks:

- Select a solver or shared allocation-heavy subsystem.
- Add failure-injection or deterministic allocation-failure tests for construction and cleanup paths.
- Document cleanup invariants for the selected subsystem.
- Keep the harness reusable but avoid expanding before the first closure is stable.

Completion criteria:

- One subsystem has tested allocation-failure behavior.
- Cleanup paths are covered and documented.
- The pattern is available for future sprints.

## Step 10: Final Claim Recalibration and Epic Closeout

Tasks:

- Re-run required local quality gates.
- Reconcile README, package docs, generated report indexes, retrospectives, and project plan status.
- Produce an Epic 15 retrospective.
- Recalibrate state-of-the-art language based only on hosted and local evidence.
- Leave an explicit residual queue for Epic 16.

Completion criteria:

- All Epic 15 deliverables have evidence.
- Unsupported claims remain explicit non-claims.
- The final documentation accurately describes the project as delivered.
