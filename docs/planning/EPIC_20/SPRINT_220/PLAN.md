# Sprint 220 Plan: Structured Local-Only Workflow Guard

**Sprint Duration:** 14 days
**Goal:** Replace fragile workflow YAML normalization with a structured guard
while preserving the decision that generated API HTML stays local-only.

**Time budget:** 10 engineering hours per day, 140 hours total. No day exceeds
12 hours; the Epic 20 sprint capacity is 168 hours. CI elapsed time is not
included in the estimates.

**Prerequisites:** Sprint 213's generated-API local-only decision and workflow
fixture corpus, the current `api-docs-local-only` Make target, and the quality
and installed-contract baseline from Sprints 217-219. Keep `docs/api/html/`
ignored, untracked, and unpublished.

**Primary scope:** `scripts/check_api_docs_local_only.sh`,
`scripts/api_docs_workflow_yaml_common.py`,
`tests/test_api_docs_local_only_guard.py`, `.github/workflows/*.yml` and
`*.yaml`, the `api-docs-local-only` validation route, and maintainer guidance.
Separate workflow structure extraction from shell-aware publication policy;
retain step boundaries and effective shell identity across both stages.

**Evidence rule:** Freeze expected classifications before replacing parsing.
Record each fixture's YAML form, effective shell, publication surface, expected
allow/reject result, and reason. A parser error, unsupported construct, or
ambiguous publication path must fail closed with an actionable diagnostic.
Record old/new differences explicitly rather than treating either result as
automatically correct. Required failed, skipped, or unavailable checks block
completion.

**Non-goals:** Publishing generated HTML, changing the product decision,
approving arbitrary workflow expressions or actions, or claiming complete
shell-language analysis. A smaller parser is useful only if the policy and
known regression coverage remain intact.

| Epic item | Project estimate | Planned days and hours |
| --- | ---: | --- |
| 220.1 Fixture Freeze | 22 h | Days 1-3: 10 + 10 + 2 |
| 220.2 Structured Parser | 34 h | Days 3-6: 8 + 10 + 10 + 6 |
| 220.3 Policy Engine | 28 h | Days 6-9: 4 + 10 + 10 + 4 |
| 220.4 Differential Regression | 20 h | Days 9-11: 6 + 10 + 4 |
| 220.5 Maintainer Guide | 18 h | Days 11-13: 6 + 10 + 2 |
| 220.6 Gate And Rollout | 18 h | Days 13-14: 8 + 10 |
| **Total** | **140 h** | **14 days at 10 h each** |

---

## Day 1: Guard And Workflow Inventory

**Title:** Guard And Workflow Inventory
**Theme:** Establish what the current local-only route actually protects.
**Time estimate:** 10 hours (220.1: 10 h)

### Tasks

1. Trace `make api-docs-local-only` through the shell guard, shared YAML
   helpers, regression suite, and `api-docs-routing` dependency.
2. Inventory every tracked `.github/workflows/*.yml` and `*.yaml` input,
   including `run`, `uses`, `with`, job/step shell defaults, and publication
   paths or staging commands.
3. Separate non-workflow checks (ignored, untracked, unstaged output and
   Doxyfile/product wording) from YAML and shell-policy checks; assign owners
   so parser replacement cannot discard them.
4. Start `WORKING_NOTES.md` with the source revision, invocation commands,
   current dependency assumptions, owner map, and risk ledger.

### Deliverables

- Inventory of real workflow inputs and guard responsibilities.
- Baseline commands and risks recorded in working notes.

### Completion Criteria

- Every current guard responsibility has a named source and test owner.
- Real `.yml` and `.yaml` workflows are included in the input inventory.

---

## Day 2: Classification Corpus Freeze

**Title:** Classification Corpus Freeze
**Theme:** Turn existing regressions into an explicit acceptance oracle.
**Time estimate:** 10 hours (220.1: 10 h)

### Tasks

1. Classify existing accepted and rejected fixtures by anchors/aliases,
   flow collections, quoting/comments, block/folded scalars, and shell or
   PowerShell command forms.
2. Record why each case passes or fails, including direct generated paths,
   broad `docs/` paths, dynamic paths, staging/archive commands, and
   upload/deploy/release consumers.
3. Identify fixtures that distinguish YAML comments from shell comments,
   referenced anchors from inert text, and commands within one step from
   commands in unrelated steps.
4. Run the current guard and regression suite to capture a reproducible
   baseline; mark any flaky or environment-sensitive fixture explicitly.

### Deliverables

- Versioned classification matrix with expected result and rationale.
- Baseline test log tied to a source revision.

### Completion Criteria

- Every existing fixture has an expected classification independent of the
  new implementation.
- Accepted cases are documented as carefully as rejected bypass cases.

---

## Day 3: Freeze Edges And Choose Parser Contract

**Title:** Freeze Edges And Choose Parser Contract
**Theme:** Finish the oracle and define the structured extraction boundary.
**Time estimate:** 10 hours (220.1: 2 h; 220.2: 8 h)

### Tasks

1. Add missing baseline fixtures for duplicate keys, malformed YAML, unknown
   aliases, merge keys, unsupported tags, and dynamic publication fields;
   freeze the expected fail-closed outcomes.
2. Evaluate a maintained YAML parser against a narrowly defined fail-closed
   grammar using current repository/CI dependency constraints, GitHub's
   workflow syntax, and the frozen corpus.
3. Specify a structured intermediate record containing workflow, job, step,
   source location, effective shell, action identity, `with` paths, and raw
   `run` scalar text; avoid concatenating unrelated steps.
4. Define how duplicate/ambiguous keys and unsupported syntax produce errors
   rather than dropping a publication-relevant field.

### Deliverables

- Frozen fixture matrix, including ambiguity and error cases.
- Parser choice and extraction contract with dependency rationale.

### Completion Criteria

- Item 220.1's 22-hour freeze is closed.
- The chosen strategy represents existing real workflows and fails closed on
  unsupported publication-relevant syntax.

---

## Day 4: Structured Workflow Reader

**Title:** Structured Workflow Reader
**Theme:** Parse workflows into bounded jobs and steps.
**Time estimate:** 10 hours (220.2: 10 h)

### Tasks

1. Implement the selected parser/dependency setup and a single entry point
   for `.yml` and `.yaml` workflow files.
2. Extract job and step mappings, `uses`, `with`, `run`, job/step shell
   defaults, and source context without losing YAML scalar style semantics.
3. Reject parse errors, duplicate publication-relevant keys, unexpected
   shapes, and unsupported tags/aliases with file and step diagnostics.
4. Add focused parser tests for ordinary real workflows plus malformed and
   ambiguous inputs before moving policy rules.

### Deliverables

- Structured workflow reader with explicit error behavior.
- Initial parser-level regression tests.

### Completion Criteria

- No relevant field is silently dropped or moved across job/step boundaries.
- All inventoried real workflows parse, or an exception is documented and
  resolved before proceeding.

---

## Day 5: Scalars, Aliases, And Shell Identity

**Title:** Scalars, Aliases, And Shell Identity
**Theme:** Preserve what the workflow runner would pass to a step.
**Time estimate:** 10 hours (220.2: 10 h)

### Tasks

1. Validate literal/folded and quoted `run` scalar rendering, including
   newline/chomping behavior and comment placement that changes shell meaning.
2. Resolve supported aliases and merge behavior only through structured
   values; reject cycles or ambiguous overrides rather than guessing.
3. Carry effective shell from explicit step, job defaults, and platform
   defaults into the policy record. Keep PowerShell backslashes and backtick
   continuations distinct from POSIX shell escaping.
4. Add parser tests for flow-sequence steps, nested flow mappings, anchors,
   comments, and Windows commands drawn from the frozen corpus.

### Deliverables

- Complete step records with rendered commands and effective shell identity.
- Scalar/alias/shell extraction tests.

### Completion Criteria

- Folded comments and Windows paths have the expected rendered meaning.
- Parsing cannot hide publication fields in an alias or flow collection.

---

## Day 6: Parser Handoff And Policy Boundary

**Title:** Parser Handoff And Policy Boundary
**Theme:** Finish extraction and establish policy inputs.
**Time estimate:** 10 hours (220.2: 6 h; 220.3: 4 h)

### Tasks

1. Connect the structured reader to the guard while retaining all non-YAML
   local-only checks and existing Make entry points.
2. Define normalized policy inputs for action identity, path-bearing fields,
   rendered command text, effective shell, and same-step or explicit
   staged-output relationships.
3. Consolidate canonical path handling for case, separators, relative and
   dot segments, globs, and dynamic expressions without normalizing away
   shell-significant characters.
4. Compare parsed records from every real workflow to the frozen inventory;
   resolve missing fields or shell mismatches.

### Deliverables

- End-to-end parser-to-policy data contract.
- Real-workflow extraction comparison and discrepancy log.

### Completion Criteria

- Item 220.2's 34-hour parser allocation is closed.
- Every policy decision can identify its originating workflow, job, and step.

---

## Day 7: Direct Publication Policy

**Title:** Direct Publication Policy
**Theme:** Apply the local-only rule to structured action and path fields.
**Time estimate:** 10 hours (220.3: 10 h)

### Tasks

1. Port direct generated-HTML reference and broad `docs/` publication-path
   checks to the structured step records.
2. Classify upload-artifact, pages/deploy, release-asset, and local/unknown
   publication actions conservatively, while allowing unrelated actions.
3. Reject dynamic or unresolved publication paths and unsupported action
   forms with a specific fail-closed diagnostic.
4. Cover quoted and flow `uses`/`with` values, aliases, mixed-case Windows
   paths, and safe non-publication paths in focused tests.

### Deliverables

- Structured direct-publication policy.
- Positive and negative action/path regressions.

### Completion Criteria

- Known publication bypasses reject and known benign actions remain allowed.
- A dynamic publication path cannot be treated as a safe literal path.

---

## Day 8: Shell Staging And Upload Policy

**Title:** Shell Staging And Upload Policy
**Theme:** Detect generated docs moved into publishable outputs.
**Time estimate:** 10 hours (220.3: 10 h)

### Tasks

1. Port shell-aware `cp`, `mv`, `tar`, archive, and equivalent staging checks
   using rendered `run` text and the effective shell.
2. Relate staged destinations to upload/deploy/release consumers without
   flattening separate steps into a fictitious single command.
3. Preserve POSIX quoting, comments, continuations, and argument boundaries;
   preserve PowerShell backslashes, backticks, and `Copy-Item` paths.
4. Add paired bypass/false-positive fixtures for folded comments, continued
   Windows paths, unrelated uploads, and inert path text.

### Deliverables

- Shell-aware staging and publication detection over structured steps.
- Paired POSIX/PowerShell and cross-step regression fixtures.

### Completion Criteria

- Known staging routes cannot publish `docs/api/html/` or a containing tree.
- An unrelated upload or a shell comment alone does not trigger rejection.

---

## Day 9: Policy Edge Cases And Differential Runner

**Title:** Policy Edge Cases And Differential Runner
**Theme:** Finish policy behavior and make old/new comparisons repeatable.
**Time estimate:** 10 hours (220.3: 4 h; 220.4: 6 h)

### Tasks

1. Complete canonical-path and command edge tests for Windows separators,
   globbed docs roots, relative paths, expression fragments, and unknown
   publication actions.
2. Build a differential runner that executes old and new guards against the
   same frozen fixture worktrees without mutating the checked-out project.
3. Capture verdict, diagnostic, workflow location, and expected result for
   each fixture in a machine-readable comparison report.
4. Triage initial disagreements as intended safety tightening, false
   positive, missed bypass, or parser error; do not suppress mismatches.

### Deliverables

- Completed structured publication policy.
- Reproducible differential report harness and first discrepancy ledger.

### Completion Criteria

- Item 220.3's 28-hour policy allocation is closed.
- Old/new verdicts are comparable for every frozen fixture.

---

## Day 10: Differential Corpus Reconciliation

**Title:** Differential Corpus Reconciliation
**Theme:** Resolve behavior changes against the frozen oracle.
**Time estimate:** 10 hours (220.4: 10 h)

### Tasks

1. Run the differential corpus across anchors, flow forms, quotes, folded
   blocks, shell comments, PowerShell, direct paths, and staged artifacts.
2. Investigate every mismatch against YAML and shell semantics; correct the
   implementation or update an expectation only with written justification.
3. Run the current guard against all real workflows and inspect any newly
   rejected step before treating it as safe.
4. Preserve coverage for ignored/untracked/unstaged HTML and Doxyfile/product
   wording while parser-related code changes.

### Deliverables

- Resolved old/new classification report with reasons for differences.
- Real-workflow validation record.

### Completion Criteria

- All frozen accepted/rejected fixtures match the independently recorded
  expected result; any intentional old/new change has a reviewed rationale.
- No real workflow is silently skipped by parser or policy.

---

## Day 11: Mutation Gate And Guide Outline

**Title:** Mutation Gate And Guide Outline
**Theme:** Prove fail-closed behavior and prepare the maintainer contract.
**Time estimate:** 10 hours (220.4: 4 h; 220.5: 6 h)

### Tasks

1. Add mutation cases for an added artifact upload, aliased/flow `with.path`,
   moved generated docs, malformed YAML, duplicate keys, and dynamic paths.
2. Include benign mutations for comments, quoted anchor text, unrelated
   actions, and separate steps so safety tightening does not overreject.
3. Finish the differential report and identify the old normalization code
   that becomes removable only after parity is demonstrated.
4. Outline parser dependency, supported syntax, unsupported constructs,
   troubleshooting commands, and escalation to an explicit publication
   decision rather than local guard bypasses.

### Deliverables

- Mutation regression results and closed discrepancy ledger.
- Maintainer guide outline with support boundaries.

### Completion Criteria

- Item 220.4's 20-hour differential allocation is closed.
- Every new mutation is asserted as allow, reject, or fail-closed error.

---

## Day 12: Maintainer Guide And Diagnostics

**Title:** Maintainer Guide And Diagnostics
**Theme:** Make the structured guard operable without weakening it.
**Time estimate:** 10 hours (220.5: 10 h)

### Tasks

1. Document the selected YAML dependency, its pinned or checked version,
   supported workflow forms, and local/CI installation path.
2. Explain effective shell selection, path normalization, known command
   coverage, and the boundary where ambiguity fails closed.
3. Document how to reproduce a failure with `make api-docs-local-only` and
   the focused Python regression suite, including workflow/step diagnostics.
4. Review README/API-reference wording to ensure it still says generated
   HTML is ignored local output, not hosted, committed, or artifact-published.

### Deliverables

- Maintainer-facing parser, policy, and repair guidance.
- Reviewed product wording and actionable failure diagnostics.

### Completion Criteria

- A maintainer can reproduce and diagnose a rejected workflow locally.
- Documentation does not imply that passing the guard authorizes HTML
  publication.

---

## Day 13: Escalation And Rollout Preparation

**Title:** Escalation And Rollout Preparation
**Theme:** Finalize guidance and retire parsing only after parity.
**Time estimate:** 10 hours (220.5: 2 h; 220.6: 8 h)

### Tasks

1. Finish the guide's fail-closed escalation path for unsupported YAML,
   unknown actions, dynamic paths, and a future explicit publication decision.
2. Remove superseded YAML-normalization branches and helpers only where the
   structured reader and frozen tests prove equivalent or safer behavior.
3. Confirm `api-docs-local-only` and `api-docs-routing` still invoke the
   intended guard/tests in both local and hosted routes.
4. Run focused guard, routing, workflow, and documentation checks; record
   exact commands and any environment prerequisites in working notes.

### Deliverables

- Trimmed guard with preserved local-only Make/CI wiring.
- Focused rollout gate and maintenance documentation.

### Completion Criteria

- Item 220.5's 18-hour guide allocation is closed.
- No old parser code is removed without a passing replacement fixture or a
  documented, reviewed fail-closed tightening.

---

## Day 14: Integrated Gate And Sprint Closeout

**Title:** Integrated Gate And Sprint Closeout
**Theme:** Validate the final guard and its unchanged publication decision.
**Time estimate:** 10 hours (220.6: 10 h)

### Tasks

1. Run the full frozen fixture and mutation suites, current-workflow guard,
   `make api-docs-local-only`, `make api-docs-routing`, and applicable docs,
   format, lint, and test gates for files actually changed.
2. Check that `docs/api/html/` remains ignored, untracked, unstaged, and
   absent from workflow upload/deploy/release paths.
3. Review the final old/new differential report, parser dependency and CI
   wiring, diagnostics, and unsupported-syntax behavior; rerun after fixes.
4. Record pass/fail evidence, residual limitations, and follow-up owners in
   `WORKING_NOTES.md`; prepare a closeout artifact without claiming hosted
   evidence before hosted checks run.

### Deliverables

- Final structured guard, regression corpus, maintainer guide, and gate log.
- Sprint closeout with explicit residual-risk and hosted-evidence status.

### Completion Criteria

- All known allowed/forbidden fixtures and current workflows classify
  correctly; ambiguous publication syntax fails closed.
- Required local gates pass and hosted gates are reported accurately; no
  generated API HTML publication is introduced.
- Item 220.6's 18-hour rollout allocation and the 140-hour sprint plan are
  closed.
