# Sprint 167 Day 12: Acceptance Criteria And Stop Conditions

## Purpose

Day 12 converts the selected Epic 15 gaps into objective completion criteria,
validation commands, hosted-evidence requirements, and stop conditions. These
claim gates are the handoff contract for Sprints 168 through 176.

The gates are intentionally scoped. Passing a selected gate proves only the
named implementation, documentation, report, platform, or product-decision
surface. It does not promote retained non-claims into supported claims.

## Acceptance Criteria Table

| Gap ID | Future owner | Acceptance criteria |
| --- | --- | --- |
| G167-01 | Sprint 168 / Sprint 169 | One performance family, matrix scope, platform, compiler/toolchain, command, runtime budget, and report path are selected; the report records methodology metadata; freshness checking is wired; README/report docs describe only the selected scope; wording explicitly rejects portable superiority. |
| G167-02 | Sprint 170 | A shared-library ABI decision record exists; build/package behavior enforces the decision; docs and package metadata agree; unsupported shared-library, dynamic ABI, runtime-loader, SONAME/install-name, and DLL/import-library claims are guarded or documented as unsupported. |
| G167-03 | Sprint 171 | One package-manager path is either proven with a concrete provider artifact and validation path or formally deferred; docs distinguish source install, CMake package, `pkg-config`, and package-manager support; provider wording does not imply ecosystem-wide distribution. |
| G167-04 | Sprint 172 | One public-header family is selected and cleaned; ownership, lifecycle, error, tolerance, workspace, threading, and declaration-order language is coherent; examples/docs remain aligned; behavior and ABI are unchanged unless explicitly planned and tested. |
| G167-05 | Sprint 173 | Generated API HTML publication status is decided; the selected hosted, committed, artifact-only, or local-only path is implemented; navigation and freshness checks reflect the decision; generated HTML is not claimed as published unless the publication path is real. |
| G167-06 | Sprint 174 | One additional external comparison family has selected fixtures, comparator, metrics, tolerances, generated report rows, freshness checks, and scoped documentation; broad external-library ecosystem parity remains unsupported. |
| G167-07 | Sprint 175 | A report/platform matrix identifies current hosted and local-only status; one report freshness lane is promoted beyond Linux or the deferral is formalized with blockers; docs distinguish selected platform freshness from broad platform/report parity. |
| G167-08 | Sprint 176 | One allocation-heavy subsystem is selected; deterministic allocation-failure or cleanup-path tests exist; cleanup invariants are documented; passing tests prove only the selected subsystem, not broad OOM safety. |
| G167-09 | Sprint 176 | README, docs indexes, evidence ledger, non-claim language, sprint retrospective, and Epic 15 retrospective align with completed evidence; unsupported broad claims remain explicit non-claims. |

## Validation Command Map

| Gap ID | Required local commands or evidence | Hosted evidence requirement |
| --- | --- | --- |
| G167-01 | Run the selected benchmark/report generation command, the selected freshness target, `make bench-fast` where applicable, and `git diff --check`. If C files change, run `make format && make lint && make test`. | Hosted workflow/job must be named for any hosted performance-publication claim. CI smoke alone is not methodology-bound publication. |
| G167-02 | Run `bash scripts/static_package_deferral_check.sh`, install/package validation affected by the decision, docs consistency checks, and full C quality gate if `.c` or `.h` files change. | Hosted package/build jobs must pass if CI, package metadata, or build-system guards change. |
| G167-03 | Run relevant install/package tests such as `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, provider proof script if added, and docs checks. | Hosted provider proof is required before claiming provider support beyond local readiness. |
| G167-04 | Run header/declaration guard if added, relevant examples, `git diff --check`, and `make format && make lint && make test` for public-header changes. | Hosted CI should pass for any merged header behavior or declaration changes. |
| G167-05 | Run `make docs` or selected docs generator, generated-doc freshness/local-only guard, docs link checks if available, and `git diff --check`. | Hosted publication must be backed by the named publication workflow or artifact policy before public docs call it hosted/published. |
| G167-06 | Run the selected comparison generator, report-index normalization/freshness target, relevant solver tests, and full C quality gate if implementation/header files change. | Hosted freshness or comparison job must pass before docs describe the comparison as hosted-maintained. |
| G167-07 | Run selected report freshness command, platform-neutral path/newline checks, and docs checks. | The promoted platform lane must pass on the named platform before claiming that platform freshness. |
| G167-08 | Run the selected failure-injection test, relevant subsystem tests, sanitizers if touched, and `make format && make lint && make test` if C/header files change. | Hosted quality jobs must pass before describing the failure-path proof as reviewed. |
| G167-09 | Run `git diff --check`, docs/link/freshness checks touched by claim updates, and full C quality gate if source/header files change during closeout. | Final PR CI should be cited only by exact workflow/job/commit scope. |

## Stop-Condition Register

| Stop ID | Applies to | Stop condition | Required response |
| --- | --- | --- | --- |
| SC-001 | All gaps | Evidence is ambiguous, generated from an untracked local artifact, or cannot be reproduced by a named command. | Stop and record the gap as local-only, advisory, or deferred until reproducible evidence exists. |
| SC-002 | All gaps | Public wording expands selected evidence into broad state-of-the-art, broad parity, broad platform support, or portable superiority. | Rewrite the claim to the selected scope or retain the explicit non-claim. |
| SC-003 | All code/header changes | `.c` or `.h` files changed and `make format && make lint && make test` has not passed. | Do not proceed to commit/PR completion until the full gate passes or the user decides how to handle failure. |
| SC-004 | Performance | Runtime is too long, variance is unbounded, metadata is missing, or the lane lacks a named platform/toolchain. | Keep performance evidence local/advisory or reduce the selected scope. |
| SC-005 | ABI/package | Docs or metadata mention shared libraries, dynamic ABI stability, runtime loaders, package managers, SONAME/install-name, DLLs, or provider distribution without a selected decision and validation. | Remove or guard the wording, or stop for product decision clarification. |
| SC-006 | Package managers | Provider proof is local-only but docs imply supported distribution. | Mark as readiness/local proof only or formal deferral. |
| SC-007 | Headers/API | Header cleanup changes declarations, structs, constants, lifecycle semantics, or ABI shape unexpectedly. | Stop and convert to explicit API-change work with tests, or preserve behavior. |
| SC-008 | Generated API | Generated HTML cannot be reproduced, published, or checked according to the selected policy. | Keep the status local-only and update docs accordingly. |
| SC-009 | Comparisons | Comparator behavior, fixture tolerances, raw vector/basis ambiguity, or report rows are unclear. | Stop and tighten fixture metrics/tolerances before claiming comparison evidence. |
| SC-010 | Platform freshness | A report passes on one platform but docs imply broad cross-platform freshness. | Scope docs to the named platform or formalize the broader deferral. |
| SC-011 | Allocation failure | Failure-injection hooks are nondeterministic or do not prove cleanup invariants. | Stop and redesign the harness or narrow the selected subsystem. |
| SC-012 | Final closeout | Retrospective or README claims exceed the final evidence ledger. | Recalibrate claims before closeout. |

## Implementation Handoff Template

Future Sprint 168-176 artifacts should include this minimum structure when
closing a selected gap:

```markdown
# Sprint <N> <Day/Artifact>: <Selected Gap>

## Selected Gap

- Gap ID:
- Ledger rows:
- Non-claims retained:
- Sprint owner:

## Scope

- In scope:
- Out of scope:
- Evidence boundary:

## Implementation Notes

- Files changed:
- Commands or hosted lanes added:
- Claim wording changed:

## Validation

- Local commands run:
- Hosted evidence expected:
- Results:

## Stop Conditions Checked

- Applicable stop IDs:
- Outcome:

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
```

## Evidence Ledger Link Updates

| Ledger row | Acceptance link |
| --- | --- |
| E15-014 | G167-01 acceptance and SC-004 govern performance report publication. |
| E15-006 / E15-007 | G167-02 acceptance and SC-005 govern shared-library and dynamic ABI decisions. |
| E15-008 | G167-03 acceptance and SC-006 govern package-manager readiness. |
| E15-010 | G167-04 acceptance and SC-007 govern public-header coherence. |
| E15-009 | G167-05 acceptance and SC-008 govern generated API HTML publication. |
| E15-012 / E15-013 / E15-018 | G167-06 acceptance and SC-009 govern bounded external comparisons. |
| E15-002 / E15-003 / E15-004 / E15-015 | G167-07 acceptance and SC-010 govern platform/report freshness claims. |
| E15-016 | G167-08 acceptance and SC-011 govern allocation-failure evidence. |
| E15-001 through E15-018 | G167-09 acceptance and SC-012 govern final claim recalibration. |

## Day 13 Handoff

Day 13 should reconcile Sprint 167 artifacts against these claim gates and
prepare the Sprint 168 handoff. The handoff should identify the preferred
performance family candidate, the exact evidence boundary, and the stop
conditions that Sprint 168 must preserve.

## Validation Notes

Day 12 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Every selected gap has objective completion criteria. | Complete | G167-01 through G167-09 have acceptance criteria and owner mappings. |
| Future sprints know which checks must pass. | Complete | Validation command map names required local commands and hosted-evidence rules for each gap. |
| Stop conditions prevent accidental claim drift. | Complete | SC-001 through SC-012 cover ambiguous evidence, over-broad claims, quality gates, package/ABI wording, generated docs, comparisons, platform freshness, allocation failure, and closeout. |
