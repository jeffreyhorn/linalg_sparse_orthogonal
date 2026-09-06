# Sprint 197 Day 9 Integrated Validation Matrix

## Purpose

Day 9 prepares the final integrated validation plan for item 206.4. It converts
the changed surfaces and evidence owners from Days 1 through 8 into focused
validation commands, full-gate triggers, expected outputs, and environment
residuals before Day 10 begins execution.

## Changed Surface Inventory

| Surface | Current branch change | Validation owner |
| --- | --- | --- |
| `docs/planning/EPIC_18/PROJECT_PLAN.md` | Added Sprint 197 Day 8 interim status snapshot. | Markdown whitespace checks and docs generation checks. |
| `docs/planning/EPIC_18/SPRINT_197/PLAN.md` | Added requested day-by-day Sprint 197 plan. | Markdown whitespace checks and sprint-artifact review. |
| `docs/planning/EPIC_18/SPRINT_197/WORKING_NOTES.md` | Added Day 1-9 working ledger, evidence notes, and status updates. | Markdown whitespace checks and sprint-artifact review. |
| `docs/planning/EPIC_18/SPRINT_197/artifacts/*.md` | Added Day 1-9 evidence artifacts. | Markdown whitespace checks and sprint-artifact review. |
| Public user docs | No Day 6 edits. | No revalidation beyond docs generation and claim scan unless edited later. |
| Maintainer/API/corpus/schema docs | No Day 7 edits. | No targeted owner guard required unless edited later. |
| C source or public/internal headers | No changes through Day 9. | Full C gate not required unless future edits touch `*.c` or `*.h`. |

## Focused Validation Command Matrix

| Surface or risk | Command | Expected result | Run timing |
| --- | --- | --- | --- |
| Markdown whitespace and patch hygiene | `git diff --check` | No trailing whitespace, conflict markers, or patch whitespace errors. | Day 9 and every later edit day. |
| API docs generation and public header coverage | `make docs-check` | Doxygen generation succeeds and `api-docs-coverage` reports all checked-in public headers covered. | Day 9 and every later docs edit day. |
| Generated API local-only policy | `make api-docs-freshness` | `docs-check` passes and generated API HTML remains ignored/untracked/un-staged. | Day 10 focused validation or any API-publication wording/policy edit. |
| Project-plan status integrity | Manual review of `PROJECT_PLAN.md` plus `SPRINT_197/artifacts/day8-project-plan-status.md` | Interim status does not mark Sprints 198-205 complete without artifacts. | Day 9 planning; revisit on Day 10 if edits occur. |
| Package-manager/Homebrew non-claims | `bash scripts/package_manager_deferral_check.sh` | Package-manager support remains guarded unless exact proof evidence exists. | Day 10 if package docs or support matrix are touched; otherwise optional focused confidence. |
| Static/shared package boundary | `bash scripts/static_package_deferral_check.sh` | Shared-library and dynamic ABI claims remain deferred. | Day 10 if install/package docs are touched; otherwise optional focused confidence. |
| Windows PowerShell ownership and claim boundaries | `make windows-powershell-guard` | Workflow snippets, selected Cholesky guarded path, PowerShell claim boundaries, and doc anchors validate. | Day 10 if Windows wording/workflow/manifests are touched; otherwise optional focused confidence. |
| Selected oracle freshness | `make report-index-oracle-freshness` | Selected local oracle reports regenerate and freshness check passes. | Day 10 only if final validation needs generated oracle proof; command may mutate generated local report outputs. |
| Selected comparison freshness | `make report-index-comparison-freshness` | Selected local comparison reports regenerate and freshness check passes. | Day 10 only if final validation needs generated comparison proof; command may mutate generated local report outputs. |
| Selected benchmark freshness | `make bench-canonical-report-freshness` | Canonical selected benchmark bundle regenerates and selected freshness check passes. | Day 10 only if final validation needs benchmark proof; command may create local benchmark artifacts. |
| Benchmark freshness tests | `make bench-canonical-report-freshness-tests` | Python benchmark freshness regressions pass. | Day 10 if benchmark docs, manifest rows, or freshness checker behavior are touched. |
| Source list ownership | `make source-list-check` | Library source registration count and source list pass. | Required if C source registration changes; optional confidence otherwise. |
| LDLT CSC helper guard | `make ldlt-csc-helper-guard` | LDLT CSC helper ownership and maintainer docs pass. | Required if LDLT helper/test surfaces change. |
| QR external reference helper guard | `make qr-external-ref-helper-guard` | QR external-reference helper ownership and maintainer docs pass. | Required if QR external-reference helper/test surfaces change. |
| QR header docs guard | `make qr-header-docs-guard` | QR header documentation guard passes. | Required if QR public header/docs ownership changes. |
| Format, lint, and unit tests | `make format && make lint && make test` | Formatting applied, strict lint passes, and test suite passes. | Required before proceeding if any `*.c` or `*.h` file is modified. |
| Reviewed compile quality path | `make quality-review-compile` | `format-check`, `source-list-check`, and `lint` pass. | Optional final confidence if source/header files change or CI parity is needed. |
| Reviewed full quality path | `make quality-review` | `format-check`, `lint`, `test`, and `deadcode-check` pass. | Optional broad final confidence; expensive and not required for docs-only Day 9. |

## Full-Gate Trigger Decision

| Trigger | Required gate |
| --- | --- |
| Any `*.c` or `*.h` file changes | Run `make format && make lint && make test` before commit/PR. |
| Library source registration changes | Run `make source-list-check`; include C full gate if files are C/header changes. |
| Public header documentation changes | Run `make docs-check`; run `make api-docs-freshness`; run relevant header-specific guards such as `make qr-header-docs-guard` when applicable; run full C gate because headers changed. |
| API publication policy changes | Run `make docs-check` and `make api-docs-freshness`; add link/publication checks if the policy introduces hosted or artifact output. |
| Windows workflow, manifest, or claim wording changes | Run `make windows-powershell-guard`; hosted Windows CI remains required for actual MSVC/PowerShell evidence. |
| Selected comparison manifest, generator, or normalizer changes | Run selected comparison focused tests plus `make report-index-comparison-freshness`; hosted platform evidence is required for platform promotion. |
| Selected benchmark manifest, report, or methodology changes | Run `make bench-canonical-report-freshness` and benchmark freshness tests; hosted evidence is required for a hosted platform claim. |
| Package/install support wording changes | Run package-manager and static-package deferral guards, install docs checks where available, and docs checks. |
| Planning-only Markdown changes | Run `git diff --check` and `make docs-check`; full C gate is not required. |

## Environment Residuals

| Evidence | Local Day 9 expectation | Residual |
| --- | --- | --- |
| Hosted Windows MSVC/CMake selected comparison evidence | Not reproducible from this macOS/Linux-oriented local checkout without the hosted Windows runner. | Treat as hosted-only evidence; do not promote Windows freshness locally. |
| Hosted PowerShell `--require-pwsh` validation | Local `pwsh` may be unavailable; the hosted lane is the evidence owner for required PowerShell availability. | Local unavailable PowerShell is an environment residual, not pass evidence. |
| Homebrew formula proof | Requires approved license metadata and local Homebrew proof environment. | No package-manager support promotion without exact proof output. |
| Hosted benchmark platform freshness | Requires hosted runner artifact and methodology metadata for the selected platform/row. | Local benchmark output is not portable performance evidence. |
| Generated report freshness commands | Commands can regenerate local report artifacts. | Generated outputs must be treated as local evidence unless tracked/promoted by a reviewed hosted lane. |
| Full C quality gates | Required only when C/header files change. | Planning-only days use docs checks unless later edits touch C/header surfaces. |

## Day 10 Validation Log Template

| Command | Surface owner | Result | Evidence path or output summary | Follow-up |
| --- | --- | --- | --- | --- |
| `git diff --check` | Patch hygiene | Pending |  |  |
| `make docs-check` | Docs/API generation | Pending |  |  |
| `make api-docs-freshness` | Generated API local-only policy | Pending |  |  |
| `make windows-powershell-guard` | Windows PowerShell ownership | Pending |  |  |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager non-claims | Pending |  |  |
| `bash scripts/static_package_deferral_check.sh` | Shared-library/dynamic ABI non-claims | Pending |  |  |
| `make report-index-comparison-freshness` | Selected comparison freshness | Pending or skipped with reason |  |  |
| `make report-index-oracle-freshness` | Selected oracle freshness | Pending or skipped with reason |  |  |
| `make bench-canonical-report-freshness` | Selected benchmark freshness | Pending or skipped with reason |  |  |
| `make format && make lint && make test` | C/header full gate | Not required unless C/header files change |  |  |

## Day 9 Decision

Day 9 does not execute costly generated-report, benchmark, Windows, or full C
quality gates. It establishes the validation plan and confirms that planning
Markdown changes require `git diff --check` and `make docs-check` now. Day 10
will run the selected focused gates and record pass/fail evidence.
