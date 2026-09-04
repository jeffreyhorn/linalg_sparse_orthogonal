# Sprint 197 Day 10 Focused Validation Log

## Purpose

Day 10 executes the focused validation gates planned on Day 9 for
final-validation item 206.4. The current branch changes are planning Markdown
and one Epic 18 project-plan interim status snapshot, with no production code,
public header, workflow, manifest, public documentation, maintainer/API,
benchmark, corpus/schema, or generated-report source changes.

## Executed Focused Gates

| Command | Surface owner | Result | Evidence summary | Follow-up |
| --- | --- | --- | --- | --- |
| `git diff --check` | Patch hygiene | Pass | No whitespace, conflict-marker, or patch hygiene errors. | Re-run after later edits. |
| `make api-docs-freshness` | Docs/API generation and generated API local-only policy | Pass | Doxygen generation succeeded; `api-docs-coverage` found 18 checked-in public headers, 18 generated reference pages, and 18 generated source pages; local-only guard confirmed generated API HTML is ignored, untracked, unstaged, and not referenced by workflow publication paths. | Re-run after later docs/API edits. |
| `make windows-powershell-guard` | Windows PowerShell ownership and selected Windows report claim boundaries | Pass | Hosted wiring, required Windows workflow steps, selected Cholesky guarded path, selected manifest references, Windows deferral record, and Windows/PowerShell claim boundaries passed. Local `pwsh` is unavailable and is reported as an environment residual; the target's intentional negative `--require-pwsh` test fails internally as expected and the overall guard exits successfully. | Re-run if Windows workflow, manifest, selected report wording, PowerShell snippets, or claim-boundary docs change. |
| `bash scripts/package_manager_deferral_check.sh` | Package-manager and Homebrew non-claim boundary | Pass | Deferral record, provider recipe absence, selected Homebrew local proof boundary, package metadata neutrality, and public package-manager non-claims passed. | Re-run if package support, Homebrew proof, README, INSTALL, or maintainer wording changes. |
| `bash scripts/static_package_deferral_check.sh` | Static/shared package and dynamic ABI non-claim boundary | Pass | Sprint 170 decision record, `BUILD_SHARED_LIBS` rejection, static target declaration, static archive/install metadata, absent shared export/ABI metadata, package metadata neutrality, support wording, Windows package non-claims, and Windows workflow package execution boundaries passed. | Re-run if package/install/shared-library/ABI wording or build metadata changes. |
| `make source-list-check` | Library source registration | Pass | Source-list guard passed with 49 library sources. | Re-run if source registration or library source list changes. |

## Skipped Gates

| Command | Reason skipped | Required later when |
| --- | --- | --- |
| `make report-index-comparison-freshness` | Current branch did not edit comparison generator, selected target manifest, normalized report code, report docs, or generated comparison artifacts; the command regenerates local comparison outputs. | Selected comparison manifests, generators, normalizer behavior, workflow artifacts, or selected comparison docs change, or final closeout explicitly needs fresh local generated comparison evidence. |
| `make report-index-oracle-freshness` | Current branch did not edit oracle generator, selected oracle manifest semantics, report docs, or generated oracle artifacts; the command regenerates local oracle outputs. | Oracle manifests, generator behavior, report-index semantics, or final closeout explicitly needs fresh local oracle evidence. |
| `make bench-canonical-report-freshness` | Current branch did not edit benchmark code, benchmark manifest rows, methodology metadata, benchmark docs, or freshness checker behavior; the command regenerates benchmark artifacts. | Benchmark platform/row evidence, benchmark docs, manifest metadata, or freshness checker behavior changes, or final closeout explicitly needs fresh local benchmark evidence. |
| `make bench-canonical-report-freshness-tests` | Current branch did not edit benchmark freshness checker code or benchmark manifest/test semantics. | Benchmark freshness logic or selected benchmark manifest metadata changes. |
| `make ldlt-csc-helper-guard` | Current branch did not edit LDLT CSC helper/test surfaces or maintainer guard wording. | LDLT CSC helper boundaries, tests, or maintainer docs change. |
| `make qr-external-ref-helper-guard` | Current branch did not edit QR external-reference helper/test surfaces or maintainer guard wording. | QR external-reference helper boundaries, tests, or maintainer docs change. |
| `make qr-header-docs-guard` | Current branch did not edit QR public headers or QR header docs. | QR public header comments or QR docs ownership changes. |
| `make format && make lint && make test` | No `*.c` or `*.h` files changed. | Any C source or public/internal header changes. |

## Environment Residuals

| Environment evidence | Day 10 result |
| --- | --- |
| Local PowerShell availability | `make windows-powershell-guard` reported `UNAVAILABLE: pwsh not found; structural checks passed`; this remains an environment residual, not local pass evidence. |
| Hosted Windows MSVC/CMake evidence | Not locally reproduced by Day 10; hosted CI remains the owner for Windows platform execution evidence. |
| Homebrew formula proof | Not run because there is no new Homebrew license metadata decision or package-support promotion on this branch. |
| Hosted benchmark platform freshness | Not run locally; benchmark platform promotion requires hosted artifact evidence and methodology metadata. |
| Generated report freshness | Local selected report freshness commands were skipped to avoid generating unrelated local artifacts for unchanged report surfaces. |

## Fixes

No fixes were required. All executed focused gates passed.

## Item 206.4 Evidence

Day 10 provides focused integrated-validation evidence for the current
planning-only branch state. It does not complete final item 206.4 because Day
11 still owns the full quality-gate decision and clean tracked-worktree
verification.
