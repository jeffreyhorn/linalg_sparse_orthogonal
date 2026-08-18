# Sprint 167 Day 2: Prior Epic Residual Audit

## Purpose

Day 2 extracts deferred work, retained non-claims, and next-epic candidates
from the Epic 13 and Epic 14 retrospectives. The goal is to separate residuals
already closed or narrowed by later work from the still-open queue that should
feed the Sprint 167 evidence ledger and Day 3 risk/value classification.

## Source Retrospectives Reviewed

| Source | Status | Day 2 use |
| --- | --- | --- |
| `docs/planning/EPIC_13/EPIC_13_RETROSPECTIVE.md` | Complete | Identifies Epic 13 earned claims, retained non-claims, highest-priority next-epic candidates, and long-horizon deferrals. |
| `docs/planning/EPIC_14/EPIC_14_RETROSPECTIVE.md` | Complete with explicit residuals | Identifies Epic 14 closures, narrowed outcomes, retained non-claims, and the final residual queue feeding Epic 15. |

## Epic 13 Priority Residual Extraction

Epic 13 handed off six highest-priority next-epic candidates:

| Epic 13 residual | Affected surface | Evidence type | Status after Epic 14 |
| --- | --- | --- | --- |
| Generated API HTML refresh | API docs, generated docs, documentation publication | local generated docs, docs checks, publication policy | Narrowed/closed as a product decision in Epic 14: generated API HTML is local-only, with source-header-first authority and recurring checks. Hosted publication remains open only if selected later. |
| Hosted promotion for selected local-only oracle/comparison rows | report freshness, hosted CI, generated evidence | hosted CI, report indexes, generated rows | Partially closed in Epic 14: selected oracle and comparison evidence gained a reviewed Linux hosted path; unselected families and broader hosted proof remain open. |
| One bounded QR comparison expansion | QR solver, external comparison, report freshness | fixture-local comparison, generated rows, tests | Closed for the selected Epic 14 target through the bounded `qr-compatible-ls` family; broad QR parity remains unsupported. |
| One bounded partial-SVD comparison publication | partial SVD, external comparison, report freshness | subspace-safe metrics, generated rows, tests | Closed for the selected Epic 14 target through `partial-svd-diag6-k2`; broad SVD parity remains unsupported. |
| Windows package parity decision | Windows package, install, pkg-config, Makefile | hosted Windows CMake proof, non-claim guards | Closed as a decision: Windows remains CMake-first/static-first; Windows Makefile and Windows `pkg-config` command parity remain explicit non-claims. |
| Next public-header cleanup batch | public API, generated docs, examples | declaration-preserving header cleanup, docs checks | Partially closed by the Sprint 164 selected batch for `sparse_matrix.h`, `sparse_iterative.h`, and `sparse_eigs.h`; other headers remain open. |

## Epic 13 Long-Horizon Deferrals

These deferrals were not fully closed by Epic 14 and should remain visible in
Epic 15 evidence work:

| Deferral | Affected surface | Evidence type | Current status |
| --- | --- | --- | --- |
| Package-manager distribution | package, install, release engineering | provider recipe, install/upgrade proof, CI | Still open. Epic 14 hardened static-first source/package metadata but did not add package-manager distribution. |
| Shared-library product support | build, install, ABI, platform loaders | CMake/Make support, exported symbols, installed shared consumers | Still open as support; Epic 14 strengthened shared-library rejection and static-first non-claims. |
| Dynamic ABI compatibility policy | public headers, binary compatibility, versioning | ABI policy, symbol/version checks, compatibility matrix | Still open. Exact package version metadata does not equal ABI stability. |
| Broad ecosystem parity | numerical comparison, package ecosystems | named external libraries, fixtures, metrics, tolerances | Still open. Epic 14 added bounded comparison families only. |
| Portable performance superiority | performance, backends, reports | hosted benchmark methodology, cross-platform evidence | Still open. Epic 14 improved local methodology metadata but did not prove hosted or portable superiority. |
| Broad state-of-the-art positioning | public claims, comparison, performance, package, API | comprehensive evidence package | Still unsupported and explicitly retained as a non-claim. |
| Typed runtime/backend API promotion | runtime/backend API, ABI scope | public API design, compatibility contract, tests | Still deferred unless tied to a bounded future API and ABI scope. |

## Epic 14 Residual Extraction

Epic 14 handed off six highest-priority next-epic candidates:

| Priority | Epic 14 residual | Affected surface | Evidence type | Day 2 status |
| ---: | --- | --- | --- | --- |
| 1 | Sprint 166 PR-hosted CI confirmation | hosted CI, merge evidence, closeout evidence | PR workflow results, hosted Linux/macOS/Windows status | Operationally post-PR because PR #184 has merged, but no Sprint 167 artifact has yet reconciled exact hosted results. Carry as evidence-confirmation work if later claims cite PR-hosted proof. |
| 2 | Hosted performance publication decision | performance reports, benchmark methodology, hosted CI | hosted report lane, artifact upload, methodology metadata | Open. Epic 14 kept benchmark/sentinel rows local-only and methodology-bound. |
| 3 | Shared-library ABI product design | build, package, ABI, loaders | ABI policy, symbol visibility, SONAME/install-name/DLL metadata, runtime-loader validation | Open. Epic 14 retained static-first support and strengthened shared-library rejection. |
| 4 | Package-manager distribution readiness | package, release, install/upgrade behavior | provider scope, package recipe, provenance, CI proof | Open. Epic 14 did not claim package-manager distribution. |
| 5 | Broader public-header cleanup batch | public API, generated docs, examples | declaration-preserving cleanup, docs checks | Open for remaining high-risk headers such as QR, SVD, ILU, IC, and LDLT. |
| 6 | Additional bounded comparison family | solver comparison, external reference, report freshness | fixtures, references, metrics, normalized rows, tests | Open. Epic 14 closed selected QR and partial-SVD additions but not further families. |

## Epic 14 Long-Horizon Deferrals

Epic 14 explicitly retained these longer-horizon deferrals:

| Deferral | Affected surface | Evidence type | Current status |
| --- | --- | --- | --- |
| Broad external-library parity | solver correctness and comparison | many named libraries, fixtures, metrics, tolerances | Still unsupported. |
| Portable performance superiority | performance and backend claims | hosted cross-platform methodology and benchmark results | Still unsupported. |
| Broad state-of-the-art positioning | public claims and product positioning | comprehensive correctness, performance, package, platform, ABI, and comparison evidence | Still unsupported. |
| Broad cross-platform generated-report parity | report freshness and CI | Linux/macOS/Windows report generation and freshness gates | Still open. |
| Windows Makefile parity | Windows package/build support | Make install/uninstall execution proof | Still unsupported. |
| Windows `pkg-config` execution parity | Windows package metadata and downstream compile | pkg-config command execution proof | Still unsupported. |
| Hosted generated API HTML publication | generated API docs and publication | hosted artifact/site and freshness proof | Still open if selected; current policy is local-only. |
| Package-manager/provider upgrade behavior | package-manager distribution | install, upgrade, version, uninstall, provenance proof | Still unsupported. |

## Duplicate And Resolved Residual Notes

| Residual theme | Duplicate sources | Resolution status |
| --- | --- | --- |
| Generated API HTML | Epic 13 API HTML refresh; Epic 14 hosted generated API publication deferral | Epic 14 closed ambiguity by choosing local-only generated HTML. Hosted publication remains a distinct optional Epic 15 gap. |
| Hosted oracle/comparison evidence | Epic 13 hosted selected rows; Epic 14 selected hosted evidence | Selected Linux hosted oracle/comparison paths were promoted in Epic 14. Broader report-family hosting and hosted performance remain open. |
| QR comparison expansion | Epic 13 QR comparison residual; Epic 14 additional bounded comparison residual | One selected QR family closed in Epic 14. The Epic 15 residual is additional family coverage, not the same QR target. |
| Partial-SVD comparison | Epic 13 partial-SVD comparison residual; Epic 14 comparison-family residual | One selected partial-SVD family closed in Epic 14. Broader or additional comparison coverage remains open. |
| Windows package parity | Epic 13 Windows package decision; Epic 14 Windows Make/pkg-config retained non-claims | Closed as a product decision, not as full Windows Make/pkg-config support. Remaining parity work is intentionally unsupported unless reselected. |
| Header cleanup | Epic 13 next header cleanup; Epic 14 broader public-header cleanup | Sprint 164 closed one selected header batch. Remaining headers are a new bounded cleanup queue. |
| Static/shared package boundary | Epic 13 shared-library support; Epic 14 shared-library ABI product design | Static-first rejection was hardened; true shared-library support and ABI design remain open product work. |
| Performance publication | Epic 13 portable performance deferral; Epic 14 hosted performance publication decision | Still open. Epic 14 improved methodology metadata but did not host performance publication. |

## Source Map For Day 3 Classification

| Residual ID | Source | Candidate gap | Affected surface | Evidence needed |
| --- | --- | --- | --- | --- |
| R167-01 | Epic 14 residual priority 1 | PR-hosted CI confirmation | CI, closeout evidence | Exact hosted Linux/macOS/Windows workflow status if future claims cite PR-specific hosted proof. |
| R167-02 | Epic 14 residual priority 2 | Hosted performance publication decision | performance, reports, CI | Hosted benchmark/report lane or explicit retained local-only non-claim. |
| R167-03 | Epic 14 residual priority 3 | Shared-library ABI product design | build, package, ABI, loaders | Product decision plus ABI policy or strengthened static-first deferral. |
| R167-04 | Epic 14 residual priority 4 | Package-manager distribution readiness | package, release, install/upgrade | Selected provider proof or explicit package-manager deferral artifact. |
| R167-05 | Epic 14 residual priority 5 | Broader public-header cleanup batch | public API, generated docs, examples | Selected header family, declaration-preserving cleanup, docs checks. |
| R167-06 | Epic 14 residual priority 6 | Additional bounded comparison family | solver comparison, reports, tests | Selected fixtures, external reference, metrics, normalized rows, freshness checks. |
| R167-07 | Epic 14 long-horizon deferral | Broad generated-report platform parity | report freshness, CI, platform | Selected report-family platform promotion or explicit deferral. |
| R167-08 | Epic 14 long-horizon deferral | Hosted generated API HTML publication | generated API docs, publication | Hosted artifact/site decision, freshness proof, or retained local-only policy. |
| R167-09 | Epic 13 and 14 deferral | Allocation/failure-path evidence | solver setup, memory management, tests | Deterministic allocation-failure tests and cleanup invariants for one subsystem. |
| R167-10 | Epic 13 and 14 deferral | Broad state-of-the-art/external parity | public claims, comparisons, performance | Comprehensive competitive evidence; currently retained as unsupported. |

## Day 3 Handoff

Day 3 should rank the residual IDs above by:

- claim risk;
- user value;
- closure feasibility within Epic 15;
- dependency on hosted CI, package decisions, generated artifacts, or source
  changes;
- whether the residual can be completely closed or should remain an explicit
  non-claim.

Day 3 should treat R167-02 through R167-06 as the primary closeable candidates
and keep R167-10 as a high-risk non-claim unless evidence scope changes
substantially.

## Validation Notes

Day 2 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Prior-epic residuals are listed with source references. | Complete | Epic 13 and Epic 14 residual tables list source retrospectives and affected surfaces. |
| Resolved and still-open residuals are separated. | Complete | Epic 13 priority residuals are classified as closed, narrowed, partially closed, or still open; Epic 14 residuals are carried into the Sprint 167 source map. |
| Residuals are ready for risk and value classification. | Complete | Residual IDs R167-01 through R167-10 provide the Day 3 classification input. |
