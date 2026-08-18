# Sprint 167 Day 10: Evidence Ledger Review And Corrections

## Purpose

Day 10 reviews the Day 9 Epic 15 evidence ledger against the current public
documentation, install/package docs, report-index conventions, CI ownership,
and explicit non-claims. The review keeps claim language conservative: selected
fixtures, hosted lanes, local report generators, and static-first packaging
must not be promoted into broader product claims without a named evidence
owner.

## Sources Reviewed

| Source area | Reviewed files or owners | Review result |
| --- | --- | --- |
| Public claims | `README.md`, `INSTALL.md`, `docs/api_reference.md`, `docs/tutorial.md`, `docs/solver_selection.md`, `docs/maintainer_guide.md` | Public language supports scoped build, install, examples, solver, and documentation claims, but not broad state-of-the-art, external parity, package-manager, or dynamic ABI claims. |
| Package metadata | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, install validation scripts | Static-first source install is supported for maintained Make/CMake paths. It does not imply shared-library support, package-manager distribution, runtime-loader behavior, or dynamic ABI stability. |
| CI evidence | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Hosted proof is platform/job scoped. Linux, macOS, and Windows evidence should name exact reviewed lanes rather than broad platform parity. |
| Report indexes | `tests/corpus/manifests/report_families.tsv`, report freshness scripts, benchmark/report scripts | Selected oracle/comparison freshness has hosted coverage where named. Broader generated report families remain selected-only, local-only, advisory, or deferred. |
| Prior residuals | Epic 13 and Epic 14 retrospectives, Sprint 167 Day 2 and Day 3 artifacts | Deferred high-risk claims remain open unless Epic 15 selects a bounded closure sprint. |
| Day 9 ledger | `artifacts/day9-evidence-ledger-draft.md` | Rows are mostly conservative, but several need sharper wording to avoid treating selected proof as broad product support. |

## Corrections Applied To The Ledger Position

| Ledger ID | Day 9 wording risk | Day 10 corrected position | Owner or disposition |
| --- | --- | --- | --- |
| E15-002 | "Linux reviewed baseline" could sound like broad Linux parity. | Keep `Partially supported / hosted-only`; supported only for named Linux workflow jobs and selected freshness lanes. | Retain as scoped platform evidence; Sprint 167 Day 11 should not select broad Linux parity as closeable. |
| E15-003 | macOS package/build wording could be read as full macOS packaging parity. | Keep `Partially supported / hosted-only`; Apple Clang and Homebrew GCC are separate scoped lanes, not package-manager or ABI proof. | Retain scoped macOS support; package-manager remains Sprint 171. |
| E15-004 | Windows CMake tier could be read as general Windows parity. | Keep `Partially supported / hosted-only`; Windows support is CMake-first and excludes Makefile, Windows `pkg-config` execution parity, DLL/shared support, and package-manager proof. | Retain scoped Windows support; package parity remains Sprint 171 or retained deferral. |
| E15-005 | Static-first install support could drift into distribution wording. | Keep `Supported for scoped paths`; source install/export proof only. | Static package boundary feeds Sprint 170 and Sprint 171. |
| E15-009 | "Local-only / deferred publication" could imply generated API HTML already has a publication decision. | Treat current support as `Local-only`; publication is a future decision, not present evidence. | Sprint 173 decides hosted, committed, artifact-only, or retained local-only status. |
| E15-012 | Maintained corpus/oracle wording could imply broad corpus completeness. | Keep `Partially supported`; hosted evidence covers selected rows only. | Sprint 174/175 may expand selected families; broad completeness remains a non-claim. |
| E15-013 | External comparison wording could imply library parity. | Keep `Partially supported`; selected fixture comparisons only. | Sprint 174 may add one bounded family without broad parity. |
| E15-014 | Benchmark smoke and local reports could imply published performance evidence. | Keep `Local-only / partially supported`; CI `bench-fast` is smoke evidence, not methodology-bound publication or superiority. | Sprints 168 and 169 own hosted performance publication decisions. |
| E15-015 | "Broad generated-report freshness" as partially supported could overstate current coverage. | Correct to `Unsupported broadly / supported only for selected rows`; all-family report freshness is not proven. | Sprint 175 selects one promotion or formal deferral. |
| E15-016 | Functional tests could be misread as deterministic allocation-failure proof. | Correct to `Deferred`; functional coverage is not failure-injection evidence. | Sprint 176 owns one selected deterministic failure-path proof. |
| E15-017 | No issue if left explicit. | Keep `Unsupported`; no unqualified state-of-the-art sparse linear algebra claim. | Retained final claim recalibration non-claim. |
| E15-018 | No issue if left explicit. | Keep `Unsupported except selected fixtures`; selected dense-helper rows do not prove ecosystem parity. | Sprint 174 may add bounded comparison evidence only. |

## Reviewed Evidence Ledger

| Ledger ID | Claim area | Reviewed status | Evidence boundary | Future owner / label |
| --- | --- | --- | --- | --- |
| E15-001 | Local build and full C test quality | Supported | Local quality commands support source/header changes when run and recorded. | Standing quality gate. |
| E15-002 | Linux reviewed baseline | Partially supported / hosted-only | Named Linux CI jobs only. | Retained scoped support. |
| E15-003 | macOS reviewed package/build tier | Partially supported / hosted-only | Named macOS CI jobs only. | Retained scoped support. |
| E15-004 | Windows reviewed CMake tier | Partially supported / hosted-only | Windows CMake and install/downstream lanes only. | Retained scoped support. |
| E15-005 | Static-first source package install | Supported for scoped paths | Source install, CMake package, and `pkg-config` metadata where validated. | Sprint 170/171 input. |
| E15-006 | Shared-library support | Unsupported / deferred | Static-only guard remains authoritative. | Sprint 170 decision. |
| E15-007 | Dynamic ABI compatibility | Unsupported / deferred | Exact version metadata is not an ABI promise. | Sprint 170 decision. |
| E15-008 | Package-manager distribution | Unsupported / deferred | No provider packaging or distribution proof exists. | Sprint 171 decision. |
| E15-009 | Generated API HTML | Local-only | Source headers and markdown docs remain authoritative; generated HTML is not published. | Sprint 173 decision. |
| E15-010 | Public API/header coherence | Partially supported | Selected header cleanup exists; broader coherence remains incomplete. | Sprint 172 selected batch. |
| E15-011 | Solver correctness | Partially supported | Test/corpus evidence is family and fixture scoped. | Future solver-family owners as selected. |
| E15-012 | Maintained corpus/oracle evidence | Partially supported | Selected QR and partial-SVD rows plus selected hosted freshness only. | Sprint 174/175 selected expansion. |
| E15-013 | Selected external comparison evidence | Partially supported | Selected QR and partial-SVD comparison rows only. | Sprint 174 bounded family. |
| E15-014 | Benchmark/performance reports | Local-only / partially supported | Local methodology rows and smoke CI only. | Sprints 168/169. |
| E15-015 | Generated-report freshness breadth | Unsupported broadly / selected-only supported | Selected freshness rows only; no all-family freshness claim. | Sprint 175. |
| E15-016 | Allocation/failure-path evidence | Deferred | No selected deterministic allocation-failure proof today. | Sprint 176. |
| E15-017 | State-of-the-art sparse linear algebra positioning | Unsupported | Evidence supports scoped maturity, not unqualified state-of-the-art status. | Retained non-claim / final recalibration. |
| E15-018 | External-library ecosystem parity | Unsupported except selected fixtures | Selected fixture comparisons only. | Retained non-claim, with Sprint 174 bounded expansion possible. |

## Explicit Non-Claim Rows

| Non-claim ID | Non-claim | Current status | Owner or retained deferral |
| --- | --- | --- | --- |
| NC-001 | Unqualified state-of-the-art sparse linear algebra status | Unsupported | Retain as final claim recalibration input. |
| NC-002 | Broad external-library parity with SuiteSparse, Eigen, SciPy, PETSc, Trilinos, LAPACK, or vendor sparse libraries | Unsupported | Retain; Sprint 174 may add only one bounded comparison family. |
| NC-003 | Portable performance superiority across platforms, compilers, matrix families, or external libraries | Unsupported | Sprints 168/169 may publish methodology-bound evidence, not superiority. |
| NC-004 | Shared-library support | Unsupported / deferred | Sprint 170 product decision. |
| NC-005 | Dynamic ABI stability or binary compatibility | Unsupported / deferred | Sprint 170 product decision. |
| NC-006 | Package-manager distribution through Homebrew, vcpkg, Conan, apt, dnf, pacman, or similar providers | Unsupported / deferred | Sprint 171 decision. |
| NC-007 | Broad platform parity across Linux, macOS, and Windows | Unsupported | Retain scoped lane language only. |
| NC-008 | Windows Makefile and Windows `pkg-config` execution parity | Unsupported / deferred | Retain unless a future Windows package sprint explicitly selects it. |
| NC-009 | Hosted or source-controlled generated API HTML publication | Unsupported today | Sprint 173 decision. |
| NC-010 | Broad all-family generated-report freshness | Unsupported | Sprint 175 selected promotion or formal deferral. |
| NC-011 | Broad allocation-failure cleanup guarantee across all solvers | Unsupported | Sprint 176 selected subsystem proof only. |
| NC-012 | Solver-family correctness beyond maintained fixture/test evidence | Unsupported | Future solver-family sprints must name fixtures, oracles, and tolerances. |

## Future Owner Map

| Future sprint | Evidence gap owned | Day 10 boundary |
| --- | --- | --- |
| Sprint 168 | Hosted performance baseline decision and first evidence gate | May promote a scoped hosted benchmark path; must not claim portable superiority. |
| Sprint 169 | Performance methodology and publication hardening | May improve repeatability and methodology metadata; must keep claims method-bound. |
| Sprint 170 | Shared-library ABI product decision | Either retain static-only with stronger guards or start a staged ABI track with explicit non-promises. |
| Sprint 171 | Package-manager readiness or deferral | Source package proof cannot be described as provider distribution. |
| Sprint 172 | Public header/API coherence batch | Header cleanup can improve usability and docs, but not ABI stability by itself. |
| Sprint 173 | Generated API HTML publication decision | Local-only generated docs remain non-public evidence unless a publication route is selected. |
| Sprint 174 | Additional bounded external comparison family | Adds selected comparison evidence only; broad ecosystem parity remains unsupported. |
| Sprint 175 | Generated report freshness breadth | Selects one report/platform promotion or records a formal deferral. |
| Sprint 176 | Deterministic allocation-failure proof | Selects one allocation-heavy subsystem; broad OOM guarantee remains unsupported. |

## Day 11 Handoff

Day 11 should select gaps using the reviewed ledger rather than the broader
candidate list alone. High-value closeable candidates are hosted performance
publication, shared/static ABI decision, package-manager deferral or provider
proof, public-header coherence, generated API publication, one bounded
comparison expansion, selected report-freshness promotion, and one
deterministic allocation-failure proof.

Day 11 should not select broad state-of-the-art positioning, broad
external-library parity, portable performance superiority, broad platform
parity, or broad all-family report freshness as closeable Epic 15 outcomes.

## Validation Notes

Day 10 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Ledger language does not overstate evidence. | Complete | Corrected statuses keep selected, local-only, hosted-only, static-first, and deferred evidence distinct. |
| Unsupported claims are explicit non-claims. | Complete | NC-001 through NC-012 cover state-of-the-art status, broad external parity, portable performance superiority, shared libraries, dynamic ABI, package managers, broad platform parity, Windows package parity, generated API publication, report freshness, allocation failure, and solver correctness breadth. |
| Every high-risk row has an owner or retained deferral. | Complete | Future owner map assigns Sprint 168 through Sprint 176 owners or retained non-claim labels. |
