# Sprint 214 Working Notes: ABI, Shared-Library And Release Readiness Decision

## Sprint Goal And Boundary

Either prove one narrowly scoped shared-library/ABI path or close a complete
release/ABI design with acceptance gates and enforceable deferral guards. The
current product contract is static-first; this sprint does not inherit shared
build, dynamic ABI, runtime-loader, package-manager distribution, or release
readiness claims from the existence of source headers or static packages.

Day 1 is evidence intake only. Day 2 owns the proof-versus-deferral decision.
No shared target, package selector, or public support wording changes before
that decision is recorded.

## Item Checklist And Evidence Owners

| Item | Evidence owner and intended validation | Status through Day 13 |
| --- | --- | --- |
| 214.1 ABI/Release Scope Decision | Day 2 decision record; compare a named-platform shared proof with design/deferral against Sprint 170 and Sprint 207 decisions, CI availability, and stop gates. | Complete; selected ABI/release design with stronger static-first deferral guards. |
| 214.2 ABI Policy Design | Days 3-5 policy artifacts; inspect installed headers, actual exported symbols, version/layout/callback/ownership risks, and platform loader metadata. | Complete as future policy design: Day 3 provisional symbols, Day 4 ABI identity/compatibility, Day 5 Linux/macOS/Windows loader gates. No dynamic ABI is implemented or supported. |
| 214.3 Build/Install Proof Or Guard | Days 5-8 Make/CMake and guard changes; validate selected configure/build/install path plus static and unsupported-path regressions. | Implemented locally: Day 6 rejects unsupported selectors; Day 7 validates installed Make/CMake prefixes; Day 8 wires 14 mutation fixtures and the deferral/docs guard into package CI, plus the Windows installed-prefix validator. Hosted runs remain unverified. |
| 214.4 Downstream Consumer Evidence | Days 8-10 out-of-tree installed consumer and negative matrix, or exact future shared-consumer acceptance tests if deferred. | Complete for the selected static scope: Day 9 recorded installed consumer output and future shared fixtures; Day 10 proved missing Make/CMake archive and package-version diagnostics while preserving positive static consumers. Dynamic loader/ABI cases remain future-only. |
| 214.5 Release Checklist | Days 11-12 checklist covering changelog, tag, source archive reproducibility, artifacts, provenance, CI evidence, and stop conditions. | Checklist design complete: Day 11 source/tag/archive plus Day 12 provenance, platform CI, approval, and rollback gates. Candidate evidence and release approval remain absent; no release claim earned. |
| 214.6 Validation And Closeout | Days 13-14 install/export, static/shared guards, docs checks, and full C/build gates as applicable; record commands and outputs. | Complete for local selected path: Day 13 integrated gates passed and Day 14 reconciles all items, rechecks affected surfaces, and records residuals. No hosted or release proof is claimed. |

## Day 1 Evidence Ledger

| Surface | Current source of truth and finding | Sprint 214 implication |
| --- | --- | --- |
| Sprint scope | `docs/planning/EPIC_19/PROJECT_PLAN.md`, Sprint 214 section, defines items 214.1-214.6 at 24, 30, 38, 28, 24, and 24 hours (168 total). `PLAN.md` allocates 14 days at 12 hours. | Scope and item traceability. |
| Prior ABI decision | `docs/planning/EPIC_15/SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md` selects static-first-only packaging and defers shared library and dynamic ABI support. | A new decision must explicitly supersede only the proven portion of that boundary. |
| Prior ABI inventory | Sprint 170 `day2-header-abi-inventory.md` and `day4-symbol-visibility.md` identify concrete public layouts, callbacks, ownership across the API, `SPARSE_IDX_BITS`, and uncurated global symbols. | Re-audit current headers and actual artifact symbols before choosing a dynamic ABI scope. Historical counts are not a current symbol allowlist. |
| Package-provider decision | `docs/planning/EPIC_19/SPRINT_207/artifacts/day5-provider-decision.md` selects continued public-provider deferral. Local Homebrew static source formula evidence is developer-mode proof only. | Shared proof, if chosen, cannot imply package-manager distribution. |
| Residual queue | `docs/planning/EPIC_18/EPIC_18_RESIDUAL_QUEUE.md` entry `E18-RQ-009` retains release, shared-library, and dynamic ABI readiness as product/platform policy work. Sprint 207 retrospective and Sprint 213 retrospective keep related claims unearned. | Record exact residuals after any narrow proof; do not infer broad readiness. |
| Make build/install | `Makefile` defines `build/libsparse_lu_ortho.a`, installs it with checked-in `include/*.h`, generated `sparse_version.h`, and generated `sparse.pc` from `sparse.pc.in`. | Static archive and Unix Make install are maintained; no shared Make target is present in the inspected contract. |
| CMake build/install | `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`, declares `add_library(sparse_lu_ortho STATIC ...)`, exports `Sparse::sparse_lu_ortho`, installs archive/headers/config files, and uses `ExactVersion` package matching. | Exact package-version matching is not dynamic ABI compatibility. Shared proof would require deliberate target/export and metadata changes. |
| Package metadata | `sparse.pc.in` describes static archive metadata and `-lsparse_lu_ortho -lm`; `cmake/SparseConfig.cmake.in` loads installed targets. | Keep Make, CMake, and `pkg-config` descriptions aligned with actual installed artifacts. |
| Static install tests | `tests/test_install.sh` checks Make install, `pkg-config`, static archive, headers, downstream behavior, and no installed `.so`/`.dylib`/`.dll`. `tests/test_cmake_install.sh` checks CMake install/export, `find_package`, exact version behavior, static imported target, and no unsupported loader/shared metadata. | Reuse as static regression gates; a selected shared proof must add scoped tests rather than silently invalidate static tests. |
| Deferral guard | `scripts/static_package_deferral_check.sh` checks the Sprint 170 record, shared configure rejection, explicit static targets, absence of shared/ABI metadata, package non-claims, and Windows boundaries. | Must be intentionally revised only if a Day 2 decision earns a narrow exception. |
| CI platform lanes | `.github/workflows/ci.yml` runs Linux reviewed package-contract proofs; `macos-ci.yml` runs reviewed Make/`pkg-config` and CMake install/export; `windows-ci.yml` runs CMake static install/downstream validation and `sparse.pc` metadata-only inspection. | CI configuration proves test ownership, not a Day 1 hosted result or shared-loader proof. Windows Make/`pkg-config` parity is unclaimed. |
| Public claim surfaces | `INSTALL.md#support-readiness-matrix`, `README.md`, and `docs/maintainer_guide.md` retain static-first and shared/ABI/package non-claims. | Later wording must match selected evidence and must not promote release readiness by implication. |
| Generated API policy | Sprint 213 retrospective retains generated Doxygen HTML as local-only output. | Release checklist must not treat generated HTML as a published or retained artifact without a separate policy change. |

## Day 3 Symbol Evidence

| Measurement or boundary | Local result / policy |
| --- | --- |
| Current library sources and archive objects | `make source-list-check` passed with 49 sources; local archive contains 49 implementation objects. |
| Direct installed-header function declarations | 165 across 18 checked-in `include/*.h` headers under default C11/32-bit-index configuration; all 165 names are defined in the local static archive. |
| Defined archive globals | 361 unique globals, including 196 not directly declared in installed headers. Static visibility is not dynamic export policy. |
| Generated version header | Compile-time macros only; no generated linker symbols or ABI epoch. |
| Provisional future dynamic subset | Exactly eight opaque-matrix/metadata C functions, listed in `artifacts/day3-symbol-visibility-policy.md`; not approved or implemented as a shared ABI. |
| Current effective shared exports | None; shared configuration remains rejected. |

The local counts are an inventory, not a checked-in ABI baseline. Day 3's
artifact records declaration owners, the exclusion rule, hidden-by-default
design, and installed-binary export tests needed before any future proof.

## Day 4 Compatibility Evidence

| Boundary | Local result / policy |
| --- | --- |
| Source and package version | `VERSION=2.2.0`, generated version macros, and CMake `ExactVersion` identify the package, not a dynamic ABI epoch. |
| Static installed consumer | `bash tests/test_cmake_install.sh` passed locally: 27 passed, 0 failed, 0 skipped, including exact-version execution and mismatch rejection. |
| Future ABI identity | No current ABI epoch or shared artifact. A future identity must bind epoch, index/scalar configuration, approved exports, platform ABI, and loader/package metadata. |
| Compatibility evidence | A successful link or same-build consumer is insufficient. Cross-version claims require a retained old executable run unchanged against a newer installed library on each claimed platform. |
| Breakage and rollback | Symbol, signature, selected layout, calling convention, ownership, width/scalar, or loader drift requires explicit review, mismatch rejection, and an epoch change or tested shim where breaking. |

The detailed classification, positive and negative fixture specifications,
baseline retention, and release blockers are in
`artifacts/day4-abi-compatibility-policy.md`.

## Day 5 Loader And Guard Evidence

| Boundary | Local result / policy |
| --- | --- |
| Selected path | Keep static-first Make/CMake install and reject shared requests; no shared target or ABI epoch is selected for implementation. |
| Linux future gate | Inspect installed ELF SONAME, dependencies, search path, export table, and runtime resolution with a named out-of-tree consumer plus missing/wrong-library failures. |
| macOS future gate | Inspect installed dylib ID, consumer load commands, RPATH, relocation, and missing/wrong-dylib behavior on a named Apple Clang lane. |
| Windows future gate | Distinguish DLL import `.lib` from static `.lib`; inspect exports/imports, controlled DLL search, CRT policy, and missing/wrong-DLL behavior on a named MSVC lane. |
| Selected implementation owner | Days 6-8 strengthen `scripts/static_package_deferral_check.sh`, Make/CMake install tests, package metadata checks, and docs non-claims with temporary negative fixtures. |
| Current guard baseline | `bash scripts/static_package_deferral_check.sh` passed locally on Day 5; this is static deferral evidence, not loader proof. |

See `artifacts/day5-loader-and-build-design.md` for exact file owners,
per-platform inspection commands, acceptance criteria, rollback, and platform
exclusions.

## Day 6 Configure And Metadata Evidence

| Boundary | Local result / implementation |
| --- | --- |
| Configure rejection | `CMakeLists.txt` now fails on package-prefixed shared selectors, four link-mode aliases, and noncanonical `BUILD_SHARED_LIBS` casing. Unsupported `OFF`/`STATIC` aliases also fail instead of being silently ignored. |
| Generated metadata | The deferral guard configures a clean default build and checks generated `sparse.pc` and `SparseConfig.cmake` for static-only scope. |
| Negative cases | The guard tests eleven unsupported selector inputs plus the existing `BUILD_SHARED_LIBS=ON` failure, checking diagnostic text. |
| Positive static build/install | `bash tests/test_cmake_install.sh` passed 27/27; `bash tests/test_install.sh` passed 23/23, including installed static consumers. |
| Boundary | No shared artifact, ABI epoch, loader proof, or public support claim was added. All execution results are local macOS evidence. |

The implementation details, command results, remaining risks, and Day 7
handoff are in `artifacts/day6-build-path-or-deferral-guard.md`.

## Day 7 Installed-Package Evidence

| Boundary | Local result / implementation |
| --- | --- |
| Make inventory | One `libsparse_lu_ortho.a`, `sparse.pc`, and 19 headers in a temporary prefix. `pkg-config` and installed basic/example consumers passed. |
| CMake inventory | Same archive, `sparse.pc`, 19 headers, `SparseConfig*.cmake`, and `SparseTargets*.cmake`; `Sparse::sparse_lu_ortho` remains `STATIC IMPORTED` and resolves to the installed archive. |
| New validator | `scripts/check_static_install_contract.py` rejects unexpected library/runtime files, DLL/import artifacts, unsupported CMake loader/selector fields, mismatched imported locations, and `sparse.pc` drift. Both install tests call it. |
| Negative fixtures | `python3 tests/test_static_install_contract.py` passed 10 tests, including Unix Make/CMake and Windows-style positive fixtures plus artifact and metadata mutations. Windows-style fixture is not hosted MSVC proof or archive-content inspection. |
| Regression results | `bash tests/test_install.sh` passed 24/24; `bash tests/test_cmake_install.sh` passed 28/28; both executed installed static consumers locally on macOS. |

The installed artifact list, negative cases, remaining optional-configuration
and Windows gaps, and Day 8 handoff are in
`artifacts/day7-install-and-metadata-contract.md`.

## Day 8 Automation And Consumer Design Evidence

| Boundary | Local result / CI ownership |
| --- | --- |
| Linux/macOS CI | Existing package jobs now execute the installed-contract mutation fixtures and explicitly run the static deferral/docs-claim guard. Make and CMake real-prefix consumer tests remain active. |
| Windows CI | The CMake install job now runs the mutation fixtures and installed-prefix validator with the Windows static `.lib` mode, before its existing PowerShell artifact and consumer checks. No Windows Make or `pkg-config` execution was added. |
| Diagnostics | Four new fixtures cover missing archive, missing `sparse.pc`, missing CMake targets export, and wrong CMake package version. Total fixture suite: 14 passing local tests. |
| Static regressions | Local Make install passed 24/24, CMake install passed 28/28, and the deferral/docs guard passed. Workflow YAML parsed; Windows PowerShell validator unit tests passed. These are local macOS results, not hosted CI results. |
| Item 214.4 design | Separate temporary install prefix, installed package resolution, static link/no dynamic loader environment, positive consumer output, and negative artifact/metadata/configuration outcomes are specified in the Day 8 artifact. |

`python3 scripts/validate_windows_powershell.py` passed structural checks but
returned 2 because `pwsh` is not installed locally. Hosted Windows execution
and static `.lib` binary-member inspection remain unverified. See
`artifacts/day8-automation-and-consumer-design.md` for CI ownership and the
exact consumer acceptance specification.

## Day 9 Installed Consumer Evidence

| Boundary | Local result / next gate |
| --- | --- |
| Environment | Darwin 24.6.0 x86_64, Apple Clang 17.0.0, CMake 4.4.3, `pkg-config` 2.5.1; no `DYLD_LIBRARY_PATH` or `LD_LIBRARY_PATH` set. |
| Make installed route | Fresh temporary prefix, installed archive/`sparse.pc`/19 headers, package flags from that prefix, two separately compiled consumers. 24/24 checks passed; basic consumer printed `2.2.0`, integer version `20200`, `nnz: 1`, `OK`; maintained example solved `[1.000000, 1.000000, 1.000000]`. |
| CMake installed route | Separate temporary prefix and out-of-tree consumer builds using `CMAKE_PREFIX_PATH`; static imported target and exact-version consumer. 28/28 checks passed, 0 skipped; both executables printed version `2.2.0`, the same solution, and `OK`. Wrong package version was rejected. |
| Symbol and metadata evidence | Local built static archive defines the used consumer symbols; successful installed links/runs prove resolution through the static package. Installed-prefix validator passed; this is not a dynamic export or loader check. |
| Deferred shared fixtures | Future Linux ELF, macOS Mach-O, and Windows PE plans now name artifact/ABI manifest inputs, installed-only build/run, export/loader inspection, missing/wrong-name failures, and retained old-executable evidence. None were executed or claimed. |

The actual commands, output, future-fixture sequencing, and Day 10 handoff
are in `artifacts/day9-installed-consumer-evidence.md`. The install tests now
print captured successful consumer output in local and hosted logs.

## Day 10 Consumer Negative Matrix Evidence

| Case | Local result / boundary |
| --- | --- |
| Missing Make archive | After both installed consumers passed, the test renamed the installed `.a`; a consumer linked with the same `pkg-config` flags failed with a `sparse_lu_ortho` library-not-found diagnostic. Archive restored before uninstall. Make install: 25 passed, 0 failed. |
| Missing CMake archive | The installed-prefix validator reported a missing regular archive; a fresh out-of-tree `find_package(Sparse)` configure rejected the missing imported target file by name. CMake install: 30 passed, 0 failed, 0 skipped. |
| Version mismatch | CMake rejected requested `2.1.0` against installed `2.2.0` with a compatible-config/version diagnostic; exact-version consumer still built and ran. This is package selection, not ABI compatibility. |
| Other selected negatives | Fourteen metadata mutation fixtures and the deferral guard passed; `BUILD_SHARED_LIBS=ON` and eleven aliases remain unsupported. |
| CI ownership | Linux/macOS workflow registrations call the changed install scripts; Windows CMake static lane calls the installed-prefix validator and fixture tests. Workflow YAML parsed. Hosted branch results remain unverified. |
| Future-only cases | Wrong architecture/platform, absent dynamic runtime path, missing shared export, old-binary/new-library compatibility, and Windows DLL/import-library failures cannot be executed under the selected static-only path. |

The full executed-versus-future matrix, diagnostics, CI registration, and
Day 11 boundary are in `artifacts/day10-consumer-negative-matrix.md`.

## Day 11 Release Source And Tag Evidence

| Input or decision | Current evidence and gate |
| --- | --- |
| Version and changelog | `VERSION=2.2.0`, with Make/CMake generated package metadata and CMake `ExactVersion` behavior. No root changelog is tracked. A release maintainer and API reviewer must approve a version-specific entry and verify installed metadata before a candidate can be selected. |
| Candidate and tag | The branch worktree is uncommitted and `git tag --list` is empty. An annotated `v<VERSION>` tag is a draft policy, not a created or historical tag. Require a reviewed clean commit, signing decision, matching tag, and immutable identity. |
| Source archive | The draft uses a pinned-commit `git archive` plus `gzip -n` candidate, with a reviewed tracked-file manifest and required build/legal inputs. Two independent clean checkouts using pinned tools must produce matching manifests and SHA-256 bytes; extraction and static build/tests must pass. None of this was executed on Day 11. |
| Generated and local output | `build/`, generated `include/sparse_version.h`, and `docs/api/html/` are ignored and not tracked. They remain excluded from source/release payloads; generated API HTML stays local-only under Sprint 213 policy. |
| Other assets | Binary and standalone docs publication are unapproved. Local Homebrew tar/checksum proof is not a release or reproducibility result; existing CI static artifacts and selected reports are not release payloads. |

The owner, inspection, pass, stop, and current state for each input and
publication surface are in `RELEASE_READINESS_CHECKLIST.md`. The rationale,
candidate method, and Day 12 handoff are in
`artifacts/day11-release-sources-and-tags.md`.

## Day 12 Provenance And CI Gate Evidence

| Gate | Current result and remaining proof |
| --- | --- |
| Source-to-binary chain | Checklist now requires one reviewed commit/tag, matching two-build source archive digests, approved source manifest, extracted static build, installed-file hashes, build flags/toolchain/dependency record, and installed consumer. No release candidate or provenance chain exists yet. |
| Package origin | Source checkout or separately approved source archive only. Local Homebrew proof, CI build outputs, binary packages, shared ABI, and public providers do not become distribution channels through this checklist. |
| Platform CI | Linux `package-contract` plus reviewed compile/CMake quality, macOS separate Make/`pkg-config` and CMake install jobs plus Apple Clang quality, and Windows MSVC CMake installed-consumer/CTest subset are named. Candidate-specific hosted run URLs/results have not been recorded. |
| Claims and generated docs | README, INSTALL, and maintainer guidance now point to or preserve the static-first release non-claim. Generated `docs/api/html/` remains local-only, with no release asset. |
| Approval and rollback | Product/release, build, platform, API/ABI, documentation, and legal/license signoffs are manual gates. The checklist requires halt/notify/new-version correction and retention of original digests; no publication or rollback was performed. |

`artifacts/day12-release-provenance-and-ci-gates.md` records the matrix and
current blockers. This is a complete release-gate specification, not evidence
that the product is release-ready.

## Day 13 Integrated Validation Evidence

| Surface | Local result and limit |
| --- | --- |
| Installed static consumers | Make/`pkg-config` install passed 25/25; CMake install/export passed 30/30 with 0 skipped. Both executed installed consumers and rejected missing archive/version failures. |
| Negative contract | Fourteen static install metadata fixtures and the static package deferral guard passed, including unsupported shared/linkage selectors and no shared/import metadata. |
| Documentation and provider | `make api-docs-local-only`, `make support-docs-guard`, and the full `make package-manager-deferral-guard` passed. The latter completed its developer-mode local Homebrew proof after a CMake source bootstrap; no public provider or release claim follows. |
| Reviewed quality | `make quality-review-compile` passed format, 49-source list, lint/cppcheck, and helper guard; `make quality-review-cmake` passed clean build, 60-test parity, and 60/60 CTest. One non-fatal warning in unchanged `tests/test_svd_partial_corpus.c:83` is recorded in the artifact. |
| Platform boundary | Windows workflow-owner regression passed locally, but no hosted Windows MSVC or candidate-specific Linux/macOS/Windows run was performed. No shared consumer was run because the selected path is deferral. |

The exact commands, outputs, skip reasons, warning, and Day 14 handoff are in
`artifacts/day13-integrated-validation.md`. Local static validation does not
close release provenance or hosted platform evidence.

## Day 14 Closeout Evidence

The Day 14 audit in `artifacts/day14-closeout-review.md` reconciles items
214.1-214.6 as completed decision/design/selected static-guard work. The
Sprint 214 residual handoff in `RESIDUAL_QUEUE.md` prioritizes candidate
release, shared product decision, ABI compatibility, platform loader and
consumer proof, hosted evidence, and separate distribution decisions. Epic 19
status now records the exact static-first outcome rather than a shared or
release promotion. Day 14 reran Make install 25/25, CMake install 30/30,
static contract fixtures 14/14, the static guard, docs-check,
api-docs-local-only, support-docs and provider-claim fixtures, format-check,
and source-list-check; all passed. The Day 13 full reviewed quality and
provider checks remain the unchanged-code baseline. Exact commands and
skip reasons are in the Day 14 artifact.

## Current Claim Boundary

| Earned current surface | Still unclaimed |
| --- | --- |
| Static archive source build and install. | Shared-library build/install/package support. |
| Unix Make install and `pkg-config` downstream validation on reviewed Linux/macOS lanes. | Stable dynamic ABI, export allowlist, ABI epoch, or cross-version compatibility. |
| CMake static install/export and installed `find_package(Sparse)` consumer proof, with exact package-version matching. | SONAME, macOS install-name/RPATH, Windows DLL/import-library behavior, runtime-loader proof, or static/shared selectors. |
| Windows CMake static install/downstream lane and metadata-only `sparse.pc` inspection. | Windows Make parity, Windows `pkg-config` command execution parity, or broad platform parity. |
| Developer-mode local Homebrew static source formula proof. | Public provider distribution, binary/release packages, or release readiness. |

## Decision Log

| Day | Decision | Rationale and next gate |
| --- | --- | --- |
| 1 | No shared-proof or deferral path selected. | Intake confirms static-first policy and unresolved ABI/loader/release questions. Day 2 must compare a bounded proof with complete design/deferral and record a single selected path or escalate a product decision. |
| 2 | Select complete ABI/release design with stronger static-first deferral guards; reject a shared proof for Sprint 214. | The candidate Linux/GCC/CMake proof lacks a defined ABI subset, curated exports, loader policy, installed dynamic consumer, and reviewed CI lane. Keep Sprint 170/207 decisions intact. See `artifacts/day2-abi-release-scope-decision.md`. |
| 4 | Define future ABI identity and compatibility gates without assigning a current epoch. | Exact package-version matching and static consumer execution do not prove dynamic or cross-version compatibility. Retain old executable baselines before any backward-compatibility claim. See `artifacts/day4-abi-compatibility-policy.md`. |
| 5 | Complete loader policy design and select static deferral guard implementation. | Linux/macOS/Windows metadata and runtime gates are future-only. Days 6-8 strengthen configure/install/metadata/docs rejection without adding a shared target or claim. See `artifacts/day5-loader-and-build-design.md`. |
| 6 | Reject unsupported package link-mode inputs rather than silently ignore them. | The selected static-first path remains unchanged. Eleven isolated negative configurations and default generated-metadata inspection now run through the existing deferral guard. See `artifacts/day6-build-path-or-deferral-guard.md`. |
| 7 | Validate installed static artifacts and package metadata together. | Shared/import artifacts, extra CMake targets, loader fields, selectors, and `sparse.pc` drift now fail isolated fixture checks; real Make/CMake static consumers remain green. See `artifacts/day7-install-and-metadata-contract.md`. |
| 8 | Enforce the selected deferral checks in package CI and define installed consumer acceptance. | Linux/macOS package lanes now run mutation fixtures and docs/non-claim guard; Windows CMake lane runs the installed-prefix validator and fixtures. Local checks pass; hosted results are not yet evidence. See `artifacts/day8-automation-and-consumer-design.md`. |
| 9 | Preserve installed static consumer evidence and specify future shared-consumer fixtures. | Both local macOS static routes passed with visible consumer values; ELF/Mach-O/PE shared fixture inputs and missing/wrong-artifact diagnostics remain future gates. See `artifacts/day9-installed-consumer-evidence.md`. |
| 10 | Exercise selected static consumer failures without promoting shared ABI claims. | Missing Make/CMake archive and package-version diagnostics passed; static consumers remain green. Dynamic architecture/loader/export cases are future gates. See `artifacts/day10-consumer-negative-matrix.md`. |
| 11 | Draft source, tag, and artifact gates without shipping. | No changelog, tag, clean candidate commit, reproducibility comparison, or release asset exists. Require a reviewed tracked-source manifest and two-build byte comparison before any source claim; keep generated HTML local-only. See `artifacts/day11-release-sources-and-tags.md`. |
| 12 | Complete item 214.5 gate design while preserving static-first non-claims. | Source-to-binary provenance, package origin, platform CI, signoff, and rollback now have explicit owners and stop conditions. Candidate-specific results and approvals remain absent. See `artifacts/day12-release-provenance-and-ci-gates.md`. |
| 13 | Validate the selected static-first path end to end locally. | Installed Make/CMake consumers, negative fixtures, deferral/docs/provider guards, reviewed compile quality, and 60/60 CTest passed. Shared ABI, hosted platform, and release readiness remain unclaimed. See `artifacts/day13-integrated-validation.md`. |
| 14 | Close the selected ABI/release decision as guarded static-first deferral. | Item audit, Epic status, release-checklist alignment, and prioritized residuals are recorded in `artifacts/day14-closeout-review.md` and `RESIDUAL_QUEUE.md`; no shared or release claim is promoted. |

## Day 2 Selected-Path Acceptance Matrix

| Workstream | Required outcome | Boundary |
| --- | --- | --- |
| ABI policy, Days 3-5 | Document selected future ABI subset, symbol/export, ownership/layout, version/break, and Linux/macOS/Windows loader requirements. | Design is not binary compatibility proof. |
| Deferral guard, Days 6-8 | Preserve `BUILD_SHARED_LIBS=ON` rejection, static-only Make/CMake install, static package metadata, and docs non-claims; add focused negative regressions. | No shared target, artifact, selector, or loader claim. |
| Consumer, Days 9-10 | Execute maintained static installed consumers; specify exact future dynamic positive/negative fixtures. | Specification is not executed shared evidence. |
| Release, Days 11-12 | Complete checklist with source/tag/artifact/provenance/CI owners and stop gates. | Checklist is not release readiness. |
| Closeout, Days 13-14 | Pass affected install/export, guard, docs, and quality checks and record residuals. | Failed required checks block closure. |

The full option comparison, future shared acceptance tests, non-claims, and
stop conditions are in the Day 2 decision artifact. No product input blocks
design/deferral work; shared-library or release promotion would require a new
explicit decision and evidence.

## Risk Register

| Risk or unknown | Why it matters | Mitigation / owner |
| --- | --- | --- |
| Public concrete structs and callbacks | A shared build would expose layouts, calling conventions, and ownership semantics as binary commitments. | Days 3-4 identify an explicit ABI subset and compatibility policy; no broad API-to-ABI inference. |
| Uncurated symbols | Archive globals are not a reviewed dynamic export list. | Day 3 inspect actual symbols and define allowlist/hidden-by-default behavior or future acceptance tests. |
| Index width and version confusion | `SPARSE_IDX_BITS` changes public types; `VERSION=2.2.0` and CMake `ExactVersion` identify source/package versions, not ABI epoch. | Day 4 separates ABI version from package version and defines mismatch tests. |
| Platform loader differences | Linux SONAME, macOS install-name/RPATH, and Windows DLL/import-library/CRT rules differ. | Day 5 records per-platform policy; claim only a platform with executed loader proof. |
| Static regression from shared work | A shared target or metadata selector could break current Make/CMake consumers. | Days 6-10 retain static install and downstream tests as mandatory gates. |
| Package-provider overclaim | A narrow shared artifact can be mistaken for a distributable package. | Keep Sprint 207 provider deferral and package guard active unless separately reopened. |
| Release checklist mistaken for readiness | A checklist is a future gate, not proof of reproducibility, provenance, or published artifacts. | Days 11-12 distinguish required evidence from executed evidence and assign blockers. |
| CI versus local evidence confusion | Workflow definitions alone do not show hosted success on a new path. | Days 8 and 13 record actual job/test outcomes and label local-only evidence precisely. |

## Validation Matrix

| Command or check | When required | Status through Day 13 |
| --- | --- | --- |
| `bash scripts/static_package_deferral_check.sh` | Current static/ABI boundary and after build/install/claim changes. | Passed locally on Days 1, 3, 5-10, 12, and 13; Day 13 includes selector and Windows scope checks. |
| `make package-manager-deferral-guard` | Provider claim boundary, especially after docs or package metadata changes. | Passed locally on Day 13, including the developer-mode Homebrew static-source proof; an earlier run had been interrupted during its CMake dependency bootstrap. Public provider support remains deferred. |
| `bash tests/test_install.sh` | Make install, `pkg-config`, static consumer, and shared-artifact regression. | Passed locally on Day 13: 25/25 with installed consumers and missing-archive link rejection. |
| `bash tests/test_cmake_install.sh` | CMake install/export, `find_package`, version, and shared-metadata regression. | Passed locally on Day 13: 30/30, 0 skipped, including version and missing imported-file diagnostics. |
| `python3 tests/test_static_install_contract.py` | Installed-prefix positive/negative fixture regressions. | Passed locally on Day 13: 14/14 fixtures. |
| `bash -n scripts/static_package_deferral_check.sh tests/test_install.sh tests/test_cmake_install.sh` | Changed Bash guard/test syntax. | Passed locally on Day 13. |
| `python3 tests/test_validate_windows_powershell.py` | Windows workflow ownership and diagnostic regression. | Passed locally on Day 13; expected negative `--require-pwsh` fixture output is not a hosted Windows parse run. |
| Relevant docs guards and `make docs-check` | Public ABI, package, release, or API-routing wording changes. | Day 13 `make api-docs-local-only` passed Doxygen coverage, direct guard, and regression suite; `make support-docs-guard` passed. Generated HTML remains ignored/local-only. |
| `python3 tests/test_package_manager_deferral_guard.py` | Provider guard regressions after package-claim wording changes. | Passed locally on Day 12; full provider guard passed on Day 13. |
| `make format && make lint && make test` | Any changed `*.c` or `*.h` file. | Not separately required through Day 13; no C/header files changed. Reviewed compile quality and full CMake/CTest suite passed, but are not relabeled as this command. |
| `make quality-review-compile`, `make quality-review-cmake`, and selected install/consumer lanes | Later build, CMake, or ABI proof changes as appropriate. | Both reviewed targets passed locally on Day 13; CMake 60/60 tests. |
| `make source-list-check` | Source/archive inventory baseline and later build changes. | Passed locally on Day 3 (49 library sources). |
| `git diff --check` plus new-file whitespace check | Every documentation edit. | Passed through Day 13; Day 14 final hygiene recorded below. |

## Changed-Surface Tracker

| Path | Day | Change | Reason |
| --- | --- | --- | --- |
| `docs/planning/EPIC_19/SPRINT_214/PLAN.md` | 0 | Added before Day 1 | Day-by-day scope and hour allocation. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 1 | Added | Evidence ledger, item checklist, risk register, decision log, validation matrix, and changed-surface tracker. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day1-abi-release-evidence-intake.md` | 1 | Added | Day 1 current-state and non-claim record. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 2 | Updated | Selected decision, acceptance matrix, and handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day2-abi-release-scope-decision.md` | 2 | Added | Option comparison, selected path, future shared-proof tests, and stop gates. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 3 | Updated | Symbol inventory, candidate export boundary, and Day 4 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day3-symbol-visibility-policy.md` | 3 | Added | Current declaration/archive inventory and future symbol-visibility acceptance policy. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 4 | Updated | Compatibility evidence, acceptance boundary, validation result, and Day 5 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day4-abi-compatibility-policy.md` | 4 | Added | Future ABI identity, compatibility matrix, retained baseline, and release blockers. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 5 | Updated | Loader policy evidence, completed ABI design status, static guard baseline, and Day 6 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day5-loader-and-build-design.md` | 5 | Added | Platform loader gates and selected static deferral implementation/negative-test design. |
| `CMakeLists.txt` | 6 | Updated | Reject unsupported package-prefixed shared and link-mode cache selectors. |
| `scripts/static_package_deferral_check.sh` | 6 | Updated | Exercise selector failures and inspect default generated static package metadata. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 6 | Updated | Configure/install evidence, validation matrix, and Day 7 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day6-build-path-or-deferral-guard.md` | 6 | Added | Implementation, focused regressions, unresolved risks, and next gates. |
| `scripts/check_static_install_contract.py` | 7 | Added | Installed-prefix artifact and generated metadata contract validator. |
| `tests/test_static_install_contract.py` | 7 | Added | Positive and mutation fixtures for Unix and Windows-style static metadata. |
| `tests/test_install.sh` | 7 | Updated | Validate real Make install against artifact/metadata contract. |
| `tests/test_cmake_install.sh` | 7 | Updated | Validate real CMake install against artifact/metadata contract. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 7 | Updated | Install evidence, negative results, and Day 8 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day7-install-and-metadata-contract.md` | 7 | Added | Artifact inventory, negative-test results, and residuals. |
| `.github/workflows/ci.yml` | 8 | Updated | Run mutation fixtures and label deferral/docs-claim proof in Linux package CI. |
| `.github/workflows/macos-ci.yml` | 8 | Updated | Run mutation fixtures and label deferral/docs-claim proof in macOS CMake package CI. |
| `.github/workflows/windows-ci.yml` | 8 | Updated | Run Windows fixture diagnostics and validate the real installed static prefix. |
| `tests/test_static_install_contract.py` | 8 | Updated | Exercise missing archive, missing package metadata, and wrong version diagnostics. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 8 | Updated | CI ownership, validation results, consumer design, and Day 9 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day8-automation-and-consumer-design.md` | 8 | Added | CI ownership, diagnostic matrix, and platform-scoped consumer acceptance specification. |
| `tests/test_install.sh` | 9 | Updated | Print successful installed Make/`pkg-config` consumer output. |
| `tests/test_cmake_install.sh` | 9 | Updated | Print successful CMake package and exact-version consumer output. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 9 | Updated | Static consumer outputs, symbol/metadata boundary, and future fixture handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day9-installed-consumer-evidence.md` | 9 | Added | Executed local static consumer evidence and future ELF/Mach-O/PE shared-consumer specification. |
| `tests/test_install.sh` | 10 | Updated | Day 10 required a missing-archive link failure; PR #237 review follow-up added installed-prefix validation before another package-flag link and narrowed the explicit-path link to missing-file diagnostics, not host-library fallback proof. |
| `tests/test_cmake_install.sh` | 10 | Updated | Require version-specific mismatch diagnostic and missing imported archive validation/configure failures. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 10 | Updated | Consumer negative evidence, validation results, and Day 11 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day10-consumer-negative-matrix.md` | 10 | Added | Executed static and future-only dynamic negative matrix with CI ownership. |
| `docs/planning/EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md` | 11 | Added | Living source/tag/artifact gates, owners, inspections, stop conditions, and Day 12 slots. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day11-release-sources-and-tags.md` | 11 | Added | Release input inventory, reproducibility acceptance, and unclaimed publication boundary. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 11 | Updated | Release-source evidence, validation status, and Day 12 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md` | 12 | Updated | Source-to-binary, platform CI, signoff, blockers, and rollback gates. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day12-release-provenance-and-ci-gates.md` | 12 | Added | Provenance and CI matrix, public wording reconciliation, stop conditions. |
| `README.md` | 12 | Updated | Make release-readiness/source-archive non-claim explicit. |
| `INSTALL.md` | 12 | Updated | Preserve user-facing release non-claim and link maintainer checklist. |
| `docs/maintainer_guide.md` | 12 | Updated | Direct reviewers to release gates without promoting deferred support. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 12 | Updated | Gate evidence, validation status, and Day 13 handoff. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day13-integrated-validation.md` | 13 | Added | Integrated static install/consumer, negative, docs, provider, and reviewed quality transcript. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 13 | Updated | Validation results, changed-surface status, and Day 14 handoff. |
| `docs/planning/EPIC_19/PROJECT_PLAN.md` | 14 | Updated | Mark Sprint 214 decision/deferral closed with exact static-only evidence and residuals. |
| `docs/planning/EPIC_19/SPRINT_214/RELEASE_READINESS_CHECKLIST.md` | 14 | Updated | Reflect passing Day 13 local checks without implying a candidate release. |
| `docs/planning/EPIC_19/SPRINT_214/RESIDUAL_QUEUE.md` | 14 | Added | Prioritize unproved release, ABI, loader, hosted, and distribution gates. |
| `docs/planning/EPIC_19/SPRINT_214/artifacts/day14-closeout-review.md` | 14 | Added | Final item audit, validation boundary, and retrospective handoff. |
| `docs/planning/EPIC_19/SPRINT_214/WORKING_NOTES.md` | 14 | Updated | Item status, closeout evidence, decision log, and changed-surface tracking. |
| `docs/planning/EPIC_19/SPRINT_214/RETROSPECTIVE.md` | Closeout | Added | Sprint outcomes, measured validation, bounded claim, lessons, and residual handoff. |

## Day 1 Handoff

Day 2 must select one path or identify the exact product input blocking a
decision. The selection must name a platform/toolchain and executable consumer
and loader gates for any shared proof. Otherwise it must specify enforceable
configure/install/docs deferral gates and exact future shared acceptance tests.

## Day 2 Handoff

Day 2 selected design/deferral after comparing a Linux/GCC/CMake-only shared
candidate. Days 3-5 should specify the ABI subset, symbol/visibility policy,
version and compatibility rule, and platform loader tests before Days 6-8
change any guard. The current static-first install/export and package-provider
boundaries remain in force.

## Day 3 Handoff

Day 4 must decide the ABI epoch/compatibility policy and assess whether the
provisional eight-symbol opaque-matrix subset can be stable across width,
scalar, ownership, error, and platform calling-convention boundaries. No
dynamic export list is approved merely because all eight functions exist in
the static archive.

## Day 4 Handoff

Day 5 must define platform-specific loader and packaging acceptance for Linux,
macOS, and Windows against the proposed ABI identity. No shared target, ABI
epoch, cross-version compatibility, or release-readiness claim was enabled by
Day 4's policy design or static CMake install test.

## Day 5 Handoff

Days 6-8 should implement the selected static-only configure, install,
metadata, and docs guard improvements with isolated negative fixtures. Keep
the maintained static imported target and consumers passing. Do not add a
shared selector, artifact, or loader support claim under this decision; the
platform-specific loader tests remain future acceptance gates.

## Day 6 Handoff

Day 7 should inventory the installed prefix and generated package files for
unselected shared artifacts, import/loader metadata, and static/shared
selectors, with isolated negative fixtures. Preserve both maintained static
consumer paths and distinguish Windows static `.lib` from a DLL import
library by metadata/content rather than suffix alone.

## Day 7 Handoff

Day 8 should review CI wiring for the new validator and fixture tests, retain
the current Linux/macOS static install lanes, and assess Windows hosted
artifact/consumer coverage without claiming DLL/import-library content proof
from the local Windows-style fixture. Public shared, loader, ABI, provider,
and release-readiness claims remain deferred.

## Day 8 Handoff

Day 9 should record exact out-of-tree static consumer commands and outputs for
the selected Make/CMake routes, then write executable future shared-consumer
fixtures without presenting them as runtime evidence. Hosted CI results must
be checked separately; the Day 8 local macOS runs and Windows-style fixtures
do not establish Windows DLL/import-library or loader behavior.

## Day 9 Handoff

Day 10 should execute selected-path negative cases for missing/wrong archive,
package version, imported location, and unsupported shared selectors while
keeping both static installed consumers green. Wrong architecture, missing
dynamic export, absent runtime path, and DLL/SONAME/install-name failures
remain explicitly future-only until a shared product path is selected and
implemented. Hosted Windows CI has not been verified by this local run.

## Day 10 Handoff

Day 11 should draft release source/tag/reproducibility gates without treating
the passing static consumer matrix as shared ABI or release-readiness proof.
The Windows hosted CMake job and the Day 8 CI edits still need branch-run
evidence; wrong-architecture and runtime-loader negatives remain future-only.

## Day 11 Handoff

Day 12 should complete the checklist's checksum/provenance, source-to-binary,
platform CI, approval, public-claim, and rollback gates. Keep the missing
changelog, clean commit, approved tag policy, reviewed manifest, and untested
two-build comparison as explicit blockers. Do not create or imply a release
asset, and do not promote local generated API HTML into a retained artifact.

## Day 12 Handoff

Day 13 should run the integrated static installed-consumer, fixture-negative,
deferral, documentation, and applicable build-quality gates and record their
actual outputs. This does not close the checklist's missing changelog,
candidate tag/archive, reproducibility comparison, source-to-binary
provenance, hosted candidate CI, or manual approvals. Keep shared-library,
dynamic ABI, public provider, binary-release, and generated API publication
claims deferred.

## Day 13 Handoff

Day 14 should audit all six Sprint 214 items, check the changed-surface and
residual queue, and re-run affected gates after any final edit. Day 13 local
selected-path checks passed, but hosted branch CI, a clean release candidate,
tag/archive reproducibility, provenance, and approvals remain unverified.
Do not convert static install or developer-mode Homebrew proof into shared
ABI, public provider, or release-readiness support.

## Day 14 Final Hygiene

Final `git diff --check` passed, the new Sprint 214 Markdown trailing-
whitespace scan found no matches, and `git diff --name-only -- '*.c' '*.h'`
was empty. The same results are recorded in the Day 14 closeout artifact.
The branch remains uncommitted for the requested later retrospective,
commit, and PR step; this is not a release candidate or tag.
