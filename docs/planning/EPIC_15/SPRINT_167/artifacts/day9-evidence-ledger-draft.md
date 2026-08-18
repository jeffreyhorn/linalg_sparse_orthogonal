# Sprint 167 Day 9: Evidence Ledger Draft

## Purpose

Day 9 creates the first Epic 15 evidence ledger draft from the prior residual,
source/header, test/corpus, CI, package, and documentation inventories. The
ledger is intentionally conservative: broad claims remain unsupported unless a
named source file, validation command, report, or hosted CI lane owns the
evidence.

Day 10 will review and correct this draft against current docs and non-claim
wording.

## Status Labels

| Label | Meaning |
| --- | --- |
| Supported | Current evidence and validation owners support the scoped claim. |
| Partially supported | Evidence exists, but only for selected fixtures, selected platforms, or selected workflows. |
| Hosted-only | Evidence depends on a hosted CI lane and should name the exact workflow/job scope. |
| Local-only | Evidence is reproducible locally but is not hosted, release-published, or broadly portable. |
| Advisory | Evidence provides metadata, navigation, or interpretation but does not prove a product claim. |
| Deferred | The project intentionally leaves the capability for a later decision or implementation. |
| Unsupported | The project must not claim this capability. |

## Draft Evidence Ledger

| Ledger ID | Claim area | Draft status | Scope supported today | Evidence owners | Validation or hosted owner | Future owner / action | Non-claims |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E15-001 | Local build and full C test quality | Supported | Local Makefile quality and test execution when the required commands pass. | `Makefile`, `README.md`, tests under `tests/` | `make format && make lint && make test`; local only unless CI job is named. | General quality gate for source/header changes. | Does not prove package-manager support, dynamic ABI, or state-of-the-art behavior. |
| E15-002 | Linux reviewed baseline | Partially supported / hosted-only | Linux reviewed Makefile compile-quality, CMake parity, dead-code, static package, and selected report-freshness lanes. | `.github/workflows/ci.yml`, `README.md`, `INSTALL.md`, `docs/maintainer_guide.md` | Linux CI jobs `lint`, `cmake-build-and-test`, `deadcode`, `package-contract`, `generated-report-freshness`. | Day 10 should verify wording does not overstate supplemental lanes. | No broad platform parity, package-manager support, dynamic ABI, portable performance, or broad report freshness. |
| E15-003 | macOS reviewed package/build tier | Partially supported / hosted-only | Apple Clang reviewed Make/CMake path plus reviewed static install/export proof; Homebrew GCC is supplemental. | `.github/workflows/macos-ci.yml`, `INSTALL.md`, `README.md` | macOS CI `build-and-test`, `install-and-pkgconfig`, `cmake-install-export`. | Keep platform tier explicit. | No broad macOS parity, shared-library packaging, dynamic ABI, package-manager support, or portable performance. |
| E15-004 | Windows reviewed CMake tier | Partially supported / hosted-only | MSVC CMake configure/build/CTest with expected count `59`, plus CMake install/downstream validation. | `.github/workflows/windows-ci.yml`, `INSTALL.md`, `README.md` | Windows CI `build-and-test` and `install-and-downstream`. | Keep CMake-first boundary explicit. | No Windows Makefile parity, Windows `pkg-config` execution parity, DLL/shared support, runtime-loader behavior, package-manager support, or broad Windows parity. |
| E15-005 | Static-first source package install | Supported for scoped paths | Static archive, public headers, generated version header, pkg-config metadata, CMake package metadata, downstream consumers, exact version behavior, uninstall cleanup. | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, `INSTALL.md` | `tests/test_install.sh`, `tests/test_cmake_install.sh`, Linux/macOS package jobs, Windows CMake package job. | Sprint 170/171 package decision work should build from this boundary. | No package-manager distribution, shared library, dynamic ABI, runtime-loader behavior, or static/shared selector UX. |
| E15-006 | Shared-library support | Unsupported / deferred | CMake intentionally rejects `BUILD_SHARED_LIBS=ON`. | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh`, `INSTALL.md`, README | `bash scripts/static_package_deferral_check.sh` | Sprint 170 decides whether to retain static-only or start a staged shared ABI track. | No `.so`, `.dylib`, `.dll`, SONAME, install-name/RPATH, import-library, or installed shared consumer support. |
| E15-007 | Dynamic ABI compatibility | Unsupported / deferred | Exact package version metadata exists, but no dynamic ABI policy exists. | `CMakeLists.txt`, `INSTALL.md`, `docs/maintainer_guide.md` | Static deferral guard and package metadata checks only prove non-claim boundaries. | Sprint 170 product decision. | No binary compatibility guarantee, symbol versioning policy, exported-symbol check, or ABI matrix. |
| E15-008 | Package-manager distribution | Unsupported / deferred | Source install and package metadata are maintained; no provider distribution is supported. | `INSTALL.md`, README, `docs/maintainer_guide.md`, Day 7 inventory | No provider validation command today. | Sprint 171 chooses provider proof or formal deferral. | No Homebrew, vcpkg, Conan, apt, dnf, pacman, upgrade, or provider provenance support. |
| E15-009 | Generated API HTML | Local-only / deferred publication | Generated API HTML is intentionally local-only and ignored; source headers are the maintained API authority. | `docs/api_reference.md`, `docs/maintainer_guide.md`, public headers, Doxygen config | `make docs`, docs checks/API coverage where selected. | Sprint 173 chooses hosted/committed/artifact-only publication or reaffirms local-only status. | No hosted API docs, source-controlled generated HTML, ABI support, package proof, or external parity. |
| E15-010 | Public API/header coherence | Partially supported | Selected prior header batches were cleaned; remaining headers still need broader cleanup. | `include/*.h`, `docs/api_reference.md`, Day 4 inventory | Declaration-preservation checks where selected; full C gate if headers change. | Sprint 172 selects one high-impact header family. | No broad API redesign or ABI guarantee from comment cleanup. |
| E15-011 | Solver correctness | Partially supported | Many solver tests exist, but broad correctness is fixture/test-scope-bound. | `tests/test_*.c`, solver docs, `docs/maintainer_guide.md` | `make test`, CTest, selected proof-owner tests. | Future sprints should tie claims to solver family and fixture evidence. | No broad sparse-matrix-family correctness, external ecosystem parity, or state-of-the-art proof. |
| E15-012 | Maintained corpus/oracle evidence | Partially supported / local-only with selected hosted lane | QR and partial-SVD fixture-local corpus/oracle rows with selected Linux hosted freshness for required rows. | `tests/corpus/**`, `scripts/run_corpus_oracle.py`, `scripts/normalize_report_index.py`, `.github/workflows/ci.yml` | `make report-index-oracle-freshness`; Linux `generated-report-freshness` job for selected rows. | Day 10 should verify selected versus unselected row wording. | No broad corpus completeness, SuiteSparse parity, platform/package/ABI/performance proof, or broad QR/SVD correctness. |
| E15-013 | Selected external comparison evidence | Partially supported / local-only with selected hosted lane | Selected `qr-minnorm`, `qr-compatible-ls`, and `partial-svd-diag6-k2` fixture-level comparisons. | `scripts/run_external_comparison.py`, dense reference helpers, `tests/corpus/manifests/report_families.tsv`, `.github/workflows/ci.yml` | `make report-index-comparison-freshness`; Linux hosted selected comparison freshness lane. | Sprint 174 selects one additional bounded comparison family. | No broad QR/SVD/partial-SVD parity, raw basis/vector identity, external-library ecosystem parity, platform/package/ABI/performance proof, or state-of-the-art evidence. |
| E15-014 | Benchmark/performance reports | Local-only / partially supported | Local benchmark rows and local sentinels have methodology metadata; CI has bench-fast smoke and compile coverage. | `benchmarks/README.md`, `Makefile`, `scripts/bench_canonical_report.sh`, performance scripts, README | `make bench-canonical-report`, `make performance-sentinels`, `make bench-fast` in Linux supplemental CI. | Sprints 168-169 choose hosted performance publication and methodology hardening. | No portable performance, backend superiority, release benchmark proof, or state-of-the-art performance. |
| E15-015 | Broad generated-report freshness | Partially supported | Selected oracle/comparison freshness exists; other generated report families remain local, advisory, or unselected. | `tests/corpus/manifests/report_families.tsv`, `scripts/normalize_report_index.py`, CI workflows | Selected freshness targets only. | Sprint 175 selects a platform/report promotion or formal deferral. | No all-family freshness, broad platform parity, or release proof. |
| E15-016 | Allocation/failure-path evidence | Deferred / partially supported | Functional tests cover many paths, but deterministic allocation-failure evidence is not yet a first-class selected proof. | Day 4 allocation inventory, solver source files, tests | No selected failure-injection gate today. | Sprint 176 selects one subsystem and adds deterministic failure-path proof. | No broad OOM/failure cleanup guarantee across all solvers. |
| E15-017 | State-of-the-art sparse linear algebra positioning | Unsupported | Current evidence supports a mature, evidence-disciplined self-contained library, not broad state-of-the-art parity. | README non-claims, Epic 13/14 retrospectives, Epic 15 review | No comprehensive competitive evidence owner today. | Keep as final claim recalibration/non-claim target. | No unqualified state-of-the-art, broad external-library parity, portable performance superiority, broad package/ABI/platform parity. |
| E15-018 | External-library ecosystem parity | Unsupported except selected fixtures | Selected dense-helper comparisons exist for narrow fixtures only. | comparison scripts, dense reference helpers, corpus docs, maintainer guide | Selected comparison freshness only. | Sprint 174 may add one bounded family, not broad parity. | No LAPACK, NumPy, SciPy, SuiteSparse, Eigen, PETSc, Trilinos, or package ecosystem parity. |

## Missing Evidence Register

| Missing evidence | Related ledger rows | Why it matters | Candidate owner |
| --- | --- | --- | --- |
| Hosted performance publication lane | E15-014, E15-017 | Without hosted performance evidence, benchmark rows remain local-only and cannot support broader performance wording. | Sprint 168 / Sprint 169 |
| Shared-library ABI product decision | E15-006, E15-007, E15-008 | Package and provider decisions depend on whether the product stays static-first or starts a shared ABI track. | Sprint 170 |
| Package-manager provider proof or formal deferral | E15-008 | Source installs do not equal package-manager distribution. | Sprint 171 |
| Next selected public-header cleanup batch | E15-010 | API docs remain partially normalized, and generated API publication should rely on cleaned header inputs. | Sprint 172 |
| Generated API HTML publication decision | E15-009 | Local-only status is clear, but hosted/committed/artifact publication remains a possible adoption gap. | Sprint 173 |
| Additional bounded comparison family | E15-013, E15-018 | More comparison breadth is needed without making broad parity claims. | Sprint 174 |
| Cross-platform report freshness promotion or deferral | E15-015 | Selected Linux hosted report freshness does not prove broad report/platform parity. | Sprint 175 |
| Deterministic allocation-failure proof | E15-016 | Manual allocation-heavy solver setup lacks one selected failure-injection evidence lane. | Sprint 176 |
| PR #184 hosted-result reconciliation | E15-002, E15-003, E15-004 | Epic 14 closeout cited PR-time hosted confirmation as residual; current ledger should avoid branch-specific claims unless exact hosted evidence is cited. | Sprint 167 Day 10/11 if needed |

## Evidence Owner Notes

| Owner class | Source examples | Ledger use |
| --- | --- | --- |
| Source-controlled command owner | `Makefile`, scripts, tests | Can support local evidence when commands are run and recorded. |
| Hosted CI owner | `.github/workflows/*.yml` | Can support hosted evidence only for exact workflow/job/commit/scope. |
| Public claim owner | `README.md`, `INSTALL.md`, `docs/api_reference.md`, `benchmarks/README.md` | Must map to evidence rows or explicit non-claims. |
| Maintainer interpretation owner | `docs/maintainer_guide.md`, corpus schemas/manifests | Defines support tiers, caveats, and claim boundaries. |
| Historical planning owner | prior epic/sprint artifacts | Provides rationale and residual history, not live support unless current evidence still validates the claim. |
| Generated local owner | ignored `build/` and `coverage/` paths | Local-only unless a hosted lane or publication decision promotes the exact artifact. |

## Day 10 Review Targets

Day 10 should check this draft for:

- any row that should be downgraded from supported to partially supported or
  local-only;
- any row that needs a sharper non-claim;
- any source path that should cite a more authoritative document;
- any missing evidence that should move into Day 11 selection;
- stale Epic 12 path references in current Sprint 167 artifacts;
- benchmark wording that could imply hosted or portable performance evidence;
- package wording that could imply package-manager distribution or dynamic ABI.

## Validation Notes

Day 9 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| All major claim categories have ledger rows. | Complete | Draft ledger covers build/test, platforms, package, ABI, package-manager, generated API, API/header, solver correctness, corpus/oracle, comparison, performance, report freshness, allocation failure, state-of-the-art, and external parity. |
| Every row has a status and evidence reference or explicit gap. | Complete | Each ledger row includes a draft status, evidence owners, validation/hosted owner, future action, and non-claims. |
| Missing evidence is visible before gap selection. | Complete | Missing evidence register identifies the main Day 11 selection candidates and unresolved proof gaps. |
