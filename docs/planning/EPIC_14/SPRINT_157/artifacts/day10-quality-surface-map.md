# Day 10 Quality Surface Map

## Scope

Day 10 maps validation expectations by change type and evidence surface before
Epic 14 implementation sprints begin. This artifact turns the Day 9 evidence
templates into concrete command expectations and review checklists.

The map is a floor, not a ceiling. If a change touches multiple surfaces, run
the union of the relevant checks.

## Change-Type Validation Matrix

| Change type | Typical files | Required validation | Supplemental validation when relevant | Stop condition |
| --- | --- | --- | --- | --- |
| Documentation-only planning work | `docs/planning/**` | `git diff --check`; direct trailing-whitespace scan on new untracked docs if they are not yet in Git diff. | Claim-sensitive `rg` scan when wording touches public support, state-of-the-art, package, ABI, performance, platform, or generated evidence. | Whitespace fails, claim wording widens without evidence, or planning artifact is used as product proof. |
| Public documentation | `README.md`, `INSTALL.md`, `docs/*.md`, `benchmarks/README.md`, `examples/README.md`, `tests/corpus/README.md` | `git diff --check`; claim-sensitive scan for unsupported public wording. | Link/anchor spot checks for changed references; generated docs/report checks if public docs cite fresh generated evidence. | Docs imply unsupported state-of-the-art, external parity, package-manager, shared-library, dynamic ABI, runtime-loader, broad Windows, or portable performance claims. |
| Public headers, even comment-only | `include/*.h`, `include/sparse_version.h.in` | `make format && make lint && make test`; before/after declaration-preservation proof for intended comment-only cleanup. | Apply Sprint 158 generated-doc policy if generated API docs are part of the touched surface. | Normalized declarations drift unexpectedly or C/header gates are not run. |
| Implementation C files | `src/*.c` | `make format && make lint && make test`. | Focused test binary or sanitizer path for touched solver/runtime area; `make source-list-check` when adding/removing library sources. | Full C gate fails or source lists diverge. |
| Tests | `tests/*.c`, `tests/*.h`, test helpers | `make format && make lint && make test`. | Focused test binary first for fast diagnosis; CMake enumeration when adding/removing tests; Windows count update if CMake registered tests change. | Make/CMake registration diverges or CI expected counts are stale. |
| Python scripts | `scripts/*.py`, `tests/*_external_dense_reference.py`, `tests/test_normalize_report_index.py` | Targeted Python self-check or test for the touched script; `git diff --check`. | `python3 scripts/validate_corpus_schema.py`, `python3 scripts/normalize_report_index.py --check`, freshness command tied to changed family. | Script behavior changes without targeted test or generated report semantics become ambiguous. |
| Shell scripts | `scripts/*.sh`, `tests/*.sh` | Targeted shell script test or dry-run where safe; `git diff --check`. | Package install scripts for install changes; sentinel/benchmark commands for report scripts. | Script cannot run in its intended shell or changes package/report claims without matching docs. |
| Build-system source lists | `Makefile`, `CMakeLists.txt`, `build-metadata/library_sources.txt` | `make source-list-check`; relevant Make/CMake build path. | `make quality-review-cmake-compile` or `make quality-review-cmake` for CMake registration changes; full C gate if C sources/headers also changed. | Makefile and CMake source/test registration diverge. |
| Package metadata and install rules | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/*.cmake.in`, `VERSION`, install docs, install scripts | Affected install proof: `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, and/or `bash scripts/static_package_deferral_check.sh`; `git diff --check`. | CMake parity checks and platform-specific CI reconciliation for workflow changes. | Static-first support is confused with shared-library, dynamic ABI, package-manager, or Windows `pkg-config` parity. |
| CI workflows | `.github/workflows/*.yml` | `git diff --check`; reconcile lane name, trigger, expected counts, support-tier comments, package/report artifact semantics, and docs references. | Local equivalent commands when available; `ctest -N` count check if test registration changes. | Expected test count, lane name, support-tier wording, or hosted artifact interpretation is inconsistent. |
| Corpus metadata | `tests/corpus/**` | `python3 scripts/validate_corpus_schema.py`; relevant focused C/script tests when expected rows change. | `make report-index-oracle-freshness` for selected oracle rows; normalizer family checks. | Expected rows are not schema-valid or local generated rows are treated as source-controlled pass evidence. |
| Generated oracle/comparison report tooling | `scripts/run_corpus_oracle.py`, `scripts/run_external_comparison.py`, `scripts/normalize_report_index.py`, report-family rows, Makefile freshness targets | Targeted script tests plus selected freshness command: `make report-index-oracle-freshness` and/or `make report-index-comparison-freshness`; `git diff --check`. | Hosted CI artifact semantics when workflows change; row-count/freshness diagnostics in sprint artifact. | Stale/missing/failing selected rows are ignored or advisory families are promoted by accident. |
| Benchmarks, sentinels, and large-matrix reports | `benchmarks/**`, `scripts/bench_canonical_report.sh`, `scripts/performance_sentinels.sh`, `scripts/large_matrix_guardrails.sh`, benchmark docs | Relevant command: `make bench-canonical-report`, `make performance-sentinels`, `make large-matrix-guardrails`, or `make bench-fast`; `git diff --check`. | `make wall-check` when touching sentinel thresholds or wall gates. | Timing rows imply portable superiority or methodology fields are missing. |
| Dead-code and coverage reports | `scripts/deadcode_*`, coverage Makefile targets, report docs | `make deadcode-report` or `make deadcode-check` for dead-code changes; `make coverage` only when coverage behavior changes and tools are available. | CI artifact/upload checks if workflow publication changes. | Coverage/dead-code rows are presented as solver correctness or product quality proof. |
| Generated API docs | `Doxyfile`, public header comments, API docs policy, `docs/api/html/` if selected | `make docs`; warning triage; page coverage check from Sprint 158 policy; `git diff --check`. | Full C/header gate if public headers changed. | Generated output is committed without policy, warnings are untriaged, or page coverage misses intended public headers. |
| Final claim audit | Public docs, maintainer guide, retrospectives, project plan | Claim-sensitive scan plus evidence-owner mapping; `git diff --check`. | Re-run selected evidence commands for every changed claim surface. | Any positive claim lacks recurring local/hosted/product-decision evidence. |

## Core Command Catalog

| Command | Primary use |
| --- | --- |
| `git diff --check` | Required for all changed tracked files. |
| `rg -n "[ \t]$" <new-untracked-doc-path> || true` | Required when new docs are untracked and therefore not covered by `git diff --check`. |
| `make format && make lint && make test` | Required when any `.c` or `.h` file changes. |
| `make source-list-check` | Source-list consistency when library sources or build metadata change. |
| `make quality-review-compile` | Reviewed Makefile compile-quality wrapper. |
| `make quality-review-cmake-compile` | CMake configure/build/CTest registration check. |
| `make quality-review-cmake` | CMake configure/build plus full CTest execution. |
| `make quality-review` | Reviewed Makefile quality path with tests and dead-code check. |
| `make quality-review-full` | Strong local baseline when broad changes justify it. |
| `bash tests/test_install.sh` | Unix Make install/`pkg-config` package proof. |
| `bash tests/test_cmake_install.sh` | CMake install/export and downstream `find_package(Sparse)` proof. |
| `bash scripts/static_package_deferral_check.sh` | Static-first package, `BUILD_SHARED_LIBS=ON` rejection, and ABI/shared metadata guard. |
| `python3 scripts/validate_corpus_schema.py` | Corpus schema and manifest validation. |
| `python3 scripts/normalize_report_index.py --check` | Normalized report-row construction validation. |
| `python3 scripts/normalize_report_index.py --check-freshness` | Cross-family freshness diagnostics. |
| `make report-index-oracle-freshness` | Selected local oracle regeneration and freshness check. |
| `make report-index-comparison-freshness` | Selected local comparison regeneration and freshness check. |
| `make bench-canonical-report` | Canonical benchmark report generation. |
| `make performance-sentinels` | Runtime sentinel report generation. |
| `make large-matrix-guardrails` | Large-matrix guardrail report generation. |
| `make bench-fast` | Fast benchmark runtime signal. |
| `make wall-check` | Bounded wall-time sentinel gate. |
| `make deadcode-report` | Classified dead-code report generation. |
| `make deadcode-check` | Dead-code report completeness check. |
| `make docs` | Doxygen API docs generation. |
| `make coverage` | Tree-mutating coverage report when coverage behavior is touched and tools are available. |

## Package And Build-System Quality Map

| Touched surface | Required checks | Review checklist |
| --- | --- | --- |
| Add/remove library source | `make source-list-check`; relevant Make/CMake build path; full C gate if C files changed. | Update `build-metadata/library_sources.txt`, `Makefile` `LIB_SRCS`, and `CMakeLists.txt` consistently. |
| Add/remove test | Full C gate; CMake registration check; Windows expected count reconciliation if CMake registered tests change. | Keep Makefile and CMake test lists aligned; update workflow `EXPECTED_WINDOWS_CTEST_COUNT` only with verified count. |
| Install rule or CMake export metadata | `bash tests/test_cmake_install.sh`; CMake parity check; `git diff --check`. | Imported target remains static, install-prefix paths do not leak source/build paths, no shared metadata appears. |
| `sparse.pc.in` | `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` if CMake install also installs `.pc`; `git diff --check`. | Description remains static archive scoped, link flags match install tests, no `Libs.private` or unsupported package/ABI wording appears. |
| `BUILD_SHARED_LIBS` rejection or package boundary docs | `bash scripts/static_package_deferral_check.sh`; affected docs checks. | Rejection names export/import, visibility, dynamic ABI, loader metadata, installed shared consumer, and runtime-loader blockers. |
| `VERSION` or generated version metadata | Install scripts and CMake package version checks. | Version remains single-sourced and exact-version package behavior is preserved. |
| README/INSTALL package wording | Docs checks plus static deferral guard if guard-controlled wording changes. | Static-first support is not widened into package-manager, shared-library, dynamic ABI, runtime-loader, or broad platform claims. |

## CI Reconciliation Checklist

When `.github/workflows/*.yml` changes, record all applicable items:

| Checklist item | Required reconciliation |
| --- | --- |
| Lane name | Name matches reviewed, supplemental, hosted, local-only, or advisory support tier. |
| Trigger | `push`/`pull_request` branches remain aligned with supported branch names. |
| Expected counts | CTest count, header count, generated row count, or report count is updated only with evidence. |
| Platform scope | Linux, macOS, and Windows comments match actual commands and runner images. |
| Package scope | Windows CMake install/downstream is not described as Windows Makefile or `pkg-config` execution parity. |
| Generated artifacts | Upload/summary behavior matches selected hosted evidence policy. |
| Advisory rows | Coverage, dead-code, benchmark, large-matrix, and optional-data rows remain advisory/supplemental unless selected. |
| Docs references | README, INSTALL, maintainer guide, benchmark docs, and corpus docs match workflow support-tier wording. |
| Local equivalent | A local equivalent command is run or explicitly recorded as unavailable. |
| Failure semantics | Stale/missing/failing selected rows fail the lane; skips/defer rows remain policy-bound. |

## Generated Evidence Quality Map

| Generated family | Freshness command | Claim-bearing status before selection | Promotion requirement |
| --- | --- | --- | --- |
| API HTML | `make docs` plus Sprint 158 warning/page checks | Local convenience unless Sprint 158 publishes or guards policy. | Publication decision, warning triage, page coverage, docs alignment. |
| Oracle rows | `make report-index-oracle-freshness` | Local-only generated compare inputs. | Hosted selected freshness lane or retained local-only policy. |
| Comparison rows | `make report-index-comparison-freshness` | Local-only generated compare inputs. | Hosted selected freshness lane plus fixture-local metric contract. |
| Normalized report index | `python3 scripts/normalize_report_index.py --check-freshness` | Local-only advisory unless required families fail. | Selected family semantics and support-tier update. |
| Canonical benchmark reports | `make bench-canonical-report` | Local-only advisory. | Methodology-bound selection, row classification, caveat fields. |
| Sentinels | `make performance-sentinels`; `make wall-check` for hard wall gate | Mixed local advisory and bounded hard gate. | Explicit selected rows and threshold semantics. |
| Large-matrix guardrails | `make large-matrix-guardrails` | Local-only reviewed/supplemental rows. | Selected guardrail policy and docs caveats. |
| Dead-code report | `make deadcode-report`; `make deadcode-check` | Local advisory with reviewed completeness checks where CI owns it. | Do not convert to solver correctness proof. |
| Coverage | `make coverage` | Supplemental tree-mutating local signal. | Do not convert to completeness or product quality claim without policy. |

## Claim-Sensitive Scan Terms

Use targeted `rg` scans when public claim wording changes. The scan should
inspect changed docs plus relevant owner docs.

| Claim risk | Example scan terms |
| --- | --- |
| State-of-the-art | `state-of-the-art|best|superior|competitive|production-grade` |
| External parity | `parity|LAPACK|SuiteSparse|Eigen|PETSc|Trilinos|SciPy|NumPy` |
| Performance | `performance|faster|speed|throughput|scalability|portable` |
| Package and ABI | `package-manager|Homebrew|apt|dnf|pacman|vcpkg|conan|shared|ABI|runtime-loader|SONAME|DLL|dylib` |
| Windows parity | `Windows|Makefile|pkg-config|parity|MSVC|MinGW` |
| Generated evidence | `generated|freshness|hosted|local-only|advisory|supplemental|coverage|dead-code|benchmark` |

Finding these terms is not a failure by itself. The check fails only when the
wording creates a positive claim without the Day 9 evidence contract fields.

## Day 11 Inputs

Day 11 should use this quality map to build the claim target register:

- accepted claims should name the required validation owner;
- rejected claims should name the quality gate or evidence gap that keeps them
  rejected;
- docs that must move together should be listed with each claim;
- C/header, package, CI, generated-report, and documentation-only changes
  should keep their validation expectations separate.

## Completion Check

- Validation expectations are explicit before implementation sprints begin.
- C/header gates are separated from documentation-only checks.
- Package and build-system changes have focused install/export and static
  deferral checks.
- CI changes have a reconciliation checklist for names, counts, support tiers,
  artifacts, and hosted-log semantics.
- Generated evidence remains local-only/advisory unless a selected promotion
  gate changes its support tier.
