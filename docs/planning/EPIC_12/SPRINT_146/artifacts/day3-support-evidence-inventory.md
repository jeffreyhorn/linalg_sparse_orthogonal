# Day 3 Support Evidence Inventory

## Scope

Day 3 inventories the non-numerical Epic 12 evidence families: report indexes,
runtime/backend governance, package and ABI posture, platform lanes, adoption
surfaces, and validation evidence. It ties each support claim to current
source-controlled owners or external CI lanes and identifies the remaining
support-tier gaps that Sprint 146 must reconcile or publish as residuals.

## Support Evidence Owner Table

| Evidence Area | Supported Claim | Source-Controlled Owner | External Or Generated Evidence | Validation Command | Support Tier |
| --- | --- | --- | --- | --- | --- |
| Report family metadata | Report families have maintained row meanings, row origins, freshness policies, support tiers, owners, claim scopes, and non-claims. | `tests/corpus/manifests/report_families.tsv`; `tests/corpus/schemas/report_index_fields.md` | normalized local indexes under `build/report-index/` | `python3 scripts/validate_corpus_schema.py`; `python3 scripts/normalize_report_index.py --check` | source-controlled metadata, mostly local-only/advisory rows |
| Report freshness diagnostics | Freshness checks distinguish source-controlled rows, generated rows, stale rows, missing generated rows, advisory rows, required-generated failures, skips, and defers. | `scripts/normalize_report_index.py`; `tests/test_normalize_report_index.py`; `tests/corpus/schemas/report_index_fields.md` | generated report indexes under ignored `build/` paths | `python3 tests/test_normalize_report_index.py`; `python3 scripts/normalize_report_index.py --check-freshness` | local generated diagnostics unless backed by hosted logs |
| Runtime/backend governance | Runtime/backend behavior has a maintained control-boundary contract for public typed options, maintainer-only controls, compatibility env vars, fallback semantics, and sentinel interpretation. | `docs/maintainer_guide.md`; `benchmarks/README.md`; `docs/algorithm.md`; `docs/cookbook.md`; `README.md`; `tests/corpus/manifests/report_families.tsv` | local sentinel output under `build/bench-reports/sentinels/` | focused backend tests; `make performance-sentinels`; report-index freshness checks | local-only governance and sentinel evidence |
| Runtime/backend sentinels | Selected sentinel rows provide local hard/advisory regression visibility without portable timing claims. | `scripts/performance_sentinels.sh`; `tests/test_normalize_report_index.py`; `tests/corpus/schemas/report_index_fields.md` | generated sentinel TSVs under ignored `build/bench-reports/sentinels/` | `make performance-sentinels`; `python3 scripts/normalize_report_index.py --family sentinel --check-freshness` | local-only; `S5` hard local gate, `S2`/`S3` advisory context |
| Static-first package contract | The project has an explicit maintained static archive package contract with install/export metadata, downstream consumer proof, exact-version proof, and unsupported shared-artifact checks. | `CMakeLists.txt`; `sparse.pc.in`; `scripts/static_package_deferral_check.sh`; `tests/test_install.sh`; `tests/test_cmake_install.sh`; `README.md`; `INSTALL.md`; `docs/maintainer_guide.md` | local install proof outputs; hosted CI package logs | `bash scripts/static_package_deferral_check.sh`; `bash tests/test_install.sh`; `bash tests/test_cmake_install.sh` | reviewed Linux source of truth, reviewed macOS static-first install/export, Windows CMake-first subset plus supplemental CMake install/downstream |
| Package report rows | Package rows identify proof-owner commands and static-first support boundaries without implying a fresh install just ran. | `tests/corpus/manifests/report_families.tsv`; `scripts/normalize_report_index.py`; `tests/test_normalize_report_index.py` | generated normalized index rows under `build/report-index/` | `python3 scripts/normalize_report_index.py --family package --check`; `--check-freshness` | source-controlled advisory metadata |
| Linux platform lane | Linux remains the strongest reviewed source-of-truth baseline for Make, CMake, package contract, quality, dead-code/report, and broad local proof. | `.github/workflows/ci.yml`; `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; report-family CI rows | hosted GitHub Actions logs | CI workflow execution plus local source-of-truth commands | reviewed source-of-truth baseline |
| macOS platform lane | macOS has reviewed static-first Make install/`pkg-config` and CMake install/export proof for the maintained static archive package contract. | `.github/workflows/macos-ci.yml`; `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; `tests/test_install.sh`; `tests/test_cmake_install.sh` | hosted `macos-latest` CI logs | hosted macOS workflow plus install scripts | reviewed static-first install/export proof |
| Windows platform lane | Windows has reviewed MSVC CMake subset proof and supplemental CMake install/downstream confidence, with staged POSIX/pthread blockers explicit. | `.github/workflows/windows-ci.yml`; `README.md`; `INSTALL.md`; `docs/maintainer_guide.md`; CMake test registration | hosted Windows CI logs | hosted Windows CMake build/test and supplemental install/downstream block | reviewed CMake-first subset plus supplemental install/downstream confidence |
| Adoption front door | README, INSTALL, examples, cookbook, solver-selection, diagnostics, static-first install, advanced controls, and selected headers now present a clearer first-use workflow without widening claims. | `README.md`; `INSTALL.md`; `examples/README.md`; `docs/cookbook.md`; `docs/solver_selection.md`; selected public headers | none, except example/install validation outputs | `make examples-build`; install validation; docs claim audit; full C gate when headers change | source-controlled documentation and public-header guidance |
| Header adoption cleanup | Four high-impact public headers were cleaned without intentional public declaration drift. | `include/sparse_matrix.h`; `include/sparse_iterative.h`; `include/sparse_qr.h`; `include/sparse_svd.h` | local declaration-preservation scans from Sprint 145 | `make format && make lint && make test` when touched | public API documentation cleanup only |
| Validation package | Sprint 146 final validation must aggregate local quality, corpus/report/package/adoption checks, and hosted CI reconciliation. | Sprint 137-145 validation artifacts; Makefile targets; shell/Python validation scripts; workflows | generated local logs and hosted CI logs | day-specific Sprint 146 validation matrix | final support depends on evidence family and platform lane |

## Report Evidence Inventory

The maintained report-index claim is source-controlled through
`tests/corpus/manifests/report_families.tsv`,
`tests/corpus/schemas/report_index_fields.md`,
`scripts/normalize_report_index.py`, and `tests/test_normalize_report_index.py`.
Report rows are not all pass/fail evidence:

- `corpus` rows are source-controlled metadata and expected-result contracts.
- `oracle`, `benchmark`, `sentinel`, `guardrail`, `deadcode`, and `coverage`
  rows are generated local evidence unless hosted logs explicitly promote the
  corresponding lane.
- `package`, `ci`, `documentation`, and `runtime_backend` rows are
  source-controlled advisory or lane-definition metadata.
- `missing_generated` rows expose absent local reports without manufacturing
  pass evidence.

Final Sprint 146 wording may claim maintained report navigation, row
semantics, and freshness diagnostics. It may not claim generated report
freshness from source-controlled rows, coverage completeness, zero dead code,
release benchmark proof, or hosted CI proof from local normalized indexes.

## Runtime And Backend Governance Inventory

Sprint 142 closed governance, not runtime performance portability. The
source-controlled support package records:

- public typed controls remain the caller-facing API surface;
- compatibility environment variables, benchmark/test opt-ins, dense-helper
  selectors, SVD low-rank env selection, FM/debug variables, OpenMP runtime
  context, and package/link controls remain maintainer-only or deferred unless
  a later sprint promotes them;
- explicit typed options precede AUTO/default semantics, compatibility env
  overrides, compile-time flags, fallback behavior, and maintainer/report
  context;
- `make performance-sentinels` owns local sentinel visibility;
- `S5` remains the hard local sentinel gate, while `S2` and `S3` are advisory
  context rows.

Final wording may claim maintained governance and local sentinel visibility.
It may not claim backend portability, optional-backend availability, portable
timing, benchmark superiority, package/ABI closure, or state-of-the-art
status.

## Package And Platform Inventory

The package decision is static-first:

- static archive install/export metadata is the maintained package contract;
- `pkg-config` metadata describes static archive package metadata;
- CMake exports provide `Sparse::sparse_lu_ortho`;
- downstream consumer proof covers Make/`pkg-config`, maintained examples,
  CMake package config, exact-version config/build/run, and mismatched-version
  rejection;
- `scripts/static_package_deferral_check.sh` guards against shared-library,
  runtime-loader, package-manager, unsupported ABI, and selector wording drift.

Platform support is tiered:

- Linux is the strongest reviewed source-of-truth baseline.
- macOS has reviewed static-first Make install/`pkg-config` and CMake
  install/export proof for the maintained static archive package contract.
- Windows is CMake-first with reviewed MSVC CMake subset proof and
  supplemental CMake install/downstream confidence.
- Windows Makefile parity, Windows `pkg-config` parity, reviewed Windows
  install-validation parity, and staged `test_threads`,
  `test_sprint4_integration`, and `test_fuzz` closure remain residuals.

Final wording may not imply shared-library ABI support, dynamic loader
support, package-manager distribution, static/shared selectors, broad platform
parity, or portable performance.

## Adoption Evidence Inventory

Sprint 145 made adoption simpler without creating new product capabilities.
The source-controlled adoption owners are:

- `README.md`: shortest first-use route and public support-tier summary;
- `INSTALL.md`: static-first install and downstream consumption;
- `examples/README.md`: maintained example ladder;
- `docs/cookbook.md`: data-first and workflow recipes;
- `docs/solver_selection.md`: problem-shape routing with bounded QR and
  partial-SVD evidence;
- selected public headers: ownership, NULL/error, QR, SVD, and iterative
  documentation cleanup;
- `tests/corpus/manifests/report_families.tsv` and
  `tests/corpus/schemas/report_index_fields.md`: adoption-adjacent report row
  coherence.

Final wording may claim a clearer first-use workflow and aligned adoption
surfaces. It may not claim tutorial completion, all-header cleanup, broad
numerical parity, Windows parity, shared-library ABI support, package-manager
distribution, portable performance, or state-of-the-art status.

## Support-Tier Gaps For Sprint 146 Reconciliation

| Gap | Current Evidence Owner | Required Sprint 146 Handling |
| --- | --- | --- |
| Hosted CI reconciliation | `.github/workflows/*.yml`; PR/CI logs outside source control | Day 6-7 must reconcile final Linux, macOS, and Windows runs or record hosted-only uncertainty. |
| Generated benchmark/coverage/dead-code/sentinel freshness | report families and generated `build/` outputs | Final closeout must avoid freshness claims unless regenerated evidence is present and checked. |
| Windows install-validation parity | Windows workflow and package docs | Keep as residual unless hosted reviewed install-validation parity is explicitly earned. |
| Windows Makefile and `pkg-config` parity | Windows workflow/docs | Keep as residual; current support remains CMake-first. |
| Windows staged pthread/POSIX tests | `test_threads`, `test_sprint4_integration`, `test_fuzz` blocker notes | Publish blockers and promotion gates. |
| Shared-library ABI and loader support | package/ABI docs and static deferral guard | Keep as residual unless a full shared-library ABI product sprint exists. |
| Package-manager distribution | package docs | Keep as residual; no package-manager availability claim. |
| Tutorial alignment | Sprint 145 residual-debt ledger | Keep as documentation residual unless directly updated before closeout. |
| Broader public-header cleanup | public headers outside Sprint 145 selection | Keep as residual with selected-header cleanup clearly bounded. |
| Portable performance and competitive positioning | benchmark/runtime docs and Epic 12 final audit | Keep as non-claim unless direct comparative evidence appears. |

## Day 4 Handoff

Day 4 should turn the Day 2 and Day 3 inventories into a final validation
baseline design. It should select the strongest feasible local commands,
identify which commands are required only when surfaces change, and define how
hosted CI evidence will be reconciled without confusing generated local rows
with source-controlled pass proof.
