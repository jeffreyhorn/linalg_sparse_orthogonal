# Sprint 177 Day 4: Repository Surface Inventory

**Sprint:** 177 - Epic 16 Baseline, Evidence Matrix & Closure Gates
**Epic source:** `docs/planning/EPIC_16/PROJECT_PLAN.md`
**Requested sprint path:** `docs/planning/EPIC_15/SPRINT_177/`
**Status:** Complete

## Purpose

Inventory the concrete repository surfaces that future Epic 16 closure targets
must touch. This artifact is intentionally an inventory only: no code,
workflow, test, or public documentation behavior was changed.

## Inventory Method

- Reviewed the Sprint 177 Day 4 plan and prior Day 2-3 residual artifacts.
- Listed current source, public header, test, script, benchmark, example,
  package, and workflow surfaces.
- Scanned build registration, generated-report, generated API, package,
  platform-tier, and deferral guard ownership.
- Ranked large source/test/documentation files by line count to identify
  review-surface risk.

## Public Header Surface

The public include surface contains 18 checked-in headers plus the generated
version template:

- Matrix/core: `sparse_types.h`, `sparse_matrix.h`, `sparse_vector.h`,
  `sparse_csr.h`, `sparse_dense.h`
- Direct solvers: `sparse_lu.h`, `sparse_lu_csr.h`, `sparse_cholesky.h`,
  `sparse_ldlt.h`, `sparse_qr.h`
- Iterative/preconditioners: `sparse_iterative.h`, `sparse_ilu.h`,
  `sparse_ic.h`
- Spectral/decompositions: `sparse_svd.h`, `sparse_eigs.h`,
  `sparse_bidiag.h`
- Ordering/analysis: `sparse_reorder.h`, `sparse_analysis.h`
- Generated version surface: `sparse_version.h.in`

Header cleanup and API-reference closure should treat these as the only
installed public API surface unless a later sprint explicitly adds a new
public header.

## Implementation Surface

The `src/` root currently contains 69 C/header files. The main ownership
clusters are:

| Cluster | Primary files | Closure relevance |
| --- | --- | --- |
| Allocation and shared internals | `sparse_alloc_internal.*`, `sparse_factor_state_internal.*` | allocation-failure evidence, cleanup contracts |
| Matrix and vector core | `sparse_matrix*.c`, `sparse_vector.c`, `sparse_csr.c`, `sparse_dense.c` | public API correctness, construction-path non-claims |
| LU and LDLT | `sparse_lu.c`, `sparse_lu_csr*.c`, `sparse_ldlt*.c` | direct solver claim boundaries |
| QR | `sparse_qr.c`, `sparse_qr_householder.c` | QR corpus and comparison evidence |
| SVD/eigs | `sparse_svd*.c`, `sparse_eigs*.c` | partial-SVD/eigensolver evidence boundaries |
| Iterative solvers | `sparse_iterative*.c`, `sparse_ilu.c`, `sparse_ic.c` | allocation-failure gate, runtime backend governance |
| Ordering and graph | `sparse_reorder*.c`, `sparse_colamd.c`, `sparse_graph*.c` | graph/order evidence and large-test surfaces |

## Test Surface

The test root contains 66 direct test scripts/sources. Important closure owners
are:

| Owner | Files | Notes |
| --- | --- | --- |
| Solver families | `test_qr.c`, `test_qr_corpus.c`, `test_svd.c`, `test_svd_partial_corpus.c`, `test_iterative.c`, direct solver tests | family-local correctness and residual evidence |
| Report freshness | `test_normalize_report_index.py`, `test_selected_comparison_workflow.py`, `test_run_external_comparison.py`, `test_bench_canonical_freshness.py` | workflow/report fail-closed contracts |
| Install/package proof | `test_install.sh`, `test_cmake_install.sh` | static-first downstream and metadata proof |
| External references | `qr_external_dense_reference.py`, `svd_external_dense_reference.py`, `lu_external_dense_reference.py` | selected comparison oracles |
| Portability | `test_threads.c`, `test_sprint4_integration.c`, `test_fuzz.c` | Windows promoted-test history and remaining portability context |

## Scripts And Generated Evidence Owners

| Surface | Owner files | Primary commands |
| --- | --- | --- |
| Normalized report index | `scripts/normalize_report_index.py`, `tests/test_normalize_report_index.py`, report manifest data | `python3 scripts/normalize_report_index.py --check-freshness` |
| Selected oracle freshness | `scripts/run_corpus_oracle.py`, corpus data, report-index tests | `make report-index-oracle-freshness` |
| Selected comparison freshness | `scripts/run_external_comparison.py`, selected reference helpers, workflow guard test | `make report-index-comparison-freshness` |
| Canonical benchmark reports | `scripts/bench_canonical_report.sh`, `scripts/check_bench_canonical_freshness.py`, benchmark docs | `make bench-canonical-report-freshness` |
| Local performance sentinels | `scripts/performance_sentinels.sh`, `scripts/wall_check.sh` | `make performance-sentinels`, `make wall-check` |
| Generated API docs | `Doxyfile`, `scripts/check_api_docs_coverage.py`, `scripts/check_api_docs_local_only.sh` | `make api-docs-freshness` |
| Corpus schema | `scripts/validate_corpus_schema.py`, corpus data | `python3 scripts/validate_corpus_schema.py` |

## Build, Package, And Workflow Owners

| Surface | Owner files | Current guard shape |
| --- | --- | --- |
| Make source/test/bench registration | `Makefile` | `LIB_SRCS`, `TEST_SRCS`, `BENCH_SRCS`, `make test`, `make lint` |
| CMake registration | `CMakeLists.txt` | static `sparse_lu_ortho`, `add_sparse_test`, install/export rules |
| Source-list parity | `build-metadata/library_sources.txt` | synchronized with Makefile and CMake library source order |
| Static-first package contract | `Makefile`, `CMakeLists.txt`, `sparse.pc.in`, `cmake/SparseConfig.cmake.in` | install tests plus static deferral guard |
| Package-manager deferral | `scripts/package_manager_deferral_check.sh`, public docs | package-provider wording and recipe non-claim guard |
| Shared-library and ABI deferral | `scripts/static_package_deferral_check.sh`, CMake rejection, public docs | `BUILD_SHARED_LIBS=ON` rejection and metadata checks |
| Linux CI | `.github/workflows/ci.yml` | reviewed quality, package, selected reports, benchmark freshness |
| macOS CI | `.github/workflows/macos-ci.yml` | reviewed static-first install/export and selected comparison freshness |
| Windows CI | `.github/workflows/windows-ci.yml` | reviewed CMake-first tests and static-first CMake install/downstream proof |

## Large Review Surfaces

The largest current review surfaces by line count are concentrated in solver
tests and generated-evidence tooling:

| File | Lines | Risk |
| --- | ---: | --- |
| `tests/test_qr.c` | 3970 | QR changes have high review cost and should prefer narrow additions. |
| `tests/test_ldlt_csc.c` | 3915 | Dense direct-solver test surface; split only with clear owner seams. |
| `tests/test_integration.c` | 3279 | Broad integration file; avoid unrelated coverage additions here. |
| `tests/test_svd.c` | 3029 | SVD coverage is large and should favor helper reuse. |
| `tests/test_ldlt.c` | 3006 | Direct solver behavior and regressions are tightly coupled. |
| `tests/test_etree.c` | 2962 | Ordering/analysis review surface is already large. |
| `tests/test_iterative.c` | 2929 | Allocation-failure proof currently lives in a large iterative test file. |
| `tests/test_graph.c` | 2764 | Graph evidence should stay family-local. |
| `tests/test_chol_csc.c` | 2554 | CSC direct solver evidence has high review surface. |
| `src/sparse_ldlt_csc.c` | 2095 | Implementation changes here need narrow validation commands. |
| `scripts/run_external_comparison.py` | 2018 | Comparison expansion can increase tooling maintenance risk. |
| `tests/test_normalize_report_index.py` | 1786 | Report-index gates are central and should avoid silent false positives. |
| `docs/maintainer_guide.md` | 1739 | Public support wording is broad and easy to drift. |

## Duplicated Registration And Drift Risks

- Library source order is represented in `Makefile`, `CMakeLists.txt`, and
  `build-metadata/library_sources.txt`; source-list drift should stay guarded.
- Test registration is split between `Makefile` `TEST_SRCS`,
  `CMakeLists.txt` `add_sparse_test`, and Windows expected CTest count.
- Selected report targets are repeated across `Makefile`, report scripts,
  normalized report tests, workflow upload blocks, and public docs.
- Static-first and package-manager non-claims are repeated in `README.md`,
  `INSTALL.md`, `docs/maintainer_guide.md`, package metadata templates,
  shell guards, and workflow comments.
- Generated API freshness depends on Doxygen inputs, public header docs,
  local-only guard scripts, and public documentation wording.
- Workflow artifact upload fail-closed behavior must stay scoped to the exact
  upload block for each promoted lane, not merely present somewhere in a file.

## Platform Support-Tier Wording Locations

Support-tier wording currently spans:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `scripts/static_package_deferral_check.sh`
- `scripts/package_manager_deferral_check.sh`
- `scripts/check_lu_header_docs_guard.sh`

Any Epic 16 claim recalibration should update these locations as one wording
set or explicitly document why a location is out of scope.

## Residual-To-Surface Map

| Residual theme | Primary files | Validation candidates |
| --- | --- | --- |
| Evidence matrix and claim gates | new Sprint 177 artifacts, `docs/maintainer_guide.md`, `README.md` | `git diff --check`, targeted docs grep |
| Generated API HTML publication | `Doxyfile`, `docs/api_reference.md`, generated API scripts | `make api-docs-freshness` |
| Hosted report freshness | workflows, `scripts/normalize_report_index.py`, report tests | `make report-index-oracle-freshness`, `make report-index-comparison-freshness` |
| Package/provider boundary | install templates, install tests, package deferral scripts | `bash tests/test_install.sh`, `bash tests/test_cmake_install.sh`, deferral scripts |
| Shared-library/ABI decision | `CMakeLists.txt`, package templates, deferral docs | `bash scripts/static_package_deferral_check.sh` |
| Allocation-failure evidence | `src/sparse_iterative.c`, `tests/test_iterative.c` | `make iterative-allocation-failure-gate`, `make test` |
| Public header/API coherence | `include/*.h`, `docs/api_reference.md`, `docs/tutorial.md` | `make docs-check`, `make api-docs-freshness` |
| Review-surface reduction | large tests, report tooling, maintainer guide | targeted file splits plus full quality checks when code changes |

## Completion Criteria Check

- Planned closure targets now map to concrete files and commands.
- Duplicated target-list and workflow/report drift risks are visible.
- Large review surfaces are identified before target selection.
- No code, workflow, test, or public documentation behavior was changed by
  this inventory.
