# Sprint 167 Day 5: Test And Corpus Surface Inventory

## Purpose

Day 5 inventories test, corpus, oracle, comparison, and report evidence
surfaces. The goal is to group coverage by solver family, separate hosted
proof from local-only/advisory evidence, and prepare concrete comparison
family candidates for later Epic 15 selection.

## Inventory Commands

The Day 5 inventory used these local repository scans:

- `find tests -maxdepth 2 -type f | sort`
- `git ls-files tests scripts docs README.md Makefile CMakeLists.txt .github/workflows`
- `sed -n '1,220p' tests/corpus/manifests/report_families.tsv`
- `sed -n '1,220p' tests/corpus/manifests/fixtures.tsv`
- `rg -n "TEST_SRCS|add_sparse_test|report-index|comparison|oracle" Makefile CMakeLists.txt .github/workflows/*.yml`
- line-count scan across `tests/test_*.c`

These scans are planning evidence only. They do not prove the tests pass on
the current branch.

## Test Surface Counts

| Surface | Count | Notes |
| --- | ---: | --- |
| Tracked C proof-owner tests matching `tests/test_*.c` | 59 | Matches current Windows CTest expected count in `.github/workflows/windows-ci.yml`. |
| Tracked test helper headers in `tests/*.h` | 13 | Shared fixtures and helper APIs for direct solvers, QR, SVD, graph, thread, and iterative tests. |
| Tracked corpus files under `tests/corpus/` | 20 | Manifests, schemas, expected rows, and corpus README. |
| External dense reference helpers | 5 | Cholesky, LDLT, LU, QR, and SVD Python helpers. |
| Corpus/report/comparison scripts | 6 | Includes schema validation, corpus oracle generation, external comparison, report normalization, benchmark report, and dead-code report scripts. |

## Solver-Family Test Map

| Family | Primary tests | Related helpers | Evidence implication |
| --- | --- | --- | --- |
| Matrix core, vector, I/O, arithmetic | `test_sparse_matrix.c`, `test_sparse_vector.c`, `test_sparse_io.c`, `test_sparse_arith.c`, `test_edge_cases.c`, `test_known_matrices.c` | `test_framework.h`, `test_integration_fixtures.h` | Core functional evidence; not package, ABI, performance, or external parity proof by itself. |
| LU and LU CSR | `test_sparse_lu.c`, `test_lu_csr.c`, `test_direct_csc_dispatch.c`, `test_direct_csc_regression.c`, `test_integration.c` | `test_direct_solver_helpers.h`, `lu_external_dense_reference.py` | Strong candidate for future bounded external comparison or allocation-failure proof. |
| Cholesky and LDLT | `test_cholesky.c`, `test_chol_csc.c`, `test_chol_csc_supernodal.c`, `test_ldlt.c`, `test_ldlt_csc.c`, `test_ldlt_backend_dispatch.c` | `test_chol_csc_supernodal_helpers.h`, `chol_external_dense_reference.py`, `ldlt_external_dense_reference.py` | Strong candidate set for direct-solver comparison, package confidence, and allocation-failure evidence. |
| QR | `test_qr.c`, `test_qr_solve.c`, `test_qr_corpus.c`, `test_sprint6_integration.c` | `test_qr_helpers.h`, `qr_external_dense_reference.py` | Existing QR corpus and comparison evidence is selected/fixture-local; good follow-up candidate if a new comparison family is selected. |
| SVD and partial SVD | `test_svd.c`, `test_svd_partial_corpus.c`, `test_bidiag.c`, `test_sprint8_integration.c` | `test_svd_helpers.h`, `test_svd_partial_helpers.h`, `test_svd_partial_shared_helpers.h`, `svd_external_dense_reference.py` | Existing partial-SVD corpus/comparison evidence is selected/fixture-local; good follow-up candidate if subspace-safe metrics can be bounded. |
| Iterative solvers and preconditioners | `test_iterative.c`, `test_minres.c`, `test_bicgstab.c`, `test_bicgstab_block.c`, `test_block_solvers.c`, `test_stagnation.c`, `test_ilu.c`, `test_ic.c`, `test_omp.c` | `test_iterative_handle_helpers.h`, `test_solver_helpers.h` | Relevant to hosted performance and backend governance; comparison expansion would need careful residual/tolerance policy. |
| Eigensolvers | `test_eigs.c`, `test_eigs_thick_restart.c`, `test_eigs_lobpcg.c` | existing test framework helpers | Relevant to numerical corpus expansion but lower priority than QR/SVD/direct-solver residuals for Epic 15. |
| Reorder, graph, symbolic analysis | `test_reorder.c`, `test_reorder_nd.c`, `test_reorder_amd_qg.c`, `test_colamd.c`, `test_etree.c`, `test_graph.c`, `test_graph_fm_buckets.c` | `test_graph_fixtures.h` | Important support algorithms; comparison/performance claims should avoid solver parity overreach. |
| Cross-solver and framework | `test_cross_solver_oracle.c`, `test_framework_optin.c`, sprint integration tests | `test_framework.h`, `test_solver_helpers.h` | Good evidence glue but not broad external parity proof. |
| Package/install | shell tests `test_install.sh`, `test_cmake_install.sh` | CMake example and installed consumers | Static-first package evidence, not package-manager, dynamic ABI, or shared-library support. |
| Report/comparison scripts | `test_normalize_report_index.py`, `test_run_external_comparison.py` | dense reference helpers and report scripts | Report infrastructure evidence; generated rows remain scoped by support tier. |

## Largest Test Files

Large test files indicate where coverage is deep but also where future changes
may have high review cost.

| Rank | Test file | Lines | Surface |
| ---: | --- | ---: | --- |
| 1 | `tests/test_qr.c` | 3970 | QR factorization and solve behavior |
| 2 | `tests/test_ldlt_csc.c` | 3915 | CSC LDLT behavior |
| 3 | `tests/test_integration.c` | 3279 | Cross-feature integration |
| 4 | `tests/test_svd.c` | 3029 | SVD behavior |
| 5 | `tests/test_ldlt.c` | 3006 | LDLT behavior |
| 6 | `tests/test_etree.c` | 2962 | Elimination tree behavior |
| 7 | `tests/test_iterative.c` | 2924 | Iterative solver behavior |
| 8 | `tests/test_graph.c` | 2764 | Graph algorithms |
| 9 | `tests/test_chol_csc.c` | 2554 | CSC Cholesky behavior |
| 10 | `tests/test_chol_csc_supernodal.c` | 2504 | Supernodal Cholesky behavior |

## Maintained Corpus Map

| Corpus surface | Source-controlled owner | Current scope |
| --- | --- | --- |
| Fixture manifest | `tests/corpus/manifests/fixtures.tsv` | Maintained QR and partial-SVD fixture rows with explicit support tiers and non-claims. |
| Generator manifest | `tests/corpus/manifests/generators.tsv` | Deterministic generated-matrix metadata and hashes. |
| Optional data manifest | `tests/corpus/manifests/optional_data.tsv` | Optional external data skip/defer policy. |
| Expected result rows | `tests/corpus/expected/*.tsv` | Source-controlled expected values and status rows for selected fixtures. |
| Report-family manifest | `tests/corpus/manifests/report_families.tsv` | Contract rows for corpus, oracle, benchmark, sentinel, package, CI, documentation, runtime backend, and comparison report families. |
| Schemas | `tests/corpus/schemas/*.md` | Field semantics for fixtures, oracle rows, and normalized report indexes. |
| Corpus README | `tests/corpus/README.md` | Interpretation rules and non-claim boundaries for generated/local/hosted rows. |

Current fixture families are concentrated in:

- QR rank-deficient rectangular fixtures;
- QR underdetermined minimum-norm fixtures;
- partial-SVD clustered/repeated fixtures;
- partial-SVD rank-deficient projector fixtures;
- partial-SVD sparse low-rank output fixtures;
- partial-SVD fail-closed/recovery fixtures.

## Oracle And Comparison Surface Map

| Surface | Command owner | Generated output | Support tier |
| --- | --- | --- | --- |
| Corpus schema validation | `python3 scripts/validate_corpus_schema.py` | n/a | Source-controlled schema/advisory proof. |
| Selected oracle freshness | `make report-index-oracle-freshness` | `build/corpus/oracle/*.tsv` and normalized report rows | Local generated by default; reviewed Linux hosted lane owns selected rows after CI passes. |
| Selected comparison freshness | `make report-index-comparison-freshness` | `build/comparison/{qr_minnorm,qr_compatible_ls,partial_svd_diag6_k2}/*` | Local generated by default; reviewed Linux hosted lane owns selected comparison families after CI passes. |
| Report normalization | `python3 scripts/normalize_report_index.py` | normalized index rows under ignored build paths | Infrastructure proof; not solver correctness by itself. |
| External comparison runner | `python3 scripts/run_external_comparison.py --target ...` | selected comparison study rows | Fixture-local comparison only. |
| Corpus oracle runner | `python3 scripts/run_corpus_oracle.py ...` | selected observed oracle rows | Fixture-local oracle only. |

## Selected Comparison Families

The current selected comparison freshness target regenerates three families:

| Target | Fixture | Comparator | Claim boundary |
| --- | --- | --- | --- |
| `qr-minnorm` | `qr_underdetermined_minnorm_2x4` | source-controlled dense QR reference helper | One local fixture-level QR minimum-norm comparison. |
| `qr-compatible-ls` | `qr_overdetermined_compatible_5x3` | source-controlled dense QR reference helper | One local fixture-level QR compatible least-squares comparison. |
| `partial-svd-diag6-k2` | `partial_svd_diag6_k2` | source-controlled dense SVD singular-value reference helper | One local fixture-level partial-SVD diagonal top-k comparison with subspace-safe metrics. |

These families do not prove broad QR, SVD, or partial-SVD correctness; raw
basis/vector identity; external-library parity; platform support; package/ABI
support; performance superiority; release readiness; or state-of-the-art
status.

## Candidate Epic 15 Comparison Families

| Priority | Candidate | Why it is a good candidate | Required evidence |
| ---: | --- | --- | --- |
| 1 | LU CSR dense solve comparison | LU CSR is large, allocation-heavy, core to direct solving, and has a dense reference helper. | Small source-controlled fixture, dense reference expected values, residual/max-delta metrics, generated comparison rows, freshness check, focused tests. |
| 2 | LDLT CSC SPD solve comparison | LDLT CSC is the largest implementation file and has existing dense reference support. | SPD or semidefinite fixture, factor/solve comparison metrics, tolerance policy, generated rows, focused tests. |
| 3 | Cholesky CSC SPD solve comparison | Cholesky is important and has dense reference helper support. | SPD fixture, solve/residual metrics, generated comparison row family, freshness check. |
| 4 | QR rank-deficient least-squares follow-up | Builds on existing QR corpus and dense QR reference helper. | New fixture distinct from prior QR families, subspace-safe metrics, non-parity wording. |
| 5 | Partial-SVD clustered/repeated follow-up | Builds on existing partial-SVD corpus and SVD dense helper. | Subspace/projector-safe metrics, repeated-spectrum caveats, generated rows, focused tests. |

Day 5 recommends LU CSR, LDLT CSC, and Cholesky CSC as the strongest
additional comparison candidates if Epic 15 wants to diversify beyond the
already-covered QR/partial-SVD comparison families.

## Platform And Test-Scope Classification

| Evidence class | Owner | Current interpretation |
| --- | --- | --- |
| Full local C quality gate | `make format && make lint && make test` | Required when `.c` or `.h` files change; local platform proof only. |
| CMake test registration | `CMakeLists.txt` and CTest | Registers 59 tests, including portable Windows-promoted tests. |
| Windows reviewed CMake subset | `.github/workflows/windows-ci.yml` | Current expected Windows CTest count is `59`; CMake-first proof, not Makefile/pkg-config parity. |
| Linux reviewed hosted oracle/comparison freshness | `.github/workflows/ci.yml` | Selected oracle and selected comparison rows only; not broad report-family hosting. |
| macOS reviewed lanes | `.github/workflows/macos-ci.yml` | Platform-specific hosted proof only for configured macOS checks. |
| Source-controlled corpus metadata | `tests/corpus/**` | Advisory or prerequisite evidence until generated rows pass. |
| Generated local report rows | ignored `build/` paths | Local-only unless a reviewed hosted lane owns the exact selected row family. |
| Package shell tests | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Static-first source/package proof; not package-manager distribution or dynamic ABI support. |
| Optional external data rows | `tests/corpus/manifests/optional_data.tsv` | Skip/defer policy only; not pass evidence. |

## Day 6 Handoff

Day 6 should inventory CI and workflows with attention to:

- exact hosted Linux, macOS, and Windows workflow owners;
- selected oracle/comparison hosted lane boundaries;
- Windows CMake-first support and retained Makefile/`pkg-config` non-claims;
- any brittle count, path, shell, or artifact naming assumptions;
- hosted performance publication gaps relevant to R167-02;
- report-family promotion candidates relevant to R167-07.

## Validation Notes

Day 5 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Test and corpus evidence is grouped by solver family. | Complete | Solver-family test map groups matrix/core, LU, Cholesky/LDLT, QR, SVD, iterative, eigs, graph/reorder, package, and report/comparison surfaces. |
| Hosted and local-only proof surfaces are distinguished. | Complete | Oracle/comparison and platform/test-scope tables separate local generated rows, advisory metadata, and reviewed hosted lanes. |
| Comparison candidates are ready for selection. | Complete | LU CSR, LDLT CSC, Cholesky CSC, QR follow-up, and partial-SVD follow-up candidates list required evidence. |
