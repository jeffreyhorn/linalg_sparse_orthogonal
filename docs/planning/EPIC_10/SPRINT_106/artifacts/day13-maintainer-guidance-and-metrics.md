# Sprint 106 Day 13: Maintainer Guidance and Metrics Update

## Scope

Day 13 updated maintainer guidance for the Sprint 106 source and test helper
ownership model, compiled before/after maintainability metrics, and recorded
the residual extraction queue for Sprint 107 planning.

## Maintainer Guidance Update

Updated `docs/maintainer_guide.md` with Sprint 106 ownership guidance for:

- `src/sparse_ldlt_csc_rowadj.c` as the LDLT CSC row-adjacency owner
- `src/sparse_qr_householder.c` and `src/sparse_qr_internal.h` as the private
  QR Householder/internal-contract seam
- `src/sparse_lu_csr_struct.c` and `src/sparse_lu_csr_internal.h` as the LU CSR
  structural storage seam
- `tests/test_graph_fixtures.h` as the graph/reorder fixture owner
- `tests/test_direct_solver_helpers.h` as the direct-solver assertion/residual
  helper owner
- `tests/test_integration_fixtures.h` as the integration progress/oracle
  fixture owner

The guidance also records the rule that future compiled helper targets require
explicit Make/CMake registration and reviewed CTest-surface reconciliation.

## Source Metrics

| owner | Sprint 106 baseline | current | delta |
|---|---:|---:|---:|
| `src/sparse_ldlt_csc.c` | 2,174 | 2,095 | -79 |
| `src/sparse_qr.c` | 1,563 | 1,448 | -115 |
| `src/sparse_lu_csr.c` | 1,665 | 1,594 | -71 |
| `src/sparse_ldlt_csc_rowadj.c` | 0 | 82 | +82 |
| `src/sparse_qr_householder.c` | 0 | 79 | +79 |
| `src/sparse_qr_internal.h` | 0 | 16 | +16 |
| `src/sparse_lu_csr_struct.c` | 0 | 57 | +57 |
| `src/sparse_lu_csr_internal.h` | 0 | 9 | +9 |

The implementation work reduced the three touched large implementation owners
by 265 lines while creating focused helper/internal owners totaling 243 lines.
The net line-count movement is deliberately small; the value is ownership
locality and narrower review/failure localization.

## Giant-Test Metrics

| owner | Sprint 106 baseline | current | delta |
|---|---:|---:|---:|
| `tests/test_graph.c` | 2,925 | 2,758 | -167 |
| `tests/test_reorder_nd.c` | 2,340 | 2,304 | -36 |
| `tests/test_lu_csr.c` | 1,899 | 1,806 | -93 |
| `tests/test_integration.c` | 3,421 | 3,279 | -142 |
| `tests/test_graph_fixtures.h` | 0 | 195 | +195 |
| `tests/test_direct_solver_helpers.h` | 0 | 93 | +93 |
| `tests/test_integration_fixtures.h` | 0 | 140 | +140 |

The fixture work reduced four giant test owners by 438 lines while creating
428 lines of focused, named helper ownership.

## Build and Test Surface Metrics

| surface | Sprint 106 baseline | current | note |
|---|---:|---:|---|
| library sources tracked by source-list checker | 42 | 45 | three new library `.c` owners |
| reviewed Makefile test binaries | 54 | 54 | unchanged registration |
| reviewed CMake tests from `ctest -N` | 54 | 54 | unchanged registration |
| new compiled test helper targets | 0 | 0 | helper extraction stayed header-only |
| new public/install headers | 0 | 0 | private contracts stayed private |

## Residual Extraction Queue

| candidate | reason to defer | suggested next step |
|---|---|---|
| `tests/test_ldlt_csc.c` direct CSC fixtures | largest proof owner remains large, but broad fixture movement risks hiding direct CSC intent | identify one narrow row-adjacency or residual/oracle helper before moving any assertions |
| `tests/test_qr.c` QR proof helpers | QR source seam was extracted, but QR proof owner still has many local fixtures | split only repeated QR matrix/vector builders with names that preserve solve/reconstruction intent |
| `tests/test_iterative.c` iterative workflow fixtures | high helper density and convergence evidence wording make broad extraction risky | extract reusable matrix/RHS builders only after preserving external-reference and convergence-claim wording |
| `tests/test_svd.c` SVD proof helpers | Sprint 103 claim wording and rank/oracle evidence are sensitive | defer until a dedicated SVD proof-owner cleanup day with full focused SVD validation |
| `src/sparse_eigs.c` orchestration | source remains large but is tied to Sprint 103 comparison surfaces | prefer helper extraction around workspace or dispatch only after a new boundary artifact |
| `src/sparse_matrix.c` matrix shell | central API/compatibility risk remains too high for incidental cleanup | reserve for an API/compatibility sprint, not opportunistic maintainability work |

## Validation

Day 13 changed documentation only. Validation:

```sh
git diff --check
rg -n "[ \t]+$" docs/maintainer_guide.md docs/planning/EPIC_10/SPRINT_106
```

Both checks passed.

## Day 13 Conclusion

Sprint 106 now has maintainer-facing guidance for the new source and test
ownership layout, metrics that show the impact and limits of the extraction
work, and an explicit residual queue suitable for Sprint 107 planning.
