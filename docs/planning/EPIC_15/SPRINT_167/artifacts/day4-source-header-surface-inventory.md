# Sprint 167 Day 4: Source And Header Surface Inventory

## Purpose

Day 4 inventories the implementation and public-header surfaces relevant to
Epic 15 gaps. The goal is to identify concrete candidates for public-header
cleanup, ABI/package decision work, additional comparison coverage, and
allocation-failure evidence.

## Inventory Commands

The Day 4 inventory used these local repository scans:

- `find src include -type f | sort`
- `for f in include/*.h src/*.c; do wc -l ...; done`
- `rg -n "\\b(malloc|calloc|realloc|free)\\b" src include`
- public-header declaration density scan across `include/*.h`
- `find examples -maxdepth 2 -type f | sort`

These scans are planning evidence only. They do not prove correctness,
coverage, ABI stability, or performance.

## Source Surface Summary

| Surface | Count |
| --- | ---: |
| Public header templates and headers in `include/` | 19 |
| Implementation/internal files in `src/` | 69 |
| Maintained example files under `examples/` | 18 |

## Largest Implementation Files

Large files are not inherently wrong, but they are the highest-friction review
and failure-path audit surfaces. They are also likely candidates for invariant
documentation, helper extraction, or targeted failure tests.

| Rank | File | Lines | Primary surface |
| ---: | --- | ---: | --- |
| 1 | `src/sparse_ldlt_csc.c` | 2095 | CSC LDLT factorization |
| 2 | `src/sparse_lu_csr.c` | 1594 | CSR LU factorization |
| 3 | `src/sparse_ldlt.c` | 1535 | Public/internal LDLT flow |
| 4 | `src/sparse_iterative.c` | 1495 | Iterative solvers |
| 5 | `src/sparse_qr.c` | 1448 | QR factorization and solve |
| 6 | `src/sparse_eigs.c` | 1336 | Eigenvalue APIs and orchestration |
| 7 | `src/sparse_svd.c` | 1319 | SVD APIs and orchestration |
| 8 | `src/sparse_chol_csc.c` | 1279 | CSC Cholesky |
| 9 | `src/sparse_matrix.c` | 1053 | Matrix core |
| 10 | `src/sparse_lu.c` | 1042 | LU factorization |

## Public Header Inventory

| Header | Lines | Approximate public declaration count | Day 4 note |
| --- | ---: | ---: | --- |
| `include/sparse_iterative.h` | 828 | 19 | Recently cleaned in Epic 14; still large and adoption-critical. |
| `include/sparse_eigs.h` | 648 | 5 | Recently cleaned in Epic 14; generated-doc and option/result wording remains important. |
| `include/sparse_matrix.h` | 609 | 25 | Recently cleaned in Epic 14; core lifecycle and ownership authority. |
| `include/sparse_analysis.h` | 488 | 6 | Analysis/report-style API with meaningful claim wording risk. |
| `include/sparse_qr.h` | 373 | 14 | High-value candidate for next cleanup because QR comparison and least-squares claims are active. |
| `include/sparse_lu.h` | 360 | 11 | High-value candidate because LU remains core and large implementation-backed. |
| `include/sparse_lu_csr.h` | 322 | 0 by simple declaration scan | Needs manual review because public declarations may use multiline/comment-heavy style. |
| `include/sparse_ldlt.h` | 315 | 7 | High-value candidate because LDLT and LDLT CSC are large allocation-heavy surfaces. |
| `include/sparse_svd.h` | 243 | 9 | High-value candidate because partial-SVD comparison and convergence claims are active. |
| `include/sparse_cholesky.h` | 227 | 3 | Candidate after LDLT/QR/SVD due to solver-family importance. |
| `include/sparse_ilu.h` | 200 | 6 | Candidate for preconditioner and failure-path clarity. |
| `include/sparse_dense.h` | 197 | 0 by simple declaration scan | Needs manual review before ranking because declarations may not match the simple scan. |
| `include/sparse_reorder.h` | 186 | 6 | Candidate where ordering behavior, stability, and comparison caveats matter. |
| `include/sparse_csr.h` | 161 | 6 | Important for compressed-format adoption and performance claims. |
| `include/sparse_ic.h` | 121 | 4 | Candidate for preconditioner setup/failure semantics. |
| `include/sparse_bidiag.h` | 72 | 2 | Smaller candidate, likely lower priority. |
| `include/sparse_vector.h` | 70 | 0 by simple declaration scan | Smaller surface; review manually if vector docs become a front-door concern. |
| `include/sparse_types.h` | 324 | 3 | ABI-sensitive type/enum surface; important for ABI decision work. |
| `include/sparse_version.h.in` | not measured with checked-in `.h` scan | n/a | Generated install/version surface, relevant to package/ABI claims. |

## Implementation Family Map

| Family | Implementation files | Public headers | Maintained examples | Epic 15 relevance |
| --- | --- | --- | --- | --- |
| Matrix core and formats | `sparse_matrix.c`, `sparse_matrix_build_internal.c`, `sparse_matrix_io.c`, `sparse_csr.c`, `sparse_vector.c`, internal matrix headers | `sparse_matrix.h`, `sparse_csr.h`, `sparse_vector.h`, `sparse_types.h` | `example_basic_solve.c`, `example_compressed_input.c`, `example_matrix_market.c` | API authority, ABI type surface, compressed-format adoption, package consumers. |
| LU and LU CSR | `sparse_lu.c`, `sparse_lu_csr.c`, `sparse_lu_csr_struct.c`, LU CSR internals | `sparse_lu.h`, `sparse_lu_csr.h` | `example_basic_solve.c` | Large implementation, high allocation density, potential failure-path candidate. |
| Cholesky and LDLT | `sparse_cholesky.c`, `sparse_chol_csc.c`, `sparse_chol_csc_supernodal.c`, `sparse_ldlt.c`, `sparse_ldlt_csc.c`, `sparse_ldlt_dense.c`, `sparse_ldlt_csc_rowadj.c`, `sparse_ldlt_csc_supernodal.c` | `sparse_cholesky.h`, `sparse_ldlt.h` | `example_ldlt.c` | High implementation size and allocation density; ABI/header cleanup and failure-path candidate. |
| QR | `sparse_qr.c`, `sparse_qr_householder.c`, QR internals | `sparse_qr.h` | `example_least_squares.c`, `example_minnorm.c` | Strong candidate for header cleanup and comparison-family selection because QR comparison claims are active. |
| SVD and partial SVD | `sparse_svd.c`, `sparse_svd_partial.c`, SVD internals | `sparse_svd.h`, `sparse_bidiag.h` | `example_svd_lowrank.c` | Strong candidate for header cleanup and comparison/failure-path evidence because partial-SVD claims are active. |
| Iterative solvers | `sparse_iterative.c`, `sparse_iterative_block.c`, `sparse_iterative_minres.c`, workspace internals | `sparse_iterative.h` | `example_iterative.c`, `example_ic_minres.c`, `example_matrix_free.c` | Recently cleaned but still large; performance and backend evidence may touch this surface. |
| Preconditioners | `sparse_ilu.c`, `sparse_ic.c` | `sparse_ilu.h`, `sparse_ic.h` | `example_ic_minres.c` | Good bounded header/failure candidates if solver families are not selected. |
| Eigensolvers | `sparse_eigs.c`, dense/workspace/selection/LOBPCG/thick-restart internals | `sparse_eigs.h` | `example_eigs.c` | Recently cleaned; large implementation remains relevant to performance/report claims. |
| Ordering and graph | `sparse_reorder.c`, `sparse_reorder_nd.c`, `sparse_reorder_amd_qg.c`, `sparse_colamd.c`, graph partitioning/coarsening/refinement files | `sparse_reorder.h`, `sparse_analysis.h` | `example_colamd.c`, `example_analysis.c` | Comparison, performance, and API caveat wording may touch ordering surfaces. |
| Dense helpers | `sparse_dense.c` | `sparse_dense.h` | comparison/oracle helpers indirectly | External comparison reference and dense fallback behavior may touch this surface. |

## Allocation-Dense Surfaces

The allocation scan counts textual occurrences of `malloc`, `calloc`,
`realloc`, and `free`; it is a heuristic for failure-path audit priority, not
a correctness result.

| Rank | File | Allocation/free mentions | Day 4 implication |
| ---: | --- | ---: | --- |
| 1 | `src/sparse_lu_csr.c` | 128 | Strong allocation-failure candidate; large implementation and core solver surface. |
| 2 | `src/sparse_ldlt_csc.c` | 125 | Strong allocation-failure candidate; largest implementation file. |
| 3 | `src/sparse_ldlt.c` | 114 | Strong candidate; public LDLT flow and allocation density overlap. |
| 4 | `src/sparse_qr.c` | 101 | Strong candidate if QR remains the comparison/header focus. |
| 5 | `src/sparse_lu.c` | 92 | Core direct solver candidate. |
| 6 | `src/sparse_etree.c` | 84 | Symbolic analysis/tree candidate, especially for Cholesky/LDLT workflows. |
| 7 | `src/sparse_svd_partial.c` | 66 | Strong candidate if partial-SVD comparison and convergence remain active. |
| 8 | `src/sparse_graph_coarsen.c` | 58 | Graph algorithm candidate, lower priority for Epic 15 focus. |
| 9 | `src/sparse_svd.c` | 54 | SVD orchestration candidate. |
| 10 | `src/sparse_reorder.c` and `src/sparse_chol_csc.c` | 51 each | Ordering/Cholesky candidates. |

## Public-Header Cleanup Candidates

| Priority | Header family | Reason | Recommended handling |
| ---: | --- | --- | --- |
| 1 | QR (`sparse_qr.h`) | Active comparison claims, least-squares/minimum-norm behavior, and a large implementation file. | Strong Sprint 172 candidate if Day 11 selects QR/API coherence. |
| 2 | SVD/partial SVD (`sparse_svd.h`, `sparse_bidiag.h`) | Active partial-SVD comparison and convergence residuals; important tolerance and output semantics. | Strong candidate if comparison and API cleanup are paired. |
| 3 | LDLT (`sparse_ldlt.h`) | Large and allocation-heavy implementation family with package/API reliability implications. | Strong candidate if failure-path evidence targets LDLT. |
| 4 | LU/LU CSR (`sparse_lu.h`, `sparse_lu_csr.h`) | Core direct solver, high allocation density, and adoption importance. | Good candidate for failure-path and header coherence work. |
| 5 | ILU/IC (`sparse_ilu.h`, `sparse_ic.h`) | Preconditioner setup and failure semantics need clear lifecycle wording. | Bounded candidate if smaller cleanup is preferred. |
| 6 | Analysis/reorder (`sparse_analysis.h`, `sparse_reorder.h`) | Ordering and analysis claims affect performance and solver-selection docs. | Candidate after direct solver/API surfaces. |
| 7 | CSR/dense/vector (`sparse_csr.h`, `sparse_dense.h`, `sparse_vector.h`) | Adoption and format interoperability surfaces, but less urgent than QR/SVD/LDLT. | Review if performance or external comparison work touches these paths. |

## Allocation-Failure Candidate Shortlist

| Priority | Subsystem | Why bounded | Risks to investigate |
| ---: | --- | --- | --- |
| 1 | LU CSR | Single major source file with highest allocation/free density and core direct-solver value. | Partial symbolic/numeric setup cleanup, permutation buffers, factor-state ownership. |
| 2 | LDLT CSC | Largest source file and nearly highest allocation/free density. | Factor cleanup, row-adjacency/supernodal handoff behavior, symbolic/numeric partial construction. |
| 3 | QR | Active comparison family and high allocation density make it useful for both correctness and failure semantics. | Workspace cleanup, rank-deficient paths, least-squares/minimum-norm setup failures. |
| 4 | LDLT generic | Public-facing LDLT orchestration with high allocation/free density. | Dispatch cleanup and dense/CSC backend ownership boundaries. |
| 5 | Partial SVD | Active Epic 15 comparison area and bounded source file. | Partial-result cleanup, convergence failure cleanup, sparse output ownership. |

## Quality-Gate Trigger Rule

Future Sprint 167 or Epic 15 work that edits any `.c` or public/internal `.h`
file must run:

```sh
make format && make lint && make test
```

Documentation-only changes should still run at least:

```sh
git diff --check
```

Additional targeted checks should be selected based on the touched surface:

| Touched surface | Supplemental checks to consider |
| --- | --- |
| Public headers | docs checks, API docs coverage, declaration-preservation scans |
| Package/build files | install validation, CMake export validation, static/shared deferral checks |
| Report scripts/indexes | normalizer tests, report freshness checks |
| Comparison/corpus files | corpus schema, oracle/comparison generator tests, freshness checks |
| Bench/performance files | benchmark report generation, sentinel checks, methodology scans |

## Day 5 Handoff

Day 5 should inventory the test and corpus surfaces with attention to:

- proof-owner tests for QR, SVD, partial SVD, LU CSR, LDLT, Cholesky, iterative
  solvers, and package behavior;
- maintained corpus manifests and generated report families;
- current external comparison fixtures and tolerance policies;
- platform-specific test exclusions or staged lanes;
- candidate tests that could own future allocation-failure or comparison
  evidence.

## Validation Notes

Day 4 changed only Sprint 167 planning artifacts. No `.c` or `.h` files were
modified, so the full C quality gate is not required for this day.

## Completion Check

| Criterion | Status | Evidence |
| --- | --- | --- |
| Source and header surfaces are mapped to Epic 15 gap candidates. | Complete | Implementation family map connects source files, public headers, examples, and Epic 15 relevance. |
| Public-header cleanup candidates are concrete. | Complete | QR, SVD/partial SVD, LDLT, LU/LU CSR, ILU/IC, analysis/reorder, and CSR/dense/vector are ranked. |
| Allocation-failure candidates are bounded enough for future sprint work. | Complete | LU CSR, LDLT CSC, QR, LDLT generic, and partial SVD are listed with specific risk areas. |
