# Sprint 118 Day 8 Product Truth Map

## Purpose

Day 8 completes the current product truth map for the post-Epic-10 baseline.
It records what the project can claim now, what remains an Epic 11 candidate
claim, and what must stay an explicit non-claim until future implementation,
proof, validation, and public wording earn a stronger statement.

The map is intentionally evidence-bound. Each current truth entry cites an
existing source, and candidate claims remain assigned to future owner sprints
rather than being promoted by planning language.

## Evidence Cross-Reference

| Evidence source | Truth supported |
|---|---|
| `README.md` | Public product identity, compressed-first positioning, solver summaries, installation summary, benchmark caveats, and support boundaries. |
| `INSTALL.md` | Static-first install contract, `pkg-config`, CMake consumer route, platform support table, and staged package limitations. |
| `docs/solver_selection.md` | User-facing solver-family routing and bounded solver recommendations. |
| `docs/tutorial.md` | First-use walkthroughs, repeated-run examples, and adoption flow support. |
| `docs/matrix_market.md` | Matrix Market load/save support, duplicate handling, and unsupported format boundaries. |
| `docs/maintainer_guide.md` | Reviewed, supplemental, and staged validation-lane interpretation. |
| `benchmarks/README.md` | Benchmark command groups, report surfaces, local-measurement caveats, and sentinel interpretation. |
| `examples/README.md` | Maintained example inventory and adoption-route evidence. |
| `include/*.h` | Public API surfaces for storage, solver, Matrix Market, graph, reorder, package, and reporting claims. |
| `src/*.c` and `src/*.h` | Implementation reality for current behavior and source-boundary caveats. |
| `tests/*.c` | Behavior proof owners, CTest membership, fixture-local evidence, and regression coverage. |
| `benchmarks/*.c` | Local benchmark and performance-sentinel driver inventory. |
| `examples/*.c` | Current example coverage for adoption claims. |
| `Makefile` | Local reviewed validation, source-list, CMake parity, benchmark, install, coverage, and report target names. |
| `CMakeLists.txt` and `cmake/` | CMake build, install, export, and downstream package-consumer truth. |
| `.github/workflows/*.yml` | Linux, macOS, and Windows reviewed or supplemental platform truth. |
| `artifacts/day2-validation-inventory.md` | Validation surface inventory and required-lane selection. |
| `artifacts/day3-baseline-quality-recheck.md` | Local reviewed baseline result: `make quality-review-full` passed, Makefile/CMake parity was `54` vs `54`, and CTest passed `54 / 54`. |
| `artifacts/day4-ci-tier-platform-truth.md` | CI-tier, platform, install, package, and staged-exclusion truth. |
| `artifacts/day5-residual-intake.md` | Residual queue, duplicate fence, and non-claim carry-forward inventory. |
| `artifacts/day6-residual-owner-map.md` | Sprint 119-127 residual ownership and proof gates. |
| `artifacts/day7-product-truth-map-design.md` | Truth-map categories, classification rules, and candidate-claim fences. |
| `docs/planning/EPIC_10/EPIC_10_RETROSPECTIVE.md` | Final Epic 10 earned claims, residuals, and non-claims. |
| `docs/planning/EPIC_10/SPRINT_117/RETROSPECTIVE.md` | Final pre-Epic-11 integration and closeout evidence. |

## Summary Truth Table

| Category | Baseline truth? | Candidate claims? | Explicit non-claims? | Owner |
|---|---|---|---|---|
| Compressed-first storage and construction | Yes, with compatibility caveats. | Yes. | Yes. | Sprints 119, 126 |
| Mutable-shell compatibility | Yes. | Yes. | Yes. | Sprints 119, 126 |
| Direct solvers | Yes, bounded by current proof breadth. | Yes. | Yes. | Sprint 120 |
| Iterative solvers | Yes, with handle-scope limits. | Yes. | Yes. | Sprint 120 |
| Eigensolver surfaces | Yes, with source-boundary and oracle caveats. | Yes. | Yes. | Sprints 119, 120 |
| SVD/QR/rank surfaces | Yes, with fixture and parity caveats. | Yes. | Yes. | Sprint 121 |
| Matrix Market I/O | Yes, bounded to documented variants. | Yes. | Yes. | Sprints 121, 126 |
| Graph and reordering | Yes, with universal-fill caveats. | Yes. | Yes. | Sprint 123 |
| Package/install/platform | Yes, static-first and tiered. | Yes. | Yes. | Sprints 124, 125 |
| Benchmark/performance | Yes, local and bounded. | Yes. | Yes. | Sprint 123 |
| Validation/reporting | Yes. | Yes. | Yes. | Sprints 122, 127 |
| Adoption/docs | Yes, dense but usable. | Yes. | Yes. | Sprint 126 |
| Explicit non-claims | Yes. | N/A. | Yes. | Sprint 127 |

## Compressed-First Storage And Construction

| Field | Current truth |
|---|---|
| Baseline truth | Compressed-first workflows are supported and preferred when callers already have CSR or CSC data. The project exposes CSR/CSC APIs, conversion routes, and one-shot solver entry points that avoid making the mutable orthogonal shell the product center for already-compressed inputs. |
| Evidence sources | `README.md`, `include/sparse_csr.h`, `include/sparse_csc.h`, `include/sparse_matrix.h`, `tests/test_csr.c`, `tests/test_integration.c`, `examples/example_compressed_input.c`, `artifacts/day3-baseline-quality-recheck.md`. |
| Caveats | The mutable sparse-matrix shell remains a supported compatibility surface. Not every operation is organized as a purely compressed-first product path, and future source-boundary work remains scheduled. |
| Epic 11 candidate claims | Stronger compressed-first public routing, clearer example placement, and source-boundary follow-through after Sprints 119 and 126. |
| Explicit non-claims | No claim of GraphBLAS parity, vendor-backend parity, or a pure compressed-only product model. |
| Owner sprint or future owner | Sprints 119 and 126. |

## Mutable-Shell Compatibility

| Field | Current truth |
|---|---|
| Baseline truth | The orthogonal linked-list sparse matrix remains a supported public compatibility representation for mutation, traversal, Matrix Market load/save, and workflows that benefit from incremental construction. |
| Evidence sources | `README.md`, `include/sparse_matrix.h`, `docs/matrix_market.md`, `tests/test_sparse_matrix.c`, `tests/test_sparse_io.c`, `examples/example_matrix_market.c`. |
| Caveats | The shell is not the performance-center claim for large compressed workloads. Factored or reordered state carries documented ownership and mutation restrictions. |
| Epic 11 candidate claims | Public docs can present the shell more explicitly as compatibility and construction support while moving performance-oriented examples toward compressed-first routes. |
| Explicit non-claims | No claim that the mutable shell is the state-of-the-art sparse storage model or the fastest route for all operations. |
| Owner sprint or future owner | Sprints 119 and 126. |

## Direct Solvers

| Field | Current truth |
|---|---|
| Baseline truth | Direct solver support includes LU, Cholesky, LDLT, QR, CSR LU, CSC Cholesky/LDLT dispatch-backed paths, one-shot entry points, and selected repeated direct lifecycle coverage. |
| Evidence sources | `README.md`, `docs/solver_selection.md`, `include/sparse_lu.h`, `include/sparse_cholesky.h`, `include/sparse_ldlt.h`, `include/sparse_qr.h`, `tests/test_sparse_lu.c`, `tests/test_lu_csr.c`, `tests/test_cholesky.c`, `tests/test_ldlt.c`, `tests/test_ldlt_csc.c`, `tests/test_qr.c`, `artifacts/day3-baseline-quality-recheck.md`. |
| Caveats | External-oracle breadth is still selected rather than comprehensive, and some proof owners remain large or shared with adjacent behavior. |
| Epic 11 candidate claims | Expanded direct-solver oracle coverage, clearer proof-owner boundaries, and regression fixture cleanup after Sprint 120. |
| Explicit non-claims | No claim of SuiteSparse, LAPACK, SciPy, or every-matrix-family parity. |
| Owner sprint or future owner | Sprint 120. |

## Iterative Solvers

| Field | Current truth |
|---|---|
| Baseline truth | Iterative solver support includes CG, GMRES, MINRES, BiCGSTAB, block variants, preconditioners, diagnostics, and selected repeated-handle workflows. |
| Evidence sources | `README.md`, `docs/solver_selection.md`, `include/sparse_iterative.h`, `tests/test_iterative.c`, `tests/test_minres.c`, `tests/test_bicgstab.c`, `tests/test_block_solvers.c`, `artifacts/day3-baseline-quality-recheck.md`. |
| Caveats | Repeated-handle support is intentionally bounded. BiCGSTAB and block iterative coverage should not be described as equivalent to every repeated-run handle path. |
| Epic 11 candidate claims | Oracle helpers, source-owner cleanup, and clearer diagnostics ownership after Sprint 120. |
| Explicit non-claims | No claim of PETSc, Trilinos, or every-preconditioner parity. |
| Owner sprint or future owner | Sprint 120. |

## Eigensolver Surfaces

| Field | Current truth |
|---|---|
| Baseline truth | Symmetric eigensolver support includes Lanczos, thick restart, LOBPCG, shift-invert routes, and a repeated-run handle surface with current tests and examples. |
| Evidence sources | `README.md`, `docs/solver_selection.md`, `include/sparse_eigs.h`, `tests/test_eigs.c`, `tests/test_eigs_thick_restart.c`, `tests/test_eigs_lobpcg.c`, `examples/README.md`, `artifacts/day6-residual-owner-map.md`. |
| Caveats | The eigensolver source boundary remains a known residual. External comparison evidence is fixture-local and should not be promoted to ARPACK-class parity. |
| Epic 11 candidate claims | Source-boundary owner extraction and external-comparison expansion after Sprints 119 and 120. |
| Explicit non-claims | No claim of ARPACK, SciPy, LAPACK, or broad non-symmetric eigensolver parity. |
| Owner sprint or future owner | Sprints 119 and 120. |

## SVD, QR, And Rank Surfaces

| Field | Current truth |
|---|---|
| Baseline truth | QR, least-squares, full and partial SVD, rank, condition-number, pseudoinverse, minimum-norm, nullspace, and low-rank approximation surfaces exist with current regression and benchmark support. |
| Evidence sources | `README.md`, `docs/solver_selection.md`, `include/sparse_qr.h`, `include/sparse_svd.h`, `tests/test_qr.c`, `tests/test_svd.c`, `benchmarks/README.md`, `artifacts/day3-baseline-quality-recheck.md`. |
| Caveats | Current evidence does not establish broad LAPACK/SciPy parity, portable performance superiority, or every rank-deficient corner. Some proof owners remain dense. |
| Epic 11 candidate claims | Dedicated oracle expansion, proof-owner split, and rank-deficient evidence cleanup after Sprint 121. |
| Explicit non-claims | No claim of LAPACK, NumPy, SciPy, or production-grade dense linear algebra replacement parity. |
| Owner sprint or future owner | Sprint 121. |

## Matrix Market I/O

| Field | Current truth |
|---|---|
| Baseline truth | Matrix Market I/O supports documented sparse coordinate workflows, including real, pattern, and symmetric variants, with explicit duplicate handling and unsupported-feature boundaries. |
| Evidence sources | `docs/matrix_market.md`, `include/sparse_matrix.h`, `tests/test_sparse_io.c`, `tests/test_csr.c`, `examples/example_matrix_market.c`, `artifacts/day5-residual-intake.md`. |
| Caveats | The project should avoid implying broad Matrix Market coverage beyond documented variants. Unsupported array, complex, hermitian, skew, and other advanced variants remain outside the current claim unless future docs and tests say otherwise. |
| Epic 11 candidate claims | Clearer Matrix Market cookbook routing and residual proof-owner cleanup after Sprints 121 and 126. |
| Explicit non-claims | No claim of complete Matrix Market format coverage or broad external corpus parity. |
| Owner sprint or future owner | Sprints 121 and 126. |

## Graph And Reordering

| Field | Current truth |
|---|---|
| Baseline truth | Graph and reordering support includes RCM, AMD, nested dissection, COLAMD-style surfaces, graph partition helpers, typed options, and regression evidence. |
| Evidence sources | `README.md`, `include/sparse_reorder.h`, `include/sparse_graph.h`, `tests/test_reorder.c`, `tests/test_reorder_nd.c`, `tests/test_reorder_amd_qg.c`, `tests/test_colamd.c`, `tests/test_graph.c`, `tests/test_graph_fm_buckets.c`, `benchmarks/README.md`. |
| Caveats | Current evidence supports bounded behavior and guardrails, not universal fill reduction, universal speedups, or all-graph scalability claims. |
| Epic 11 candidate claims | Large-matrix sentinels, graph/reorder guardrails, and benchmark/report cleanup after Sprint 123. |
| Explicit non-claims | No claim of universal reorder/fill superiority, METIS parity, or every sparse-graph workload parity. |
| Owner sprint or future owner | Sprint 123. |

## Package, Install, And Platform

| Field | Current truth |
|---|---|
| Baseline truth | The maintained packaging story is static-first. The project supports local static install, `pkg-config`, and CMake `find_package(Sparse)` consumer routes. Linux is the strongest reviewed source of truth, macOS carries reviewed Apple Clang plus supplemental lanes, and Windows remains a reviewed MSVC CMake consumer subset. |
| Evidence sources | `INSTALL.md`, `README.md`, `Makefile`, `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml`, `artifacts/day4-ci-tier-platform-truth.md`. |
| Caveats | Windows expected CTest membership remains lower than Linux local parity. Windows Makefile parity, install-validation parity, thread/fuzz/property lanes, and full CTest parity remain staged or unclaimed. |
| Epic 11 candidate claims | ABI and package-productization decision work in Sprint 124 and platform validation expansion in Sprint 125. |
| Explicit non-claims | No shared-library dynamic ABI guarantee, package-manager support, or symmetric Linux/macOS/Windows parity claim. |
| Owner sprint or future owner | Sprints 124 and 125. |

## Benchmark And Performance

| Field | Current truth |
|---|---|
| Baseline truth | Benchmark drivers, report targets, and local performance-sentinel surfaces exist and can support local trend monitoring and bounded report generation. |
| Evidence sources | `benchmarks/README.md`, `benchmarks/*.c`, `Makefile`, `artifacts/day3-baseline-quality-recheck.md`, `artifacts/day7-product-truth-map-design.md`. |
| Caveats | Benchmark results are local measurements and guardrails. They do not establish portable superiority, cross-hardware performance claims, or vendor-backend parity. |
| Epic 11 candidate claims | Performance sentinel and report architecture cleanup after Sprint 123. |
| Explicit non-claims | No portable speedup, state-of-the-art performance, or universal scalability claim. |
| Owner sprint or future owner | Sprint 123. |

## Validation And Reporting

| Field | Current truth |
|---|---|
| Baseline truth | The current local reviewed baseline passed `make quality-review-full`, including Makefile reviewed validation, CMake configure/build, CTest registration, Makefile/CMake test-count parity, and full CTest execution. |
| Evidence sources | `artifacts/day2-validation-inventory.md`, `artifacts/day3-baseline-quality-recheck.md`, `docs/maintainer_guide.md`, `Makefile`, `CMakeLists.txt`. |
| Caveats | The Day 3 local run is not CI platform proof. Supplemental install, benchmark, sanitizer, coverage, package, and workflow lanes were intentionally skipped because their surfaces were not changed. |
| Epic 11 candidate claims | Report classification, evidence templates, corpus/report indexing, and final claim audit cleanup after Sprints 122 and 127. |
| Explicit non-claims | No claim that all supplemental lanes are always reviewed gates or that local validation proves platform parity. |
| Owner sprint or future owner | Sprints 122 and 127. |

## Adoption And Documentation

| Field | Current truth |
|---|---|
| Baseline truth | Users have public entry routes through the README, install guide, solver-selection guide, tutorial, Matrix Market docs, examples, and maintainer guide. |
| Evidence sources | `README.md`, `INSTALL.md`, `docs/solver_selection.md`, `docs/tutorial.md`, `docs/matrix_market.md`, `examples/README.md`, `docs/maintainer_guide.md`. |
| Caveats | Documentation remains dense. Some example and cookbook flows can be made more task-oriented before stronger adoption claims are earned. |
| Epic 11 candidate claims | Algorithm/docs split, cookbook-first examples, and reduced first-use friction after Sprint 126. |
| Explicit non-claims | No claim that the project has complete novice onboarding, package-manager adoption, or ecosystem-level documentation parity. |
| Owner sprint or future owner | Sprint 126. |

## Baseline Claim List

- Compressed-first workflows are supported and preferred when callers already
  have CSR or CSC data.
- The mutable orthogonal linked-list shell remains a supported compatibility
  and construction surface.
- One-shot direct solvers and selected repeated-run direct workflows are
  supported.
- Iterative solvers, diagnostics, preconditioners, and selected repeated
  handles are supported within documented limits.
- Symmetric eigensolver, SVD, QR, rank, pseudoinverse, low-rank, and related
  workflows exist with current regression evidence.
- Matrix Market I/O is supported for documented sparse coordinate workflows.
- Graph and reordering helpers exist with bounded regression and benchmark
  evidence.
- Static-first install supports local archive, `pkg-config`, and CMake package
  consumer routes.
- Linux is the strongest reviewed validation source; macOS and Windows have
  tiered reviewed or supplemental surfaces.
- The current local reviewed baseline passed with `54` CMake registrations,
  Makefile/CMake parity at `54` vs `54`, and `54 / 54` CTest passing.

## Epic 11 Candidate Claim List

- Source-boundary movement improves maintainability after Sprint 119.
- Direct, iterative, eigensolver, SVD, QR, and rank proof owners improve after
  Sprints 120-121.
- Corpus, report, and evidence-template architecture becomes easier to audit
  after Sprint 122.
- Benchmark and performance guardrails become more coherent after Sprint 123.
- Package and ABI productization decisions are explicit after Sprint 124.
- Platform install and validation boundaries improve after Sprint 125.
- Adoption docs, examples, and cookbook routes become easier to scan after
  Sprint 126.
- Final public claims and non-claims are recalibrated after Sprint 127.

## Explicit Non-Claim List

- The project is not claiming to be a broad state-of-the-art replacement for
  mature sparse linear algebra ecosystems.
- The project is not claiming SuiteSparse, PETSc, Trilinos, ARPACK, LAPACK,
  NumPy/SciPy, GraphBLAS, or vendor-backend parity.
- The project is not claiming every solver family has broad external-oracle
  coverage.
- The project is not claiming portable performance superiority.
- The project is not claiming universal reorder or fill-reduction superiority.
- The project is not claiming shared-library dynamic ABI stability.
- The project is not claiming package-manager distribution support.
- The project is not claiming symmetric Linux, macOS, and Windows reviewed
  parity.
- The project is not claiming Windows Makefile, install-validation,
  thread/fuzz/property, or full CTest parity.
- The project is not claiming GPU support.
- The project is not claiming distributed-memory support.
- The project is not claiming broad complex or mixed-precision maturity.

## Day 13 Drift-Audit Inputs

| Audit input | Day 13 use |
|---|---|
| Baseline claim list | Check public wording for overstatement or stale caveats. |
| Epic 11 candidate claim list | Ensure future-sprint language stays fenced until evidence exists. |
| Explicit non-claim list | Ensure README, install, docs, examples, and reports do not imply unearned support. |
| Evidence cross-reference | Confirm every public claim can point to docs, tests, validation, or owner artifacts. |
| Package/platform truth | Prevent symmetric platform or ABI/package-manager drift. |
| Benchmark/performance truth | Prevent portable-performance or state-of-the-art speed drift. |

## Completion Criteria Check

| Criterion | Status |
|---|---|
| Item 3 is complete. | Complete. |
| Every current truth entry cites an evidence source. | Complete. |
| Public baseline claims are listed. | Complete. |
| Candidate claims are listed and fenced. | Complete. |
| Explicit non-claims are listed. | Complete. |
| Evidence cross-reference table is present. | Complete. |
| Public claims and non-claims are ready for Day 12 template work and Day 13 drift audit. | Complete. |
