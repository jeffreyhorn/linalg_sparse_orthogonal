# Sprint 100 Day 5 External Comparison & Benchmark Baseline

## Purpose

Day 5 inventories the current comparison, benchmark, coverage, and reporting
surfaces. This artifact should feed the Sprint 100 state-of-the-art target,
the Sprint 102-105 evidence work, and the Sprint 109 competitive calibration.

## Benchmark Surface Map

| surface | owner | current interpretation |
|---|---|---|
| compile-only benchmark gate | `make tooling-build`, `make bench-build`, `make lint`, `make quality-review-compile` | catches benchmark/example compile drift without executing long workloads |
| `bench-fast` | Makefile, selected benchmark binaries | bounded PR-time runtime signal; not a portable performance claim |
| `bench-canonical-report` | `scripts/bench_canonical_report.sh` | threshold-free canonical snapshot for local/CI-friendly before/after comparison |
| `bench-reorder-sprint86` | `benchmarks/bench_reorder.c` | bounded two-fixture reorder/fill evidence lane; historical target name |
| `bench-suitesparse` | `benchmarks/bench_main.c` | LU/Cholesky SuiteSparse smoke comparison through `bench_main` |
| `bench-eigs` | `benchmarks/bench_eigs.c` | eigensolver backend/preconditioner sweep; broader exploratory comparison lane |
| `wall-check` | Makefile, `bench_amd_qg`, `bench_reorder` | narrow thresholded wall-clock regression signal with machine-class assumptions |

## Maintained Benchmark Category Split

| category | binaries | interpretation |
|---|---|---|
| canonical maintained measurement surface | `bench_refactor_csc`, `bench_chol_csc`, `bench_iterative_reuse`, `bench_eigs_reuse` | strongest benchmark-side workflow/performance evidence for retained public lifecycle surfaces |
| regression-sensitive runtime lane | `bench_scaling`, `bench_fillin`, `bench_colamd`, `bench_reorder --skip-factor`, adjacent `bench_amd_qg` | useful local/CI runtime signal; not a portable superiority claim |
| exploratory or broader comparison lane | `bench_main`, `bench_convergence`, `bench_svd`, `bench_bicgstab`, `bench_eigs`, broader `bench_reorder` | useful investigation and comparison surface without defining the compact maintained benchmark face |

## Canonical Report Surface

Command:

```sh
make bench-canonical-report
```

Outputs under `build/bench-reports/canonical/`:

| artifact | meaning |
|---|---|
| `bench_refactor_csc.csv` | repeated-run direct lifecycle / CSC comparison snapshot |
| `bench_chol_csc.csv` | CSC Cholesky backend-aware measurement snapshot |
| `bench_iterative_reuse.csv` | iterative handle reuse measurement snapshot |
| `bench_eigs_reuse.csv` | eigensolver handle reuse measurement snapshot |
| `manifest.txt` | command mapping, label, timestamp, branch, commit, and artifact inventory |
| `index.tsv` | machine-readable row per emitted canonical artifact |

Interpretation:

- threshold-free local snapshot
- suitable for branch-to-branch or run-to-run comparison
- not a pass/fail timing gate
- not a portable performance guarantee

## Reorder/Fill Evidence Surface

Command:

```sh
make bench-reorder-sprint86
```

Expansion:

```sh
bench_reorder --sprint86-slice --skip-factor
```

Bounded interpretation:

- fixture slice: `bcsstk14`, `Pres_Poisson`
- primary fill field: `nnz_L`
- timing field: `reorder_ms`, interpreted as local context only
- `--skip-factor` keeps the lane bounded by avoiding multi-minute factor
  execution
- target name is historical and should not be read as a broad Sprint 86 claim

## Maintained External Dense-Reference Lanes

| family | owner | fixtures | validation command | earned claim |
|---|---|---|---|---|
| Cholesky CSC SPD solve | `tests/test_chol_csc.c`, `tests/chol_external_dense_reference.py` | SuiteSparse `nos4`, `bcsstk04` | `make build/test_chol_csc && ./build/test_chol_csc` | bounded external-process dense-reference solve comparison for named SPD fixtures |
| LDLT CSC indefinite solve | `tests/test_ldlt_csc.c`, `tests/ldlt_external_dense_reference.py` | deterministic `kkt5`, `kkt10` | `make build/test_ldlt_csc && ./build/test_ldlt_csc` | bounded external-process dense-reference solve comparison for named deterministic KKT fixtures |

Explicit limits:

- not broad direct-solver ecosystem parity
- not external factorization parity
- not proof of internal pivot, fill, or CSC-layout equivalence
- not every-solver-family external correctness comparison

## Internal Cross-Checks and Non-External Evidence

| family | current non-external evidence type | examples |
|---|---|---|
| LU / Cholesky / QR / LDLT wrappers | internal direct-solver residuals and cross-family comparisons | SuiteSparse tests, integration tests, QR-vs-LU, Cholesky-vs-CG, LDLT lifecycle checks |
| iterative solvers | residual/convergence tests and direct-solver comparisons | CG/GMRES/MINRES/BiCGSTAB tests, stagnation and residual-history tests |
| eigensolvers | residual checks, backend parity, SVD cross-checks | grow-m, thick-restart, LOBPCG, shift-invert, residual/refinement tests |
| SVD | reconstruction, orthogonality, rank, partial-vs-full comparisons | full/partial SVD tests, low-rank and pseudoinverse tests |
| reorder/graph | fill, permutation, partition, determinism, and runtime-local measurements | AMD/COLAMD/ND/graph tests and `bench_reorder` |

These surfaces are valuable, but they are not a replacement for maintained
external oracle comparison where Epic 10 wants to earn stronger claims.

## Coverage Surface

| target | backend | status | notes |
|---|---|---|---|
| `make coverage` | auto-selects `coverage-lcov` or `coverage-gcovr` | supplemental | tree-mutating; rebuilds tests with coverage instrumentation |
| `make coverage-lcov` | GCC/lcov/genhtml | supplemental | Linux CI uses this path; removes tests and benchmarks from source coverage report |
| `make coverage-gcovr` | Apple Clang gcov emulation via gcovr | supplemental | macOS-friendly fallback for Apple Clang `.gcno` format |

Current threshold:

- `COV_THRESHOLD = 80`

Interpretation:

- coverage remains a live supplemental signal
- coverage is not part of `make quality-review-full`
- operators returning from coverage to normal direct/reviewed paths should run
  `make clean`
- Day 5 did not run coverage because this day owns topology, not coverage
  generation

## Comparison Gap Table

| area | current status | Epic 10 gap |
|---|---|---|
| Cholesky CSC | bounded external dense-reference lane on `nos4` and `bcsstk04` | broaden fixture taxonomy only with explicit proof architecture |
| LDLT CSC | bounded external dense-reference lane on `kkt5` and `kkt10` | broader Matrix Market / indefinite corpus comparison remains open |
| LU / CSR LU | internal residuals and SuiteSparse validation | no maintained external LU oracle lane |
| QR | internal reconstruction, rank, solve, and QR-vs-LU comparisons | no maintained external QR oracle lane |
| SVD / partial SVD | internal reconstruction, orthogonality, rank, partial-vs-full checks | no maintained external SVD reference lane |
| iterative solvers | residual, convergence, stagnation, and direct-solver comparisons | no maintained external iterative comparison architecture |
| eigensolvers / LOBPCG | residual, backend parity, refinement, SVD cross-checks | no maintained external ARPACK/Spectra/LOBPCG comparison architecture |
| reorder/fill | bounded `bench_reorder` and test evidence | no broad generated reorder/fill report target or ecosystem comparison |
| graph partitioning | internal determinism, balance, separator, and fixture checks | no external graph partitioning comparison claim |
| coverage | supplemental 80% tree-mutating lane | not reviewed universal gate |

## Day 5 Conclusion

The project has a mature benchmark/reporting topology and two real maintained
external dense-reference lanes. It does not yet have broad external oracle
coverage across solver families, nor portable performance claims. Epic 10
should treat every uncovered comparison row above as a candidate work item,
not as an existing product claim.

