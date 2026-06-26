# Benchmarks

Permanent benchmark drivers for the sparse linear algebra library.
Built via `make bench`; invoked individually via `make bench-<name>`
or by running the binary in `build/` directly.

## Compile-only gate

Routine local quality checks should now catch benchmark/example compile
drift without executing the long-running benchmark workloads:

- `make tooling-build`
  - builds all benchmark and example binaries
  - does not run them
- `make lint`
  - now includes the same compile-only tooling gate before the existing
    source-level lint passes
- `make quality-review-compile`
  - reviewed local compile-quality wrapper
  - routes through `format-check` + `lint`, so benchmark/example compile drift
    is covered there as well

Focused subsets remain available:

- `make bench-build`
- `make examples-build`

For repository-wide reviewed-baseline, dead-code, and maintainer-policy
interpretation, use the top-level [README](../README.md) and the
[Maintainer Guide](../docs/maintainer_guide.md). This file stays focused on
benchmark-local command usage and surface-specific behavior.

## Reorder coverage

The benchmark entry points intentionally expose different reorder surfaces
depending on what the underlying factorization path actually supports:

- `bench_main --reorder none|rcm|amd|nd`
  - solver-harness entry point for LU and `--cholesky`
  - intentionally does not accept `colamd`
  - LU / Cholesky factorization options use symmetric reorderings only
  - `--help` and invalid `--reorder` errors now explicitly point users to
    `bench_reorder` / `bench_colamd` for COLAMD comparisons
- `bench_reorder`
  - cross-ordering comparison harness for `none`, `rcm`, `amd`, `colamd`,
    and `nd`
  - supports both direct reorder calls and `--reorder-via-analyze`
  - supports `--sprint86-slice` for the bounded Sprint 86 runtime corpus
    (`bcsstk14`, `Pres_Poisson`)
- `bench_colamd` and `example_colamd`
  - QR-focused comparison tools for `none`, `amd`, and `colamd`
  - use the same lowercase mode labels as the benchmark CLI
- `bench_chol_csc` and `bench_ldlt_csc`
  - backend-comparison tools, not general reorder sweeps
  - keep their fixed reorder choices in the code path being compared so
    backend timings stay like-for-like

| Binary                 | Topic                                                   | Smoke target              |
|------------------------|---------------------------------------------------------|---------------------------|
| `bench_main`           | One-shot LU / Cholesky / SpMV / iterative harness       | `make bench-suitesparse`  |
| `bench_scaling`        | LU scaling sweep                                        | (in `make bench`)         |
| `bench_fillin`         | Fill-in vs reordering quality                           | (in `make bench`)         |
| `bench_convergence`    | Iterative-solver convergence rates                      | (in `make bench`)         |
| `bench_svd`            | Sparse SVD (bidiagonalisation + QR)                     | (in `make bench`)         |
| `bench_refactor`       | Direct repeated-run lifecycle: Cholesky analyze once / refactor many | (in `make bench`) |
| `bench_refactor_csc`   | Direct repeated-run lifecycle proof: SPD Cholesky by default, plus optional indefinite LDL^T KKT mode | (in `make bench`) |
| `bench_iterative_reuse`| Public repeated-run iterative handle proof: CG, GMRES, MINRES | (in `make bench`)    |
| `bench_eigs_reuse`     | Public repeated-run eigensolver handle proof: grow-m, thick-restart, explicit LOBPCG | (in `make bench`) |
| `bench_colamd`         | QR/COLAMD ordering quality                              | (in `make bench`)         |
| `bench_bicgstab`       | BiCGStab convergence                                    | (in `make bench`)         |
| `bench_chol_csc`       | CSC Cholesky backend comparison                         | (in `make bench`)         |
| `bench_ldlt_csc`       | LDL^T linked-list vs CSC + dispatch                     | (in `make bench`)         |
| `bench_eigs`           | Symmetric eigensolver backend sweep                     | `make bench-eigs`         |

## Workflow groups

The shipped benchmark surfaces are easiest to read in four bounded groups:

- one-shot compatibility/comparison:
  - `bench_main`
  - `bench_scaling`
  - `bench_fillin`
  - `bench_convergence`
  - `bench_svd`
  - `bench_colamd`
  - `bench_bicgstab`
  - `bench_chol_csc`
  - `bench_ldlt_csc`
  - `bench_eigs`
- direct repeated-run lifecycle:
  - `bench_refactor`
  - `bench_refactor_csc`
- iterative public-handle reuse:
  - `bench_iterative_reuse`
- eigensolver public-handle reuse:
  - `bench_eigs_reuse`

## Current maintained category split

The Sprint 65 maintained performance-governance split is:

- canonical maintained proof surface:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- regression-sensitive runtime lane:
  - `bench_scaling`
  - `bench_fillin`
  - `bench_colamd`
  - `bench_reorder --skip-factor`
  - `bench_amd_qg` remains adjacent to this lane but should stay bounded
- exploratory or broader comparison lane:
  - `bench_main`
  - `bench_convergence`
  - `bench_svd`
  - `bench_bicgstab`
  - `bench_eigs`
  - broader `bench_reorder`

Interpretation:

- the canonical maintained surface is where Sprint 65 normalization and later
  efficiency follow-through should stay centered
- the runtime lane is useful for current-branch checks but should not be
  marketed as portable performance proof
- the exploratory lane stays valuable without defining the compact maintained
  benchmark face of the repo
- examples remain the API-adoption teaching surface; benchmarks remain the
  workflow/performance proof surface

For threshold-free local or CI-friendly reporting on the maintained canonical
surface, use:

- `make bench-canonical-report`
- optionally set `BENCH_CANONICAL_REPORT_LABEL=<label>` on that command to
  attach a bounded comparison label to the bundle metadata

That target writes one CSV per canonical maintained benchmark under:

- `build/bench-reports/canonical/`

plus a bounded bundle-level metadata surface:

- `manifest.txt`
  - exact command mapping
  - explicit artifact inventory
  - generated timestamp
  - bounded report label from `BENCH_CANONICAL_REPORT_LABEL`
  - git commit / branch when locally available
- `index.tsv`
  - one structured row per emitted canonical artifact
  - keeps the same bounded canonical surface identity and command mapping in a
    machine-readable comparison form

This is intentionally not a pass/fail timing gate:

- compare the emitted CSV rows across branches or runs
- treat it as artifact-friendly reporting, not portable performance proof
- use the bundle metadata to make before/after or cross-branch snapshots
  easier to line up without widening the benchmark claim surface
- keep `bench-fast` as the bounded runtime lane and `wall-check` as the narrow
  thresholded regression gate that already has a justified machine-class
  baseline
- for the bounded Sprint 86 ND runtime rerun slice, use:
  - `make bench-reorder-sprint86`
  - this expands to `bench_reorder --sprint86-slice --skip-factor`
  - it is branch-local evidence for the touched ND lane, not canonical proof

The two refactor benchmarks remain the strongest benchmark-side adoption
surfaces for the public repeated-run direct lifecycle:

- `bench_refactor`
  - compares one-shot Cholesky factorization against the analyze-once /
    factor-many direct path on same-pattern value changes
  - reports one-shot average, one-time analysis cost, initial numeric factor,
    average later refactor cost, repeated-run average cost, speedup, and final
    residual
- `bench_refactor_csc`
  - default mode keeps the SPD / Cholesky repeated-run workflow and compares
    the public repeated-run path against the direct CSC/supernodal completion
    path
  - `--indefinite-kkt` switches to a synthetic above-threshold KKT saddle-point
    workload and compares the public repeated-run LDL^T path against the direct
    resolved-analysis CSC completion path
  - reads as the main throughput/proof surface for the large-`n` CSC-backed
    repeated-run direct lane, not as the error-path contract surface; failed
    refactor preservation stays owned by `tests/test_integration.c`
  - the family-local large-`n` analysis-backed CSC helper parity stays owned
    by `tests/test_chol_csc.c`, not by this benchmark surface
  - the bounded seeded generative large-`n` lifecycle follow-through stays
    owned by `tests/test_fuzz.c`, not by this benchmark surface
  - reports CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `ldlt_dense_backend_request`
    - `ldlt_dense_backend_selected`
    - `ldlt_dense_backend_fallback`
    - `analyze_ms`
    - `refactor_public_ms`
    - `refactor_csc_ms`
    - `solve_public_ms`
    - `solve_csc_ms`
    - `speedup_refactor`
    - `res_public`
    - `res_csc`
  - interpret the LDL^T backend fields narrowly:
    - SPD / Cholesky rows report `n/a` in those three columns
    - `--indefinite-kkt` reports the normalized backend request, the current
      env/probe-derived backend name, and whether the widened LDL^T selector
      reported fallback to builtin
    - this is bounded observability for the retained repeated-run LDL^T lane,
      not a broad portability or performance claim

Read that support split narrowly:

- examples such as `example_analysis` stay the adoption entry points
- `bench_refactor` / `bench_refactor_csc` stay the retained
  workflow/performance proof surfaces after adoption
- tests still own regression/oracle/property guarantees for the large-`n`
  CSC-backed lifecycle lane

`bench_chol_csc` remains the maintained benchmark-side proof surface for the
first Sprint 64 backend-aware Cholesky CSC lane:

- it still compares linked-list, CSC scalar, and CSC supernodal timings on
  one fixed AMD-reordered workload so fallback and accelerated paths stay
  comparable
- each CSV row now also reports:
  - `benchmark`
  - `category`
  - `scenario`
  - `csc_scalar_path`
  - `csc_supernodal_path`
  - `csc_supernodal_dense_kernel`
  - `csc_supernodal_panel_solver`
- the path columns stay intentionally stable at:
  - `scalar`
  - `supernodal`
- `csc_supernodal_dense_kernel` identifies the active dense-kernel descriptor
  behind the supernodal lane; on the current default build it reports
  `builtin`
- `csc_supernodal_panel_solver` identifies whether the supernodal lane has
  the batched panel-solve callback required by the Sprint 75 Day 7 kernel
  landing; on the current default build it reports `batched_panel`
- this keeps the Sprint 64 benchmark refresh bounded to path measurability and
  truthfulness, not broad benchmark-governance churn
- the Sprint 75 Day 10 public callback/runtime semantics remain test-owned in
  `tests/test_integration.c`; this benchmark stays a measurement surface, not
  the owner of progress/cancel truth
- it is not the owner of the Sprint 68 staged public-path oracle/parity lane
  or the bounded seeded lifecycle property lane; those remain test-owned in
  `tests/test_integration.c` and `tests/test_fuzz.c`

So the benchmark-side reading stays:

- benchmarks = proof and measurement surfaces
- examples = adoption surfaces
- tests = regression/oracle/property owners

The two reuse benchmarks stay intentionally narrow and should be read as public
handle-path proof surfaces, not broad solver bake-offs:

- `bench_iterative_reuse`
  - compares one-shot and explicit public-handle repeated-run paths for:
    - `CG`
    - `GMRES`
    - `MINRES`
  - now reports stable CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `solver`
    - `one_shot_total_ms`
    - `reuse_total_ms`
    - `speedup`
    - last-run iteration / residual / convergence and status fields
  - intentionally does not claim public repeated-run-handle support for:
    - `BiCGSTAB`
    - block iterative workflows
- `bench_eigs_reuse`
  - compares one-shot and explicit public-handle repeated-run paths for:
    - grow-m Lanczos
    - thick-restart Lanczos
    - explicit LOBPCG
  - now reports stable CSV rows with:
    - `benchmark`
    - `category`
    - `matrix`
    - `scenario`
    - `backend`
    - `one_shot_median_ms`
    - `reuse_median_ms`
    - `speedup`
    - last-run iteration / convergence / residual / basis-size fields
    - retained agreement fields:
      - `lambda_max_diff`
      - `residual_diff`
      - `backend_used`
  - intentionally stays narrower than `bench_eigs`, which remains the broader
    backend/preconditioner sweep harness

## bench_main

Main solver-harness benchmark for:

- LU solve timing
- Cholesky solve timing via `--cholesky`
- SpMV-only timing via `--spmv`
- iterative-solver timing via `--iterative`

### CLI notes

```
bench_main [matrix.mtx]
bench_main --dir PATH
bench_main --size N
bench_main --help
```

Important CLI behavior:

- `--help` / `-h` now prints the live usage block
- malformed numeric or enum-like arguments fail with explicit flag-local
  diagnostics
- missing option values fail explicitly instead of silently falling through
- conflicting modes such as `--spmv --iterative` are rejected
- `--reorder` accepts only:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- unsupported `colamd` requests are intentionally redirected to:
  - `bench_reorder`
  - `bench_colamd`

## bench_eigs

Drives the three symmetric eigensolver backends — grow-m Lanczos
(`SPARSE_EIGS_BACKEND_LANCZOS`), Wu/Simon thick-restart Lanczos
(`_LANCZOS_THICK_RESTART`), and Knyazev LOBPCG (`_LOBPCG`) — across
the standard SuiteSparse + KKT corpus, with optional preconditioner
sweeps (NONE / IC0 / LDLT) for the LOBPCG branch.

### CLI summary

```
bench_eigs --sweep default [--csv] [--repeats N]   # full corpus sweep
bench_eigs --compare       [--csv] [--repeats N]   # 3-backend × 3-precond pivot
bench_eigs --matrix <path> --k N --which {LARGEST|SMALLEST|NEAREST}
                             [--sigma F] [--backend B] [--precond P]
                             [--block-size N] [--tol F] [--max-iters N]
                             [--csv] [--repeats N]
bench_eigs --help                                   # full help
```

When no mode flag is given, `--sweep default` runs. `--repeats`
defaults to 3 for the smoke target; bump to 5 when collecting more
stable local timing numbers.

### CSV schema

For `--sweep` and `--matrix` (one row per (matrix, k, which, backend,
precond) combination):

```
matrix, n, k, which, sigma, backend, precond,
iterations, peak_basis, wall_ms, residual, status
```

For `--compare` (one row per (matrix, k, which, precond), three
backend triples per row):

```
matrix, n, k, which, sigma, precond,
growing_m_iters, growing_m_wall_ms, growing_m_residual, growing_m_status,
thick_iters,     thick_wall_ms,     thick_residual,     thick_status,
lobpcg_iters,    lobpcg_wall_ms,    lobpcg_residual,    lobpcg_status
```

The `backend` / `precond` columns echo the configuration; `wall_ms`
is the median of `--repeats` runs; `peak_basis` is the doubles-times-
n peak Lanczos basis as exposed via `sparse_eigs_t.peak_basis_size`;
`status` is `OK` for converged or `NOT_CONVERGED` / etc. for
diagnosed failures.

### Default sweep

Runs the following grid:

- (nos4 n=100, bcsstk04 n=132) × {LARGEST, SMALLEST} × {k=3, k=5} × {3 backends}
- bcsstk14 n=1806 × LARGEST × {k=3, k=5} × {3 backends} (SMALLEST excluded —
  bottom of spectrum is too clustered for un-preconditioned convergence
  in a smoke-target runtime budget)
- KKT-150 (synthetic indefinite saddle-point) × NEAREST_SIGMA at σ=0
  × {3 backends}

About 33 rows total; ~15 seconds at `--repeats 2` on a 2025
M2-class development machine, ~25 seconds at `--repeats 3`.

### Compare mode

Smaller focused corpus (3 entries) × 3 preconditioners (NONE / IC0 /
LDLT), pivoted so each row shows the three backends side-by-side.
Useful for comparing how the preconditioner choice changes LOBPCG
iteration count and convergence status on the same matrix/workload.
