# Benchmarks

Permanent benchmark drivers for the sparse linear algebra library.
Built via `make bench`; invoked individually via `make bench-<name>`
or by running the binary in `build/` directly.

| Binary              | Topic                                       | Smoke target          |
|---------------------|---------------------------------------------|-----------------------|
| `bench_main`        | LU decomposition over the SuiteSparse corpus | `make bench-suitesparse` |
| `bench_scaling`     | LU scaling sweep                             | (in `make bench`)     |
| `bench_fillin`      | Fill-in vs reordering quality                | (in `make bench`)     |
| `bench_convergence` | Iterative-solver convergence rates           | (in `make bench`)     |
| `bench_svd`         | Sparse SVD (bidiagonalisation + QR)          | (in `make bench`)     |
| `bench_refactor`    | LDL^T re-factor with cached symbolic         | (in `make bench`)     |
| `bench_refactor_csc`| Same but the CSC supernodal kernel           | (in `make bench`)     |
| `bench_colamd`      | COLAMD ordering quality                      | (in `make bench`)     |
| `bench_bicgstab`    | BiCGStab convergence                         | (in `make bench`)     |
| `bench_chol_csc`    | CSC Cholesky (Sprint 18)                     | (in `make bench`)     |
| `bench_ldlt_csc`    | LDL^T linked-list vs CSC + dispatch          | (in `make bench`)     |
| `bench_eigs`        | Symmetric eigensolver (3 backends)           | `make bench-eigs`     |

## bench_eigs (Sprint 21 Day 11)

Drives the three Sprint 20/21 eigensolver backends — grow-m Lanczos
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

When no mode flag is given, `--sweep default` runs.  `--repeats`
defaults to 3 for the smoke target; bump to 5 when capturing
recorded numbers (e.g. for `docs/planning/EPIC_2/SPRINT_21/bench_day14.txt`).

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
Demonstrates the Day 9 preconditioning speedup (e.g. bcsstk04
SMALLEST k=3: vanilla LOBPCG 800 iters NOT_CONVERGED → IC0 LOBPCG
62 iters OK → LDLT LOBPCG 8 iters OK on the captured run).
