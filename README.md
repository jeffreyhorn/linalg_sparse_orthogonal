# linalg_sparse_orthogonal

A C library for sparse matrices using the **orthogonal linked-list** (cross-linked) representation, with direct and iterative linear system solvers.

## Start Here

Use this README as the short front door. It gets you from local build to first
solve, then routes deeper support detail to the maintained docs that own it.

1. **Build locally:** run `make`, then `make examples` if you want the shipped
   example binaries.
2. **Run the first maintained solve:** use `./build/example_basic_solve` from
   [examples/README.md#start-here](examples/README.md#start-here), or use the
   inline [Quick Start](#quick-start) when you want to paste a tiny program.
3. **Start from your data format:** if your matrix already exists as CSR, CSC,
   or Matrix Market input, use the
   [cookbook first-use ladder](docs/cookbook.md#first-use-ladder) and
   `./build/example_compressed_input`.
4. **Choose the solver family:** use [Choose a Workflow](#choose-a-workflow)
   for the compact route, then
   [docs/solver_selection.md#choose-the-smallest-workflow](docs/solver_selection.md#choose-the-smallest-workflow)
   for the fuller decision tree.
5. **Inspect local diagnostics:** keep return codes, `NULL` constructor
   results, residuals, convergence status, rank diagnostics, and benchmark
   measurements tied to the workflow that produced them. The maintained
   example handoff is
   [examples/README.md#diagnostics-handoff](examples/README.md#diagnostics-handoff).
6. **Install only when you need a downstream consumer:** use
   [Installation](#installation), then [INSTALL.md#start-here](INSTALL.md#start-here)
   for static-first install and package details.
7. **Escalate after the first workflow works:** runtime/backend controls,
   benchmarks, report indexes, API reference, platform tiers, and maintainer
   evidence live in
   [Runtime And Backend Controls](#runtime-and-backend-controls),
   [benchmarks/README.md](benchmarks/README.md),
   [docs/api_reference.md](docs/api_reference.md), and
   [docs/maintainer_guide.md](docs/maintainer_guide.md).

## Adoption Map

| Need | Start here | Then use |
|---|---|---|
| Smallest local build and solve | [examples/README.md#start-here](examples/README.md#start-here) | [Quick Start](#quick-start) |
| Problem-shape decision tree | [Choose a Workflow](#choose-a-workflow) | [docs/solver_selection.md#choose-the-smallest-workflow](docs/solver_selection.md#choose-the-smallest-workflow) |
| CSR, CSC, or Matrix Market first-use path | [docs/cookbook.md#first-use-ladder](docs/cookbook.md#first-use-ladder) | Maintained examples linked from the cookbook |
| Diagnostics after a first run | [examples/README.md#diagnostics-handoff](examples/README.md#diagnostics-handoff) | [docs/solver_selection.md#diagnostics-handoff](docs/solver_selection.md#diagnostics-handoff) |
| Installed consumer setup | [Installation](#installation) | [INSTALL.md#start-here](INSTALL.md#start-here) |
| Exact API declarations and ownership contracts | [docs/api_reference.md](docs/api_reference.md) | Public headers under [`include/`](include/) |
| Local benchmark/report interpretation | [benchmarks/README.md](benchmarks/README.md) | Generated report index/manifest artifacts |
| Current algorithm behavior | [docs/algorithm.md](docs/algorithm.md) | [docs/algorithm_history.md](docs/algorithm_history.md) for historical measurement notes |
| Maintainer quality policy | [docs/maintainer_guide.md](docs/maintainer_guide.md) | Sprint planning artifacts when historical traceability is needed |

## Current Capabilities

The inventory below is a capability reference. For first use, follow
[Start Here](#start-here) and widen into these details only when the workflow
needs them.

### Core Data Structure
- **Orthogonal linked-list storage** — each non-zero is linked into both its row list and column list, enabling efficient row and column traversal
- **Slab pool allocator** with free-list for fast node allocation and reuse

### Direct Solvers
- **One-shot direct solves** — LU, Cholesky, LDL^T, and QR remain the default public entry points for most callers.
- **Explicit repeated-run direct lifecycle** — `sparse_analyze()` → `sparse_factor_numeric()` → `sparse_factor_solve()`, then `sparse_refactor_numeric()` between later `sparse_factor_solve()` calls, supports analyze-once / factor-many workflows when the sparsity pattern stays fixed.
- **Dispatch-backed direct kernels** — CSR LU plus CSC Cholesky and LDL^T provide faster large-matrix paths behind the existing public APIs.
- **Multi-RHS and refinement support** — block solves, iterative refinement, rank diagnostics, and minimum-norm QR paths stay available without changing the one-shot-first workflow.

### Singular Value Decomposition (SVD)
- **Full and partial SVD** — dense full SVD plus Lanczos-based partial SVD for the largest singular values.
- **Conditioning and inverse-style tools** — numerical rank, condition number estimation, pseudoinverse, and low-rank approximation.
- **Low-rank output choices** — dense and sparse low-rank approximations stay available under the same public SVD surface.
- **Maintained partial-SVD corpus proof** — generated diagonal fixtures now cover the 8x6 clustered/repeated top-3 lane plus Sprint 151 rank-deficient rectangular projectors, sparse low-rank output, and non-repeated fail-closed recovery rows. This is fixture-local evidence, not broad partial-SVD correctness, raw singular-vector identity, external-library parity, performance, platform/package/ABI, or state-of-the-art support.

### Iterative Solvers
- **Core one-shot solvers** — CG for SPD systems, GMRES for general unsymmetric systems, MINRES for symmetric indefinite systems, and BiCGSTAB as a one-shot compatibility path.
- **Repeated-run iterative handles** — explicit reusable-handle support is intentionally bounded to `CG`, `GMRES`, and `MINRES`.
- **Preconditioning and matrix-free variants** — ILU(0), ILUT, IC(0), and matrix-free callbacks stay available without changing the public workflow boundary.
- **Diagnostics** — residual histories, stagnation detection, and breakdown reporting remain built into the iterative solver surfaces.

### Eigenvalue Infrastructure
- **Symmetric tridiagonal QR algorithm** — implicit QR with Wilkinson shifts and deflation (eigenvalues via `tridiag_qr_eigenvalues`, eigenpairs via `tridiag_qr_eigenpairs`)
- **2×2 symmetric eigensolver** — numerically stable quadratic formula
- **Dense matrix utilities** — Givens rotations, matrix-matrix/vector multiply

### Sparse Symmetric Eigensolver
- **`sparse_eigs_sym`** — k extreme or near-sigma eigenpairs of a symmetric sparse matrix. Three concrete backends are available behind `opts->backend = SPARSE_EIGS_BACKEND_AUTO` (overridable explicitly):
  - **Grow-m Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS`) — full MGS reorthogonalization with a growing-subspace outer loop. Peak memory `O(m_cap · n)`. AUTO default for `n < SPARSE_EIGS_THICK_RESTART_THRESHOLD` (500).
  - **Wu/Simon thick-restart Lanczos** (`SPARSE_EIGS_BACKEND_LANCZOS_THICK_RESTART`) — preserves the converged Ritz subspace in a compact arrowhead basis between restart phases; peak memory `O((k + m_restart) · n)` regardless of total iteration count. AUTO default for `n ≥ 500` when no preconditioner is supplied.
  - **LOBPCG** (`SPARSE_EIGS_BACKEND_LOBPCG`) — Knyazev's Locally Optimal Block Preconditioned Conjugate Gradient with block Rayleigh-Ritz over the `[X | W | P]` subspace, BLOPEX-style conditioning guard, and per-column soft-locking. It composes with the existing IC(0) and LDL^T preconditioner callbacks and AUTO routes here for `n ≥ SPARSE_EIGS_LOBPCG_AUTO_N_THRESHOLD` (1000) when `opts->precond != NULL` and the block size is at least 4.
- Three `which` modes across all backends — `LARGEST`, `SMALLEST`, `NEAREST_SIGMA`. Shift-invert composes with `sparse_ldlt_factor_opts` and the existing LDL^T backend dispatch.
- **Parallel MGS reorthogonalization** — both Lanczos backends parallelise the inner-product / daxpy bodies under `-DSPARSE_OPENMP`, gated on `n ≥ SPARSE_EIGS_OMP_REORTH_MIN_N` (500) so small problems don't pay OMP fork/join overhead.
- **Ritz-pair output** — optional eigenvectors via `compute_vectors = 1`; `result.residual_norm` reports the maximum relative Ritz residual across the converged pairs.
- **Observability** — `result.used_csc_path_ldlt` reports the LDL^T backend chosen on the shift-invert path; `result.peak_basis_size` reports the simultaneously-live `V` columns for memory budgeting; `result.backend_used` reports which backend AUTO actually picked.
- **Optional inverse-iteration refinement post-pass** — `opts->refine = 1` runs Rayleigh-quotient inverse iteration on each converged Ritz pair, refactoring `(A − λ_j I) = L D L^T` per-pair via `sparse_ldlt_factor_opts` (with retry under a small perturbation if the shifted matrix is singular at the converged Ritz value). Default off; budget-bound via `opts->refine_max_iters` (default 5).
- **Preconditioning evidence** — fixture-level LOBPCG runs show IC(0) and
  LDL^T preconditioners reducing iterations on the maintained bcsstk04
  SMALLEST-eigenpair case; treat those rows as local benchmark evidence, not a
  portable speedup guarantee.
- **`bench_eigs`** — permanent benchmark driver at `benchmarks/bench_eigs.c`; CLI with `--sweep default`, `--compare`, and `--matrix <path>` modes, CSV output, and configurable `--repeats`. `benchmarks/README.md` documents the current CLI and CSV schema.

**Picking a backend** — pass `SPARSE_EIGS_BACKEND_AUTO` (the zero default) and let the library choose: small problems run on grow-m Lanczos; medium-to-large problems route to thick-restart Lanczos for the bounded memory; large problems with a preconditioner route to LOBPCG.  Override with an explicit `opts->backend` when profiling or when the workload differs from the bench-corpus heuristics.

### Matrix Operations
- **Sparse matrix-vector product** (SpMV) with optional OpenMP parallelization
- **Block SpMV** — sparse matrix times dense block Y = A·X for nrhs vectors (`sparse_matvec_block`)
- **Sparse matrix-matrix multiply** — C = A*B via Gustavson's algorithm (`sparse_matmul`)
- **Sparse transpose** — compute A^T as a new matrix (`sparse_transpose`)
- **Matrix arithmetic** — scalar scaling (`sparse_scale`) and addition (`sparse_add`)
- **Infinity norm** with internal caching (`sparse_norminf`)

### Symbolic Analysis & Refactorization
- **Elimination tree** computation via Liu's algorithm with path compression
- **Symbolic Cholesky/LU factorization** — predict exact symbolic structure for Cholesky (upper bound on stored numeric factor when dropping is enabled) or upper-bound sparsity structure for LU, without numeric work
- **Analyze-once, factor-many workflow** — `sparse_analyze()` → `sparse_factor_numeric()` → `sparse_refactor_numeric()` for repeated solves with the same sparsity pattern but different values
- **Column counts** — predict symbolic nnz per column of L for pre-allocation (upper bound on stored numeric counts when dropping is enabled)

### Reordering & Preconditioning
- **Fill-reducing reordering** — Reverse Cuthill-McKee (RCM); Approximate Minimum Degree (AMD, ~5·nnz + 6·n + 1 initial integer workspace, growing on demand when fill-in pushes adjacency past the initial bound — scales to large structurally regular fixtures without the earlier bitset's O(n²/64) penalty); Nested Dissection (ND, multilevel vertex separator — best on 2D / 3D PDE meshes); and Column Approximate Minimum Degree (COLAMD) for unsymmetric/QR problems. On the explicit `sparse_analyze()` lifecycle, `sparse_analysis_opts_t.reorder_opts` now carries the shipped typed analysis-time controls for supernodal etree postorder and the highest-value ND routing/coarsening knobs. Legacy `SPARSE_SUPERNODAL_POSTORDER` / `SPARSE_ND_*` env vars remain compatibility overrides only when those typed fields are left unspecified. `bench_reorder --reorder-via-analyze` exercises the analyze-time dispatch path directly from the bench harness.
- **Condition number estimation** — Hager/Higham 1-norm estimator from LU or LDL^T factors, quick R-diagonal estimator from QR

### I/O & Interop
- **Matrix Market I/O** — load and save `.mtx` files through
  `sparse_load_mm(...)` and `sparse_save_mm(...)`; see
  [docs/matrix_market.md](docs/matrix_market.md) for the exact supported
  coordinate formats, duplicate-entry behavior, ownership, and errno contract
- **CSR/CSC export plus compressed-first construction** — convert to/from compressed sparse row/column formats and enter the one-shot direct workflow directly from caller-owned compressed data

### Quality
- **Thread-safe** — concurrent solves on shared factored matrices, per-matrix pool allocators
- **Parallel SpMV** — OpenMP row-wise parallelization (compile with `-DSPARSE_OPENMP`)
- **errno capture** for I/O errors (`sparse_errno`)
- **Progress / cancel callbacks** — `sparse_progress_cb_t` plus
  `opts->progress_cb` / `opts->progress_user` are available across the public
  LU, Cholesky, LDL^T, QR, iterative, and supported eigensolver paths. Callback
  signatures emit `phase`, `step`, `total`, and `elapsed_s`; a non-zero return
  cancels with `SPARSE_ERR_CANCELLED`. See the relevant option headers for
  family-local cancellation and input-mutation contracts.
- **Continuous integration** — support tiers are summarized in
  [INSTALL.md#supported-platforms](INSTALL.md#supported-platforms) and owned
  in detail by [docs/maintainer_guide.md](docs/maintainer_guide.md). In short:
  Linux is the strongest reviewed source of truth, macOS carries reviewed
  static-first install/export proof plus reviewed hosted selected comparison
  freshness for selected generated artifacts. Windows remains CMake-first with
  promoted `test_threads`, `test_sprint4_integration`, and `test_fuzz` CTest
  targets plus reviewed CMake install/downstream validation for the
  static-first package surface.
  Windows still does not claim Makefile parity, `pkg-config` execution parity,
  package-manager support, shared-library support, dynamic ABI support,
  runtime-loader behavior, report freshness, or broad Windows parity.
  Benchmark/report rows remain bounded local evidence rather than portable
  performance claims.

## Choose a Workflow

Start with the smallest surface that matches your real workload:

- **One-shot direct solve:** use LU, Cholesky, LDL^T, or QR when you are
  solving once or only occasionally.
  - Use LU for general square systems; singular systems report
    `SPARSE_ERR_SINGULAR`.
  - Use Cholesky for symmetric positive-definite systems; non-SPD systems
    report `SPARSE_ERR_NOT_SPD`.
  - Use LDL^T for symmetric indefinite systems where the LDL^T API is the
    natural model.
  - Use QR for rectangular or rank-deficient least-squares workflows. The
    maintained corpus proof currently covers a bounded QR family owned by
    [`tests/test_qr_corpus.c`](tests/test_qr_corpus.c): one Sprint 139
    rank-deficient 6x4 nullspace seed plus Sprint 150 rank-deficient
    duplicate/dependent-row and underdetermined minimum-norm fixtures. This is
    not a broad QR or external-library parity claim.
- **Compressed-first one-shot direct entry:** use
  `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)` when your
  matrix already lives in compressed sparse storage and you want a one-shot
  direct path without treating linked-list mutation as the starting point.
- **Stable-pattern repeated direct solve:** use
  `sparse_analyze()` → `sparse_factor_numeric()` →
  `sparse_factor_solve()`, then `sparse_refactor_numeric()` between later
  solves as values change.
- **Repeated-run iterative handle:** use the explicit handle path only for
  `CG`, `GMRES`, or `MINRES` when the problem dimension is stable and
  workspace reuse matters.
- **Repeated-run eigensolver handle:** use the explicit handle path for
  grow-m Lanczos, thick-restart Lanczos, or explicit `LOBPCG` when you are
  reusing the same dimension and want persistent workspace.

Start from these shipped references:

- `example_basic_solve` for the smallest one-shot direct path
- `example_compressed_input` for caller-owned CSR/CSC array input
- `example_analysis` for the analyze-once / factor-many direct lifecycle
- [examples/README.md#start-here](examples/README.md#start-here) for the
  maintained first-use ladder and expected first outputs
- [docs/solver_selection.md](docs/solver_selection.md) for matrix-format and
  solver-family selection
- [docs/cookbook.md#first-use-ladder](docs/cookbook.md#first-use-ladder) for
  data-first CSR, CSC, and Matrix Market routing
- [docs/tutorial.md](docs/tutorial.md) for the fuller repeated-run direct flow

When to widen beyond the first examples:

- examples teach the API workflow
- benchmarks measure retained workflow/performance behavior on the current
  machine, compiler, dependency, fixture, and configuration
- tests own regression, oracle, and property guarantees
- `make bench-canonical-report` writes one bounded snapshot of the maintained
  benchmark surface with generated `index.tsv` / `manifest.txt` methodology
  context and is intentionally not a pass/fail timing gate; unselected
  canonical rows are `status=measurement`, `support_tier=local_only`, and
  `claim_boundary=local_threshold_free`
- `make bench-canonical-report-freshness` regenerates that canonical bundle
  and checks only the selected `bench_refactor_csc` row for
  `nos4.mtx --repeat 1`; the reviewed Linux hosted performance lane runs the
  same selected-row freshness check with hosted metadata on that selected row
  only, still without a timing threshold or portable performance claim
- `make performance-sentinels` writes a local sentinel bundle: its hard
  pass/fail behavior is limited to the existing S5 wall-check lane and the S6
  selected `bench_refactor_csc` local smoke ceiling, while Cholesky CSC and
  LDLT KKT rows are threshold-free measurement context; S5/S6 rows carry
  baseline provenance, while S2/S3 rows carry backend-context caveats rather
  than pass/fail meaning

Selected performance evidence path:

| Need | Start here | Detailed interpretation |
| --- | --- | --- |
| Selected hosted/local freshness | `make bench-canonical-report-freshness` | [benchmarks/README.md#report-index-handoff](benchmarks/README.md#report-index-handoff) |
| Local selected regression smoke gate | `make performance-sentinels` | [benchmarks/README.md#report-index-handoff](benchmarks/README.md#report-index-handoff) |
| Cross-report navigation | `python3 scripts/normalize_report_index.py --check-freshness` | [docs/maintainer_guide.md#normalized-report-index-workflow](docs/maintainer_guide.md#normalized-report-index-workflow) |

Generated report artifacts remain under ignored `build/` paths. Regenerate
them before interpreting rows, and treat them as navigation or local evidence
within their recorded fixture, command, branch, build, and machine context.

If you still need the original coefficient view later, start one-shot direct
paths from a fresh matrix or a fresh `sparse_copy()`.

For direct-solver evidence boundaries and current test ownership, use the
[Maintainer Guide](docs/maintainer_guide.md). The README keeps the adoption
path focused on choosing and running supported public workflows.

## Runtime And Backend Controls

Use public typed options when caller-owned backend or analysis policy matters:

- `sparse_cholesky_opts_t.backend` for Cholesky linked-list/CSC dispatch.
- `sparse_ldlt_opts_t.backend` for LDL^T linked-list/CSC dispatch.
- `sparse_eigs_opts_t.backend` for symmetric eigensolver AUTO/Lanczos/
  thick-restart/LOBPCG dispatch.
- `sparse_analysis_opts_t.reorder_opts` for the shipped analysis-time
  supernodal postorder and ND routing/coarsening controls.

Leave these fields zero-initialized for default/AUTO behavior. Explicit typed
values take precedence over legacy compatibility environment variables where
both exist.

Environment variables such as `SPARSE_CHOL_DENSE_BACKEND`,
`SPARSE_LDLT_DENSE_BACKEND`, `SPARSE_SVD_LOWRANK_OUTER`, FM debug/profile
knobs, `SPARSE_OPENMP`, `OMP_NUM_THREADS`, and test/benchmark opt-ins are
maintainer, build, runtime-context, or report controls. They are useful for
local diagnostics and generated report context, but they are not new public
typed APIs, ABI guarantees, package guarantees, platform parity claims, or
portable performance claims.

Runtime/backend sentinels follow the same boundary: `S5` is the existing
local `wall-check` hard gate, while `S2` Cholesky CSC and `S3` LDLT KKT rows
are threshold-free local context rows in
`build/bench-reports/sentinels/sentinels.tsv`. Generated benchmark,
sentinel, and normalized report-index artifacts stay under ignored `build/`
paths and are not hosted CI proof, package proof, ABI proof, runtime-loader
proof, external-library parity, OpenMP speedup evidence, backend superiority
evidence, or state-of-the-art evidence.

## Building

Most first-time local adoption only needs:

```bash
make
make test
```

Use `make tooling-build` when you want the example and benchmark binaries
without running them yet. Use [INSTALL.md](INSTALL.md) when you need
cross-platform install, downstream-consumer, or install-support detail.

### With Make (recommended)

```bash
make            # build library
make tooling-build  # compile benchmark/example binaries without running them
make lint       # strict compile + static analysis (includes tooling-build)
make quality-review-compile  # reviewed format-check + source-list-check + lint wrapper
make test       # run all unit tests
make quality-review  # reviewed format-check + lint + test + deadcode-check
make quality-review-full  # strongest local reviewed baseline: quality-review + quality-review-cmake
make warning-workflow WARNING_WORKFLOW_LABEL=label  # authoritative repository-wide warning inventory capture
make quality-review-cmake-compile  # reviewed CMake configure + rebuild + ctest -N
make quality-review-cmake  # reviewed CMake configure + rebuild + ctest -N + ctest
make deadcode   # refresh raw dead-code evidence in build/deadcode/
make deadcode-report  # generate classified dead-code report.md / report.tsv
make deadcode-check   # verify report completeness invariants
python3 scripts/normalize_report_index.py --check  # validate normalized report-row construction
python3 scripts/normalize_report_index.py --check-freshness  # inspect report freshness diagnostics
make report-index-oracle-freshness      # selected QR/partial-SVD oracle freshness, mirrored by reviewed Linux hosted CI
make report-index-comparison-freshness  # selected QR + partial-SVD + LU comparison freshness, mirrored by reviewed Linux/macOS hosted CI
make bench-canonical-report-freshness   # selected bench_refactor_csc report freshness, mirrored by reviewed Linux hosted CI
make bench      # run benchmarks
make bench-canonical-report  # write one CSV per canonical maintained benchmark under build/bench-reports/canonical/
make performance-sentinels  # local sentinel bundle: S5/S6 hard gates + threshold-free Cholesky CSC/LDLT KKT context
make large-matrix-guardrails  # generated guardrail index/manifest plus reviewed structural report artifacts
make examples   # build standalone example programs
make docs       # generate Doxygen API reference (requires doxygen)
make docs-check # generate and check local Doxygen API page coverage
make api-docs-freshness # selected local Doxygen freshness plus local-only staging guard
# API reference entry point: docs/api_reference.md
make omp        # build and test with OpenMP-enabled parallel SpMV
make sanitize   # build with undefined-behavior sanitizer
make coverage   # default line-coverage report on the active test surface (80% threshold; backend auto-selected)
make install    # install to PREFIX (default /usr/local)
make uninstall  # remove installed files
make clean      # remove build artifacts
```

The normalized report index is a maintainer navigation and freshness aid. It
does not replace the underlying validation commands and does not turn local
benchmark, coverage, dead-code, comparison, or package metadata rows into
release proof. The reviewed Linux hosted report-freshness lane runs only the
selected oracle gate and the selected comparison gate above. The reviewed macOS
hosted report-freshness lane runs only the selected comparison gate above. The
comparison gate is limited to QR minimum-norm, QR compatible least-squares,
partial-SVD diag6 k2, and linked-list LU nonsymmetric square-solve generated
rows and artifacts; it does not promote broad report-index freshness, selected
oracle freshness on macOS, Windows report freshness, or any unselected
local-only family.

The reviewed Linux hosted selected-performance lane runs only the selected
`bench_refactor_csc` canonical row for `nos4.mtx --repeat 1` through
`make bench-canonical-report` and
`scripts/check_bench_canonical_freshness.py --mode hosted`. It checks artifact
presence, selected row identity, methodology metadata, manifest agreement, and
`hosted_selected_threshold_free` claim boundaries. It does not compare timing
values, set a regression threshold, claim portable speed, promote the other
canonical benchmark rows, or provide external-library, package, ABI, broad
platform, release, or state-of-the-art evidence.

### With CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build .
ctest           # run tests
cmake --install .   # install (supports find_package(Sparse))
```

See [INSTALL.md](INSTALL.md) for detailed cross-platform instructions.

### Compiler Requirements

- C11-compatible compiler (GCC, Clang, etc.)
- Standard math library (`-lm`)

## Quick Start

Use this path if you want to paste one tiny hand-written direct solve. For the
maintained runnable ladder and expected first output, use
[examples/README.md#start-here](examples/README.md#start-here). If your
coefficients already exist as CSR or CSC arrays, skip incremental insertion and
start from `sparse_create_from_csr(...)`, `sparse_create_from_csc(...)`, or the
diagnostic `sparse_from_csr(...)` and `sparse_from_csc(...)` constructors
instead.

```c
#include "sparse_matrix.h"
#include "sparse_lu.h"
#include <stdio.h>

int main(void)
{
    /* Create a 3x3 system: A*x = b */
    SparseMatrix *A = sparse_create(3, 3);
    sparse_insert(A, 0, 0, 2.0);  sparse_insert(A, 0, 1, 1.0);
    sparse_insert(A, 1, 0, 1.0);  sparse_insert(A, 1, 1, 3.0);  sparse_insert(A, 1, 2, 1.0);
    sparse_insert(A, 2, 1, 1.0);  sparse_insert(A, 2, 2, 4.0);

    double b[] = {5.0, 10.0, 13.0};  /* known solution: x = [1, 2, 3] */
    double x[3];

    /* Factor and solve */
    SparseMatrix *LU = sparse_copy(A);
    sparse_lu_factor(LU, SPARSE_PIVOT_PARTIAL, 1e-12);
    sparse_lu_solve(LU, b, x);

    printf("x = [%.6f, %.6f, %.6f]\n", x[0], x[1], x[2]);

    /* Optional: iterative refinement */
    sparse_lu_refine(A, LU, b, x, 5, 1e-15);

    sparse_free(LU);
    sparse_free(A);
    return 0;
}
```

Compile and link:

```bash
make
cc -Iinclude -o example example.c -Lbuild -lsparse_lu_ortho -lm
```

Next steps after the first solve:

- stay on the one-shot path for small or occasional direct solves
- move to a compressed-first one-shot entry path when your matrix already
  arrives as CSR or CSC data; see
  [docs/cookbook.md#first-use-ladder](docs/cookbook.md#first-use-ladder)
- move to [Repeated-Run Direct Workflow](#repeated-run-direct-workflow) when
  the sparsity pattern is stable across many value changes
- move to [Iterative Solver Example](#iterative-solver-example) when the
  matrix/system type makes iterative workflows a better fit
- use [examples/README.md#diagnostics-handoff](examples/README.md#diagnostics-handoff)
  when you need to interpret return codes, residuals, convergence status, or
  rank diagnostics; use
  [docs/solver_selection.md#diagnostics-handoff](docs/solver_selection.md#diagnostics-handoff)
  when you need to decide whether diagnostics justify changing solver family,
  backend, preconditioner, tolerance, or benchmark settings
- use [Installation](#installation) and [INSTALL.md](INSTALL.md) when you need
  installed consumer workflows instead of local build-tree linking

If your coefficients already live in compressed sparse storage, the smallest
one-shot direct entry path is now:

- `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)` to build the
  public matrix shell from caller-owned compressed data
- then the usual one-shot direct family API on that matrix shell

That keeps the linked-list shell as the mutable compatibility owner, but it
avoids forcing compressed-input callers to think of incremental shell mutation
as their natural starting point. The constructor copies the compressed arrays;
the caller still owns and may later change or free those arrays without
changing the returned matrix. Use `sparse_from_csr(...)` or
`sparse_from_csc(...)` when call-site error handling needs an explicit
`sparse_err_t` diagnostic instead of a `NULL` constructor result.

### Iterative Solver Example

```c
#include "sparse_matrix.h"
#include "sparse_iterative.h"
#include "sparse_ilu.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    /* Load a matrix from Matrix Market file */
    SparseMatrix *A = NULL;
    if (sparse_load_mm(&A, "matrix.mtx") != SPARSE_OK) {
        fprintf(stderr, "Failed to load matrix\n");
        return 1;
    }
    int n = sparse_rows(A);

    double *b = malloc(n * sizeof(double));
    double *x = calloc(n, sizeof(double));  /* zero initial guess */
    /* ... set up b ... */

    /* ILU(0) preconditioned GMRES */
    sparse_ilu_t ilu;
    if (sparse_ilu_factor(A, &ilu) != SPARSE_OK) {
        fprintf(stderr, "ILU factorization failed\n");
        free(b); free(x); sparse_free(A);
        return 1;
    }

    sparse_gmres_opts_t opts = {
        .max_iter = 1000,
        .restart = 50,
        .tol = 1e-10,
    };
    sparse_iter_result_t result;
    sparse_err_t err = sparse_solve_gmres(A, b, x, &opts,
                                           sparse_ilu_precond, &ilu, &result);

    if (err == SPARSE_OK)
        printf("Converged in %d iterations, residual = %e\n",
               result.iterations, result.residual_norm);
    else
        printf("Solver returned: %s\n", sparse_strerror(err));

    sparse_ilu_free(&ilu);
    free(b); free(x);
    sparse_free(A);
}
```

Use IC(0) with SPD iterative workflows and ILU(0) / ILUT with general or
indefinite-system workflows. Preconditioner setup routines expect the original
matrix state with identity permutations, so if a matrix may already have been
factored or reordered, start from a fresh `sparse_copy()` of the original.

### Repeated-Run Lifecycle Handles

The library exposes an explicit repeated-run handle path for callers solving
many same-dimension problems while wanting to preserve allocation capacity
between runs.

The one-shot APIs remain fully supported:

- `sparse_solve_cg(...)`
- `sparse_solve_gmres(...)`
- `sparse_solve_minres(...)`
- `sparse_eigs_sym(...)`

Use them when:

- you are solving once or only occasionally
- simplicity matters more than workspace reuse
- you do not want to manage a prepare / free lifecycle explicitly

Use the explicit handle path when:

- the problem dimension is stable across repeated solves
- allocator churn or repeated workspace setup is worth avoiding
- you want one caller-owned object whose capacity can be prepared once and
  reused across runs

The public repeated-run iterative surface is:

- `sparse_iter_handle_t`
- `sparse_iter_handle_init(...)`
- `sparse_iter_handle_prepare_cg(...)`
- `sparse_iter_handle_prepare_gmres(...)`
- `sparse_iter_handle_prepare_minres(...)`
- `sparse_solve_cg_with_handle(...)`
- `sparse_solve_gmres_with_handle(...)`
- `sparse_solve_minres_with_handle(...)`
- `sparse_iter_handle_free(...)`

The public repeated-run eigensolver surface is:

- `sparse_eigs_handle_t`
- `sparse_eigs_handle_init(...)`
- `sparse_eigs_handle_prepare(...)`
- `sparse_eigs_sym_with_handle(...)`
- `sparse_eigs_handle_free(...)`

The lifecycle contract is:

1. zero-initialize the handle or call the init helper
2. optionally prepare it for the stable dimension / working-set shape
3. run the corresponding `*_with_handle(...)` entry as many times as needed
4. free the handle when done

Important behavior:

- reusing a handle preserves allocation capacity, not old numerical iteration
  state
- re-preparing may grow capacity and discards prior Krylov / Ritz /
  search-direction state
- public repeated-run iterative handles are intentionally limited to:
  - `CG`
  - `GMRES`
  - `MINRES`
- the library does not expose public repeated-run handles for:
  - `BiCGSTAB`
  - block iterative workflows
- existing one-shot entries remain the compatibility path and are not
  deprecated

### Repeated-Run Direct Workflow

The direct-solver side uses a different public repeated-run shape from the
iterative and eigensolver handles. The explicit repeated-run direct path is:

- `sparse_analysis_t`
- `sparse_factors_t`
- `sparse_analyze(...)`
- `sparse_factor_numeric(...)`
- `sparse_factor_solve(...)`
- `sparse_refactor_numeric(...)`
- `sparse_analysis_free(...)`
- `sparse_factor_free(...)`

The intended lifecycle is:

1. zero-initialize `sparse_analysis_t` and `sparse_factors_t`
2. analyze once for the chosen direct family
3. factor / solve
4. refactor / solve many on same-pattern value changes
5. free both objects explicitly when done

Important behavior:

- the one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- the compressed-first one-shot entry path is:
  - build the public matrix shell from caller-owned CSR/CSC data with
    `sparse_create_from_csr(...)` or `sparse_create_from_csc(...)`
  - then use the one-shot direct family API when you only need occasional
    solves
- repeated direct reuse preserves symbolic/permutation setup, not old numeric
  factor contents
- failed `sparse_refactor_numeric(...)` calls preserve the previously usable
  factor state on the public repeated-run direct path
- that same old-factor-preservation rule now holds on the large-`n`
  CSC-backed Cholesky lane as well; same-pattern non-SPD retries and obvious
  nnz drift reject instead of silently destroying the previous factor
- `sparse_refactor_numeric(...)` is the public same-pattern numeric-refresh
  path, not a general “accept any changed matrix” rebuild path
- the library now rejects obvious gross-structure drift cheaply, but it does
  not promise a full structural-pattern verifier

## API Overview

| Header | Purpose |
|--------|---------|
| [`sparse_types.h`](include/sparse_types.h) | `idx_t`, error codes (`sparse_err_t`), pivot/reorder strategies, version macros |
| [`sparse_matrix.h`](include/sparse_matrix.h) | Sparse matrix lifecycle, element access, SpMV, block SpMV, Matrix Market I/O |
| [`sparse_lu.h`](include/sparse_lu.h) | LU factorization, solve, block solve, condition estimation, iterative refinement |
| [`sparse_lu_csr.h`](include/sparse_lu_csr.h) | CSR LU working format — conversion, scatter-gather elimination, dense block detection, block solve |
| [`sparse_cholesky.h`](include/sparse_cholesky.h) | Cholesky factorization and solve for SPD matrices |
| [`sparse_ldlt.h`](include/sparse_ldlt.h) | LDL^T factorization with Bunch-Kaufman pivoting for symmetric indefinite matrices |
| [`sparse_analysis.h`](include/sparse_analysis.h) | Symbolic analysis, numeric factorization, refactorization (analyze-once workflow) |
| [`sparse_iterative.h`](include/sparse_iterative.h) | CG, GMRES, MINRES, BiCGSTAB; block CG/GMRES/MINRES; GMRES left/right preconditioning; explicit repeated-run handles for CG/GMRES/MINRES |
| [`sparse_ilu.h`](include/sparse_ilu.h) | ILU(0) and ILUT incomplete factorization preconditioners |
| [`sparse_ic.h`](include/sparse_ic.h) | IC(0) incomplete Cholesky preconditioner for SPD systems |
| [`sparse_qr.h`](include/sparse_qr.h) | Column-pivoted QR factorization, least-squares, rank, null space, refinement |
| [`sparse_dense.h`](include/sparse_dense.h) | Dense matrix utilities, Givens rotations, 2×2 eigensolver, tridiag QR |
| [`sparse_bidiag.h`](include/sparse_bidiag.h) | Householder bidiagonalization (SVD preprocessing) |
| [`sparse_csr.h`](include/sparse_csr.h) | CSR/CSC compressed format conversion plus compressed-first matrix construction |
| [`sparse_reorder.h`](include/sparse_reorder.h) | Fill-reducing reordering (RCM, AMD, ND, COLAMD), permutation, bandwidth |
| [`sparse_svd.h`](include/sparse_svd.h) | SVD, partial SVD, condition number, pseudoinverse, low-rank approximation |
| [`sparse_eigs.h`](include/sparse_eigs.h) | Sparse symmetric eigensolver — Lanczos/LOBPCG backends, shift-invert mode, Ritz pairs, explicit repeated-run handle |
| [`sparse_vector.h`](include/sparse_vector.h) | Dense vector utilities (norms, axpy, dot product) |

### Key Functions

**Matrix lifecycle:**
- `sparse_create(rows, cols)` — create an empty matrix
- `sparse_free(mat)` — free all memory
- `sparse_copy(mat)` — deep copy

**Element access:**
- `sparse_insert(mat, row, col, val)` — insert or update (inserting 0.0 removes)
- `sparse_get_phys(mat, row, col)` — read at physical index
- `sparse_get(mat, row, col)` / `sparse_set(mat, row, col, val)` — logical (through permutations)

**Solving linear systems:**
- `sparse_lu_factor(mat, pivot, tol)` — in-place LU decomposition
- `sparse_lu_factor_opts(mat, &opts)` — LU with optional fill-reducing reordering (RCM/AMD/ND)
- `sparse_lu_solve(mat, b, x)` — solve using factored matrix (auto-unpermutes if reordered)
- `sparse_lu_condest(A, LU, &cond)` — estimate 1-norm condition number from LU factors
- `sparse_lu_refine(A, LU, b, x, max_iters, tol)` — iterative refinement

**CSR LU (high-performance path):**
- `lu_csr_from_sparse(A, fill_factor, &csr)` — convert to CSR working format
- `lu_csr_eliminate(csr, tol, drop_tol, piv)` — scatter-gather LU elimination
- `lu_csr_eliminate_block(csr, tol, drop_tol, min_block, piv)` — with dense block optimization
- `lu_csr_solve(csr, piv, b, x)` — forward/backward substitution in CSR
- `lu_csr_solve_block(csr, piv, B, nrhs, X)` — block solve for multiple RHS
- `lu_csr_factor_solve(A, b, x, tol)` — one-shot convert + factor + solve
- `lu_detect_dense_blocks(csr, min_size, threshold, &blocks, &nblocks)` — supernodal dense block detection

**Cholesky (SPD matrices):**
- `sparse_cholesky_factor(mat)` — in-place A = L·L^T
- `sparse_cholesky_factor_opts(mat, &opts)` — with optional AMD/RCM/ND reordering
- `sparse_cholesky_solve(mat, b, x)` — solve using Cholesky factors

**LDL^T (symmetric indefinite matrices):**
- `sparse_ldlt_factor(A, &ldlt)` — P·A·P^T = L·D·L^T with Bunch-Kaufman 1x1/2x2 pivoting
- `sparse_ldlt_factor_opts(A, &opts, &ldlt)` — with optional AMD/RCM/ND fill-reducing reordering
- `sparse_ldlt_solve(&ldlt, b, x)` — solve using LDL^T factors (auto-unpermutes)
- `sparse_ldlt_inertia(&ldlt, &pos, &neg, &zero)` — eigenvalue sign count from D blocks
- `sparse_ldlt_refine(A, &ldlt, b, x, max_iters, tol)` — iterative refinement
- `sparse_ldlt_condest(A, &ldlt, &cond)` — 1-norm condition estimate via Hager/Higham
- `sparse_ldlt_free(&ldlt)` — free factorization data

**Symmetric eigensolvers:**
- `sparse_eigs_sym(A, k, &opts, &result)` — k extreme or near-sigma eigenpairs of symmetric A through the public AUTO/Lanczos/thick-restart/LOBPCG backend surface
- `sparse_eigs_handle_init(&handle)` / `sparse_eigs_handle_prepare(&handle, n, k, &opts)` / `sparse_eigs_sym_with_handle(A, k, &opts, &result, &handle)` / `sparse_eigs_handle_free(&handle)` — explicit repeated-run lifecycle path for stable-dimension symmetric eigensolves
- `opts.which` = `SPARSE_EIGS_LARGEST` / `_SMALLEST` / `_NEAREST_SIGMA`; the shift-invert mode composes with `sparse_ldlt_factor_opts`
- `opts.compute_vectors = 1` populates `result.eigenvectors` (column-major, caller-owned); `result.used_csc_path_ldlt` reports the inner LDL^T backend for shift-invert; `result.backend_used` records the concrete backend selected on successful AUTO calls

**Symbolic analysis & refactorization:**
- `sparse_analyze(A, &opts, &analysis)` — compute elimination tree, column counts, symbolic structure
- `sparse_factor_numeric(A, &analysis, &factors)` — numeric-only factorization using precomputed analysis
- `sparse_refactor_numeric(A_new, &analysis, &factors)` — refactor with new values (same pattern)
- `sparse_factor_solve(&factors, &analysis, b, x)` — solve using factors with auto-permutation
- `sparse_analysis_free(&analysis)` / `sparse_factor_free(&factors)` — cleanup

The direct repeated-run contract is therefore:

- analyze once
- factor / solve
- refactor / solve many

with reuse preserving symbolic/permutation setup rather than stale numeric
factor contents.

**QR factorization (rectangular & rank-deficient):**
- `sparse_qr_factor(A, &qr)` — column-pivoted QR: A*P = Q*R
- `sparse_qr_factor_opts(A, &opts, &qr)` — with optional AMD column reordering
- `sparse_qr_solve(&qr, b, x, &residual)` — least-squares for overdetermined systems; basic solution for underdetermined systems
- `sparse_qr_apply_q(&qr, transpose, x, y)` — apply Q or Q^T to a vector
- `sparse_qr_rank(&qr, tol)` — numerical rank estimation
- `sparse_qr_nullspace(&qr, tol, basis, &ndim)` — null-space basis extraction
- `sparse_qr_solve_minnorm(A, b, x, &opts)` — minimum 2-norm solution for underdetermined systems
- `sparse_qr_diag_r(&qr, diag)` — extract R diagonal for rank inspection
- `sparse_qr_rank_info(&qr, tol, &info)` — comprehensive rank diagnostics with condition estimate
- `sparse_qr_condest(&qr)` — quick condition estimate from R diagonal
- `sparse_qr_refine_minnorm(A, b, x, iters, &resid, &opts)` — iterative refinement for minimum-norm
- `sparse_qr_free(&qr)` — free QR factors
- `sparse_reorder_colamd(A, perm)` — COLAMD column ordering for unsymmetric/QR (handles rectangular)

The maintained QR corpus proof covers six fixture-local rows:
`qr_rank_deficient_6x4_nullspace_v1`, `qr_rankdef_duplicate_5x4_v1`,
`qr_rankdef_dependent_row_4x3_v1`, `qr_underdetermined_minnorm_2x4`,
`qr_minnorm_3x6_exact_values`, and `qr_minnorm_5x10_exact_values`. The
source-controlled proof owner is
[`tests/test_qr_corpus.c`](tests/test_qr_corpus.c), and the opt-in local
oracle/report freshness gate is `make report-index-oracle-freshness`. That
selected oracle gate and the split oracle artifacts are also run in the
reviewed Linux hosted report-freshness lane. The selected comparison freshness
gate is
`make report-index-comparison-freshness`, which checks selected fixture-local
QR minimum-norm and compatible least-squares comparisons, the selected
fixture-local partial-SVD diagonal top-k comparison for `partial_svd_diag6_k2`,
and the selected fixture-local linked-list LU square-solve comparison for
`lu_nonsym_square_5` against the selected source-controlled dense reference
helpers. The same gate is mirrored by reviewed Linux and macOS hosted
report-freshness lanes for selected comparison artifacts only. These gates do
not prove raw QR basis parity, raw singular-vector identity, broad LU or
nonsymmetric solve correctness, LU CSR parity, broad
rank-threshold policy, broad rank-deficient solve, broad minimum-norm
behavior, broad SVD or partial-SVD correctness, SuiteSparse, LAPACK, NumPy,
SciPy, Windows report freshness, broad platform parity, performance,
package/ABI, release, or
state-of-the-art evidence.

**SVD:**
- `sparse_svd_compute(A, &opts, &svd)` — full SVD: A = U·Σ·V^T (singular values only or with vectors)
- `sparse_svd_partial(A, k, &opts, &svd)` — k largest singular values via Lanczos bidiagonalization; maintained corpus evidence is fixture-local to clustered/repeated top-k, rank-deficient projector, sparse low-rank output, and fail-closed recovery rows
- `sparse_cond(A, &err)` — 2-norm condition number via SVD
- `sparse_svd_rank(A, tol, &rank)` — numerical rank estimation
- `sparse_pinv(A, tol, &pinv)` — Moore-Penrose pseudoinverse
- `sparse_svd_lowrank(A, k, &dense)` — best rank-k approximation (dense output)
- `sparse_svd_lowrank_sparse(A, k, drop_tol, &sparse)` — best rank-k approximation (sparse output)
- `sparse_svd_free(&svd)` — free SVD result

**Iterative solvers:**
- `sparse_solve_cg(A, b, x, &opts, precond, ctx, &result)` — Preconditioned Conjugate Gradient (SPD only)
- `sparse_iter_handle_init(&handle)` / `sparse_iter_handle_prepare_cg(&handle, n)` / `sparse_solve_cg_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` / `sparse_iter_handle_free(&handle)` — explicit repeated-run CG lifecycle path
- `sparse_solve_gmres(A, b, x, &opts, precond, ctx, &result)` — Restarted GMRES(k) with left/right preconditioning
- `sparse_iter_handle_prepare_gmres(&handle, n, restart)` / `sparse_solve_gmres_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` — explicit repeated-run GMRES lifecycle path
- `sparse_iter_handle_prepare_minres(&handle, n)` / `sparse_solve_minres_with_handle(A, b, x, &opts, precond, ctx, &result, &handle)` — explicit repeated-run MINRES lifecycle path
- `sparse_cg_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block CG for multiple RHS
- `sparse_gmres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block GMRES for multiple RHS
- `sparse_solve_cg_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free CG
- `sparse_solve_gmres_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free GMRES
- `sparse_solve_minres(A, b, x, &opts, precond, ctx, &result)` — MINRES for symmetric (possibly indefinite) systems
- `sparse_minres_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block MINRES for multiple RHS
- `sparse_solve_bicgstab(A, b, x, &opts, precond, ctx, &result)` — BiCGSTAB for general nonsymmetric systems
- `sparse_bicgstab_solve_block(A, B, nrhs, X, &opts, precond, ctx, &result)` — Block BiCGSTAB for multiple RHS
- `sparse_solve_bicgstab_mf(matvec, ctx, n, b, x, &opts, precond, ctx, &result)` — Matrix-free BiCGSTAB

The public repeated-run iterative handle support remains intentionally bounded
to `CG`, `GMRES`, and `MINRES`; `BiCGSTAB` and block iterative workflows
remain one-shot compatibility surfaces.

**ILU(0) / ILUT preconditioners:**
- `sparse_ilu_factor(A, &ilu)` — ILU(0) factorization (no fill-in beyond A's pattern)
- `sparse_ilut_factor(A, &opts, &ilu)` — ILUT with threshold dropping and controlled fill-in
- `sparse_ilu_solve(&ilu, r, z)` — apply preconditioner: solve L*U*z = r
- `sparse_ilu_precond` / `sparse_ilut_precond` — callbacks compatible with `sparse_precond_fn`
- `sparse_ilu_free(&ilu)` — free ILU/ILUT factors

**IC(0) preconditioner (incomplete Cholesky):**
- `sparse_ic_factor(A, &ic)` — IC(0) factorization for SPD matrices (L*L^T ≈ A, no fill-in)
- `sparse_ic_solve(&ic, r, z)` — apply preconditioner: solve L*L^T*z = r
- `sparse_ic_precond` — callback compatible with `sparse_precond_fn`
- `sparse_ic_free(&ic)` — free IC(0) factors

**Fill-reducing reordering:**
- `sparse_reorder_rcm(A, perm)` — Reverse Cuthill-McKee ordering
- `sparse_reorder_amd(A, perm)` — Approximate Minimum Degree ordering
- `sparse_reorder_nd(A, perm)` — Nested Dissection (multilevel vertex-separator); best on 2D / 3D PDE meshes
- `sparse_permute(A, row_perm, col_perm, &B)` — apply permutation
- `sparse_bandwidth(A)` — compute matrix bandwidth

**Matrix arithmetic:**
- `sparse_matmul(A, B, &C)` — sparse matrix-matrix multiply (Gustavson's algorithm)
- `sparse_scale(mat, alpha)` — in-place scalar multiplication
- `sparse_add(A, B, alpha, beta, &C)` — C = alpha*A + beta*B
- `sparse_add_inplace(A, B, alpha, beta)` — A = alpha*A + beta*B
- `sparse_norminf(mat, &norm)` — infinity norm (cached)

**I/O and format conversion:**
- `sparse_save_mm(mat, filename)` / `sparse_load_mm(&mat, filename)` — Matrix Market format
- `sparse_to_csr(mat, &csr)` / `sparse_create_from_csr(csr)` — CSR export and compressed-first construction; free exported storage with `sparse_csr_free(csr)`
- `sparse_to_csc(mat, &csc)` / `sparse_create_from_csc(csc)` — CSC export and compressed-first construction; free exported storage with `sparse_csc_free(csc)`
- `sparse_from_csr(csr, &mat)` / `sparse_from_csc(csc, &mat)` — diagnostic compressed-first constructors when you need explicit `sparse_err_t` import status; on success the caller owns the returned `SparseMatrix`, and the input arrays remain caller-owned
- `sparse_errno()` — retrieve system errno after I/O failure

All functions return `sparse_err_t` error codes (except accessors that return values directly). See `sparse_strerror()` for human-readable error messages.

## Performance Characteristics

The README keeps only the high-level performance story. Benchmark command
syntax, CSV schemas, current benchmark grouping, and measurement caveats live in
[benchmarks/README.md](benchmarks/README.md).

For n x n sparse matrices with nnz non-zeros:

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert/Remove | O(row_nnz + col_nnz) | Maintains sorted row and column order |
| MatVec | O(nnz) | Parallel with OpenMP when enabled |
| LU Factor | O(n^3) worst case | Usually much better on sparse structured matrices |
| Solve | O(nnz_LU) | Forward/back substitution |
| QR Factor | O(mn^2) | Householder transformations |
| SVD | O(mn^2) | Bidiagonalization plus QR |

Current benchmark surfaces cover:

- one-shot LU, Cholesky, LDL^T, SVD, eigensolver, SpMV, and iterative
  comparison paths;
- repeated-run direct workflows through `bench_refactor` and
  `bench_refactor_csc`;
- repeated-run iterative and eigensolver handle workflows through
  `bench_iterative_reuse` and `bench_eigs_reuse`;
- dispatch-backed CSR LU and CSC Cholesky/LDL^T paths that avoid linked-list
  pointer chasing on large matrix workloads.

Use `make bench-canonical-report` for one bounded local snapshot of the
maintained benchmark surface. Treat emitted benchmark rows as branch-local
measurement artifacts, not portable performance guarantees; the generated
`index.tsv` records methodology fields such as support tier, claim boundary,
repeat semantics, warmup and variance state, baseline, threshold, and
methodology notes. Use `make bench-canonical-report-freshness` for the
selected `bench_refactor_csc` row freshness check. Use
`make performance-sentinels` when you need the bounded local sentinel bundle:
it reports the existing hard `wall-check` gate, the S6 selected-lane local
smoke ceiling, and threshold-free Cholesky CSC / LDLT KKT backend context
under the current backend and thread settings. See
[benchmarks/README.md#report-index-handoff](benchmarks/README.md#report-index-handoff)
for generated row interpretation.

## Thread Safety

The library is safe for concurrent use under the following contract:

| Operation | Thread-safe? | Notes |
|-----------|:---:|-------|
| Concurrent solves on the same factored matrix | Yes | Solve reads `factor_norm` and linked-list structure (immutable after factorization) |
| Concurrent `sparse_norminf()` on the same matrix | Yes | `cached_norm` is `_Atomic double` with relaxed ordering; idempotent computation |
| Concurrent factorization of different matrices | Yes | Each matrix has its own pool allocator |
| Concurrent read-only access (nnz, get, matvec) | Yes | No shared mutable state |
| `sparse_errno()` | Yes | Uses `_Thread_local` storage |
| Concurrent mutation of the same matrix | **No** | Insert/remove/factor on a shared matrix requires external synchronization |
| Factorization concurrent with solve on same matrix | **No** | Factorization mutates structure; solve must wait until factorization completes |

**Mutable fields in SparseMatrix:**

| Field | Mutated by | Thread safety |
|-------|-----------|---------------|
| `cached_norm` | `sparse_norminf()` | `_Atomic double` — safe for concurrent reads/writes |
| `factor_norm` | Factorization functions | Written once during factorization, read during solve — no race (factorization completes before solve) |
| Pool, `row_headers`, `col_headers`, `nnz` | `sparse_insert()`, `sparse_remove()`, factorization | Not atomic — requires external synchronization or `SPARSE_MUTEX` |
| Permutation arrays | Factorization functions | Single-threaded context only |

**Optional mutex support:** Compile with `-DSPARSE_MUTEX` and `-pthread` to add per-matrix mutex locking on `sparse_insert()` and `sparse_remove()`. This serializes concurrent insert/remove calls on the same matrix. Note: factorization (`sparse_lu_factor`, `sparse_cholesky_factor`) is not mutex-protected and must not be called concurrently on the same matrix. Not recommended — prefer separate matrices per thread.

## Known Limitations

- **Default reviewed width remains 32-bit.** The shipped reviewed build uses
  `SPARSE_IDX_BITS=32`, so `idx_t` still limits matrix dimensions and nonzero
  counts to ~2.1 billion. Wider indices are now a bounded compile-time seam
  through `SPARSE_IDX_BITS=64`; downstream callers must rebuild against that
  same width contract.
- **In-place factorization.** `sparse_lu_factor` and `sparse_cholesky_factor` overwrite the matrix; always work on a copy if you need the original. (The CSR path via `lu_csr_factor_solve` does not modify the input.)
- **Factored-state validation.** Solve functions check an internal `factored` flag and return `SPARSE_ERR_BADARG` if the matrix has not been factored. Modifying a factored matrix (insert/remove) clears the flag. For externally-constructed factors (e.g., imported from CSR), call `sparse_mark_factored()` before solving.
- **Scalar support is still real-only.** The current dense-scalar public seam
  is named `sparse_scalar_t`. The shared matrix-shell helper and storage/build
  seam plus the iterative, eigensolver, and QR public buffer seams now route
  through that alias, but the shipped contract still binds it to real double
  precision only. This is bounded preparation for later widening, not a claim
  of complex or broad generic-scalar support today.

## Testing and Quality

The default local regression path is:

```bash
make test
```

Common focused quality commands are:

```bash
make smoke
make lint
make quality-review
make quality-review-full
make sanitize
make asan
make tsan
make coverage
```

On macOS with Apple Clang, `make asan` can hang because of a toolchain/runtime
limitation. Use GCC or LLVM clang for ASan on macOS, or use `make sanitize` for
the maintained UBSan path.

The test framework also supports opt-in non-default coverage when fixture size,
runtime, or intentionally non-default behavior makes a check unsuitable for the
ordinary default suite:

```bash
SPARSE_TEST_SLOW=1 make test
SPARSE_TEST_EXPERIMENTAL=1 make test
SPARSE_TEST_LARGE=1 make test
```

Use `make quality-review-full` for the strongest local reviewed baseline. For
exact wrapper expansion, dead-code workflow meaning, warning authority, and
cross-platform reviewed/staged interpretation, use the
[Maintainer Guide](docs/maintainer_guide.md) and the executable targets in the
`Makefile`.

Historical measurements, retired targets, and old sprint evidence belong in
`docs/planning/` artifacts, not in the README front door.

## Project Structure

```
linalg_sparse_orthogonal/
├── include/              Public headers
│   ├── sparse_types.h        Error codes, index type (includes sparse_version.h)
│   ├── sparse_version.h      Version macros (generated from VERSION file)
│   ├── sparse_matrix.h       Core data structure, SpMV, block SpMV, I/O
│   ├── sparse_lu.h           LU factorization, solve, block solve
│   ├── sparse_lu_csr.h       CSR LU — scatter-gather elimination, dense blocks
│   ├── sparse_cholesky.h     Cholesky factorization and solve
│   ├── sparse_iterative.h    CG, GMRES, MINRES, BiCGSTAB; block variants; GMRES left/right precond; repeated-run handles for CG/GMRES/MINRES
│   ├── sparse_ilu.h          ILU(0) and ILUT preconditioners
│   ├── sparse_ic.h           IC(0) incomplete Cholesky preconditioner
│   ├── sparse_qr.h           QR factorization, least-squares, rank, null space
│   ├── sparse_dense.h        Dense utilities, Givens, eigensolvers
│   ├── sparse_bidiag.h       Householder bidiagonalization
│   ├── sparse_csr.h          CSR/CSC conversion
│   ├── sparse_reorder.h      Fill-reducing reordering (RCM, AMD, ND, COLAMD)
│   ├── sparse_svd.h          SVD, condition number, pseudoinverse, low-rank
│   └── sparse_vector.h       Dense vector utilities
├── src/                  Library implementation
├── tests/                Unit tests
├── cmake/                CMake config templates
├── examples/             Standalone example programs and CMake integration example
├── benchmarks/           Performance benchmarks and workflow-specific reuse drivers
├── docs/                 Algorithm/format documentation + planning
│   └── planning/         Sprint plans, retrospectives, and project plans
├── INSTALL.md            Cross-platform installation guide
├── sparse.pc.in          pkg-config template
└── archive/              Original prototype files
```

## Installation

Use [INSTALL.md#start-here](INSTALL.md#start-here) for platform-specific
setup, staged installs, downstream consumer workflows, and install-surface
validation. The README keeps only the shortest static-first local summary:

```bash
# Makefile
make && make test && make install PREFIX=/usr/local

# CMake
cmake -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build
cmake --install build
```

After installation, downstream projects can use either `pkg-config` or
`find_package(Sparse)` against the maintained static package surface:

- `pkg-config --cflags --libs sparse`
- `find_package(Sparse REQUIRED)` plus
  `target_link_libraries(... Sparse::sparse_lu_ortho)`

The installed `sparse.pc` metadata is intentionally static-archive scoped, and
the install proof checks downstream compile/link/run behavior plus exact
package version handling. The Unix proof validates installed include and
library paths by filesystem identity so staged-prefix spelling differences do
not masquerade as package failures. These checks are package proof, not
package-manager distribution or dynamic-loader evidence.

Package-manager support is not currently provided; use source install via Make
or CMake and see `INSTALL.md` for the exact package boundary.

On Windows, `sparse.pc` is installed and inspected as static package metadata
by the reviewed CMake install/downstream lane. That lane does not run
`pkg-config` and does not claim Windows Makefile or `pkg-config` execution
parity.

Shared-library packaging is intentionally deferred; the maintained install
contract is the static archive surface described in [INSTALL.md](INSTALL.md).
CMake rejects `BUILD_SHARED_LIBS=ON` rather than silently treating a
shared-library request as supported, and the rejection names the missing
export/import, symbol visibility, dynamic ABI, platform loader metadata,
installed shared consumer, and runtime-loader validation policies.
The canonical Sprint 170 package and ABI product decision is recorded in
`docs/planning/EPIC_15/SPRINT_170/artifacts/day9-shared-library-abi-product-decision.md`.

## Documentation

- [Tutorial](docs/tutorial.md) — fuller user walkthrough for repeated-run and API workflows
- [Cookbook](docs/cookbook.md) — compressed-first direct, iterative, Matrix Market, SVD, eigensolver, and benchmark handoff paths
- [Examples](examples/README.md) — shipped example binaries and local usage references
- [Benchmarks](benchmarks/README.md) — benchmark commands, CSV fields, and measurement interpretation
- [Algorithm Reference](docs/algorithm.md) — current data structures,
  solver algorithms, compressed formats, and complexity notes
- [Matrix Market Format](docs/matrix_market.md) — supported features and limitations
- [Maintainer Guide](docs/maintainer_guide.md) — repository-wide quality-contract interpretation and documentation ownership
- [Installation Guide](INSTALL.md) — cross-platform build and install instructions

## License

This project is for research and educational purposes.
