# Sprint 104 Day 6 OpenMP and Threading Audit

## Purpose

Day 6 inventories the current OpenMP, pthread, thread-local override, and
runtime-control surfaces before any cleanup. The goal is to rank cleanup
candidates for Day 7 without changing behavior, weakening compatibility
knobs, or confusing serial and OpenMP validation lanes.

## Executive Summary

The project does not currently expose a public thread-pool or runtime-thread
control API. Parallelism is a compile-time opt-in through `SPARSE_OPENMP`, with
OpenMP pragmas limited to:

- linked-list sparse matrix-vector products in `src/sparse_matrix.c`;
- eigensolver MGS reorthogonalization in `src/sparse_eigs.c`.

Most runtime variability comes from process-global environment variables and
thread-local graph/reorder override scopes. The compatibility risk is not the
OpenMP code itself; it is the mix of process-global env parsing, test-time env
mutation, and thread-local internal overrides that can look like a unified
runtime system but are actually independent mechanisms.

## OpenMP Compile-Time and Runtime Inventory

| surface | owner | current behavior | compatibility note |
|---|---|---|---|
| `SPARSE_OPENMP` Makefile flag | `Makefile` | adds `-DSPARSE_OPENMP` and platform OpenMP linker flags | opt-in only; serial build remains default |
| `SPARSE_OPENMP` CMake option | `CMakeLists.txt` | finds OpenMP, publishes compile definition, links OpenMP target | opt-in only; reviewed CMake consumer path can stay serial |
| `sparse_matvec()` | `src/sparse_matrix.c` | `#pragma omp parallel for schedule(dynamic, 64)` over logical rows | each row writes one `y[i]`; no user-visible thread count control |
| `sparse_matvec_block()` | `src/sparse_matrix.c` | same row-parallel pattern for block RHS output | row partitions are independent; inner RHS loop stays serial |
| `s21_mgs_reorth()` | `src/sparse_eigs.c` | OpenMP reduction/daxpy over vector length inside serial MGS outer loop | keeps MGS stability; threshold defaults to 500 |
| `SPARSE_EIGS_OMP_REORTH_MIN_N` | `src/sparse_eigs.c` | compile-time threshold for eigensolver OpenMP reorth | compile-time only; not a runtime env knob |
| `test_omp` | `tests/test_omp.c` | correctness suite that reports OpenMP status | can run in serial or OpenMP builds |
| TSan OpenMP lane | `.github/workflows/ci.yml` and `Makefile` | Ubuntu/clang OpenMP eigensolver checks with suppressions | protects reorth paths, not every OpenMP consumer |
| benchmark OpenMP label | `benchmarks/bench_main.c` | prints OpenMP max thread count when compiled with OpenMP | diagnostic only |

## Public Thread-Control Surface

There is no public `sparse_set_num_threads`, thread-pool handle, scheduler
policy object, or per-call thread limit. The only user-facing controls are
external runtime mechanisms:

- build with or without `SPARSE_OPENMP`;
- set OpenMP runtime variables such as `OMP_NUM_THREADS` before execution;
- optionally build with `SPARSE_MUTEX` for per-matrix insert/remove locking.

This is a coherent small-library stance, but it should be documented as such
before adding any new runtime knobs.

## Thread-Local State Map

| state | owner | role | risk |
|---|---|---|---|
| `last_errno` | `src/sparse_types.c` | per-thread captured system errno | low; intentionally thread-local diagnostic |
| ND profiling accumulators | `src/sparse_reorder_nd.c` | per-thread `SPARSE_ND_PROFILE` counters | medium; env read is process-global but accumulation is thread-local |
| ND policy override scopes | `src/sparse_reorder_nd.c` | typed policy overrides mapped into graph internals | medium; scope begin/end must stay balanced |
| graph coarsening overrides | `src/sparse_graph_coarsen.c` | forced HEM fallback, typed coarsening, floor ratio, HCC debug | medium; thread-local protects concurrent partition calls |
| FM runtime controls | `src/sparse_graph_refine.c` | FIFO, annealing, thick-restart, gain-noise strategy state | medium; many env-derived modes flow through one thread-local runtime |
| separator lift overrides | `src/sparse_graph_separator.c` | typed separator strategy/weight | medium; interacts with ND typed policy |
| coarsest bisection override | `src/sparse_graph_bisect.c` | typed bisection policy | medium; interacts with root/coarsest env controls |

These thread-local scopes are internal correctness tools. They are not public
thread controls and should not be documented as user-level threading features.

## Process-Global Environment Controls

The source tree currently reads these `SPARSE_*` environment variables:

| variable | domain | current meaning |
|---|---|---|
| `SPARSE_CHOL_DENSE_BACKEND` | dense backend | best-effort Cholesky dense-kernel request; unknown falls back to builtin |
| `SPARSE_LDLT_DENSE_BACKEND` | dense backend | best-effort LDLT dense-factor request; unknown falls back to builtin |
| `SPARSE_ND_PROFILE` | profiling | enables nested-dissection phase timing |
| `SPARSE_QG_PROFILE` | profiling | enables quotient-graph AMD timing |
| `SPARSE_ND_ROOT_BISECT` | graph/reorder | root bisection policy compatibility override |
| `SPARSE_ND_ROOT_BISECT_MAX_N` | graph/reorder | root spectral size cap |
| `SPARSE_ND_COARSENING` | graph/reorder | heavy-edge vs HCC coarsening compatibility override |
| `SPARSE_ND_COARSEN_FLOOR_RATIO` | graph/reorder | coarsest-level size control |
| `SPARSE_ND_COARSENING_CV_FALLTHROUGH` | graph/reorder | HCC-to-HEM fall-through threshold |
| `SPARSE_ND_COARSEST_BISECTION` | graph/reorder | coarsest bisection compatibility override |
| `SPARSE_ND_SEP_LIFT_STRATEGY` | graph/reorder | separator lift strategy |
| `SPARSE_ND_SEP_LIFT_WEIGHT` | graph/reorder | separator lift weight policy |
| `SPARSE_SUPERNODAL_POSTORDER` | analysis/reorder | supernodal etree postorder opt-in |
| `SPARSE_ND_SUPERNODAL_POSTORDER` | analysis/reorder | legacy alias for supernodal postorder |
| `SPARSE_FM_FINEST_STRATEGY` | graph/FM | baseline, FIFO, annealing, thick-restart, ensemble mode |
| `SPARSE_FM_ANNEALING_SCHEDULE` | graph/FM | annealing schedule |
| `SPARSE_FM_THICK_RESTART_PERTURB` | graph/FM | thick-restart perturbation |
| `SPARSE_FM_GAIN_NOISE_SCHEDULE` | graph/FM | gain-noise schedule |
| `SPARSE_FM_ENSEMBLE_STRATEGIES` | graph/FM | ensemble strategy list |
| `SPARSE_HCC_DEBUG` | graph/debug | enables HCC debug output |
| `SPARSE_SVD_LOWRANK_OUTER` | SVD | low-rank SVD outer-loop tuning |
| `SPARSE_TEST_LARGE` | tests | test fixture size opt-in |

Process-global env variables are acceptable compatibility and benchmark knobs,
but they are poor per-call controls. Tests that mutate them must clean up in the
same test body and should not be parallelized at process level without isolation.

## Nested-Parallelism Risk Table

| area | current parallelism | nested risk | Day 7 disposition |
|---|---|---|---|
| SpMV | OpenMP row loop in `sparse_matvec` and `sparse_matvec_block` | medium when called from iterative solvers, eigensolvers, SVD, or graph spectral bisection under an already parallel caller | document and add guard comments before changing scheduling |
| CG, GMRES, BiCGSTAB, MINRES | call SpMV repeatedly, no own OpenMP regions found | medium through SpMV only | keep serial solver loops; avoid adding outer OpenMP until runtime policy exists |
| Lanczos / thick restart | OpenMP inner reorth loops plus SpMV calls | medium because one iteration can call both SpMV and reorth | preserve serial outer loops and threshold gate |
| LOBPCG | block operations and SpMV, no local OpenMP regions found | low-to-medium through SpMV | no cleanup needed before broader runtime API |
| SVD | composes with eigensolver/SpMV paths | medium through eigensolver and SpMV | validate through existing eigensolver/SVD tests after runtime changes |
| Cholesky / LDLT CSC | no OpenMP regions found in direct solver kernels | low | do not add dense-kernel OpenMP without backend/runtime policy |
| dense backend dispatch | env-selected builtin/optional provider | low for threading; medium for env mutation in tests | preserve Day 5 fallback tests |
| ND / graph partition | no OpenMP regions found; uses thread-local policy scopes | low for nested OpenMP, medium for scope balance | cleanup should target env/scope documentation and tests |
| FM ensemble strategy | algorithmically tries multiple strategies, not OpenMP | medium if later parallelized internally | keep advisory; do not assume current thread pool |
| benchmarks/examples | may call any above path | medium for interpreting timings under OpenMP env | print/record build and thread context when timing matters |

## Compatibility-Sensitive Behavior to Protect

- Serial builds must remain the default and must not require OpenMP headers or
  runtime libraries.
- OpenMP builds must continue to compile through Make and CMake.
- `OMP_NUM_THREADS` and related OpenMP runtime variables should remain external
  runtime controls; the library should not silently override them.
- `SPARSE_MUTEX` remains a separate per-matrix mutation-safety build option,
  not an OpenMP feature.
- Thread-local graph/reorder override scopes must remain balanced and local to
  the calling thread.
- Dense backend env requests remain best-effort with builtin fallback.
- Test-only env mutations must remain isolated and cleaned up.
- Windows reviewed CMake scope can stay serial unless a future sprint explicitly
  expands OpenMP validation on Windows.

## Cleanup Priority List

| priority | candidate | reason | validation cost |
|---|---|---|---|
| P0 | Add an explicit maintainer note distinguishing compile-time OpenMP, process-global env knobs, and thread-local internal overrides | removes product-model ambiguity without behavior change | docs hygiene |
| P0 | Add comments near OpenMP SpMV/reorth sites stating they rely on external OpenMP runtime control and do not own thread count | prevents accidental nested-thread API claims | full gate if comments touch `.c` |
| P1 | Add focused tests or docs for env cleanup expectations around graph/reorder tests | reduces process-global env leakage risk | test-specific or docs hygiene depending on scope |
| P1 | Inventory and group `SPARSE_*` env variables in one maintainer/runtime document | makes compatibility knobs auditable | docs hygiene |
| P2 | Consider internal helper wrappers for env parsing with consistent invalid-value handling | reduces repeated parsing logic but touches behavior-adjacent code | full C quality gate |
| P2 | Add optional benchmark output of OpenMP build status and `omp_get_max_threads()` to relevant benchmarks beyond `bench_main` | improves timing interpretation | benchmark build plus full C quality gate |
| P3 | Design a public runtime/thread-control API | high product and ABI risk; defer until after evidence contract | dedicated project-plan item |
| P3 | Add outer OpenMP to solvers/direct kernels | nested parallelism and oversubscription risk | broad performance and correctness matrix |

## Validation Needs

For docs-only cleanup:

- `git diff --check`
- trailing-whitespace scan on touched planning/runtime docs

For `.c` or `.h` cleanup touching comments only:

- `make format && make lint && make test`
- focused owner test where applicable

For OpenMP behavior changes:

- serial `make format && make lint && make test`
- OpenMP build of affected tests (`make omp` or direct OpenMP target)
- `tests/test_omp`
- eigensolver and SVD focused tests if reorth or SpMV semantics change
- CI TSan OpenMP lane or local equivalent where available

For graph/reorder env or thread-local scope changes:

- focused graph/reorder tests:
  - `build/test_graph`
  - `build/test_reorder_nd`
  - `build/test_reorder_amd_qg`
- full quality gate after any `.c` or `.h` touch

## Day 7 Recommendation

Day 7 should begin with documentation and comments, not behavior changes:

1. Record the runtime-control model in a maintainer-facing document or local
   Sprint artifact.
2. Add narrow comments at the two OpenMP implementation sites only if they
   clarify ownership without restating the code.
3. Avoid adding public thread-count APIs, env-to-thread-count translation, or
   new OpenMP regions.
4. Keep graph/reorder cleanup focused on env/scope clarity and test isolation.
5. Run the full C quality gate if Day 7 touches any `.c` or `.h` file.

## Completion Check

| criterion | status |
|---|---|
| OpenMP compile-time guards inventoried | complete |
| runtime environment controls inventoried | complete |
| thread-local and process-global behavior mapped | complete |
| nested-parallelism risks classified | complete |
| cleanup candidates ranked before code changes | complete |
| serial and OpenMP validation needs identified | complete |
