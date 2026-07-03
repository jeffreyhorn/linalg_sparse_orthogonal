# Sprint 104 Day 8 Performance Sentinel Design

## Purpose

Day 8 selects bounded local performance sentinels for Sprint 104 without
turning local wall-clock timing into a portable performance claim. The selected
sentinels are intended to catch obvious same-machine regressions in hot paths
that already have maintained benchmark surfaces.

## Design Principles

- Sentinels detect local regressions; they do not prove cross-platform speed.
- Tests remain the owners of correctness, oracle, and property guarantees.
- Benchmark residual columns remain measurement context, not oracle proof.
- Serial build behavior remains the baseline unless a sentinel explicitly
  records an OpenMP build and OpenMP runtime context.
- Thresholds must be wide enough to survive normal workstation and CI variance.
- Missing optional fixtures or unavailable optional runtimes should skip with
  explicit diagnostics, not silently pass as successful timing evidence.
- Day 9 implementation should prefer parsing existing benchmark output over
  adding new timing code.

## Selected Sentinel Path List

| ID | path | benchmark owner | purpose | fixture scope |
|---|---|---|---|---|
| S1 | repeated-run direct lifecycle | `bench_refactor_csc` | detect large regressions in public analyze/factor/refactor direct workflow | synthetic/default benchmark fixture plus optional `--indefinite-kkt` row when cheap enough |
| S2 | Cholesky CSC backend comparison | `bench_chol_csc` | detect large regressions in linked-list vs CSC scalar/supernodal factor/solve timings and descriptor reporting | one named small/default SPD fixture or synthetic small corpus subset |
| S3 | public iterative handle reuse | `bench_iterative_reuse` | detect large regressions in CG/GMRES/MINRES one-shot vs reusable-handle workflow | built-in deterministic synthetic fixtures |
| S4 | public eigensolver handle reuse | `bench_eigs_reuse` | detect large regressions in grow-m, thick-restart, and explicit LOBPCG reusable-handle workflow | built-in deterministic fixtures |
| S5 | reorder wall-check | `make wall-check` | retain the existing thresholded qg-AMD / Pres_Poisson AMD / Pres_Poisson ND regression gate | existing bcsstk14 and Pres_Poisson fixture slice |

S1-S4 align with the current canonical maintained performance surface. S5
stays in the existing regression-sensitive runtime lane and should not be
treated as a general reorder benchmark claim.

## Deferred Sentinel Candidates

| candidate | reason deferred |
|---|---|
| full `make bench` | too slow and too broad for a bounded local sentinel |
| `bench_convergence` | exploratory convergence table; solver correctness already test-owned |
| `bench_svd` | broad exploratory SVD profiling; Sprint 103 tests own bounded SVD claims |
| `bench_eigs --compare` | useful backend comparison, but broader than reusable-handle regression scope |
| OpenMP-vs-serial speedup gate | machine/runtime dependent; needs fresh OpenMP baseline and runner classification |
| optional dense backend speedup gate | provider availability and dynamic probing make it non-portable |

## Measurement Contract

Each Day 9 sentinel run should record:

- git branch and commit when available;
- platform string from `uname -a` where available;
- compiler identity when available;
- build mode: serial or `SPARSE_OPENMP`;
- `OMP_NUM_THREADS` when set, otherwise `unset`;
- dense backend env values:
  - `SPARSE_CHOL_DENSE_BACKEND`
  - `SPARSE_LDLT_DENSE_BACKEND`
- benchmark command exactly as executed;
- benchmark CSV header and parsed metric names;
- skip reason for any omitted lane.

The sentinel should not rewrite benchmark CSV schemas unless a Day 9
implementation proves the existing fields are insufficient.

## Fixture and Command Design

| ID | proposed command | primary parsed fields | warm-up rule |
|---|---|---|---|
| S1 | `build/bench_refactor_csc` | `refactor_public_ms`, `refactor_csc_ms`, `solve_public_ms`, `solve_csc_ms`, `speedup_refactor`, backend selected/fallback fields | one untimed pre-run only if Day 9 wrapper can do it without changing benchmark output |
| S2 | `build/bench_chol_csc --repeat 3 tests/data/suitesparse/nos4.mtx` or a single synthetic fallback if file missing | linked-list/CSC/supernodal factor/solve fields, speedup fields, dense-kernel descriptor fields | benchmark repeat handles averaging; no extra warm-up required initially |
| S3 | `build/bench_iterative_reuse` | one-shot total, reuse total, speedup, last-run status/residual fields | benchmark owns repeat policy |
| S4 | `build/bench_eigs_reuse` | one-shot median, reuse median, speedup, residual/agreement fields, `backend_used` | benchmark owns repeat policy |
| S5 | `make wall-check` | existing parsed `reorder_ms` fields and per-key gates | existing script owns temporary captures |

If Day 9 needs a shorter first batch, implement S2 plus S5 first because they
exercise backend descriptor context and an already thresholded timing gate.

## Variance and Threshold Policy

The threshold policy should be conservative:

| metric family | initial threshold | rationale |
|---|---:|---|
| existing wall-check qg-AMD / AMD | existing `2.0x` | already documented in `scripts/wall_check.sh` |
| existing wall-check Pres_Poisson ND | existing `1.5x` | already accounts for measured ND variance |
| canonical benchmark factor/refactor/solve timing | proposed `2.5x` over branch-local baseline | avoids noise while catching large regressions |
| canonical benchmark speedup field | fail only if speedup collapses below `0.5x` of local baseline and absolute timing also regresses | prevents ratio-only false positives on tiny timings |
| residual/agreement fields | parse and report only unless a benchmark already documents a local bound | tests own correctness thresholds |
| optional backend selected/fallback fields | compare exact strings to the current run expectation, not timing | provider availability is not portable |

Day 9 should not introduce hard fail thresholds for S1-S4 until it also lands a
machine-local baseline file or a two-run same-worktree comparison workflow.
The first implementation can emit structured sentinel rows and reserve hard
fail behavior for S5.

## Skip Behavior

Skip explicitly when:

- a required benchmark binary is missing and cannot be built;
- a required fixture file is missing;
- `SPARSE_OPENMP` comparison is requested but the binary was not built with
  OpenMP;
- an optional dense backend provider is requested but the benchmark reports
  builtin fallback;
- a benchmark returns a documented unsupported status for the selected fixture.

Skip output must include the lane ID, command, and reason. Skips should not be
reported as passes.

## Output Fields for Day 9

The first Day 9 sentinel wrapper should produce one compact TSV or CSV report
with these columns:

| field | meaning |
|---|---|
| `sentinel_id` | S1-S5 |
| `status` | `pass`, `fail`, `skip`, or `report` |
| `command` | exact command |
| `build_mode` | `serial`, `openmp`, or `unknown` |
| `omp_num_threads` | value or `unset` |
| `matrix_or_fixture` | benchmark row identity |
| `metric` | parsed metric name |
| `value` | parsed current value |
| `baseline` | optional local baseline value or `n/a` |
| `threshold` | optional threshold or `n/a` |
| `notes` | bounded explanation |

Use `report` for threshold-free canonical rows. Use `pass` / `fail` only for
lanes with a declared threshold, starting with S5.

## Validation Plan

For Day 8 design-only work:

- `git diff --check`
- trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

For Day 9 docs/scripts only:

- docs hygiene checks
- shell syntax check for any new shell script
- focused dry-run or smoke run of the new sentinel wrapper

For Day 9 benchmark `.c` changes:

- focused benchmark build for touched binary
- representative smoke command
- `make format && make lint && make test`

For Day 9 Makefile/script changes:

- focused target invocation
- `make lint` if the target affects build tooling
- docs hygiene for changed docs

## Non-Claims

This design does not claim:

- local sentinels are portable timing proofs;
- CI runners are stable enough for tight benchmark thresholds;
- optional acceleration is present;
- OpenMP builds are always faster than serial builds;
- benchmark residual columns replace tests;
- canonical report artifacts are pass/fail quality gates.

## Day 9 Implementation Recommendation

Day 9 should implement the smallest useful batch:

1. Add a maintainer-only sentinel wrapper that runs S5 and optionally captures
   threshold-free S2 output.
2. Write structured output under `build/bench-reports/sentinels/`.
3. Keep hard failure behavior limited to the existing `wall-check` lane.
4. Record runtime context before benchmark rows.
5. Avoid changing benchmark CSV schemas unless the wrapper cannot parse current
   fields.

## Completion Check

| criterion | status |
|---|---|
| selected hot paths listed | complete |
| sentinel goals scoped to local regression detection | complete |
| fixtures and command candidates selected | complete |
| variance and threshold policy defined | complete |
| skip behavior defined | complete |
| output fields proposed | complete |
| benchmark wording protected from portable-performance claims | complete |
