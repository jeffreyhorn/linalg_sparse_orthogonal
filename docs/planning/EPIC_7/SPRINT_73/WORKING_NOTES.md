# Sprint 73 Working Notes

## Day 1 - Scope Audit and Baseline Setup

### Goal

Turn the Sprint 73 project-plan scope plus the Sprint 70 and Sprint 72
handoff into one bounded configuration-modernization sprint, with the
strongest live control surfaces and non-goal fence fixed before deeper audit
begins.

### Actions

1. Re-read the Sprint 73 section of `docs/planning/EPIC_7/PROJECT_PLAN.md`,
   the Sprint 70 architecture contract, the Sprint 72 retrospective, and the
   Sprint 72 closeout artifact.
2. Reconfirm the preserved Sprint 73 constraints:
   - no generic env-var purge detached from real ownership cost
   - no public capability widening disguised as configuration cleanup
   - no platform/install/reviewed-contract widening
   - no broad backend or product-model rewrite hidden inside control-surface
     modernization
3. Reconfirm the strongest local reviewed baseline shape from:
   - `make -n quality-review-full`
   - `make quality-review-cmake-compile`
4. Capture the live Day 1 hotspot map across the strongest likely Sprint 73
   configuration surfaces.
5. Record the intended Sprint 73 workstreams, touch surfaces, and proof-risk
   surfaces before Day 2 validation work begins.

### Findings

#### 1. Sprint 73 now starts from a precise residual-control queue

Sprint 73 does not need another broad Epic 7 planning reset and it does not
need another public-surface cleanup pass.

The strongest next queue is explicitly:

- residual env-var inventory and rerank
- typed versus internal-policy ownership design
- FM/graph policy integration
- debug/profile surface rationalization
- docs/maintainer/header follow-through only where landed behavior truly moves
  the contract
- focused regression coverage and validation

#### 2. The Sprint 70 non-goal fence remains the right constraint set

The live repo state still supports the same fence:

- no repo-wide env-var purge detached from real ownership cost
- no fake capability or product widening through undocumented internal knobs
- no platform/install/reviewed-contract widening
- no broad backend or product-model rewrite hidden inside configuration work

That means Sprint 73 should stay bounded to the strongest residual control
surfaces instead of trying to clean every process-global switch at once.

#### 3. The strongest live configuration pressure is concentrated in graph/FM,
ND, and residual advanced-control seams

The live `getenv(...)` and residual compatibility map is concentrated in:

- graph/FM strategy and pass-count surfaces in `src/sparse_graph.c`
  - `SPARSE_FM_FINEST_STRATEGY`
  - `SPARSE_FM_ENSEMBLE_STRATEGIES`
  - `SPARSE_FM_FINEST_PASSES`
  - `SPARSE_FM_INTERMEDIATE_PASSES`
  - `SPARSE_FM_ENSEMBLE_DEBUG`
  - `SPARSE_FM_THICK_RESTART_DEBUG`
- FM refinement and developer/debug surfaces in `src/sparse_graph_refine.c`
  - `SPARSE_FM_ANNEALING_SCHEDULE`
  - `SPARSE_FM_THICK_RESTART_PERTURB`
  - `SPARSE_FM_GAIN_NOISE_SCHEDULE`
  - `SPARSE_FM_ANNEALING_DEBUG`
  - `SPARSE_FM_GAIN_NOISE_DEBUG`
- ND/coarsening and profile surfaces in:
  - `src/sparse_reorder_nd.c`
  - `src/sparse_graph_coarsen.c`
  - `src/sparse_graph_separator.c`
  - `src/sparse_graph_bisect.c`
  with controls such as:
    - `SPARSE_ND_ROOT_BISECT`
    - `SPARSE_ND_COARSENING`
    - `SPARSE_ND_COARSEST_BISECTION`
    - `SPARSE_ND_ROOT_BISECT_MAX_N`
    - `SPARSE_ND_COARSEN_FLOOR_RATIO`
    - `SPARSE_ND_COARSENING_CV_FALLTHROUGH`
    - `SPARSE_ND_SEP_LIFT_STRATEGY`
    - `SPARSE_ND_SEP_LIFT_WEIGHT`
    - `SPARSE_ND_PROFILE`
- residual reorder/profile compatibility in `src/sparse_reorder_amd_qg.c`
  - `SPARSE_QG_PROFILE`
- residual analysis/SVD compatibility in:
  - `src/sparse_analysis.c`
    - `SPARSE_SUPERNODAL_POSTORDER`
    - `SPARSE_ND_SUPERNODAL_POSTORDER`
  - `src/sparse_svd.c`
    - `SPARSE_SVD_LOWRANK_OUTER`

This is the right Day 1 narrowing: Sprint 73 should start from graph/FM and ND
policy convergence, with debug/profile and residual compatibility spill as the
second modernization lane.

#### 4. The strongest likely Sprint 73 touch surfaces are now explicit

Raw Day 1 `wc -l` counts from the live tree:

##### Maintained public/policy surfaces

- `docs/maintainer_guide.md` = `585`
- `include/sparse_analysis.h` = `499`

##### Configuration-modernization implementation seams

- `src/sparse_graph.c` = `821`
- `src/sparse_graph_internal.h` = `850`
- `src/sparse_reorder_nd.c` = `739`
- `src/sparse_graph_refine.c` = `629`
- `src/sparse_graph_coarsen.c` = `641`
- `src/sparse_graph_separator.c` = `297`
- `src/sparse_graph_bisect.c` = `528`
- `src/sparse_reorder_amd_qg.c` = `611`
- `src/sparse_svd.c` = `1319`

##### Strongest proof and reporting surfaces

- `tests/test_graph.c` = `2900`
- `tests/test_reorder_nd.c` = `2262`
- `tests/test_integration.c` = `2448`
- `tests/test_fuzz.c` = `651`
- `examples/example_analysis.c` = `210`
- `benchmarks/bench_reorder.c` = `321`

#### 5. The strongest reviewed baseline remains intact

The local reviewed baseline remains unchanged:

- strongest local reviewed baseline:
  - `make quality-review-full`
- reviewed CMake parity was re-materialized live:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps Sprint 73 aligned with the Sprint 70 truthfulness fence before any
configuration precedence work lands.

### Validation

This was a docs-only Day 1 baseline/setup pass, so I did not run
`make format`, `make lint`, or `make test`.

I did recheck the reviewed baseline shape and parity anchors with:

- `make -n quality-review-full`
- `make quality-review-cmake-compile`

I also captured the live Day 1 raw `wc -l` hotspot measurements and the
residual `getenv(...)`/compatibility map across the strongest likely Sprint 73
configuration surfaces.

### Day 1 Exit State

Sprint 73 Day 1 closes with:

1. one configuration-modernization starting queue
2. one preserved Sprint 70 non-goal fence
3. one live reviewed baseline anchor
4. one ranked live residual-control hotspot map

## Day 2 - Validation Baseline and Truth-Surface Recheck

### Goal

Reconfirm the Sprint 73 implementation-day validation contract and fix the
highest-signal rerun set before any configuration-modernization batch lands.

### Actions

1. Reconfirm the strongest local reviewed baseline wording:
   - `make quality-review-full`
   - reviewed CMake parity anchor
2. Reconfirm the Sprint 73 authority split:
   - `*.c` / `*.h` landing days require `make format`, `make lint`, and
     `make test`
   - substantial architecture or precedence-boundary batches default to
     `make quality-review-full`
   - docs-only audit/design/review days use targeted sanity checks only
3. Recheck the live proof surfaces Sprint 73 is most likely to stress:
   - graph/reorder proof owners
   - integration and compatibility proof owners
   - representative examples
   - maintained reorder/reporting surfaces
   - install/package proof scripts
4. Refresh the targeted rerun set most likely to matter in Sprint 73.
5. Record the authoritative validation split in the working notes.

### Findings

#### 1. The strongest reviewed baseline remains unchanged

Sprint 73 still inherits the same strongest local reviewed baseline:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

That keeps the sprint aligned with the Sprint 70 truthfulness fence before any
typed/default/env precedence or policy-ownership work lands.

#### 2. The Sprint 73 authority split is now explicit before code work

The Day 2 recheck fixes the same three-part validation split Sprint 70 and
Sprint 72 used:

- bounded `*.c` / `*.h` landing days:
  - `make format`
  - `make lint`
  - `make test`
- substantial architecture or precedence-boundary batches:
  - `make quality-review-full`
- docs-only audit/design/review days:
  - targeted sanity checks only

That is the right split for Sprint 73 because the likely work crosses graph,
reorder, compatibility, and control-precedence boundaries rather than one tiny
file-local helper seam.

#### 3. The live proof-surface split is now fixed for Sprint 73

The Day 2 recheck shows this live local split:

- the reviewed CMake tree currently owns the key proof-owner tests,
  representative examples, and reorder benchmark binaries most relevant to
  Sprint 73:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_framework_optin`
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- maintained install/package proof remains script-owned:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- the root `build/` tree is not currently carrying the usual maintained
  benchmark binaries such as:
  - `build/bench_reorder`
  - `build/bench_refactor_csc`
  - `build/bench_chol_csc`
  - `build/bench_iterative_reuse`
  - `build/bench_eigs_reuse`

That truth matters: Sprint 73 should anchor its Day 2 rerun set to the live
reviewed CMake tree and the maintained proof scripts, rather than assuming the
root benchmark binaries are materialized right now.

#### 4. The high-signal Sprint 73 rerun set is now explicit

The strongest likely rerun set for Sprint 73 is:

- graph/FM proof owners:
  - `./build/quality-review-cmake/test_graph`
  - `./build/quality-review-cmake/test_graph_fm_buckets`
- reorder/precedence proof owners:
  - `./build/quality-review-cmake/test_reorder_nd`
  - `./build/quality-review-cmake/test_integration`
- compatibility/support proof owners:
  - `./build/quality-review-cmake/test_fuzz`
  - `./build/quality-review-cmake/test_framework_optin`
- representative adoption surfaces:
  - `./build/quality-review-cmake/example_analysis`
  - `./build/quality-review-cmake/example_basic_solve`
- maintained reorder/reporting surfaces currently materialized in the reviewed
  tree:
  - `./build/quality-review-cmake/bench_reorder`
  - `./build/quality-review-cmake/bench_amd_qg`
- maintained install/package proof scripts:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

This is the right Day 2 fix: the rerun contract is now tied to the live
configuration-risk surface and the current local binary split, not to a stale
assumption about which benchmark binaries happen to exist in `build/`.

### Validation

This was a docs-only Day 2 pass, so I did not run `make format`, `make lint`,
or `make test`.

I did recheck the reviewed baseline and proof-surface split with:

- `ctest -N --test-dir build/quality-review-cmake`
- direct existence checks on the reviewed CMake proof/test/example/benchmark
  binaries
- direct existence checks on the root `build/` benchmark binaries
- direct existence checks on the install/package regression scripts

### Day 2 Exit State

Sprint 73 Day 2 closes with:

1. one explicit implementation-day validation split
2. one stable reviewed CMake parity anchor
3. one truthful live proof-surface map
4. one exact rerun set for the strongest likely Sprint 73 configuration lanes
