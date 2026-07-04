# Sprint 104 Working Notes

## Sprint Context

Sprint 104 implements "Performance Backend & Parallel Runtime Modernization"
from `docs/planning/EPIC_10/PROJECT_PLAN.md`. The sprint establishes a clearer
contract for builtin dense kernels, optional acceleration, backend descriptors,
OpenMP behavior, benchmark reporting, and bounded local performance sentinels.

The sprint does not start from a performance-superiority claim. It starts from
the Sprint 100 evidence templates and the Sprint 102-103 comparison discipline:
runtime and benchmark evidence must name the command, fixture scope, backend
state, thread context, reviewed status, unsupported cases, and remaining
non-claims before it can support wording beyond local diagnostics.

## Validation Rules

Validation must scale with the touched surface:

| touched surface | required validation |
|---|---|
| planning documentation only | `git diff --check`; trailing-whitespace scan on touched planning files |
| public documentation only | `git diff --check`; trailing-whitespace scan on touched docs |
| benchmark docs or scripts | focused benchmark/report command where runnable; docs hygiene |
| helper script only | focused helper invocation, if executable; docs hygiene |
| test `.c` file | focused test binary; `make format`; `make lint`; `make test` |
| library `.c` or public `.h` file | focused affected tests; `make format`; `make lint`; `make test` |
| build or CMake surface | focused Make/CMake configure or build check plus any code-touch gate |
| workflow or package surface | focused workflow/package command where runnable plus any code-touch gate |

If any `.c` or `.h` file is modified, the full required quality chain is:

```sh
make format && make lint && make test
```

All required checks must pass before closeout or PR creation.

## Claim Boundaries

Sprint 104 may earn only bounded backend/runtime and local regression evidence
tied to named commands, fixtures, backend state, thread settings, validation
commands, and unsupported cases.

Sprint 104 must not claim:

- portable timing superiority across machines, compilers, operating systems, or
  optional backend configurations;
- broad vendor backend parity;
- GPU or distributed-memory backend support;
- universal OpenMP scalability;
- that benchmark-local residual fields replace test or oracle correctness
  ownership;
- that a performance sentinel is a broad benchmark superiority claim;
- that optional acceleration is always present, faster, or preferred over the
  builtin fallback;
- that local benchmark output supports public state-of-the-art replacement
  language.

## Day 1 - Scope and Runtime Baseline

### Goal

Convert the Sprint 104 project-plan section and prior Epic 10 handoffs into a
bounded backend/runtime modernization package with clear workstreams, evidence
rules, and validation expectations.

### Actions

- Re-read the Sprint 104 section of
  `docs/planning/EPIC_10/PROJECT_PLAN.md`.
- Re-read Sprint 100 benchmark and performance-sentinel guardrails:
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day10-benchmark-coverage-performance-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/benchmark-interpretation-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/templates/performance-sentinel-template.md`
  - `docs/planning/EPIC_10/SPRINT_100/artifacts/day13-claim-non-goal-register.md`
- Re-read Sprint 102 and Sprint 103 closeout handoffs:
  - `docs/planning/EPIC_10/SPRINT_102/artifacts/day14-closeout-and-handoff.md`
  - `docs/planning/EPIC_10/SPRINT_103/artifacts/day14-closeout-and-handoff.md`
- Created the Sprint 104 artifacts directory.
- Recorded authoritative Day 1 inputs in
  `artifacts/day1-authoritative-inputs.txt`.
- Recorded the Sprint 104 scope baseline, workstream ownership, validation
  matrix, and claim boundaries in `artifacts/day1-runtime-baseline.md`.

### Findings

- Sprint 100 requires benchmark and sentinel evidence to name command, fixture
  scope, metrics and units, threshold or report-only status, reviewed status,
  backend/thread context, unsupported cases, and explicit non-claims.
- Sprint 102 and Sprint 103 both warn against promoting bounded correctness or
  comparison evidence into broad ecosystem, backend, or performance claims.
- Sprint 104 should audit backend consumers and runtime controls before source
  edits so descriptor, OpenMP, and sentinel work remains tied to actual owner
  surfaces.
- Performance sentinel work must begin with purpose, baseline, threshold
  source, machine-class assumptions, skip behavior, and non-claims.
- Benchmark reporting alignment should separate local timing, fill or memory,
  residual/correctness context, optional backend state, and thread settings.

### Validation Expectations

- Day 1 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no
  matches.

### Day 1 Exit State

Day 1 is complete. Sprint 104 now has working notes, authoritative inputs,
scope baseline, workstream ownership, validation expectations, and preserved
Sprint 100 benchmark/sentinel claim boundaries.

## Day 2 - Backend Consumer Audit

### Goal

Inventory source files that consume dense kernels, backend selectors,
decomposition helpers, optional acceleration hooks, OpenMP controls, benchmark
fields, and example-facing backend behavior before runtime contract design
starts.

### Actions

- Re-read Sprint 104 Day 2 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Searched source, headers, tests, benchmarks, examples, and maintainer docs
  for backend descriptors, backend enums, dense-kernel selectors, OpenMP
  controls, env knobs, and benchmark fields.
- Inspected the concrete Cholesky and LDLT dense-kernel seams:
  - `src/sparse_chol_csc_internal.h`
  - `src/sparse_dense.c`
  - `src/sparse_ldlt_csc_internal.h`
  - `src/sparse_ldlt_dense.c`
  - `src/sparse_ldlt_csc_supernodal.c`
- Inspected public selector and observability surfaces:
  - `include/sparse_ldlt.h`
  - `include/sparse_eigs.h`
  - `src/sparse_cholesky.c`
  - `src/sparse_eigs.c`
- Inspected OpenMP runtime surfaces:
  - `src/sparse_matrix.c`
  - `src/sparse_eigs.c`
  - `Makefile`
  - `CMakeLists.txt`
  - `.github/workflows/ci.yml`
- Inspected benchmark and docs reporting fields:
  - `benchmarks/README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/bench_chol_csc.c`
  - `benchmarks/bench_refactor_csc.c`
  - `benchmarks/bench_eigs.c`
  - `benchmarks/bench_eigs_reuse.c`
- Wrote the Day 2 audit in
  `artifacts/day2-backend-consumer-audit.md`.

### Findings

- The main optional dense-backend seams are Cholesky CSC supernodal dense
  kernels and LDLT CSC dense Bunch-Kaufman factorization.
- Both dense-backend seams preserve builtin fallback as the shipped default,
  but they expose optional backend selection through separate env vars:
  `SPARSE_CHOL_DENSE_BACKEND` and `SPARSE_LDLT_DENSE_BACKEND`.
- Cholesky exposes an internal `chol_dense_kernels_t` descriptor with function
  pointers and a `name`; LDLT exposes `ldlt_dense_factor_selected(...)` and
  `ldlt_dense_factor_backend_name()`.
- Cholesky, LDLT, and eigensolver public selectors already have distinct
  backend telemetry:
  - Cholesky: backend enum and benchmark path/kernel fields
  - LDLT: backend enum and optional `used_csc_path`
  - eigensolver: `backend_used` and `used_csc_path_ldlt`
- OpenMP behavior is currently compile-time enabled through `SPARSE_OPENMP`;
  SpMV/block SpMV and Lanczos MGS inner loops are the important consumers.
- Runtime control risk is concentrated in env-selected dense backends,
  OpenMP thread context, graph/ND compatibility env vars, and thread-local
  graph/FM overrides.
- Benchmark fields already report several backend/runtime values, but Day 3
  needs to define which fields are user-facing contract, benchmark-only
  context, or internal diagnostics.

### Validation Expectations

- Day 2 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no
  matches.

### Day 2 Exit State

Day 2 is complete. Sprint 104 now has a backend consumer inventory, fallback
map, optional acceleration point list, runtime-control risk list, and Day 3
runtime-contract handoff.

## Day 3 - Runtime Contract Design

### Goal

Define runtime expectations for builtin kernels, optional backends, OpenMP,
nested parallelism, and observability before descriptor or threading changes
begin.

### Actions

- Re-read Sprint 104 Day 3 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 2 backend consumer audit and converted its risks into
  contract decisions.
- Defined builtin dense kernels as the portable baseline for Cholesky CSC,
  LDLT CSC, QR, SVD, and eigensolver paths.
- Defined optional dense-backend selection semantics for:
  - `SPARSE_CHOL_DENSE_BACKEND`
  - `SPARSE_LDLT_DENSE_BACKEND`
- Defined public selector semantics for:
  - `sparse_cholesky_opts_t::backend`
  - `sparse_ldlt_opts_t::backend`
  - `sparse_eigs_opts_t::backend`
- Defined OpenMP expectations for serial builds, OpenMP builds, MGS gating,
  thread-count disclosure, and nested-parallelism non-claims.
- Split observability by audience:
  - public API users
  - test owners
  - benchmark owners
  - maintainers
- Wrote the Day 3 runtime contract design in
  `artifacts/day3-runtime-contract-design.md`.

### Findings

- Builtin fallback must remain the primary product truth. Optional dense
  backend requests are best-effort and must not become required for
  correctness, installation, or supported use.
- Silent fallback to builtin is current behavior for optional dense backend
  requests. Day 4/5 can either document that explicitly or add stronger
  selected/fallback telemetry where local patterns justify it.
- Cholesky, LDLT, and eigensolver backend selectors are not vendor selectors;
  they choose library algorithm/path implementations.
- OpenMP is compile-time optional. Serial behavior remains the reference
  behavior, and interpreted timing must disclose runtime thread context.
- Nested parallelism is a non-contract for Sprint 104: no broad speedup,
  scheduling determinism, or OpenMP-plus-BLAS interaction claim is earned.
- Benchmark residuals remain context. Correctness and oracle ownership still
  belongs to tests and external-reference artifacts.

### Validation Expectations

- Day 3 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no
  matches.

### Day 3 Exit State

Day 3 is complete. Sprint 104 now has a runtime contract for builtin fallback,
optional backend request/selection/fallback behavior, OpenMP and
nested-parallelism expectations, and diagnostic-surface ownership.

## Day 4 - Descriptor Surface Boundary

### Goal

Freeze backend descriptor and selection-surface changes before source edits by
deciding which changes are public API, internal-only, test-support, or
documentation-only.

### Actions

- Re-read Sprint 104 Day 4 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Compared the Day 2 backend consumer audit with the Day 3 runtime contract.
- Classified descriptor and selector surfaces:
  - public API selectors and telemetry
  - internal dense-backend descriptors
  - test-only overrides
  - benchmark diagnostics
  - compatibility env hooks
  - out-of-scope graph/ND runtime controls
- Rejected public API, ABI, and enum changes for the Day 5 descriptor batch.
- Preserved silent builtin fallback as current product behavior unless Day 5
  adds stronger diagnostic reporting without changing behavior.
- Defined compatibility requirements for callers, tests, examples, benchmarks,
  Windows, and environment cleanup.
- Wrote the Day 4 descriptor boundary in
  `artifacts/day4-descriptor-surface-boundary.md`.

### Findings

- Day 5 should stay inside existing internal, test, benchmark, and docs
  surfaces. A public vendor-backend API is out of scope.
- Existing public selectors choose library algorithm/path implementations, not
  vendor providers, and should not change layout or enum values.
- Cholesky and LDLT optional dense-backend seams can be documented as
  best-effort request/selected/fallback behavior without requiring ABI changes.
- Any benchmark `.c` or source `.c` change must trigger the full
  `make format && make lint && make test` quality gate.
- Graph/ND env controls are intentionally left to Day 6 threading/runtime
  cleanup, not Day 5 dense-backend descriptor work.

### Validation Expectations

- Day 4 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no
  matches.

### Day 4 Exit State

Day 4 is complete. Sprint 104 now has a descriptor-surface boundary,
compatibility checklist, focused validation plan, and scoped Day 5
implementation sequence.

## Day 5 - Backend Descriptor Batch

### Goal

Implement the selected descriptor/status proof batch without weakening builtin
fallback semantics or widening public backend APIs.

### Actions

- Re-read Sprint 104 Day 5 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 4 descriptor boundary and selected the smallest
  behavior-preserving implementation batch.
- Added a Cholesky CSC dense-backend fallback test:
  - `test_supernodal_dense_backend_invalid_env_falls_back_to_builtin`
- Added an LDLT dense-backend fallback test:
  - `test_ldlt_dense_backend_invalid_env_falls_back_to_builtin`
- Registered both tests with their existing dense-backend test groups.
- Wrote the Day 5 implementation artifact in
  `artifacts/day5-backend-descriptor-batch.md`.

### Findings

- Existing builtin, external, and Accelerate tests already covered the main
  optional dense-backend request paths.
- The missing explicit contract was invalid/unknown optional backend request
  behavior.
- The new tests prove that unknown `SPARSE_CHOL_DENSE_BACKEND` and
  `SPARSE_LDLT_DENSE_BACKEND` values fall back to builtin rather than becoming
  hard runtime failures.
- No public API, ABI, enum, library source behavior, benchmark output, or
  example change was needed for this batch.

### Validation Expectations

- Day 5 changes `.c` test files and planning documentation.
- Required checks:
  - `make build/test_chol_csc_supernodal build/test_ldlt`
  - `./build/test_chol_csc_supernodal`
  - `./build/test_ldlt`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on touched test files and
    `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `make build/test_chol_csc_supernodal build/test_ldlt`: passed.
- `./build/test_chol_csc_supernodal`: passed; 62 tests, 0 failures, 0
  skips, 8170 assertions.
- `./build/test_ldlt`: passed; 89 tests, 0 failures, 0 skips, 912
  assertions.
- `make format && make lint && make test`: passed.

### Day 5 Exit State

Day 5 is complete. Sprint 104 now has focused invalid-request fallback tests
for the Cholesky and LDLT optional dense-backend seams, with builtin fallback
preserved as the portable product truth.

## Day 6 - OpenMP and Threading Audit

### Goal

Inventory OpenMP, thread-local override, process-global env, and nested
parallelism behavior before Day 7 cleanup begins.

### Actions

- Re-read Sprint 104 Day 6 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Scanned Make, CMake, source, tests, benchmarks, CI, and algorithm docs for
  OpenMP, pthread, thread-local, and runtime-control surfaces.
- Confirmed OpenMP pragmas are limited to linked-list SpMV and eigensolver MGS
  reorthogonalization.
- Confirmed there is no public thread-pool, thread-count, or per-call scheduler
  API.
- Mapped graph/reorder thread-local override scopes separately from
  process-global `SPARSE_*` environment variables.
- Wrote the Day 6 audit artifact in
  `artifacts/day6-openmp-threading-audit.md`.

### Findings

- Serial builds remain the product default; OpenMP is opt-in through
  `SPARSE_OPENMP`.
- OpenMP runtime control is external to the library through normal OpenMP
  runtime variables such as `OMP_NUM_THREADS`.
- `SPARSE_MUTEX` is a separate mutation-safety compile-time feature, not an
  OpenMP runtime-control feature.
- Process-global env variables are used heavily for compatibility and
  benchmark controls; tests that mutate them must keep cleanup local.
- Thread-local graph/reorder override scopes protect concurrent calls but are
  internal coordination mechanisms, not public threading features.
- Nested parallelism risk is mostly indirect through SpMV calls inside
  iterative, eigensolver, SVD, and graph spectral paths.

### Validation Expectations

- Day 6 is documentation-only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 6 Exit State

Day 6 is complete. Sprint 104 now has a ranked OpenMP/threading cleanup audit
ready for Day 7, with compatibility-sensitive serial, OpenMP, env, and
thread-local behavior explicitly protected.

## Day 7 - OpenMP and Threading Cleanup

### Goal

Implement the highest-value low-risk cleanup from the Day 6 audit while
preserving serial-build behavior, public option semantics, and existing
runtime-control ownership.

### Actions

- Re-read Sprint 104 Day 7 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 6 OpenMP/threading audit and selected the P0 cleanup:
  clarify runtime-control ownership without behavior changes.
- Added implementation comments beside the OpenMP SpMV and MGS reorth owners:
  - `src/sparse_matrix.c`
  - `src/sparse_eigs.c`
- Added user-facing OpenMP runtime-control and nested-parallelism guidance to
  `docs/algorithm.md`.
- Added maintainer-facing OpenMP/runtime-control policy and validation
  interpretation to `docs/maintainer_guide.md`.
- Wrote the Day 7 cleanup artifact in
  `artifacts/day7-threading-cleanup.md`.

### Findings

- No public thread-count API is needed for this cleanup.
- `OMP_NUM_THREADS` and other OpenMP runtime settings remain the right control
  surface for OpenMP builds.
- `SPARSE_*` compatibility env vars should not be translated into OpenMP team
  size.
- Thread-local graph/reorder override scopes remain internal coordination
  mechanisms and should not be presented as user-facing thread controls.
- Because `.c` files changed, the full C quality gate is required even though
  the source cleanup is comment-only.

### Validation Expectations

- Day 7 changes `.c` files and documentation.
- Required checks:
  - `make build/test_omp build/test_eigs`
  - `./build/test_omp`
  - `./build/test_eigs`
  - `make format && make lint && make test`
  - `git diff --check`
  - trailing-whitespace scan on touched source/docs and
    `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `make build/test_omp build/test_eigs`: passed.
- `./build/test_omp`: passed; 12 tests, 0 failures, 0 skips, 831
  assertions.
- `./build/test_eigs`: passed; 31 tests, 0 failures, 0 skips, 310
  assertions.
- `make format && make lint && make test`: passed.

### Day 7 Exit State

Day 7 is complete. Sprint 104 now has a behavior-preserving threading cleanup
patch that clarifies OpenMP runtime ownership, separates `SPARSE_*`
compatibility knobs from thread-count control, and gives Day 8 a clearer
performance-sentinel baseline.

## Day 8 - Performance Sentinel Design

### Goal

Select bounded local performance sentinels for hot paths without framing local
timing as portable performance superiority.

### Actions

- Re-read Sprint 104 Day 8 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 2 backend consumer audit and Day 7 threading cleanup
  handoff.
- Reviewed current benchmark governance in `benchmarks/README.md`,
  `docs/maintainer_guide.md`, `Makefile`, `scripts/wall_check.sh`, and the
  maintained benchmark source comments.
- Selected sentinel candidates from the existing canonical maintained
  performance surface and existing thresholded runtime lane.
- Defined fixture choices, measurement context, variance policy, threshold
  stance, skip behavior, output fields, and Day 9 implementation guidance.
- Wrote the Day 8 sentinel design artifact in
  `artifacts/day8-performance-sentinel-design.md`.

### Findings

- The current canonical maintained performance surface is the right source for
  threshold-free sentinel reporting:
  - `bench_refactor_csc`
  - `bench_chol_csc`
  - `bench_iterative_reuse`
  - `bench_eigs_reuse`
- The existing `make wall-check` lane remains the only already-justified hard
  threshold gate.
- Day 9 should avoid adding hard timing thresholds for new canonical lanes
  until a local baseline or same-worktree comparison workflow exists.
- Sentinel output must record build mode, OpenMP runtime context, dense backend
  env values, command identity, and skip reasons.
- Benchmark residual fields should remain measurement context; tests still own
  correctness and oracle claims.

### Validation Expectations

- Day 8 is documentation-only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 8 Exit State

Day 8 is complete. Sprint 104 now has a bounded sentinel design that selects
local regression lanes, preserves existing benchmark-governance boundaries,
and gives Day 9 a conservative implementation sequence.

## Day 9 - Performance Sentinel Batch

### Goal

Add the first bounded local performance sentinel batch without introducing
portable performance claims or new uncalibrated timing thresholds.

### Actions

- Re-read Sprint 104 Day 9 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 8 performance sentinel design and selected the smallest
  useful implementation batch:
  - S5 existing `wall-check` hard gate
  - S2 threshold-free Cholesky CSC report rows
- Added `scripts/performance_sentinels.sh`.
- Added `make performance-sentinels`.
- Ran the new target and inspected generated output in
  `build/bench-reports/sentinels/`.
- Wrote the Day 9 implementation artifact in
  `artifacts/day9-performance-sentinel-batch.md`.

### Findings

- The existing `wall-check` lane remains the only hard timing threshold in this
  batch.
- The Cholesky CSC lane emits `report` rows, not pass/fail timing results.
- Generated output records build mode, `OMP_NUM_THREADS`, dense backend env
  values, command identity, metrics, baselines, thresholds, and notes.
- Missing binaries, missing fixtures, or missing baselines emit explicit skip
  rows where practical.
- S1, S3, and S4 remain selected for future expansion, but Day 9 does not add
  thresholds for them without a local baseline workflow.

### Representative Output

- `make performance-sentinels`: passed.
- Generated:
  - `build/bench-reports/sentinels/sentinels.tsv`
  - `build/bench-reports/sentinels/manifest.txt`
  - `build/bench-reports/sentinels/wall_check.txt`
  - `build/bench-reports/sentinels/bench_chol_csc_nos4.csv`
- Representative S5 rows:
  - `bcsstk14 qg_amd_reorder_ms = 68.6 ms` with baseline `130 ms`,
    threshold `2x`
  - `Pres_Poisson amd_reorder_ms = 4437.4 ms` with baseline `8000 ms`,
    threshold `2x`
  - `Pres_Poisson nd_reorder_ms = 4110.4 ms` with baseline `47055 ms`,
    threshold `1.5x`
- Representative S2 rows:
  - `nos4.mtx factor_ll_ms = 0.313`
  - `nos4.mtx factor_csc_ms = 0.381`
  - `nos4.mtx factor_csc_sn_ms = 0.357`
  - `nos4.mtx speedup_csc = 0.82`
  - `nos4.mtx speedup_csc_sn = 0.88`

### Validation Expectations

- Day 9 changes shell script, Makefile, and planning documentation.
- No `.c` or `.h` files are changed for Day 9.
- Required checks:
  - `bash -n scripts/performance_sentinels.sh`
  - `make performance-sentinels`
  - inspect generated sentinel report
  - `make lint`
  - `git diff --check`
  - trailing-whitespace scan on touched script, Makefile, and
    `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `bash -n scripts/performance_sentinels.sh`: passed.
- `make performance-sentinels`: passed.
- Generated sentinel report inspected:
  - `build/bench-reports/sentinels/sentinels.tsv`
  - `build/bench-reports/sentinels/manifest.txt`
- `make lint`: passed.

### Day 9 Exit State

Day 9 is complete. Sprint 104 now has a bounded local performance sentinel
wrapper that combines the existing hard wall-check gate with threshold-free
Cholesky CSC backend-aware timing rows.

## Day 10 - Benchmark Reporting Audit

### Goal

Audit benchmark scripts, benchmark docs, README references, maintainer docs,
and planning artifacts against the Day 3 runtime contract and Day 8 sentinel
design before public documentation wording changes begin.

### Actions

- Re-read Sprint 104 Day 10 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 3 runtime contract and Day 8 performance sentinel design.
- Inspected benchmark/reporting surfaces:
  - `README.md`
  - `benchmarks/README.md`
  - `docs/algorithm.md`
  - `docs/maintainer_guide.md`
  - `Makefile`
  - `scripts/bench_canonical_report.sh`
  - `scripts/performance_sentinels.sh`
  - `scripts/wall_check.sh`
- Compared benchmark wording and generated report fields to the Sprint 104
  builtin fallback, optional acceleration, OpenMP runtime, and sentinel
  non-claim contracts.
- Wrote the Day 10 benchmark reporting audit in
  `artifacts/day10-benchmark-reporting-audit.md`.

### Findings

- The main benchmark governance surfaces are already aligned around
  branch-local measurements, threshold-free canonical reports, and narrow
  hard-fail gates.
- The Day 9 `performance-sentinels` target is documented in Makefile/script
  comments but is not yet reflected in the benchmark README, maintainer guide,
  README command list, or algorithm wall-check notes.
- `scripts/bench_canonical_report.sh` still emits `category=proof` in generated
  metadata, which can be misread as timing proof even though the surrounding
  docs describe a threshold-free local snapshot.
- OpenMP benchmark wording should continue to say that selected paths may be
  parallelized under `SPARSE_OPENMP`, while interpreted timing must disclose
  runtime thread settings.
- Optional dense backend wording should pair requested/selected/fallback
  context with the builtin portable baseline.
- Benchmark residual and agreement fields remain diagnostic context; tests and
  external oracle artifacts still own correctness claims.

### Validation Expectations

- Day 10 changes planning documentation only.
- Required checks:
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 10 Exit State

Day 10 is complete. Sprint 104 now has a benchmark reporting audit that
inventories the benchmark/documentation surfaces, separates local timing from
portable claims, preserves builtin fallback and OpenMP runtime wording rules,
and gives Day 11 a concrete documentation update plan.

## Day 11 - Benchmark Reporting Alignment

### Goal

Apply the Day 10 wording rules to selected benchmark, maintainer, README, and
report-generation surfaces so performance wording and generated labels match
the actual backend/runtime evidence.

### Actions

- Re-read Sprint 104 Day 11 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 10 benchmark reporting audit and used its documentation
  update plan as the implementation checklist.
- Updated `README.md` to include `make performance-sentinels` in:
  - the workflow proof-owner summary
  - the Make command list
  - the high-level performance section
- Updated `benchmarks/README.md` with the sentinel bundle artifact list,
  S5 hard-gate scope, S2 threshold-free Cholesky CSC scope, skip behavior,
  backend/thread context, and non-claims.
- Updated `docs/maintainer_guide.md` benchmark governance with
  `performance-sentinels` ownership and interpretation rules.
- Updated `docs/algorithm.md` performance regression gate notes to connect the
  historical `wall-check` gate with the Sprint 104 sentinel bundle.
- Updated `scripts/bench_canonical_report.sh` so generated canonical report
  metadata says `category=measurement` instead of `category=proof`.
- Wrote the Day 11 alignment artifact in
  `artifacts/day11-benchmark-reporting-alignment.md`.

### Findings

- The misleading canonical report label was a direct reporting defect: the
  generated artifacts are threshold-free local measurement snapshots, not
  timing proofs.
- The new sentinel bundle can be documented without changing its Day 9
  execution behavior.
- The updated docs keep `wall-check` as the only current hard timing gate.
- Cholesky CSC sentinel rows remain threshold-free and must be interpreted
  with recorded backend env values, selected dense-kernel fields, build mode,
  and OpenMP thread settings.
- No `.c` or `.h` files are changed for Day 11.

### Validation Expectations

- Day 11 changes documentation and shell script report text.
- Required checks:
  - `bash -n scripts/bench_canonical_report.sh`
  - `make bench-canonical-report`
  - inspect generated canonical metadata for `category=measurement`
  - `make performance-sentinels`
  - inspect generated sentinel artifacts
  - `git diff --check`
  - trailing-whitespace scan on touched docs, script, and
    `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `bash -n scripts/bench_canonical_report.sh`: passed.
- `make bench-canonical-report`: passed.
- Generated canonical metadata inspected:
  - `build/bench-reports/canonical/index.tsv` uses `category=measurement`.
  - `build/bench-reports/canonical/manifest.txt` uses
    `category=measurement`.
- `make performance-sentinels`: passed.
- Generated sentinel artifacts inspected:
  - `build/bench-reports/sentinels/sentinels.tsv`
  - `build/bench-reports/sentinels/manifest.txt`
  - `build/bench-reports/sentinels/wall_check.txt`
  - `build/bench-reports/sentinels/bench_chol_csc_nos4.csv`
- `git diff --check`: passed.
- `rg -n "[ \t]+$" README.md benchmarks/README.md docs/algorithm.md
  docs/maintainer_guide.md scripts/bench_canonical_report.sh
  docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 11 Exit State

Day 11 is complete. Sprint 104 now has aligned benchmark reporting docs and
generated canonical report labels: `performance-sentinels` is documented as a
bounded local sentinel bundle, `wall-check` remains the only hard timing gate,
S2 Cholesky CSC rows remain threshold-free context, and canonical report
metadata now uses `category=measurement`.

## Day 12 - Cross-Platform Runtime Review

### Goal

Review backend/runtime behavior across local, CI, Windows, serial, OpenMP, and
optional-acceleration contexts before Sprint 104 closeout.

### Actions

- Re-read Sprint 104 Day 12 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 11 benchmark reporting alignment artifact.
- Inspected runtime/platform surfaces:
  - `Makefile`
  - `CMakeLists.txt`
  - `.github/workflows/ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `README.md`
  - `docs/maintainer_guide.md`
  - `benchmarks/README.md`
  - `scripts/performance_sentinels.sh`
  - `scripts/bench_canonical_report.sh`
- Ran a local CMake configure check in `build/day12-platform-review`.
- Ran `ctest -N` on the Day 12 CMake build tree to inspect registered tests.
- Wrote the Day 12 cross-platform runtime review artifact in
  `artifacts/day12-cross-platform-runtime-review.md`.

### Findings

- Local POSIX CMake registration reports 54 tests.
- Windows CI expects 51 tests, and the 3-test delta is explained by staged
  exclusions for:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`
- Linux remains the strongest reviewed CI source with enforced Makefile
  compile-quality, CMake parity, and dead-code paths.
- macOS enforces the Apple Clang reviewed path and wall-check/sanitize signals;
  Homebrew GCC and install/pkg-config remain supplemental.
- Windows remains a reviewed CMake-first consumer subset only; it does not
  claim Makefile parity, benchmark parity, fuzz/property coverage, or separate
  install-validation parity.
- OpenMP remains opt-in in Make and CMake; serial remains the default.
- The Day 11 benchmark/sentinel wording now matches the platform-runtime
  interpretation, so Day 12 records an explicit no-change decision.

### Validation Expectations

- Day 12 changes planning documentation only.
- Required checks:
  - `cmake -S . -B build/day12-platform-review`
  - `ctest -N --test-dir build/day12-platform-review`
  - `git diff --check`
  - trailing-whitespace scan on `docs/planning/EPIC_10/SPRINT_104`

### Validation Results

- `cmake -S . -B build/day12-platform-review`: passed.
- `ctest -N --test-dir build/day12-platform-review`: passed; registered
  `Total Tests: 54`.
- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 12 Exit State

Day 12 is complete. Sprint 104 now has a cross-platform runtime review that
maps local and CI validation surfaces, documents the POSIX/Windows CTest count
split, confirms serial/OpenMP/optional-backend interpretation remains coherent,
and records an explicit no-change decision for source, workflow, and public
documentation surfaces.

## Day 13 - Validation Reconciliation

### Goal

Reconcile Sprint 104 artifacts with the final implementation, docs, scripts,
and validation state before closeout.

### Actions

- Re-read Sprint 104 Day 13 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Reviewed final changed surfaces with `git diff --stat` and focused diffs for
  source, tests, Makefile, and report scripts.
- Reconciled Day 1 through Day 12 artifacts against the implemented source,
  test, docs, benchmark, and script changes.
- Re-ran focused backend, OpenMP, eigensolver, canonical report, and sentinel
  validation.
- Ran the required full quality gate for the branch because `.c` files were
  modified:

```sh
make format && make lint && make test
```

- Wrote the Day 13 validation reconciliation artifact in
  `artifacts/day13-validation-reconciliation.md`.

### Findings

- The implementation matches the artifact claims:
  - invalid optional Cholesky dense backend env requests fall back to builtin;
  - invalid optional LDLT dense backend env requests fall back to builtin;
  - OpenMP thread ownership remains with the OpenMP runtime;
  - canonical report metadata uses `category=measurement`;
  - `performance-sentinels` emits S5 pass/fail rows and S2 report-only rows.
- Final report rows still record serial build context, `OMP_NUM_THREADS=unset`,
  dense backend env values, and builtin Cholesky dense-kernel context.
- No Sprint 104 artifact claims portable timing superiority, broad vendor
  backend parity, public thread-control APIs, or Windows Makefile/install
  parity.

### Validation Results

- `bash -n scripts/performance_sentinels.sh && bash -n
  scripts/bench_canonical_report.sh`: passed.
- `make build/test_chol_csc_supernodal build/test_ldlt build/test_omp
  build/test_eigs`: passed; binaries were up to date.
- `./build/test_chol_csc_supernodal`: passed; 62 tests, 0 failures, 0 skips,
  8170 assertions.
- `./build/test_ldlt`: passed; 89 tests, 0 failures, 0 skips, 912 assertions.
- `./build/test_omp`: passed; 12 tests, 0 failures, 0 skips, 831 assertions.
- `./build/test_eigs`: passed; 31 tests, 0 failures, 0 skips, 310 assertions.
- `make bench-canonical-report`: passed.
- Canonical metadata inspection: passed; `index.tsv` and `manifest.txt` use
  `category=measurement`.
- `make performance-sentinels`: passed.
- Sentinel artifact inspection: passed; S5 emitted pass rows and S2 emitted
  report-only rows with backend/thread context.
- `make format && make lint && make test`: passed.

### Known Limitations

- `performance-sentinels` remains local regression evidence, not portable
  performance evidence.
- S2 Cholesky CSC rows are threshold-free until a future local-baseline or
  same-worktree comparison design exists.
- Optional dense backend requests remain best-effort and fallback to builtin.
- OpenMP remains opt-in and runtime-owned by the OpenMP runtime.
- Windows remains the reviewed CMake-first consumer subset only.

### Sprint 105 Handoff Candidates

- Preserve benchmark/sentinel wording whenever generated fields change.
- Keep Windows expected CTest count updates tied to explicit staged-exclusion
  decisions.
- Add hard thresholds for S1/S3/S4 only after a fresh variance and baseline
  design.
- Keep optional backend widening local to concrete Cholesky/LDLT seams unless
  a future sprint validates a broader provider model.
- Keep OpenMP runtime ownership explicit near any future parallel region.

### Day 13 Exit State

Day 13 is complete. Sprint 104 now has final validation reconciliation, a
focused command log, full quality-gate proof, known limitations, and concrete
Sprint 105 handoff candidates ready for closeout.

## Day 14 - Sprint Closeout and Handoff

### Goal

Close Sprint 104 with validated evidence, bounded backend/runtime claims, and
a concrete Sprint 105 handoff queue.

### Actions

- Re-read Sprint 104 Day 14 in
  `docs/planning/EPIC_10/SPRINT_104/PLAN.md`.
- Re-read the Day 13 validation reconciliation artifact.
- Confirmed every day-level deliverable has a closeout status.
- Summarized Sprint 104 outcomes across:
  - backend descriptor and optional acceleration behavior;
  - OpenMP runtime-control ownership;
  - local performance sentinels;
  - benchmark reporting alignment;
  - cross-platform runtime scope.
- Recorded residuals and non-claims for performance sentinels, optional dense
  backends, OpenMP, Windows reviewed scope, and benchmark wording drift.
- Wrote the Day 14 closeout and Sprint 105 handoff artifact in
  `artifacts/day14-closeout-and-handoff.md`.

### Findings

- Sprint 104 completed all planned day-level deliverables.
- Source and test changes remained scoped to:
  - invalid optional dense backend fallback tests;
  - OpenMP runtime ownership comments.
- Build/reporting changes remained scoped to:
  - `make performance-sentinels`;
  - canonical report metadata using `category=measurement`.
- Public and maintainer docs now separate local timing, optional acceleration,
  OpenMP runtime context, and reviewed platform scope.
- The Day 13 full quality gate remains the authoritative final source/test
  validation for Sprint 104.
- Day 14 changed planning documentation only.

### Final Sprint 104 Validation Summary

- `bash -n scripts/performance_sentinels.sh && bash -n
  scripts/bench_canonical_report.sh`: passed.
- `./build/test_chol_csc_supernodal`: passed.
- `./build/test_ldlt`: passed.
- `./build/test_omp`: passed.
- `./build/test_eigs`: passed.
- `make bench-canonical-report`: passed.
- `make performance-sentinels`: passed.
- `make format && make lint && make test`: passed.

### Sprint 105 Handoff Queue

- Preserve benchmark/sentinel claim boundaries when touching reordering, graph,
  or large-matrix evidence.
- Treat local timing as command/fixture/backend/thread-context evidence.
- Keep `performance-sentinels` hard-fail behavior limited to S5 unless a
  future baseline design justifies additional thresholds.
- Check POSIX CMake registration and Windows expected CTest count when adding
  tests.
- Keep OpenMP ownership wording near new parallel regions and benchmark
  interpretation.
- Do not widen optional dense backend language beyond current Cholesky/LDLT
  seams without matching tests, docs, and platform-scope updates.

### Validation Results

- `git diff --check`: passed.
- `rg -n "[ \t]+$" docs/planning/EPIC_10/SPRINT_104`: passed; no matches.

### Day 14 Exit State

Day 14 is complete. Sprint 104 is closed out with a validated summary,
explicit non-claims, residual queue, and Sprint 105 handoff queue. Day 13
remains the final source/test validation proof, and Day 14 documentation
hygiene passed after closeout edits.
