# Sprint 52 Working Notes

# Day 1 - baseline and scope

Date: 2026-06-01

## Goal

Start Sprint 52 from the real Sprint 51 Phase 1 lifecycle implementation
state, not from a reopened design discussion.

## Inputs reviewed

- `docs/planning/EPIC_5/PROJECT_PLAN.md`
  - Sprint 52 section
- `docs/planning/EPIC_5/SPRINT_52/PLAN.md`
- `docs/planning/EPIC_5/SPRINT_51/artifacts/day14-closeout-and-handoff.md`
- `docs/planning/EPIC_5/SPRINT_51/RETROSPECTIVE.md`

## Baseline conclusions

### 1. Sprint 52 starts from a preserved reviewed baseline, not from repair work

The strongest local reviewed baseline is still:

- `make quality-review-full`

The maintained truthfulness anchor remains explicit and measurable:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

The reviewed wrapper surface remains the authoritative local closeout path for
substantial lifecycle batches.

### 2. Sprint 52 inherits a real implemented Phase 1 lifecycle surface

Sprint 51 already landed the first implemented public direct-lifecycle package:

- shared/family public header contract refresh in:
  - `include/sparse_analysis.h`
  - `include/sparse_lu.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`
- bounded LU lifecycle integration through the shared path where the default
  option surface fit cleanly
- shared Cholesky lifecycle routing through the public
  `sparse_analyze(...)` + `sparse_factor_numeric(...)` path
- shared LDL^T lifecycle routing through the same public lifecycle path
- focused lifecycle regression coverage in `tests/test_integration.c`
- aligned repeated-run caller-surface docs in:
  - `examples/README.md`
  - `benchmarks/README.md`

Interpretation:

- Sprint 52 is not about exposing the first public lifecycle path
- Sprint 52 is about deepening the already-landed analysis/factor/refactor
  workflow so it behaves more like the first-class direct repeated-run path

### 3. The main Phase 2 work is deeper integration, not new abstraction

The Sprint 52 project-plan items reduce cleanly to:

- audit the live `sparse_analysis_t` / `sparse_factors_t` path
- reduce avoidable fallback into one-shot symbolic work
- tighten refactor semantics and result reuse
- refresh factor-many benchmark proof
- update the strongest repeated-run docs/example surfaces as needed
- expand regression proof around the deeper lifecycle path

Interpretation:

- the sprint should stay analysis/factors-centric
- this is not the place to invent a new generic direct handle
- this is not the place to expose raw CSC/native storage or reopen the Sprint
  50 non-goal fence

### 4. The strongest code hotspots are now explicit

Current high-value lifecycle hotspots by file size and role:

- public contract:
  - `include/sparse_analysis.h` = `355`
  - `include/sparse_lu.h` = `337`
  - `include/sparse_cholesky.h` = `204`
  - `include/sparse_ldlt.h` = `320`
- implementation:
  - `src/sparse_analysis.c` = `626`
  - `src/sparse_lu.c` = `1040`
  - `src/sparse_cholesky.c` = `514`
  - `src/sparse_ldlt.c` = `1494`
- proof / adoption:
  - `tests/test_integration.c` = `1314`
  - `benchmarks/bench_refactor.c` = `159`
  - `benchmarks/bench_refactor_csc.c` = `388`
  - `examples/example_analysis.c` = `191`

Interpretation:

- the highest-risk implementation seams are still concentrated in
  `src/sparse_analysis.c`, `src/sparse_lu.c`, and `src/sparse_ldlt.c`
- the strongest proof and adoption surfaces are still
  `tests/test_integration.c`, `bench_refactor*`, and `example_analysis`

### 5. The preserved compatibility fence remains the key control point

Sprint 52 still inherits the Sprint 50-51 fence:

- one-shot LU / Cholesky / LDL^T APIs remain first-class peer entry points
- one-shot usage remains the simple/default path for one-off solves
- repeated direct runs remain centered on:
  - `sparse_analysis_t`
  - `sparse_factors_t`
  - `sparse_analyze(...)`
  - `sparse_factor_numeric(...)`
  - `sparse_factor_solve(...)`
  - `sparse_refactor_numeric(...)`
- reuse preserves symbolic/permutation/setup state, not old numeric factor
  contents
- no raw internal CSC/native storage exposure
- no broad generic direct-handle redesign
- no structural-pattern verifier redesign

Interpretation:

- the Phase 2 integration work must strengthen the real repeated-run path
  without changing the basic public compatibility story

## Day 1 outcome

Sprint 52 now starts from a concrete Phase 2 baseline:

- reviewed baseline and parity anchor rechecked
- Sprint 51 implementation handoff reduced to a bounded Phase 2 queue
- strongest direct-lifecycle hotspots named explicitly
- preserved compatibility and non-goal fence restated before deeper code work

That is enough to move into the Day 2 validation baseline and touched-surface
recheck without reopening Sprint 50 or Sprint 51 design decisions.

# Day 2 - validation baseline

Date: 2026-06-01

## Goal

Reconfirm the reviewed local baseline and fix the exact Sprint 52 rerun set
before any deeper analysis/refactor code work begins.

## Validation baseline conclusions

### 1. The strongest reviewed local baseline remains unchanged

The authoritative local reviewed closeout command is still:

- `make quality-review-full`

The wrapper wording remains aligned with the current repo state:

- `quality-review-full: strongest local reviewed baseline`

Interpretation:

- Sprint 52 should continue to treat `make quality-review-full` as the
  strongest local reviewed baseline on substantial lifecycle batches
- there is no need to invent a new Sprint 52-specific validation authority

### 2. The main truthfulness anchor remains exact

The maintained reviewed CMake parity anchor is still:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- the Phase 2 lifecycle work starts from the same parity/truthfulness anchor
  Sprint 51 closed on
- Day 13 validation later in the sprint should continue to use that exact count
  as the key anchor

### 3. The code-day gate remains fixed

For later `*.c` / `*.h` lifecycle batches, the mandatory code-day gate remains:

- `make format`
- `make lint`
- `make test`

The stronger default for substantial public direct-lifecycle batches remains:

- `make quality-review-full`

Interpretation:

- Sprint 52 should not weaken its code-day validation just because the
  public lifecycle surface already exists
- deeper integration work still needs the full baseline gate

### 4. The targeted Sprint 52 rerun set is now explicit

The highest-value direct-lifecycle follow-ons remain present in the current
`build/` tree:

- `./build/example_analysis`
- `./build/bench_refactor`
- `./build/bench_refactor_csc`
- `./build/test_integration`
- `./build/test_cholesky`
- `./build/test_ldlt`
- `./build/test_etree`
- `./build/test_chol_csc`
- `./build/test_ldlt_csc`

Interpretation:

- Sprint 52 already knows the strongest repeated-run direct example surface
- Sprint 52 already knows the strongest factor-many benchmark surfaces
- Sprint 52 already knows the strongest direct lifecycle and factor-family
  regression surfaces

### 5. The docs-only vs code-day boundary is clean

Docs-only audit/design/narrowing days:

- preserve wording/truthfulness anchors
- run only targeted sanity checks when needed

Code-touch analysis/refactor lifecycle days:

- run `make format`
- run `make lint`
- run `make test`
- default to `make quality-review-full` for substantial public/direct-lifecycle
  batches

Interpretation:

- the validation contract is fixed before Sprint 52 begins changing lifecycle
  behavior
- later implementation days do not need to renegotiate their gate

## Day 2 outcome

Sprint 52 now has an explicit validation contract before deeper integration
begins:

- strongest reviewed local baseline rechecked
- reviewed CMake parity anchor rechecked at `53`
- mandatory code-day gate restated
- stronger reviewed default restated
- targeted direct-lifecycle rerun set fixed

That is enough to move into the Day 3 analysis/factors contract audit without
validation ambiguity.
