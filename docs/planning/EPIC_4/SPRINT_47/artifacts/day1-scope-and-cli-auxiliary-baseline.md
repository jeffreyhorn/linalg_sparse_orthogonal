# Sprint 47 Day 1 Artifact: Scope and CLI/Auxiliary Baseline

## Purpose

Capture the Sprint 47 starting baseline before shared CLI parsing helpers,
`bench_main` modernization, reorder-mode parity cleanup, example safety work,
auxiliary tooling cleanup, and benchmark/example docs refresh begin.

## Starting Truth

Sprint 47 starts from a stable preserved Sprint 40/41/45/46 baseline:

- strongest local reviewed baseline already exists:
  - `make quality-review-full`
- reviewed CMake parity remains explicit and measurable:
  - `ctest -N --test-dir build/quality-review-cmake` = `53`
- maintained dead-code surfaces already exist:
  - `make deadcode-report`
  - `make deadcode-check`
- dead-code execution remains serialized because `deadcode*` still shares:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- Sprint 41 already left a reusable internal safety/helper layer:
  - `src/sparse_alloc_internal.h`
  - `src/sparse_alloc_internal.c`
- Sprint 45 and Sprint 46 already left repeated-run benchmark precedents:
  - `benchmarks/bench_iterative_reuse.c`
  - `benchmarks/bench_eigs_reuse.c`
- `bench_eigs.c` already demonstrates the newer benchmark-side parsing style:
  - checked `strtol` / `strtod`
  - explicit usage text
  - explicit enum-like parser helpers

This means Sprint 47 is not opening with solver-baseline repair or major
library architecture work. It is opening with bounded benchmark/example/tooling
cleanup on top of a preserved reviewed baseline and already-proven safety and
repeated-run helper patterns.

## Day 1 Workstreams

Sprint 47 Day 1 confirms the sprint's eight bounded workstreams:

1. CLI/helper seam inventory
2. shared parsing-helper design
3. `bench_main` modernization
4. reorder-mode parity cleanup
5. example safety audit
6. auxiliary tooling cleanup
7. benchmark/example docs refresh
8. validation closeout

These come directly from the Sprint 47 section of
`docs/planning/EPIC_4/PROJECT_PLAN.md` and stay consistent with the earlier
Epic 4 rule that usability and maintainability cleanup should land through
bounded internal/helper improvements rather than broad framework churn.

## Highest-Value Authoritative Inputs

### Epic 4 planning and architecture inputs

- `docs/planning/EPIC_4/PROJECT_PLAN.md`
- `docs/planning/EPIC_4/SPRINT_47/PLAN.md`
- `docs/planning/EPIC_4/SPRINT_46/artifacts/day14-closeout-and-handoff.md`

### Inherited execution-rule inputs

- `docs/planning/EPIC_4/SPRINT_41/artifacts/day12-safety-style-and-prep-rules.md`
- `docs/planning/EPIC_4/SPRINT_40/artifacts/day13-validation-anchor-and-command-matrix.md`
- `src/sparse_alloc_internal.h`
- `src/sparse_alloc_internal.c`

### Inherited reviewed-quality / policy inputs

- `README.md`
- `Makefile`
- `CMakeLists.txt`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

### Highest-risk Day 1 benchmark/example/tooling inputs

- `benchmarks/bench_main.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_iterative_reuse.c`
- `benchmarks/bench_eigs_reuse.c`
- `examples/example_eigs.c`
- `examples/example_iterative.c`
- `examples/example_matrix_free.c`
- `scripts/deadcode_report.py`
- `scripts/deadcode_workflow.sh`

## Highest-Value Day 1 Conclusions

### 1. Sprint 47 is an auxiliary-surface modernization sprint, not a core solver sprint

The preserve-not-reopen boundary is explicit:

- preserve Sprint 40 validation-anchor truth
- reuse Sprint 41 safety helper conventions
- preserve the current validated benchmark/example behavior while improving
  parsing and auxiliary safety
- avoid broad benchmark framework redesign
- avoid unrelated public library API churn

### 2. `bench_main.c` is the primary parser-modernization hotspot

The live repo now shows:

- `benchmarks/bench_main.c` = `774` lines
- it still carries ad hoc flag handling through:
  - `atoi(...)` for `--spmv-iters`
  - `atoi(...)` for `--size`
  - `atoi(...)` for `--repeat`
  - inline string matching for reorder and pivot modes
- it currently advertises reorder modes:
  - `none`
  - `rcm`
  - `amd`
  - `nd`
- while the broader library reorder surface already includes:
  - `SPARSE_REORDER_COLAMD`

That makes `bench_main` the main Sprint 47 landing zone for:

- shared positive/bounded integer parsing
- clearer error reporting
- supported-mode reconciliation
- emitted-label parity cleanup

### 3. The benchmark peer surface is uneven rather than uniformly outdated

The live benchmark surface already splits into two different classes:

- newer / stronger benchmark CLI surface:
  - `benchmarks/bench_eigs.c` = `958`
  - checked `strtol` / `strtod`
  - explicit usage/help text
  - explicit enum-like parse helpers
- older / simpler benchmark surface:
  - `benchmarks/bench_main.c` = `774`
  - older `atoi`-style parsing
  - narrower inline mode parsing

That means Sprint 47 should not treat every benchmark binary as equally stale.
`bench_eigs.c` is more useful as a comparison point for parser shape and error
reporting than as the first modernization target.

### 4. The example surface is a bounded safety-follow-on, not the main parser problem

The main maintained examples in the repeated-run and iterative/eigensolver area
are comparatively small:

- `examples/example_eigs.c` = `284`
- `examples/example_iterative.c` = `144`
- `examples/example_matrix_free.c` = `122`

These are real Sprint 47 targets, but they are not the main complexity driver.
The stronger Day 1 interpretation is:

- examples should be audited after the parser/helper seam exists
- example cleanup should stay bounded to touched safety/helper conventions
- Sprint 47 should avoid turning the example batch into a broad teaching-style
  rewrite

### 5. The main auxiliary tooling seam is script-side safety and support-code alignment

The inherited script-side hotspot surface is explicit:

- `scripts/deadcode_report.py` = `523`
- `scripts/deadcode_workflow.sh` = `189`

These are large enough to matter, but Day 1 evidence does not justify treating
them as a separate framework project. The correct framing is:

- align touched parsing/safety/support patterns with Sprint 41 conventions
- keep auxiliary cleanup bounded to real helper/safety inconsistencies
- preserve the already-honest dead-code workflow contract

### 6. The front-half order of the sprint is fixed

The correct early sprint order is:

1. baseline and seam inventory
2. shared parsing-helper design
3. parser modernization in `bench_main`
4. reorder-mode parity cleanup
5. example audit and bounded example cleanup
6. auxiliary tooling cleanup

That ordering preserves Sprint 40's core rule: structural or usability cleanup
should be guided by measured live seams and an explicit validation anchor
before broader surface work lands.
