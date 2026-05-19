# Sprint 34 Working Notes

## Day 1

**Objective:** Convert the Sprint 33 handoff into a precise Sprint 34 build-quality enforcement baseline by reconfirming the inherited validated-state invariants, auditing the current Makefile/CMake/CI quality entry points, and naming the first concrete implementation surfaces before any enforcement wiring begins.

### Commands Run

1. Read Sprint 34 scope and Sprint 33 handoff inputs:
   - `git branch --show-current`
   - `git status --short --branch`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
   - `sed -n '151,220p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_33/HANDOFF.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_33/RETROSPECTIVE.md`
2. Inspect current Makefile, CMake, and CI quality-entry surfaces:
   - `rg -n "deadcode|tooling-build|lint:|check:|format:|test:|Werror|compile_commands|ctest|wall-check" Makefile CMakeLists.txt .github/workflows -g '!build/**'`
   - `sed -n '200,540p' Makefile`
   - `sed -n '180,320p' CMakeLists.txt`
   - `sed -n '1,220p' .github/workflows/ci.yml`
   - `sed -n '1,180p' .github/workflows/macos-ci.yml`
   - `sed -n '1,220p' .github/workflows/windows-ci.yml`
   - `ls .github/workflows | sort`
3. Reconfirm currently visible validation/dead-code baseline artifacts:
   - `ctest -N --test-dir build/sprint33-day1-cmake`
   - `git rev-parse --short HEAD`
   - `command -v cppcheck && command -v xunused && command -v clang-tidy`
   - `sed -n '1,260p' build/deadcode/report.md`
   - `sed -n '1,240p' build/deadcode/coverage-notes.txt`

### Day 1 Baseline Findings

- Sprint 34 starts from the Sprint 33 validated close exactly as intended:
  - `make format`: previously validated passing at Sprint 33 close
  - `make lint`: previously validated passing at Sprint 33 close
  - `make test`: previously validated passing at Sprint 33 close
  - `ctest -N --test-dir build/sprint33-day1-cmake`: `53` registered tests
  - full `ctest`: `53 / 53` passing at Sprint 33 close
  - `definitely-unused-internal-candidate`: `0`
- Current branch head at Day 1 baseline capture: `f1fe556`
- Unlike Sprint 33 Day 1, the local tool prerequisite gap is now closed:
  - `cppcheck`: present at `/usr/local/bin/cppcheck`
  - `xunused`: present at `/Users/jeff/.local/bin/xunused`
  - `clang-tidy`: present at `/usr/local/opt/llvm/bin/clang-tidy`

### Current Quality-Target Inventory

#### Makefile quality path

- formatting:
  - `format`
  - `format-check`
- test/runtime path:
  - `test`
  - `check = format-check + lint + test`
- compile-only tooling gate:
  - `bench-build`
  - `examples-build`
  - `tooling-build`
- lint/static-analysis path:
  - `lint` depends on `build/include/sparse_version.h` and `tooling-build`
  - strict `src/*.c` syntax-only compile under `-Werror`
  - `clang-tidy` on `src/*.c`
  - `cppcheck` on `src/` and `tests/`
- dead-code path:
  - `deadcode-compile-db`
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
- other quality-adjacent maintained gate:
  - `wall-check`

Interpretation:

- the repo already has real quality-target precedent in both warning and dead-code space
- but those paths are still parallel surfaces, not yet an intentionally integrated Sprint 34 enforcement flow
- `lint` remains library-centric for strict warnings because the `-Werror` compile step only covers `src/*.c`, while benchmarks/examples are only compile-checked indirectly through `tooling-build`

#### CMake quality path

- `ctest -N --test-dir build/sprint33-day1-cmake` currently exposes `53` active tests
- the current CMakeLists registers library tests for CTest but does **not** register benchmarks/examples under `ctest`
- CMake currently builds:
  - `13` benchmark translation units into the compile database
  - `6` example translation units into the compile database
- this remains narrower than the broader Makefile bench/example surface documented in Sprint 33

Interpretation:

- the CMake path remains the authoritative auditable view of the active executed test suite
- Sprint 34 still has to decide which compile-quality guarantees need true Make/CMake parity and which remain documented exclusions in phase 1

#### CI workflow path

- `.github/workflows/ci.yml`
  - Ubuntu `build-and-test`: `make test`, `make sanitize`, `make asan`, `make bench-build`, `make bench-fast`
  - Ubuntu `cmake-build-and-test`: configure, build, `ctest`, and Makefile/CMake test-count parity check
  - Ubuntu `tsan`
  - Ubuntu `lint`: `make format-check` and `make lint`
  - Ubuntu `coverage`
- `.github/workflows/macos-ci.yml`
  - macOS compiler matrix: build, `make test`, `make wall-check`, Apple Clang `make sanitize`
  - install/pkg-config verification job
- `.github/workflows/windows-ci.yml`
  - MSVC configure, build, and `ctest`

Interpretation:

- CI already enforces formatting, lint, tests, sanitizer, and wall-time signals
- CI does **not** yet run `deadcode`, `deadcode-report`, or `deadcode-check`
- Sprint 34 Day 1 therefore starts from a real CI quality baseline, but not yet from CI-backed dead-code enforcement

### Inherited Dead-Code Limitations Still In Force

- current dead-code report buckets from `build/deadcode/report.md`:
  - `coverage-gap`: `7`
  - `public-surface-review`: `4`
  - `secondary-candidate-signal`: `35`
  - `non-deadcode-static-analysis-noise`: `6`
  - `definitely-unused-internal-candidate`: `0`
- current compile-db coverage gap from `build/deadcode/coverage-notes.txt`:
  - missing benchmark: `bench_svd`
  - missing examples:
    - `example_basic_solve`
    - `example_condition`
    - `example_iterative`
    - `example_least_squares`
    - `example_matrix_free`
    - `example_svd_lowrank`
- current execution-model limitation from Sprint 33 handoff:
  - `deadcode*` targets still share `build/deadcode-cmake`
  - `deadcode*` targets still share `build/deadcode/`
  - they remain serial/operator-invoked targets unless Sprint 34 isolates those paths

### Likely First Implementation Surfaces

1. Makefile wiring
   - `Makefile` lint/check/deadcode target region
   - interaction between `tooling-build`, `lint`, `check`, and `deadcode*`
2. Dead-code helper scripts
   - `scripts/deadcode_workflow.sh`
   - `scripts/deadcode_report.py`
3. CMake parity surface
   - `CMakeLists.txt` benchmark/example registration and compile-db coverage boundaries
   - `build/sprint33-day1-cmake` as the current authoritative test-registry path
4. CI workflows
   - `.github/workflows/ci.yml` as the likeliest first Sprint 34 enforcement insertion point
   - `.github/workflows/macos-ci.yml` and `.github/workflows/windows-ci.yml` as portability/parity follow-up surfaces

### Day 1 Interpretation

- Sprint 34 does **not** begin with inherited cleanup debt. It begins with an enforcement-shaping problem:
  - current warning-clean and truthfulness invariants already hold
  - dead-code tooling exists and passes locally
  - CI already has multiple quality gates
  - but warning enforcement, dead-code enforcement, and CMake/CI parity still live on partially separate tracks
- The highest-value Day 1 distinction is now explicit:
  - Sprint 34 is about turning prior cleanup results into repeatable gates
  - not about reopening Sprint 32 warning cleanup or Sprint 33 speculative code-removal work
- The two inherited hard constraints remain unchanged from Sprint 33:
  - dead-code compile-db coverage is incomplete
  - dead-code execution is still safest when serialized

### Day 1 Outputs

- `artifacts/day1-enforcement-baseline.md`
- `artifacts/day1-quality-target-inventory.txt`
