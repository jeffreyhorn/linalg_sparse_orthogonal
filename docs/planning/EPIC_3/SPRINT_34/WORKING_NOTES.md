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

## Day 2

**Objective:** Audit the current warning and compile-quality behavior across the Makefile, Apple Clang/CMake, benchmark/example compile-only, and dead-code paths so Sprint 34 can define a truthful phase-1 enforcement scope before target wiring begins.

### Commands Run

1. Re-read Sprint 34 Day 1 baseline and Day 2 plan scope:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `sed -n '33,120p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
2. Inspect current Makefile compile rules, target lists, and dry-run behavior:
   - `rg -n "^TEST_SRCS|^BENCH_SRCS|^EX_SRCS|^LIB_SRCS|^CFLAGS|^INCLUDE|^CC \\?=|^SYSROOT|^BUILD_DIR|^BUILDDIR" Makefile`
   - `sed -n '1,220p' Makefile`
   - `make -n lint`
   - `make -n tooling-build`
   - `make -n deadcode-report`
   - `make -n deadcode-check`
3. Inspect current CMake warning configuration and compile-command surfaces:
   - `sed -n '1,180p' CMakeLists.txt`
   - `python3 - <<'PY' ... build/sprint33-day1-cmake/compile_commands.json ... PY`
   - `python3 - <<'PY' ... build/deadcode-cmake/compile_commands.json ... PY`
   - `python3 - <<'PY' ... count compile_commands by top-level dir ... PY`
   - `rg -n "add_executable\\((bench_|example_)|add_sparse_test\\(" CMakeLists.txt`
4. Refresh exact repo-area file counts for comparison with CMake/dead-code coverage:
   - `find src -maxdepth 1 -name '*.c' | wc -l`
   - `find tests -maxdepth 1 -name '*.c' | wc -l`
   - `find benchmarks -maxdepth 1 -name '*.c' | wc -l`
   - `find examples -maxdepth 1 -name '*.c' | wc -l`

### Day 2 Audit Findings

#### 1. The repo currently has three different compile-quality models, not one

**Makefile `lint` model**

- compiler: local default `cc` (`Apple Clang` on this branch state)
- strict library-only compile check:
  - `-Wall`
  - `-Wextra`
  - `-Wpedantic`
  - `-Wshadow`
  - `-Wconversion`
  - `-Wstrict-prototypes`
  - `-Wformat=2`
  - `-Werror`
- scope:
  - `src/*.c` only
  - syntax-only (`-fsyntax-only`)
- supporting checks:
  - `clang-tidy` on `src/*.c`
  - `cppcheck` on `src/` and `tests/`

**Makefile `tooling-build` model**

- compile-only coverage for:
  - all `14` benchmark binaries
  - all `12` example binaries
- warning flags inherited from base `CFLAGS`:
  - `-Wall`
  - `-Wextra`
  - `-Wpedantic`
  - `-Wshadow`
  - `-Wconversion`
- not included here:
  - `-Werror`
  - `-Wformat=2`
  - `-Wdouble-promotion`
- interpretation:
  - good compile-coverage surface
  - not yet a true “warning-clean gate” in the same sense as `lint`

**CMake / dead-code compile-db model**

- compiler on the current local Apple Clang path:
  - `/usr/bin/cc`
- warning flags for non-MSVC builds:
  - `-Wall`
  - `-Wextra`
  - `-Wpedantic`
  - `-Wshadow`
  - `-Wconversion`
  - `-Wdouble-promotion`
  - `-Wformat=2`
  - `-Wno-unused-parameter`
- not included:
  - `-Werror`
  - `-Wstrict-prototypes`
- current covered translation-unit counts:
  - `src`: `25`
  - `tests`: `53`
  - `benchmarks`: `13`
  - `examples`: `6`

Interpretation:

- the Makefile path is stricter on library compile failure behavior because it uses `-Werror`
- the CMake/dead-code path is broader on warning categories for normal compilation because it includes `-Wdouble-promotion` and `-Wformat=2` across the covered tree
- the benchmark/example surface is broader under the Makefile than under the current CMake/dead-code compilation database

#### 2. The current target groups split naturally into four Sprint 34 categories

**Category A: first-class warning-gate candidates**

- `src/*.c`
- rationale:
  - already has an established strict `-Werror` path in `make lint`
  - current scope is well-defined and low-ambiguity
  - already exercised in Ubuntu `lint`

**Category B: compile-quality reviewed targets, but not yet full `-Werror` targets**

- all benchmark binaries via `bench-build`
- all example binaries via `examples-build`
- rationale:
  - compile-only coverage already exists and is cheap enough to keep in the normal path
  - this surface matters to Sprint 31/Sprint 35 contracts
  - current flags are not yet normalized with the stricter library gate

**Category C: key tests / active suite parity targets**

- current active CMake suite (`53` locally on the Apple Clang build tree)
- `ctest -N` and full `ctest`
- rationale:
  - this is the authoritative auditable executed-suite view
  - the suite is essential to preserve, but current Makefile warning-gate infrastructure does not expose a dedicated strict compile-only test gate yet

**Category D: dead-code support targets**

- `deadcode-compile-db`
- `deadcode`
- `deadcode-report`
- `deadcode-check`
- rationale:
  - these are build-quality support targets, but not direct warning gates
  - they should be integrated intentionally, not conflated with warning cleanliness

#### 3. The current compiler differences divide into real phase-1 scope boundaries versus acceptable temporary exclusions

**Real phase-1 scope boundary**

- MSVC / Windows warning parity
- evidence:
  - `CMakeLists.txt` switches from gcc/clang-style flags to `/W3`
  - POSIX-only benches are already gated off on `WIN32`
- interpretation:
  - Windows remains important for portability validation
  - but it is not a truthful first phase for a single shared “warning-clean” contract because the warning model and target surface are materially different

**Real phase-1 tooling limitation**

- dead-code compile-db gap:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- interpretation:
  - this is not a reason to block Sprint 34
  - but it must stay explicit in any dead-code enforcement or parity story

**Acceptable phase-1 exclusion**

- full test-tree `-Werror` gate in the Makefile path
- interpretation:
  - the tests are still compiled and executed routinely
  - the CMake path still exposes the active suite with the stricter normal warning set
  - but Sprint 34 should treat a dedicated test warning gate as future wiring, not a prerequisite for beginning enforcement

**Acceptable phase-1 distinction, not a blocker**

- Apple Clang / Linux gcc-style warning flag differences inside CI
- interpretation:
  - the shared Makefile target model already gives a portable first pass on POSIX compilers
  - Sprint 34 can start with the strictest reliable reviewed combinations rather than pretending every compiler has identical signal today

#### 4. Current CI already reinforces some of the desired scope, but not all of it

- Ubuntu `lint` already runs:
  - `make format-check`
  - `make lint`
- that means CI already enforces:
  - `src/*.c` under `-Werror`
  - tooling compile coverage via `tooling-build`
  - `clang-tidy`
  - `cppcheck`
- but CI does **not** yet enforce:
  - `deadcode`
  - `deadcode-report`
  - `deadcode-check`
- macOS CI currently reinforces:
  - runtime build/test
  - `wall-check`
  - Apple Clang `sanitize`
- Windows CI currently reinforces:
  - configure/build/test portability only

Interpretation:

- Sprint 34 does not need to invent phase-1 quality enforcement from scratch
- it needs to unify and clarify the enforcement story that already exists in partial form

### Day 2 First-Phase Enforcement Matrix

| Surface | Current command/path | Current warning behavior | Recommended Sprint 34 phase-1 role |
|---|---|---|---|
| Library sources `src/*.c` | `make lint` | strict compile under `-Werror`, plus `clang-tidy` and `cppcheck` | authoritative local warning gate |
| Benchmarks `bench_*` | `make tooling-build` / `make bench-build` | compile-only under base warnings, no `-Werror` | reviewed compile-quality gate |
| Examples `example_*` | `make tooling-build` / `make examples-build` | compile-only under base warnings, no `-Werror` | reviewed compile-quality gate |
| Active test suite | `make test`, `ctest -N`, full `ctest` | compiled/executed routinely, but no dedicated Makefile strict warning gate | parity and suite-truthfulness target |
| CMake Apple Clang full-tree path | `cmake --build ... --clean-first` plus `ctest` | broad warnings, no `-Werror`, narrower bench/example coverage | authoritative cross-check and parity path |
| Dead-code workflow | `make deadcode-report`, `make deadcode-check` | report completeness, not warning cleanliness | separate quality-flow gate |
| Windows / MSVC | `windows-ci.yml` CMake build/test | `/W3`, different target surface | phase-1 portability cross-check, not shared warning gate |

### Day 2 Include / Exclude Guidance

**Include early in Sprint 34**

- library compile cleanliness through `make lint`
- benchmark/example compile coverage through `tooling-build`
- active-suite visibility through `ctest -N` and full `ctest`
- dead-code completeness through `deadcode-check`, with the Sprint 33 coverage-gap and serialization limitations preserved explicitly

**Do not force into the first warning-gate contract yet**

- MSVC equivalence with the POSIX warning contract
- a full Makefile-side strict warning gate for every test translation unit
- pretending the current dead-code compile-db covers all benchmark/example programs

### Day 2 Interpretation

- Sprint 34 phase 1 should start from the current strongest truthful split:
  - `src/*.c` already has a real strict gate
  - benchmarks/examples already have compile-only coverage
  - the CMake Apple Clang path already provides the whole-suite parity view
  - dead-code already has a completeness gate, but not yet a CI-safe execution model
- The main design problem for Day 3 is therefore not “what should we check?”
- It is:
  - how to turn these existing partial checks into one coherent local contract
  - without overstating what Windows, the full test tree, or the dead-code compile-db currently prove

### Day 2 Outputs

- `artifacts/day2-warning-gate-audit.md`
