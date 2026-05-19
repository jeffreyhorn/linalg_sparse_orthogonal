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

## Day 3

**Objective:** Turn the Day 2 audit into a concrete Sprint 34 phase-1 enforcement contract by deciding which commands are authoritative, how compile-only benchmark/example checks relate to runtime tests, and where dead-code checks sit in the quality flow without blurring the warning-clean story.

### Commands Run

1. Re-read the Day 2 audit and Sprint 34 Day 3 scope:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,420p' docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `sed -n '55,155p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/artifacts/day2-warning-gate-audit.md`
2. Refresh current maintainer/operator quality guidance:
   - `sed -n '90,150p' README.md`
   - `sed -n '490,660p' README.md`
   - `rg -n "deadcode|tooling-build|lint|check|ctest -N|compile-quality|warning gate" README.md docs/algorithm.md docs/planning/EPIC_3 -g '!docs/planning/EPIC_3/SPRINT_34/**'`

### Day 3 Design Decisions

#### 1. Phase-1 enforcement must preserve the repo's existing top-level command contract

Sprint 34 should preserve these user-facing meanings:

- `make lint`
  - remains the main local compile/static-analysis path
- `make test`
  - remains the runtime execution path
- `ctest -N` and full `ctest`
  - remain the authoritative active-suite visibility path
- `make deadcode-report` / `make deadcode-check`
  - remain dead-code reporting/completeness commands, not generic warning gates

Interpretation:

- Sprint 34 should harden and connect these commands
- it should **not** redefine them in a way that makes the command names misleading
- that argues for adding a new reviewed-target aggregate layer above existing commands rather than forcing one existing target to mean everything

#### 2. The authoritative phase-1 warning contract is split intentionally across compile-quality and runtime-quality layers

**Authoritative compile-quality layer**

- library sources:
  - `make lint`
  - authoritative for strict warning cleanliness on `src/*.c`
- benchmark/example compile-only reviewed surface:
  - `make tooling-build`
  - authoritative for "these entry points still compile under the maintained warning set"

**Authoritative runtime/suite layer**

- `make test`
  - authoritative for Makefile-side runtime regression protection
- `ctest -N`
  - authoritative registry view of the currently active CMake suite
- full `ctest`
  - authoritative CMake-side executed-suite cross-check

Interpretation:

- Sprint 34 phase 1 should not pretend compile-only entry points and runtime test execution are the same kind of signal
- compile-quality and runtime-quality should be siblings in the aggregate contract, not collapsed into one command's semantics

#### 3. Compile-only benchmark/example enforcement should stay compile-only in phase 1

Chosen rule:

- benchmark/example compile-quality checks remain separate from execution
- `bench-fast`, `bench`, `examples`, and `wall-check` stay runtime/perf/operator paths, not compile-quality prerequisites

Reasoning:

- Sprint 31 already established the compile-only tooling gate specifically to avoid conflating compile drift with slow benchmark execution
- folding benchmark/example execution into a warning-clean contract would make the gate slower, noisier, and harder to reason about
- the compile-quality question for Sprint 34 is:
  - "do these reviewed entry points still build under the maintained warning set?"
- not:
  - "did every benchmark runtime and wall-time check also run?"

Implication for Day 4-6:

- the phase-1 Makefile contract should treat `tooling-build` as a compile-quality dependency
- it should not absorb `bench-fast`, `wall-check`, or full benchmark/example execution into the same target

#### 4. Dead-code stays separate from warning cleanliness in phase 1, but should become a sibling quality gate

Chosen rule:

- do **not** fold `deadcode-check` directly into the warning-clean definition
- do **not** rename or reinterpret it as a warning gate
- do treat it as a sibling build-quality invariant in the higher-level reviewed-target flow

Reasoning:

- the dead-code workflow currently proves report completeness and classification integrity, not "zero findings"
- it still carries the Sprint 33 compile-db gap and serial-execution limitation
- mixing it directly into the warning-clean definition would blur two different failure meanings:
  - compile-quality regression
  - dead-code report/infrastructure incompleteness

Implication for Day 4-6:

- the new aggregate quality target should sequence warning/compile-quality and dead-code checks as separate named steps
- dead-code should remain separately invocable for maintainers
- if integrated into a higher-level target, the output must make the category boundary obvious

#### 5. The phase-1 contract should have one authoritative local path and one authoritative parity path

**Authoritative local path**

Chosen components:

1. `make format-check`
2. `make lint`
3. `make test`
4. `make deadcode-check`

Role:

- this is the reviewed local quality contract Sprint 34 should harden first
- it preserves the Sprint 32 expectation that `make lint` and `make test` remain in the normal path
- it keeps dead-code separate but adjacent

**Authoritative parity path**

Chosen components:

1. serialized Apple Clang CMake rebuild
2. `ctest -N`
3. full `ctest`
4. dead-code coverage-gap visibility through generated reporting

Role:

- this is the cross-check/parity contract
- it confirms that the Makefile-side local path has not drifted away from the maintained CMake path
- it does **not** need to replicate every Makefile target one-for-one on Day 3

Interpretation:

- Sprint 34 should aim for "one primary local enforcement story plus one primary parity story"
- not "every path does everything"

#### 6. The phase-1 aggregate target should be additive, not a semantic rewrite of `check`

Chosen design direction:

- keep current `check = format-check + lint + test` intact unless there is a compelling reason to broaden it later
- add a new reviewed-target aggregate target in Days 4-6 for Sprint 34's expanded contract

Reasoning:

- `check` already has a stable meaning in the repo
- `deadcode-check` has a stable meaning in the repo
- overloading `check` immediately would make it harder to distinguish old behavior from Sprint 34 behavior in docs, CI, and operator usage

Design consequence:

- Day 4 should design a new top-level aggregate with stepwise, attributable output
- that aggregate can depend on or sequence:
  - `format-check`
  - `lint`
  - `test`
  - `deadcode-check`
- while preserving direct access to each underlying command

#### 7. Explicit phase-1 inclusion and exclusion list

**Include in the first Sprint 34 enforcement contract**

- `src/*.c` strict compile cleanliness through `make lint`
- benchmark/example compile-only reviewed coverage through `tooling-build`
- runtime regression protection through `make test`
- active-suite registry/execution visibility through `ctest -N` and full `ctest`
- dead-code report completeness through `make deadcode-check`

**Keep as supporting or deferred, not primary phase-1 gate semantics**

- Windows `/W3` warning behavior
- full Makefile-side strict warning gate for all test translation units
- full bench/example runtime execution
- `wall-check`
- compile-db expansion beyond the current Sprint 33 exclusion list
- any interpretation of `deadcode-check` as "zero dead-code findings"

### Day 3 Command-Level Contract

| Layer | Command(s) | Phase-1 status | Meaning |
|---|---|---|---|
| formatting | `make format-check` | authoritative local prerequisite | source formatting is clean |
| library compile warnings | `make lint` | authoritative local warning gate | `src/*.c` clean under strict compile + static analysis; includes compile-only tooling build |
| runtime tests | `make test` | authoritative local runtime gate | Makefile-side test binaries build and pass |
| dead-code completeness | `make deadcode-check` | authoritative local sibling gate | dead-code report regenerated and completeness invariants hold |
| CMake suite registry | `ctest -N` | authoritative parity/supporting gate | active CMake suite remains visible and countable |
| CMake suite execution | full `ctest` | authoritative parity/supporting gate | CMake-executed suite still passes |
| dead-code coverage gap | `build/deadcode/coverage-notes.txt` via `deadcode-report` | explicit preserved limitation | known bench/example exclusions remain truthful until broadened |

### Day 3 Interpretation

- Sprint 34 phase 1 should not define "warning-clean" as one overloaded command.
- It should define a reviewed quality contract with:
  - a strict local compile/static-analysis path
  - a separate runtime test path
  - a separate dead-code completeness path
  - a maintained CMake parity path
- That gives Day 4 a concrete design target:
  - add an aggregate layer that sequences these checks cleanly
  - without destroying the meaning of `lint`, `test`, `check`, or `deadcode-check`

### Day 3 Outputs

- `artifacts/day3-warning-gate-design.md`

## Day 4

**Objective:** Turn the Day 3 contract into a concrete Makefile target topology by deciding the new phase-1 aggregate target names, dependency style, operator-facing output shape, and dead-code integration model before any Makefile edits land.

### Commands Run

1. Re-read the current Sprint 34 contract and Day 4 scope:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,520p' docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `sed -n '95,210p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/artifacts/day3-warning-gate-design.md`
2. Inspect the current Makefile quality-target region and target graph behavior:
   - `sed -n '230,520p' Makefile`
   - `make -n check`
   - `make -n deadcode`
   - `make -n deadcode-report`
   - `make -n deadcode-check`
   - `python3 - <<'PY' ... print line numbers for tooling-build / format-check / lint / check / deadcode* ... PY`
3. Refresh prior layered-target precedent from Sprint 33:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/artifacts/day4-tooling-integration-design.md`
4. Reconfirm current operator-facing docs that Day 4 must preserve:
   - `rg -n "reviewed-target|quality-flow|deadcode-check|tooling-build:|check:" docs/planning/EPIC_3 README.md -g '!docs/planning/EPIC_3/SPRINT_34/**'`

### Day 4 Target-Graph Findings

- current stable quality targets and line anchors:
  - `tooling-build`: line `247`
  - `format-check`: line `422`
  - `lint`: line `428`
  - `check`: line `446`
  - `deadcode`: line `503`
  - `deadcode-report`: line `506`
  - `deadcode-check`: line `511`
- current `check` dry run proves the inherited layering:
  - formatting first
  - then `lint` with `tooling-build` folded inside it
  - then full test-binary build + execution
- current `deadcode*` dry runs prove that:
  - `deadcode` = raw workflow/stamp
  - `deadcode-report` = raw workflow + report generation
  - `deadcode-check` = report generation + completeness verification
- important operational constraint:
  - prerequisite-only aggregation would allow `make -j` to run sibling quality targets concurrently
  - that is acceptable for `check` today
  - but it is a bad fit for Sprint 34 because `deadcode*` still shares `build/deadcode-cmake` and `build/deadcode/`

### Day 4 Design Decisions

#### 1. Sprint 34 should add a new reviewed-quality aggregate instead of mutating `check`

Chosen operator-facing target:

- `quality-review`

Chosen rule:

- keep `check` unchanged as `format-check + lint + test`
- add `quality-review` as the new Sprint 34 reviewed local quality contract

Reasoning:

- `check` already has stable repo semantics
- `quality-review` communicates that this is the broader reviewed-target contract, not the historical minimal aggregate
- this avoids silent behavior drift for existing users and CI jobs that currently rely on `check`

#### 2. The aggregate should be implemented as a serial wrapper target, not as a pure prerequisite fan-out

Chosen implementation shape:

- `.PHONY: quality-review`
- recipe body invokes subordinate targets in order using recursive `$(MAKE)` calls

Recommended sequence:

1. `format-check`
2. `lint`
3. `test`
4. `deadcode-check`

Reasoning:

- preserves strict step order
- makes failure attribution obvious
- prevents accidental parallel execution of `deadcode*` under `make -j`
- avoids the ambiguity of a large prerequisite fan-out where it is unclear which phase failed without reading dense raw output

Day 4 implication:

- `quality-review` should be a wrapper target, not a large dependency list

#### 3. The Sprint 34 topology should expose one focused compile-only reviewed target in addition to the full aggregate

Chosen supporting target:

- `quality-review-compile`

Chosen meaning:

- formatting + compile/static-analysis reviewed surface only
- intended sequence:
  1. `format-check`
  2. `lint`

Reasoning:

- Day 3 made the compile/static-analysis layer explicit
- local developers and later CI work need a fast reviewed-target subset that stops before runtime tests and dead-code reporting
- this keeps the target graph extension-friendly for Sprint 34 CI work

Chosen non-goal:

- do not add redundant wrappers for `test` or `deadcode-check` yet; those targets already exist with clear meanings

#### 4. `lint` should remain the place where benchmark/example compile coverage enters the flow

Chosen rule:

- do not split `tooling-build` back out of `lint`
- do not create a parallel compile-only bench/example branch outside `lint` for the main reviewed aggregate

Reasoning:

- Sprint 31 intentionally wired compile-only tooling coverage into `lint`
- that existing layering already answers the Day 3 compile-quality contract for benchmarks/examples
- adding a second path to build the same binaries in the same aggregate would create duplicate work and muddier output

Implication:

- `quality-review-compile` and `quality-review` should both enter compile-only benchmark/example coverage through `lint`

#### 5. Dead-code should be integrated only at the top reviewed aggregate layer in phase 1

Chosen rule:

- `quality-review-compile` stops at `lint`
- `quality-review` adds:
  - `test`
  - `deadcode-check`

Reasoning:

- dead-code is a sibling quality category, not compile-only warning logic
- this keeps the compile subset fast and conceptually clean
- it makes the full reviewed aggregate the only place where the serial dead-code step is forced

#### 6. Failure-output design should use explicit phase banners

Chosen output shape:

- before each recursive `$(MAKE)` call, emit a short banner such as:
  - `== quality-review: format-check ==`
  - `== quality-review: lint ==`
  - `== quality-review: test ==`
  - `== quality-review: deadcode-check ==`

Reasoning:

- keeps output attributable even when downstream targets are verbose
- avoids the need for users to infer which phase failed from raw tool stderr
- fits the Sprint 34 goal of actionable operator-facing quality gates

Chosen non-goal:

- no deep shell orchestration
- no opaque `&&` chains
- no duplicated internal tool commands inside the wrapper target

#### 7. Day 4 dead-code integration rule must preserve the Sprint 33 serial constraint explicitly

Chosen rule:

- `quality-review` runs `deadcode-check` as a terminal serial step
- it does not list `deadcode-check` as a parallelizable sibling prerequisite
- the target documentation should explicitly say that the reviewed aggregate preserves the current shared-build-tree dead-code execution model

Reasoning:

- this is the minimal truthful phase-1 integration
- it keeps the dead-code flow in the quality path
- but it does not pretend the underlying workflow is already parallel-safe

#### 8. Recommended Day 5 / Day 6 implementation split

**Day 5 Batch I**

- add:
  - `quality-review-compile`
  - `quality-review`
- implement serial wrapper recipes with phase banners
- validate focused target behavior and output

**Day 6 Batch II**

- tighten dependency/variable reuse if needed
- document the new targets
- decide whether any small helper variables/macros are warranted to keep the recipes readable
- validate the full reviewed path end to end

### Day 4 Proposed Target Topology

| Target | Type | Phase-1 meaning | Uses existing targets |
|---|---|---|---|
| `check` | unchanged inherited aggregate | existing repo quality shortcut | `format-check`, `lint`, `test` |
| `quality-review-compile` | new serial wrapper | reviewed formatting + compile/static-analysis contract | `format-check`, `lint` |
| `quality-review` | new serial wrapper | reviewed full local quality contract | `format-check`, `lint`, `test`, `deadcode-check` |
| `deadcode-check` | unchanged inherited target | dead-code completeness invariant | `deadcode-report` + check script |

### Day 4 Implementation Guidance

- prefer recursive `$(MAKE)` step calls over a large prerequisite list
- keep the wrappers thin and readable
- reuse existing targets rather than copying their internals
- do not rename `check`, `lint`, `test`, `deadcode-report`, or `deadcode-check`
- do not wire the new targets into CI on Day 4; leave that to the later Sprint 34 CI design/implementation days

### Day 4 Interpretation

- The right Sprint 34 Makefile move is additive, serial, and explicit.
- additive:
  - preserve the existing target contract
- serial:
  - respect the dead-code shared-path constraint and keep failure attribution clean
- explicit:
  - expose reviewed-target aggregates by name instead of hiding broader behavior behind `check`

### Day 4 Outputs

- `artifacts/day4-makefile-enforcement-design.md`

## Day 5

**Objective:** Implement the Day 4 Makefile wrapper targets for Sprint 34 phase-1 reviewed quality enforcement, validate their sequencing and output shape, and confirm that the new aggregate layer works without changing the meanings of the inherited quality targets.

### Commands Run

1. Re-read the Day 4 design and current Makefile quality-target region:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '400,530p' Makefile`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_34/artifacts/day4-makefile-enforcement-design.md`
2. Implement the new wrapper targets:
   - `apply_patch` on `Makefile`
3. Validate target shape and line placement:
   - `make -n quality-review-compile`
   - `make -n quality-review`
   - `python3 - <<'PY' ... print line numbers for quality-review-compile / quality-review ... PY`
4. Validate live behavior:
   - `make quality-review-compile`
   - `make quality-review`
5. Capture post-change status:
   - `git diff --stat`
   - `git status --short`

### Day 5 Implementation Result

- Added two new serial reviewed-quality wrapper targets in `Makefile`:
  - `quality-review-compile`
  - `quality-review`
- Target anchors after implementation:
  - `quality-review-compile`: line `455`
  - `quality-review`: line `462`
- Existing target semantics were preserved:
  - `check` unchanged
  - `lint` unchanged
  - `test` unchanged
  - `deadcode-check` unchanged

### Implemented Wrapper Behavior

**`quality-review-compile`**

- bannered serial sequence:
  1. `format-check`
  2. `lint`

**`quality-review`**

- bannered serial sequence:
  1. `format-check`
  2. `lint`
  3. `test`
  4. `deadcode-check`

### Day 5 Validation Results

**Dry-run validation**

- `make -n quality-review-compile`
  - showed the intended phase banners plus recursive `make format-check` then `make lint`
- `make -n quality-review`
  - showed the intended phase banners plus recursive `make format-check`, `make lint`, `make test`, then `make deadcode-check`

Interpretation:

- the target graph and operator-facing sequencing match the Day 4 design
- the new wrappers reuse existing targets rather than duplicating tool commands internally

**Live validation**

- `make quality-review-compile`: passed
  - validated the wrapper banners
  - validated real execution of `format-check`
  - validated real execution of `lint`, including:
    - `tooling-build`
    - strict `src/*.c` warning gate
    - `clang-tidy`
    - `cppcheck`
- `make quality-review`: passed
  - validated the full wrapper sequence end to end
  - passed through:
    - `format-check`
    - `lint`
    - `test`
    - `deadcode-check`

Most important Day 5 proof:

- the full wrapper reaches the final dead-code phase in the intended serial order:
  - `== quality-review: deadcode-check ==`
  - then the inherited dead-code workflow/configure/report/check flow
  - then `deadcode-check: report completeness checks passed.`

### Day 5 Interpretation

- The Sprint 34 reviewed-quality layer now exists as real Makefile behavior, not just design text.
- The implementation stayed additive:
  - no legacy target meaning was repurposed
  - no parallel prerequisite fan-out was introduced for the dead-code step
- The new wrappers are already usable for local enforcement and later CI integration work.

### Day 5 Outputs

- `artifacts/day5-makefile-enforcement-batch1.md`

## Day 6

**Objective:** Tighten the Day 5 reviewed-quality path by documenting the new wrapper targets in the maintained README contract, making the serial dead-code constraint explicit at the Makefile target level, and revalidating the full reviewed path on the updated graph.

### Commands Run

1. Re-read the Day 5 result and inspect the current reviewed target/doc surfaces:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '430,540p' Makefile`
   - `sed -n '100,145p' README.md`
   - `sed -n '618,670p' README.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/artifacts/day5-makefile-enforcement-batch1.md`
2. Implement the Day 6 tightening/documentation updates:
   - `apply_patch` on `Makefile`
   - `apply_patch` on `README.md`
3. Validate the updated graph and docs shape:
   - `make -n -j2 quality-review`
   - `python3 - <<'PY' ... print .NOTPARALLEL and quality-review line anchors ... PY`
   - `sed -n '103,120p' README.md`
   - `sed -n '660,705p' README.md`
4. Validate the full reviewed path on the updated graph:
   - `make -j2 quality-review`
5. Capture post-change state:
   - `git status --short`
   - `git diff --stat`

### Day 6 Implementation Result

- `Makefile`
  - added an explicit `.NOTPARALLEL` declaration for:
    - `quality-review-compile`
    - `quality-review`
    - `deadcode`
    - `deadcode-report`
    - `deadcode-check`
- current relevant line anchors:
  - `.NOTPARALLEL`: line `454`
  - `quality-review-compile`: line `456`
  - `quality-review`: line `463`
- `README.md`
  - added `quality-review-compile` and `quality-review` to the main "With Make" command list
  - added a dedicated "Reviewed Local Quality Path" section describing:
    - exact wrapper sequences
    - the fact that the wrappers are additive
    - the fact that `check`, `lint`, `test`, and `deadcode-check` keep their existing meanings

### Day 6 Validation Results

**Graph/documentation validation**

- `make -n -j2 quality-review`
  - still showed the intended bannered phase order:
    - `format-check`
    - `lint`
    - `test`
    - `deadcode-check`
- README now exposes the new reviewed targets in the maintained build instructions and explains their meaning near the dead-code workflow section

Interpretation:

- the new Sprint 34 wrapper targets are no longer just Makefile-internal knowledge
- they are part of the documented local quality contract

**Live validation**

- `make -j2 quality-review`: passed

Important observed behavior:

- the banner order remained serial and attributable under a `-j2` invocation
- `lint` still entered benchmark/example compile coverage through its inherited `tooling-build` dependency
- `test` still ran after the lint phase completed
- `deadcode-check` still ran as the terminal reviewed step
- because no dead-code inputs changed after the earlier Day 5 run, the final `deadcode-check` phase reused the existing report stamp and completed immediately with:
  - `deadcode-check: report completeness checks passed.`

Day 6 implication:

- the reviewed path is now both documented and dependency-tight enough to avoid needless dead-code reruns when the report stamp is already fresh
- the remaining limitation is still the Sprint 33 one:
  - separate concurrent shell invocations can still race on the shared `build/deadcode-cmake` / `build/deadcode/` paths
  - but the reviewed aggregate itself no longer exposes a parallel sibling path for those targets

### Day 6 Interpretation

- Sprint 34 now has a completed local reviewed-quality path, not just a first-pass wrapper implementation.
- completed means:
  - additive target contract
  - explicit serial dead-code guard at the Makefile target level
  - maintained README documentation
  - revalidated full reviewed-path execution on the updated graph

### Day 6 Outputs

- `artifacts/day6-makefile-enforcement-batch2.md`

## Day 7

**Objective:** Audit what the current CMake path already proves for Sprint 34 reviewed-target enforcement, separate true CMake-path parity from Makefile-only quality responsibilities, and write the concrete Day 8 implementation contract.

### Commands Run

1. Re-read the Sprint 34 Day 7 scope and the inherited Sprint 33 CMake/dead-code constraints:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_33/HANDOFF.md`
2. Inspect the current CMake target graph and active-suite surface:
   - `sed -n '1,260p' CMakeLists.txt`
   - `sed -n '261,420p' CMakeLists.txt`
   - `ctest -N --test-dir build/sprint33-day1-cmake`
   - `find build/sprint33-day1-cmake -maxdepth 2 -type f | sort | sed -n '1,200p'`
3. Measure the current compile-db coverage against the maintained bench/example inventories:
   - `python3 - <<'PY' ... count build/deadcode-cmake/compile_commands.json by top-level directory ... PY`
   - `python3 - <<'PY' ... list all bench_*.c and example_*.c files ... PY`
   - `python3 - <<'PY' ... list benchmark/example files present in build/sprint33-day1-cmake/compile_commands.json ... PY`
   - `sed -n '1,220p' build/deadcode/coverage-notes.txt`
   - `sed -n '1,220p' build/deadcode/report.md`
4. Cross-check the current Makefile/README quality-contract boundary:
   - `sed -n '420,560p' Makefile`
   - `rg -n "deadcode|quality-review|tooling-build|format-check|lint:|check:" Makefile README.md docs/planning/EPIC_3/PROJECT_PLAN.md`
5. Record the Day 7 parity design:
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/artifacts/day7-cmake-parity-design.md`

### Day 7 Audit Result

Current `build/sprint33-day1-cmake` proof surface:

- `ctest -N` still shows `53` registered tests.
- the generated CMake tree currently builds:
  - library: `libsparse_lu_ortho.a`
  - tests: `53`
  - benchmarks: `13`
  - examples: `6`
- the CMake compile database currently covers:
  - `src`: `25`
  - `tests`: `53`
  - `benchmarks`: `13`
  - `examples`: `6`

Current CMake-path coverage gaps versus the maintained Makefile tooling surface:

- benchmarks missing from the CMake compile-db:
  - `bench_svd`
- examples missing from the CMake compile-db:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

What the current CMake path does prove:

- active-suite registration remains auditable through `ctest -N`
- active-suite execution remains auditable through full `ctest`
- the CMake-defined reviewed target set can be rebuilt cleanly from one generated build tree
- that build tree exercises a broader warning family than the Makefile lint syntax-only pass, including the Sprint 32 baseline categories preserved in `CMakeLists.txt`

What the current CMake path does not prove:

- it does not replace `make format-check`
- it does not replace `make lint`'s `clang-tidy` / `cppcheck` phases
- it does not replace `make deadcode-check`
- it does not cover the full Makefile benchmark/example compile-only surface because the Sprint 33 compile-db exclusion list still exists
- it does not make benchmark/example runtime execution part of Sprint 34's reviewed compile-quality contract

### Day 7 Design Decision

Day 8 should implement dedicated CMake-parity wrapper targets rather than changing the meaning of any existing local quality target.

Chosen target shape:

- `quality-review-cmake-compile`
  - configure a dedicated build tree:
    - `QUALITY_REVIEW_CMAKE_DIR ?= build/quality-review-cmake`
  - run:
    - `cmake -S . -B $(QUALITY_REVIEW_CMAKE_DIR) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
    - `cmake --build $(QUALITY_REVIEW_CMAKE_DIR) --parallel 1 --clean-first`
    - `ctest -N --test-dir $(QUALITY_REVIEW_CMAKE_DIR)`
- `quality-review-cmake`
  - run `quality-review-cmake-compile`
  - then run:
    - `ctest --test-dir $(QUALITY_REVIEW_CMAKE_DIR) --output-on-failure`

Why this shape:

- it mirrors the Sprint 34 Day 5 split between compile-oriented review and full review
- it keeps `ctest -N` as an explicit auditable active-suite check instead of burying it in implementation details
- it keeps the full CTest execution step separate and attributable
- it avoids overloading the historical `build/sprint33-day1-cmake` artifact tree with new reviewed-quality semantics

### Day 7 Parity Boundary

Direct Sprint 34 CMake parity target:

- clean rebuild of the CMake-defined reviewed set
- `ctest -N`
- full `ctest`

Documented Makefile-authoritative sibling checks:

- `format-check`
- `lint` static-analysis phases
- `deadcode-check`
- compile-only coverage for:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Dead-code parity rule for Day 8:

- do not silently imply the dead-code compile-db gap is closed
- keep the exclusion list explicit in:
  - `build/deadcode/coverage-notes.txt`
  - `build/deadcode/report.md`
  - maintainer-facing documentation
- treat compile-db broadening as later work unless Day 8 can do it without destabilizing the reviewed target contract

### Day 7 Interpretation

- Sprint 34 does not need to force all quality semantics through CMake to achieve phase-1 parity.
- It does need a first-class, named CMake reviewed path so the project cannot drift into an effectively Make-only build/test contract.
- The truthful Day 8 scope is therefore:
  - add named CMake reviewed wrappers
  - validate them end to end
  - preserve the Sprint 33 dead-code coverage gap as an explicit limitation rather than hiding it

### Day 7 Outputs

- `artifacts/day7-cmake-parity-design.md`

## Day 8

**Objective:** Implement the reviewed CMake parity path from Day 7, validate the new wrapper targets end to end, and record which Sprint 34 quality guarantees now have named Make/CMake reviewed flows versus preserved documented limitations.

### Commands Run

1. Re-read the Day 7 design contract and the current wrapper surface:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/artifacts/day7-cmake-parity-design.md`
   - `sed -n '430,560p' Makefile`
   - `sed -n '100,140p' README.md`
   - `sed -n '670,725p' README.md`
2. Implement the Day 8 parity wrappers and maintained docs updates:
   - `apply_patch` on `Makefile`
   - `apply_patch` on `README.md`
3. Validate the target graph and README contract:
   - `make -n quality-review-cmake-compile`
   - `make -n quality-review-cmake`
   - `sed -n '100,118p' README.md`
   - `sed -n '670,710p' README.md`
4. Validate the live reviewed CMake parity path:
   - `make quality-review-cmake-compile`
   - `make quality-review-cmake`
5. Record the Day 8 implementation result:
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/artifacts/day8-cmake-parity-implementation.md`

### Day 8 Implementation Result

- `Makefile`
  - added:
    - `QUALITY_REVIEW_CMAKE_DIR ?= build/quality-review-cmake`
    - `quality-review-cmake-compile`
    - `quality-review-cmake`
  - extended `.NOTPARALLEL` so the reviewed CMake wrappers also stay serial under `make -j`
- `README.md`
  - added both reviewed CMake targets to the maintained "With Make" command list
  - extended the "Reviewed Local Quality Path" section with:
    - exact command sequence for `quality-review-cmake-compile`
    - exact command sequence for `quality-review-cmake`
    - the explicit boundary that these CMake wrappers do not replace Makefile-authoritative formatter, static-analysis, or dead-code checks

Implemented target behavior:

- `quality-review-cmake-compile`
  - configure:
    - `cmake -S . -B build/quality-review-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  - rebuild:
    - `cmake --build build/quality-review-cmake --parallel 1 --clean-first`
  - active-suite audit:
    - `ctest -N --test-dir build/quality-review-cmake`
- `quality-review-cmake`
  - runs `quality-review-cmake-compile`
  - then runs:
    - `ctest --test-dir build/quality-review-cmake --output-on-failure`

### Day 8 Validation Results

**Graph/contract validation**

- `make -n quality-review-cmake-compile`
  - showed the intended three explicit phases:
    - configure
    - build
    - `ctest -N`
- `make -n quality-review-cmake`
  - showed the intended reviewed flow:
    - recursive `quality-review-cmake-compile`
    - then full `ctest`
- README now exposes both reviewed CMake targets in the maintained operator-facing command list and documents the boundary versus the Makefile-reviewed path

**Live validation**

- `make quality-review-cmake-compile`: passed
  - configured `build/quality-review-cmake`
  - completed a clean serialized rebuild of the full CMake-defined reviewed set
  - `ctest -N` reported `53` registered tests
- `make quality-review-cmake`: passed
  - reran the reviewed compile path on the dedicated build tree
  - full `ctest` passed:
    - `53 / 53` tests passed
    - reported real time: `236.99 sec`

Observed reviewed CMake parity surface after implementation:

- direct CMake-reviewed guarantees now exist for:
  - clean rebuild of the CMake-defined reviewed target set
  - active-suite registration via `ctest -N`
  - active-suite execution via full `ctest`
- preserved Makefile-authoritative sibling checks remain:
  - `format-check`
  - `lint` static-analysis phases
  - `deadcode-check`

Preserved documented limitation:

- the Sprint 33 dead-code compile-db gap remains unchanged and explicit:
  - missing benchmark:
    - `bench_svd`
  - missing examples:
    - `example_basic_solve`
    - `example_condition`
    - `example_iterative`
    - `example_least_squares`
    - `example_matrix_free`
    - `example_svd_lowrank`

### Day 8 Interpretation

- Sprint 34 now has named reviewed paths on both sides of the build split:
  - Makefile reviewed path:
    - `quality-review-compile`
    - `quality-review`
  - CMake reviewed parity path:
    - `quality-review-cmake-compile`
    - `quality-review-cmake`
- This closes the main Day 7 concern that the project could drift into an effectively Make-only reviewed build/test contract.
- It does not yet close the dead-code compile-db coverage gap, which remains a truthful documented limitation rather than an implied guarantee.

### Day 8 Outputs

- `artifacts/day8-cmake-parity-implementation.md`

## Day 9

**Objective:** Audit the existing GitHub Actions job surface, choose the narrow Sprint 34 phase-1 CI enforcement set, and write the workflow contract that Days 10 and 12 should implement without reintroducing dead-code flakiness.

### Commands Run

1. Re-read the Sprint 34 Day 9 scope and current branch state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '220,340p' docs/planning/EPIC_3/SPRINT_34/PLAN.md`
2. Inventory the existing CI workflows:
   - `find .github -maxdepth 3 -type f | sort`
   - `sed -n '1,260p' .github/workflows/ci.yml`
   - `sed -n '1,260p' .github/workflows/macos-ci.yml`
   - `sed -n '1,260p' .github/workflows/windows-ci.yml`
3. Cross-check the local reviewed-target contract and dead-code execution model:
   - `rg -n "quality-review|quality-review-cmake|deadcode|wall-check|format-check|lint|xunused|cppcheck" Makefile README.md .github/workflows`
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,260p' scripts/deadcode_report.py`
4. Record the Day 9 CI design:
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/artifacts/day9-ci-enforcement-design.md`

### Day 9 Audit Result

Current workflow split:

- `.github/workflows/ci.yml`
  - `build-and-test`
    - `make test`
    - `make sanitize`
    - `make asan`
    - `make bench-build`
    - `make bench-fast`
  - `cmake-build-and-test`
    - CMake configure
    - CMake build
    - full `ctest`
    - Makefile/CMake registered-test-count comparison
  - `tsan`
    - Linux thread / OpenMP TSan coverage
  - `lint`
    - `make format-check`
    - `make lint`
  - `coverage`
    - `make coverage`
- `.github/workflows/macos-ci.yml`
  - compiler matrix on `macos-latest`
  - `make`
  - `make test`
  - `make wall-check`
  - Apple Clang `make sanitize`
- `.github/workflows/windows-ci.yml`
  - MSVC CMake configure/build/ctest only

What already aligns with Sprint 34:

- Linux `lint` is already the closest CI analogue to the reviewed Makefile compile-quality path.
- Linux `cmake-build-and-test` is already the closest CI analogue to the reviewed CMake parity path.
- Linux CI is the most realistic first place to add dead-code enforcement because:
  - `cppcheck` already ships there
  - the current dead-code scripts are POSIX-oriented
  - it avoids dragging the initial xunused/toolchain provisioning burden across macOS and Windows in phase 1

What should stay out of phase 1:

- Linux `build-and-test`
  - keep it focused on runtime tests, sanitizers, and benchmark runtime coverage
- Linux `tsan`
  - keep it focused on race-detection scope
- Linux `coverage`
  - keep it focused on coverage production
- macOS matrix jobs
  - keep them focused on portability and `wall-check`
- Windows CMake job
  - keep it focused on MSVC build/test parity rather than dead-code / strict-warning enforcement semantics that are not yet standardized there

### Day 9 Design Decision

Sprint 34 phase-1 CI enforcement should be Linux-first and split across three explicit job responsibilities:

1. reviewed Makefile compile-quality job
2. reviewed CMake parity job
3. serialized dead-code job

Chosen workflow targets by job:

- Linux `lint` job
  - replace the separate:
    - `make format-check`
    - `make lint`
  - with:
    - `make quality-review-compile`
- Linux `cmake-build-and-test` job
  - replace the step-by-step configure/build/test flow with:
    - `make quality-review-cmake`
- new Linux `deadcode` job
  - install:
    - `cppcheck`
    - CMake/Ninja/LLVM/Clang inputs needed to build `xunused`
  - build/install `xunused` in-job on the runner
  - run:
    - `make deadcode-report`
    - `make deadcode-check`
  - upload:
    - `build/deadcode/report.md`
    - `build/deadcode/report.tsv`
    - `build/deadcode/coverage-notes.txt`
    - raw `cppcheck.txt`
    - raw `xunused.txt`
    - on `if: always()`

### Day 9 Job-Selection Matrix

- enforce now:
  - Linux `lint`
    - reviewed compile-quality wrapper
  - Linux `cmake-build-and-test`
    - reviewed CMake parity wrapper
  - Linux `deadcode`
    - report + completeness enforcement
- preserve unchanged for phase 1:
  - Linux `build-and-test`
  - Linux `tsan`
  - Linux `coverage`
  - macOS matrix jobs
  - Windows build/test job

Reasoning:

- these three Linux jobs are narrow enough to stay attributable
- they cover the new Sprint 34 reviewed paths directly
- they avoid conflating warning/dead-code enforcement with sanitizer, benchmark-runtime, portability, or MSVC-specific concerns

### Day 9 Dead-Code CI Execution Model

The Sprint 33 dead-code serial constraint must remain explicit in CI:

- do not run `deadcode`, `deadcode-report`, and `deadcode-check` as sibling parallel branches
- do not put the dead-code job in a matrix
- keep one job, one runner, one artifact tree:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- run the workflow in serial order:
  - tool install / xunused build
  - `make deadcode-report`
  - `make deadcode-check`
  - artifact upload with `if: always()`

This preserves the current execution assumption without pretending the shared-path limitation is already solved.

### Day 9 Failure-Output Expectations

Reviewed compile-quality job:

- failure should surface the existing `quality-review-compile` banners so the operator can distinguish:
  - `format-check`
  - `lint`

Reviewed CMake parity job:

- failure should surface the existing `quality-review-cmake` banners so the operator can distinguish:
  - configure
  - clean build
  - `ctest -N`
  - full `ctest`

Dead-code job:

- step names should remain explicit:
  - install dead-code tools
  - build/install `xunused`
  - generate dead-code report
  - validate dead-code report
  - upload dead-code artifacts
- `upload-artifact` must run on `if: always()` so a failure still leaves:
  - classified report
  - raw tool output
  - coverage-gap note
- failure interpretation should stay actionable:
  - missing tool / xunused build failure
  - raw workflow generation failure
  - report completeness invariant failure

### Day 9 Interpretation

- Sprint 34 phase 1 should not try to make every existing CI leg enforce every new local reviewed path.
- It should wire the reviewed paths into the jobs already closest to their responsibility:
  - `quality-review-compile` into Linux `lint`
  - `quality-review-cmake` into Linux `cmake-build-and-test`
  - `deadcode-report` / `deadcode-check` into one serialized Linux job
- This keeps CI narrow, attributable, and compatible with the inherited dead-code execution constraint.

### Day 9 Outputs

- `artifacts/day9-ci-enforcement-design.md`

## Day 10

**Objective:** Implement the phase-1 Linux CI enforcement plan from Day 9 in `.github/workflows/ci.yml`, preserve the dead-code serial execution model, and record the intentionally deferred CI scope.

### Commands Run

1. Re-read the Day 9 CI contract and current workflow state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_34/artifacts/day9-ci-enforcement-design.md`
   - `sed -n '1,260p' .github/workflows/ci.yml`
2. Implement the Day 10 CI wiring in the Linux workflow only:
   - `apply_patch` on `.github/workflows/ci.yml`
3. Validate the workflow file shape locally:
   - `sed -n '1,240p' .github/workflows/ci.yml`
   - `python3 - <<'PY' ... print line anchors for cmake-build-and-test / lint / deadcode jobs ... PY`
4. Record the Day 10 implementation result:
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/WORKING_NOTES.md`
   - `apply_patch` on `docs/planning/EPIC_3/SPRINT_34/artifacts/day10-ci-enforcement-implementation.md`

### Day 10 Implementation Result

- `.github/workflows/ci.yml`
  - `cmake-build-and-test`
    - replaced the hand-written configure/build/ctest/test-count steps with:
      - `make quality-review-cmake`
  - `lint`
    - kept the tool-install step
    - replaced the separate:
      - `make format-check`
      - `make lint`
    - with:
      - `make quality-review-compile`
  - added new Linux `deadcode` job
    - install step:
      - `cppcheck`
      - `clang-18`
      - `llvm-18-dev`
      - `libclang-18-dev`
      - `ninja-build`
    - build/install `xunused` from upstream source into `$HOME/.local`
    - run:
      - `make deadcode-report`
      - `make deadcode-check`
    - upload on `if: always()`:
      - `build/deadcode/report.md`
      - `build/deadcode/report.tsv`
      - `build/deadcode/coverage-notes.txt`
      - `build/deadcode/cppcheck.txt`
      - `build/deadcode/xunused.txt`

### Day 10 Validation Results

Local workflow-file validation:

- `sed -n '1,240p' .github/workflows/ci.yml`
  - confirmed the intended Linux-only scope
  - confirmed no edits to `build-and-test`, `tsan`, or `coverage`
- anchor check:
  - `cmake-build-and-test`: line `37`
  - `lint`: line `124`
  - `deadcode`: line `137`

Interpretation:

- the CI implementation landed exactly on the three Day 9 target jobs
- macOS and Windows remained untouched
- the dead-code flow remains a single non-matrix Linux job, which preserves the Sprint 33 serial execution assumption

### Day 10 Deferred Scope

Still intentionally deferred after this implementation:

- Linux `build-and-test`
  - unchanged runtime / sanitizer / benchmark-runtime role
- Linux `tsan`
  - unchanged race-detection role
- Linux `coverage`
  - unchanged coverage-production role
- macOS workflow
  - unchanged portability / `wall-check` role
- Windows workflow
  - unchanged MSVC CMake parity role
- dead-code compile-db coverage gap
  - still preserved and reported, not implicitly closed

### Day 10 Interpretation

- Sprint 34 now has a first CI enforcement pass that directly exercises the new reviewed paths rather than duplicating their internals in workflow YAML.
- The implementation stays phase-1 narrow:
  - Makefile reviewed compile-quality wrapper
  - CMake reviewed parity wrapper
  - serialized dead-code reporting/check job
- It also preserves the inherited dead-code non-flakiness assumption by keeping one runner, one artifact tree, and always-on artifact upload for failure triage.

### Day 10 Outputs

- `artifacts/day10-ci-enforcement-implementation.md`
