# Sprint 33 Working Notes

## Day 1

**Objective:** Convert the Sprint 32 handoff into a precise Sprint 33 dead-code baseline by confirming the inherited clean-state invariants, auditing the current Makefile/CMake/static-analysis entry points, generating a dedicated `compile_commands.json` path for later `xunused` work, and naming the first low-risk cleanup surfaces before any code removal begins.

### Commands Run

1. Read Sprint 33 scope and Sprint 32 handoff inputs:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_33/PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/HANDOFF.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_32/RETROSPECTIVE.md`
2. Inspect current build and analysis entry points:
   - `rg -n "deadcode|cppcheck|clang-tidy|compile_commands|xunused" Makefile CMakeLists.txt scripts tests src include docs/planning/EPIC_3 -g '!docs/planning/EPIC_3/SPRINT_33/**'`
   - `sed -n '1,260p' Makefile`
   - `sed -n '400,470p' Makefile`
   - `sed -n '1,280p' CMakeLists.txt`
   - `find build -maxdepth 3 \( -name compile_commands.json -o -name '*.json' \) | sort`
3. Reconfirm the current validated test-registry baseline and local tool availability:
   - `ctest -N --test-dir build/sprint32-day1-cmake`
   - `command -v cppcheck || true; command -v xunused || true; command -v clang-tidy || true`
4. Generate a dedicated Sprint 33 compilation database path:
   - `cmake -S . -B build/sprint33-day1-cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
   - `wc -l build/sprint33-day1-cmake/compile_commands.json`
   - `grep -c '"file":' build/sprint33-day1-cmake/compile_commands.json`
   - `sed -n '1,20p' build/sprint33-day1-cmake/compile_commands.json`
5. Derive baseline scope counts and coverage gaps:
   - `find src -maxdepth 1 -name '*.c' | wc -l`
   - `find tests -maxdepth 1 -name '*.c' | wc -l`
   - `find benchmarks -maxdepth 1 -name '*.c' | wc -l`
   - `find examples -maxdepth 1 -name '*.c' | wc -l`
   - `grep '"file":' build/sprint33-day1-cmake/compile_commands.json | awk '...'`
   - `printf '%s\n' benchmarks/*.c | sed 's#benchmarks/##; s#\.c$##' | sort > /tmp/s33_bench_all.txt`
   - `rg -o "add_executable\\((bench_[^ )]+)" -r '$1' CMakeLists.txt | sort > /tmp/s33_bench_cmake.txt`
   - `comm -23 /tmp/s33_bench_all.txt /tmp/s33_bench_cmake.txt`
   - `printf '%s\n' examples/*.c | sed 's#examples/##; s#\.c$##' | sort > /tmp/s33_ex_all.txt`
   - `rg -o "add_executable\\((example_[^ )]+)" -r '$1' CMakeLists.txt | sort > /tmp/s33_ex_cmake.txt`
   - `comm -23 /tmp/s33_ex_all.txt /tmp/s33_ex_cmake.txt`
   - `ls src/*internal*.h src/*errno*.h 2>/dev/null | sort`
   - `rg -n "static [^;]*\\(" tests benchmarks examples src -g '*.c' -g '*.h' | head -n 120`

### Baseline Findings

- Sprint 33 starts from the validated Sprint 32 clean-state contract, not from residual warning debt:
  - full-tree warnings: `0`
  - dormant-scaffold debt: `0`
  - active `ctest` registry: `53`
  - opt-in truthfulness policy already in force through `RUN_TEST_SLOW(...)`, `RUN_TEST_EXPERIMENTAL(...)`, `SPARSE_TEST_SLOW=1`, and `SPARSE_TEST_EXPERIMENTAL=1`
- Current branch head at Day 1 baseline capture: `ff3cfe6`
- There is no existing dead-code Makefile support yet:
  - no `deadcode`
  - no `deadcode-report`
  - no `deadcode-check`
- Existing static-analysis flow today is still `lint`-centric:
  - `tooling-build`
  - strict `src/*.c` compile with `-Werror`
  - `clang-tidy` over `src/*.c`
  - `cppcheck` over `src/` and `tests/`
- Before Day 1, there was no `compile_commands.json` in any existing build tree under `build/`.
- A dedicated Day 1 CMake configure path now exists:
  - `build/sprint33-day1-cmake/compile_commands.json`
  - total translation units recorded: `97`
- Local tool availability at Day 1:
  - `cppcheck`: present at `/usr/local/bin/cppcheck`
  - `clang-tidy`: present at `/usr/local/opt/llvm/bin/clang-tidy`
  - `xunused`: not installed / not found in `PATH`

### Scope Counts

Repository `.c` file counts at Day 1:

- `src`: `25`
- `tests`: `54`
- `benchmarks`: `14`
- `examples`: `12`

Day 1 `compile_commands.json` coverage by area:

- `src`: `25`
- `tests`: `53`
- `benchmarks`: `13`
- `examples`: `6`

Interpretation:

- the compilation database fully covers the library sources
- it covers the current active CTest registry exactly (`53` test translation units)
- it does **not** cover the full Makefile-only tooling surface yet

### Coverage Gaps Relevant To Sprint 33

Compared with the current Makefile source lists and directories:

- benchmark source present in the repo but absent from CMake `compile_commands.json`:
  - `bench_svd`
- example sources present in the repo but absent from CMake `compile_commands.json`:
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Day 1 implication:

- a naive `xunused build/compile_commands.json` pass would reason about a narrower bench/example surface than the Makefile currently builds
- Sprint 33 should document or resolve that coverage mismatch before treating `xunused` output as a complete first-pass dead-code signal

### Likely First-Pass Candidate Areas

These are **candidate audit surfaces**, not Day 1 dead-code findings:

1. `tests/`
   - largest low-risk internal area by file count (`54` `.c` files)
   - already governed by Sprint 32 truthfulness constraints
   - must preserve active and opt-in test coverage, including `tests/test_framework_optin.c`
2. `benchmarks/` and `examples/`
   - likely place for private one-off helpers, legacy fixture code, and stale experiment scaffolding
   - need careful handling because current `compile_commands.json` under-covers them relative to the Makefile
3. private helper layer in `src/`
   - internal-header surfaces currently include:
     - `src/sparse_analysis_internal.h`
     - `src/sparse_bicgstab_internal.h`
     - `src/sparse_chol_csc_internal.h`
     - `src/sparse_colamd_internal.h`
     - `src/sparse_eigs_internal.h`
     - `src/sparse_errno_internal.h`
     - `src/sparse_graph_internal.h`
     - `src/sparse_ldlt_csc_internal.h`
     - `src/sparse_matrix_internal.h`
     - `src/sparse_reorder_amd_qg_internal.h`
     - `src/sparse_reorder_nd_internal.h`
     - `src/sparse_svd_internal.h`
   - these are appropriate later cleanup candidates only after Sprint 33 establishes the tool/report policy
4. candidate public API or documented surface
   - must remain a separate audit queue
   - not appropriate for the first “definitely-unused internal code” removal pass

### Day 1 Interpretation

- Day 1 did **not** discover inherited cleanup debt from Sprint 32; it confirmed that Sprint 33 starts from a clean quality baseline and a live truthfulness policy.
- The first real Sprint 33 infrastructure gap is not warning debt but tooling completeness:
  - no dead-code targets exist yet
  - `xunused` is currently a local prerequisite gap
  - `compile_commands.json` must be generated deliberately rather than assumed
- The second key Day 1 finding is scope coverage mismatch:
  - CMake gives Sprint 33 a usable `compile_commands.json`
  - but that database does not yet cover every benchmark/example source the Makefile builds
- That means Day 2 and Day 3 should define the policy and evidence standard before any deletion work begins, and Day 4/Day 5 must treat compilation-database coverage as part of the infrastructure task, not as an afterthought.

### Day 1 Outputs

- `artifacts/day1-dead-code-baseline.md`
- `artifacts/day1-scope-counts.txt`
- `artifacts/day1-tooling-inventory.txt`
