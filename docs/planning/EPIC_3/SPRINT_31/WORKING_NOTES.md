# Sprint 31 Working Notes

## Day 1

**Objective:** Convert the Sprint 30 benchmark/example handoff into a precise Sprint 31 inventory by rerunning the authoritative Apple Clang CMake baseline for the current branch, inspecting the named Sprint 31 files, and recording the current benchmark CLI/help behavior before any edits begin.

### Commands Run

1. Read Sprint 31 scope and Sprint 30 handoff inputs:
   - `sed -n '43,75p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/HANDOFF.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_30/artifacts/day11-tooling-warning-triage.md`
2. Inspect Sprint 31 target files and benchmark build definitions:
   - `sed -n '1,260p' benchmarks/bench_main.c`
   - `sed -n '1,260p' benchmarks/bench_convergence.c`
   - `sed -n '1,260p' benchmarks/bench_reorder.c`
   - `sed -n '1,220p' benchmarks/bench_colamd.c`
   - `sed -n '1,220p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,220p' benchmarks/bench_ldlt_csc.c`
   - `sed -n '1,220p' examples/example_colamd.c`
   - `rg -n "bench_main|bench_reorder|bench_colamd|bench_chol_csc|bench_ldlt_csc|example_colamd" CMakeLists.txt benchmarks`
3. Inspect parser and help behavior around reorder-related CLI surfaces:
   - `rg -n "usage|reorder|strcmp\\(argv|--help|fprintf\\(stderr|SPARSE_REORDER_" benchmarks/bench_main.c benchmarks/bench_reorder.c benchmarks/bench_colamd.c benchmarks/bench_chol_csc.c benchmarks/bench_ldlt_csc.c examples/example_colamd.c`
   - `sed -n '260,520p' benchmarks/bench_main.c`
   - `sed -n '220,380p' benchmarks/bench_reorder.c`
4. Run a clean serialized Apple Clang CMake baseline capture for the current branch:
   - `rm -rf build/sprint31-day1-cmake`
   - `cmake -S . -B build/sprint31-day1-cmake`
   - `cmake --build build/sprint31-day1-cmake --parallel 1`
5. Derive Sprint 31 benchmark/example warning counts from the new CMake stderr artifact:
   - counts by file
   - counts by warning class
   - counts by file and warning class

All command stdout/stderr and derived counts were written under `docs/planning/EPIC_3/SPRINT_31/artifacts/`.

### Raw Counts

- CMake configure warnings: `0`
- CMake full-build warnings: `112`
- benchmark warnings: `13`
- example warnings: `1`
- benchmark/example warnings combined: `14`
- warning-bearing benchmark/example files: `6`

### Sprint 31 File Inventory

- `benchmarks/bench_main.c`
  - `5` warnings
  - stale usage text still advertises only `rcm|amd|none`
  - parser still rejects `colamd` and `nd`
  - `reorder_name()` is not exhaustive for current supported reorder values
  - still uses positional options-struct initialization
- `benchmarks/bench_convergence.c`
  - `2` warnings
  - `snprintf` visibility issue under `_POSIX_C_SOURCE 199309L`
  - one residual `NAN` / `double` promotion warning
- `benchmarks/bench_colamd.c`
  - `3` warnings
  - QR options still use positional initialization
- `benchmarks/bench_chol_csc.c`
  - `1` warning
  - Cholesky options still use positional initialization
- `benchmarks/bench_ldlt_csc.c`
  - `2` warnings
  - LDLT options still use positional initialization
- `examples/example_colamd.c`
  - `1` warning
  - public COLAMD example still teaches positional QR options initialization

### Current Benchmark CLI/Help State

- `bench_main` remains the main Sprint 31 correctness target:
  - top-of-file usage text advertises `--reorder rcm|amd|none`
  - parser accepts only `none`, `rcm`, and `amd`
  - unknown-reorder error text still says to use only `none`, `rcm`, or `amd`
  - `reorder_name()` handles `NONE`, `RCM`, and `AMD`, but not `COLAMD` or `ND`
- `bench_reorder` already exposes the broader reorder surface Sprint 31 needs to align to:
  - benchmark set includes `NONE`, `RCM`, `AMD`, `COLAMD`, and `ND`
  - CLI supports `--nd-threshold`, `--skip-factor`, `--reorder-via-analyze`, and `--only`
  - output schema is a stable CSV header rather than ad hoc help text
- the specialized benchmark/example programs are narrower tools, but they reinforce the same public API expectations:
  - `bench_colamd` and `example_colamd` both assume `COLAMD` is a normal public reorder choice
  - `bench_chol_csc` and `bench_ldlt_csc` are backend-comparison tools rather than general reorder harnesses

### Day 1 Interpretation

- The Sprint 31 warning inventory on the current branch matches the Sprint 30 handoff exactly: the queue is still the same six files and the same `14` benchmark/example warnings.
- `bench_main.c` is still the highest-signal Sprint 31 file because it combines user-visible CLI/help drift, one `-Wswitch` coverage warning, one portability warning, and initializer drift in the project’s main benchmark harness.
- The benchmark/example portability debt is still tightly scoped to `bench_main.c` and `bench_convergence.c`.
- The designated-initializer cleanup is still mechanical and well bounded: it covers the same four benchmark/example files identified by Sprint 30.
- No new benchmark/example warning sites appeared between Sprint 30 close and Sprint 31 Day 1.

### Day 1 Outputs

- `artifacts/day1-benchmark-tooling-baseline.md`
- `artifacts/day1-tooling-warning-counts-by-file.txt`
- `artifacts/day1-tooling-warning-counts-by-class.txt`
- `artifacts/day1-tooling-warning-counts-by-file-and-class.txt`
- raw configure/build stdout/stderr logs for the clean Day 1 CMake rerun

## Day 2

**Objective:** Audit `bench_main` against the actual reorder API and the project’s benchmark reference tools, then write the canonical Sprint 31 CLI contract before any code edits begin.

### Commands Run

1. Re-read current benchmark-facing docs and `bench_main` CLI surfaces:
   - `sed -n '1,220p' benchmarks/README.md`
   - `sed -n '1,120p' benchmarks/bench_main.c`
   - `sed -n '620,690p' benchmarks/bench_main.c`
   - `sed -n '1,120p' benchmarks/bench_reorder.c`
2. Re-read public reorder contracts in the headers:
   - `rg -n "SPARSE_REORDER_(NONE|RCM|AMD|COLAMD|ND)|reorder" include/sparse_*.h include/*.h`
   - `sed -n '1,140p' include/sparse_lu.h`
   - `sed -n '1,140p' include/sparse_cholesky.h`
   - `sed -n '1,120p' include/sparse_qr.h`
3. Verify the actual implementation-side reorder acceptance for the factorization paths `bench_main` calls:
   - `rg -n "SPARSE_REORDER_COLAMD|switch \\(.*reorder|opts->reorder|reorder == SPARSE_REORDER_COLAMD|sparse_reorder_colamd" src include`
   - `sed -n '248,320p' src/sparse_lu.c`
   - `sed -n '276,330p' src/sparse_cholesky.c`
   - `sed -n '1040,1095p' src/sparse_ldlt.c`
   - `sed -n '160,185p' src/sparse_analysis.c`

### Day 2 Audit Result

`bench_main` is out of sync with the current LU/Cholesky solver contracts, but not in the exact way Sprint 30’s early handoff phrasing implied.

Observed benchmark-tool state:

- `bench_main` usage text advertises `none|rcm|amd`
- `bench_main` parser accepts only `none`, `rcm`, and `amd`
- `bench_main` `reorder_name()` prints only `none`, `rcm`, and `amd`
- `bench_reorder` exposes `NONE`, `RCM`, `AMD`, `COLAMD`, and `ND`

Observed public/API state:

- `sparse_lu_opts_t` documents and is implemented for `NONE`, `RCM`, `AMD`, and `ND`
- `sparse_cholesky_opts_t` documents and is implemented for `NONE`, `RCM`, `AMD`, and `ND`
- `sparse_ldlt_opts_t` documents and is implemented for `NONE`, `RCM`, `AMD`, and `ND`
- `sparse_qr_opts_t` documents `COLAMD` as recommended and also accepts `AMD`, `RCM`, and `ND`
- `sparse_analysis()` accepts `COLAMD`, but only through its own analysis-time symmetric-permutation path

Important Day 2 clarification:

- `COLAMD` is **not** currently accepted by the LU / Cholesky / LDLT factorization option paths that `bench_main` uses
- those implementations reject `SPARSE_REORDER_COLAMD` with `SPARSE_ERR_BADARG`
- `ND` is the real missing reorder mode for `bench_main`’s current solver modes

### Canonical `bench_main` Contract Chosen For Sprint 31

For the existing `bench_main` LU and `--cholesky` solver modes:

- accepted reorder spellings should be `none`, `rcm`, `amd`, and `nd`
- printed reorder labels should be `none`, `rcm`, `amd`, and `nd`
- help text should advertise exactly those four values
- unknown-reorder diagnostics should name exactly those four values

`COLAMD` decision:

- do **not** add `colamd` to `bench_main`’s current LU/Cholesky reorder parser in Sprint 31
- reason: the underlying solver option paths used by `bench_main` do not actually support it
- if the project later wants a main benchmark harness mode that exercises `COLAMD`, that should be a QR-oriented or analyze-oriented path rather than a misleading parser-only addition

### Day 2 Interpretation

- Sprint 31 Day 3 should fix `bench_main` to match the real solver contract, not the broader enum surface.
- The most important user-visible drift is still real: `bench_main` is stale because it omits `nd`.
- The apparent `colamd` gap is a deeper contract mismatch between broad enum availability and the narrower LU/Cholesky reorder implementations.
- This audit removed the main Day 3 ambiguity: `bench_main` should align to `none|rcm|amd|nd`, not claim unsupported `colamd` behavior.

### Day 2 Outputs

- `artifacts/day2-bench-main-cli-audit.md`

## Day 3

**Objective:** Implement the first `bench_main` contract-alignment batch by adding the real missing LU/Cholesky reorder mode (`nd`), updating usage and error text to match that contract, and re-running a clean build plus CLI smoke checks to capture the before/after behavior.

### Files Edited

- `benchmarks/bench_main.c`

### Changes Made

1. Updated the file-header usage text from `rcm|amd|none` to `none|rcm|amd|nd`.
2. Extended `reorder_name()` to print:
   - `none`
   - `rcm`
   - `amd`
   - `colamd`
   - `nd`
3. Extended the `--reorder` parser to accept `nd`.
4. Updated the unknown-reorder diagnostic to name the canonical Sprint 31 solver-harness contract:
   - `none`
   - `rcm`
   - `amd`
   - `nd`

Important implementation note:

- `reorder_name()` now handles `COLAMD` even though the parser still rejects `colamd`.
- This is intentional.
- The helper should be exhaustive for the enum to avoid stale-label drift and compile warnings.
- The parser remains aligned to the narrower LU/Cholesky solver contract established in Day 2.

### Validation Run

1. Formatting:
   - `make format`
2. Clean comparable rebuild:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`
3. Behavioral smoke checks:
   - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder nd`
   - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder colamd`

### Validation Results

Post-Day-3 clean CMake rebuild:

- full-tree warnings: `111` (down from `112`)
- benchmark/example warnings: `13` (down from `14`)
- `bench_main.c` warnings: `4` (down from `5`)

Post-Day-3 benchmark/example warning classes:

- `-Wmissing-field-initializers`: `10`
- `-Wimplicit-function-declaration`: `2`
- `-Wdouble-promotion`: `1`
- `-Wswitch`: `0`

Remaining `bench_main.c` warnings:

- `3` `-Wmissing-field-initializers`
- `1` `-Wimplicit-function-declaration`

Behavioral smoke-check results:

- `--reorder nd` now succeeds and prints `Reorder: nd`
- `--reorder colamd` still exits with an error, but the message now reflects the canonical contract:
  - `Error: unknown reorder mode 'colamd' (use 'none', 'rcm', 'amd', or 'nd')`

### Day 3 Interpretation

- The highest-signal `bench_main` contract drift is now partly closed:
  - `nd` is supported in help text
  - `nd` is supported in the parser
  - `nd` is supported in printed labels
- The stale `-Wswitch` warning is gone.
- `bench_main` now matches the real LU/Cholesky reorder contract identified in Day 2.
- The remaining `bench_main` warning debt is mechanical:
  - positional options initialization
  - `snprintf` visibility under `_POSIX_C_SOURCE 199309L`

### Day 3 Outputs

- `artifacts/day3-bench-main-sync.md`
- `artifacts/day3-tooling-warning-counts-by-file.txt`
- `artifacts/day3-tooling-warning-counts-by-class.txt`
- `artifacts/day3-tooling-warning-counts-by-file-and-class.txt`
- `artifacts/day3-cmake-build.stdout.txt`
- `artifacts/day3-cmake-build.stderr.txt`
- `artifacts/day3-bench-main-nd.stdout.txt`
- `artifacts/day3-bench-main-nd.stderr.txt`
- `artifacts/day3-bench-main-colamd.stdout.txt`
- `artifacts/day3-bench-main-colamd.stderr.txt`

## Day 4

**Objective:** Align reorder-mode presentation across `bench_main`, `bench_reorder`, and the specialized COLAMD benchmark/example programs so the tools use the same user-facing spelling conventions after Day 3’s contract fix.

### Files Edited

- `benchmarks/bench_reorder.c`
- `benchmarks/bench_colamd.c`
- `examples/example_colamd.c`

### Commands Run

1. Inspect reorder-related help/output surfaces:
   - `rg -n "reorder|COLAMD|AMD|RCM|ND|none|natural|usage|CSV|help" benchmarks/bench_main.c benchmarks/bench_reorder.c benchmarks/bench_colamd.c examples/example_colamd.c benchmarks/README.md`
   - `sed -n '1,220p' benchmarks/bench_colamd.c`
   - `sed -n '1,220p' examples/example_colamd.c`
   - `sed -n '1,120p' benchmarks/bench_reorder.c`
2. Capture live pre-edit behavior:
   - `./build/sprint31-day1-cmake/bench_reorder --only nos4 --skip-factor`
   - `./build/sprint31-day1-cmake/bench_colamd`
3. Apply consistency edits.
4. Validation:
   - `make format`
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --target bench_reorder bench_colamd example_colamd`
   - `./build/sprint31-day1-cmake/bench_reorder --only nos4 --skip-factor`
   - `./build/sprint31-day1-cmake/bench_colamd`
   - `./build/sprint31-day1-cmake/example_colamd`

### Changes Made

1. `bench_reorder.c`
   - changed reorder labels in the CSV output from uppercase enum-style names to lowercase CLI-style spellings:
     - `NONE -> none`
     - `RCM -> rcm`
     - `AMD -> amd`
     - `COLAMD -> colamd`
     - `ND -> nd`
   - updated the file-header description to use the same lowercase spelling set
2. `bench_colamd.c`
   - changed the benchmark title from `COLAMD vs AMD vs Natural` to `colamd vs amd vs none`
   - changed the header columns from `nnz(R)nat` / `nnz(R)col` to `nnz(R)none` / `nnz(R)colamd`
   - changed percentage text from `vs natural` to `vs none`
3. `example_colamd.c`
   - changed the explanatory comparison text from `Natural ordering` / `COLAMD ordering` to:
     - `none (natural order)`
     - `colamd`

### Validation Results

Observed post-edit output:

- `bench_main`
  - already used lowercase CLI-style spellings after Day 3
- `bench_reorder`
  - now prints CSV rows like:
    - `nos4,100,none,...`
    - `nos4,100,rcm,...`
    - `nos4,100,amd,...`
    - `nos4,100,colamd,...`
    - `nos4,100,nd,...`
- `bench_colamd`
  - now titles the comparison as `colamd vs amd vs none`
  - now headers the columns as `nnz(R)none`, `nnz(R)amd`, `nnz(R)colamd`
  - now prints percentage text as `vs none`
- `example_colamd`
  - now prints `none (natural order)` and `colamd` in the LU fill-in comparison

Targeted rebuild result:

- `bench_reorder`, `bench_colamd`, and `example_colamd` rebuilt successfully
- no new warning class was introduced by the Day 4 presentation cleanup

### Day 4 Interpretation

- Day 4 closed the remaining user-facing naming drift after Day 3.
- The main benchmark tools now present reorder modes using the same lowercase spellings users type on the CLI.
- The specialized COLAMD tools no longer mix `none` row labels with `natural`/`nat` headers.
- This was a presentation-level cleanup only; no reorder behavior or supported-mode surface changed.

### Day 4 Outputs

- `artifacts/day4-benchmark-behavior-consistency.md`
- `artifacts/day4-build.stdout.txt`
- `artifacts/day4-build.stderr.txt`
- `artifacts/day4-bench-reorder-nos4.stdout.txt`
- `artifacts/day4-bench-reorder-nos4.stderr.txt`
- `artifacts/day4-bench-colamd.stdout.txt`
- `artifacts/day4-bench-colamd.stderr.txt`
- `artifacts/day4-example-colamd.stdout.txt`
- `artifacts/day4-example-colamd.stderr.txt`

## Day 5

**Objective:** Audit the benchmark/example portability warnings identified in Sprint 30, confirm whether they are caused by missing headers or by feature-test macro selection, and choose the standard portability pattern for the Day 6 implementation batch.

### Commands Run

1. Scan benchmark/example entry points for `_POSIX_C_SOURCE`, `clock_gettime`, `snprintf`, and `dirent` usage:
   - `rg -n "_POSIX_C_SOURCE|snprintf\\(|clock_gettime\\(|timespec|dirent.h" benchmarks examples`
2. Inspect the two priority warning files:
   - `sed -n '1,120p' benchmarks/bench_main.c`
   - `sed -n '330,390p' benchmarks/bench_main.c`
   - `sed -n '1,120p' benchmarks/bench_convergence.c`
   - `sed -n '360,400p' benchmarks/bench_convergence.c`
3. Inspect nearby benchmark entry points that share the same timer / feature-macro pattern:
   - `sed -n '1,80p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,80p' benchmarks/bench_ldlt_csc.c`
   - `sed -n '1,80p' benchmarks/bench_refactor_csc.c`

### Day 5 Audit Result

The benchmark portability issue is a feature-test-macro problem, not a missing-header problem.

Confirmed in both warning-bearing priority files:

- `benchmarks/bench_main.c`
  - defines `_POSIX_C_SOURCE 199309L`
  - includes `<stdio.h>`
  - uses `clock_gettime`
  - later uses `snprintf`
- `benchmarks/bench_convergence.c`
  - defines `_POSIX_C_SOURCE 199309L`
  - includes `<stdio.h>`
  - uses `clock_gettime`
  - later uses `snprintf`

Interpretation:

- the current warnings are not caused by forgetting to include `<stdio.h>`
- the declaration is hidden because the file asks for a very low POSIX surface (`199309L`) and then later uses a function that is gated beyond that surface on the current Apple Clang / libc path
- both files need a portability fix that preserves `clock_gettime` availability while also exposing the standard C/POSIX declarations they already use

### Nearby Benchmark Pattern Audit

Files with the same `_POSIX_C_SOURCE 199309L` + `clock_gettime` timer pattern:

- `benchmarks/bench_scaling.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_svd.c`

Important distinction:

- only `bench_main.c` and `bench_convergence.c` currently combine that pattern with `snprintf` and therefore emit the current portability warning
- the other benchmark entry points do not currently fail on the same warning class, but they already use the same old feature-test baseline and would become split-style outliers if Sprint 31 raises the macro only in two files

Examples audit:

- no Sprint 31 example file currently shows the same `_POSIX_C_SOURCE` / `snprintf` portability pattern
- `examples/example_colamd.c` is part of the initializer-cleanup queue, not the Day 5 portability queue

### Chosen Portability Pattern For Day 6

Chosen standard:

- use `_POSIX_C_SOURCE 200809L` in benchmark entry points that rely on POSIX timer interfaces and standard library functions like `snprintf`

Why this pattern wins:

- preserves `clock_gettime` visibility
- exposes the declarations the benchmark entry points already use
- removes the current Apple Clang implicit-declaration warning without adding local ad hoc declarations
- is a cleaner project-wide benchmark convention than keeping a 199309L baseline in some files and a newer surface in others

Rejected alternatives:

- removing `_POSIX_C_SOURCE` entirely
  - too risky because the benchmark timer helpers rely on `clock_gettime`
- adding manual `snprintf` declarations
  - wrong layer; treats the symptom rather than the feature-surface mismatch
- fixing only the two warninging files and leaving the rest on `199309L`
  - technically workable, but leaves the benchmark directory with split portability conventions for no real benefit

### Day 5 Implementation Scope For Day 6

Priority Day 6 files:

- `benchmarks/bench_main.c`
- `benchmarks/bench_convergence.c`

Adjacent benchmark entry points that share the same timer/macro pattern and should be considered for the same fix style while the portability batch is open:

- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_scaling.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_svd.c`

Day 5 conclusion:

- the root cause is confirmed
- the portability fix style is chosen
- Day 6 can proceed as a narrow source-edit batch instead of another investigation pass

### Day 5 Outputs

- `artifacts/day5-portability-audit.md`

## Day 6

**Objective:** Apply the Day 5 portability fix pattern by raising the benchmark entry-point feature-test baseline to `_POSIX_C_SOURCE 200809L`, then rerun the clean CMake benchmark baseline to verify that the Apple Clang implicit-declaration warnings are removed without introducing new warning classes.

### Files Edited

- `benchmarks/bench_scaling.c`
- `benchmarks/bench_chol_csc.c`
- `benchmarks/bench_main.c`
- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_convergence.c`
- `benchmarks/bench_eigs.c`
- `benchmarks/bench_refactor_csc.c`
- `benchmarks/bench_refactor.c`
- `benchmarks/bench_bicgstab.c`
- `benchmarks/bench_svd.c`

### Commands Run

1. Reconfirm the benchmark entry points that still use the old POSIX timer baseline:
   - `rg -n "#define _POSIX_C_SOURCE 199309L" benchmarks`
2. Raise the benchmark entry-point feature-test macro to the Day 5 chosen standard:
   - `_POSIX_C_SOURCE 199309L -> 200809L`
3. Format the tree:
   - `make format`
4. Run a clean comparable rebuild against the established Sprint 31 baseline build directory:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`
5. Re-derive benchmark/example warning counts from the captured CMake stderr log:
   - counts by file
   - counts by warning class
   - counts by file and warning class
6. Reconfirm the new benchmark-wide macro convention:
   - `rg -n "#define _POSIX_C_SOURCE 200809L" benchmarks`

### Changes Made

1. Raised the benchmark entry-point POSIX feature-test macro from `199309L` to `200809L` in all ten files that shared the same `clock_gettime`-driven timer pattern.
2. Kept the implementation narrow:
   - no benchmark logic changed
   - no CLI/help contract changed
   - no initializer cleanup was bundled into this batch
3. Standardized the benchmark directory on one portability convention instead of fixing only the two currently warninging files.

### Validation Results

Post-Day-6 clean CMake rebuild:

- full-tree warnings: `109` (down from `111` on Day 3, down from `112` on Day 1)
- benchmark/example warnings: `11` (down from `13` on Day 3, down from `14` on Day 1)
- `bench_main.c` warnings: `3` (down from `4`)
- `bench_convergence.c` warnings: `1` (down from `2`)

Post-Day-6 benchmark/example warning classes:

- `-Wmissing-field-initializers`: `10`
- `-Wdouble-promotion`: `1`
- `-Wimplicit-function-declaration`: `0`

Remaining benchmark/example warning-bearing files:

- `benchmarks/bench_chol_csc.c`: `1` `-Wmissing-field-initializers`
- `benchmarks/bench_colamd.c`: `3` `-Wmissing-field-initializers`
- `benchmarks/bench_convergence.c`: `1` `-Wdouble-promotion`
- `benchmarks/bench_ldlt_csc.c`: `2` `-Wmissing-field-initializers`
- `benchmarks/bench_main.c`: `3` `-Wmissing-field-initializers`
- `examples/example_colamd.c`: `1` `-Wmissing-field-initializers`

Key portability result:

- the Apple Clang `snprintf` implicit-declaration warnings are gone from both `bench_main.c` and `bench_convergence.c`
- no new warning class was introduced by raising the POSIX feature-test surface

### Day 6 Interpretation

- Day 6 closed the active Sprint 31 portability warning class exactly as Day 5 predicted.
- The benchmark directory now uses one clearer `_POSIX_C_SOURCE 200809L` convention for timer-based entry points instead of a split `199309L` / `200809L` mix.
- The remaining Sprint 31 benchmark/example warning queue is now purely mechanical:
  - designated-initializer cleanup in five files
  - one residual `-Wdouble-promotion` site in `bench_convergence.c`

### Day 6 Outputs

- `artifacts/day6-portability-fixes.md`
- `artifacts/day6-cmake-build.stdout.txt`
- `artifacts/day6-cmake-build.stderr.txt`
- `artifacts/day6-tooling-warning-counts-by-file.txt`
- `artifacts/day6-tooling-warning-counts-by-class.txt`
- `artifacts/day6-tooling-warning-counts-by-file-and-class.txt`

## Day 7

**Objective:** Replace the first benchmark batch of brittle positional options initialization with designated initializers in `bench_colamd.c` and `bench_chol_csc.c`, then verify the warning reduction and preserve the existing benchmark behavior.

### Files Edited

- `benchmarks/bench_colamd.c`
- `benchmarks/bench_chol_csc.c`

### Commands Run

1. Reconfirm the warning-bearing positional initializers in the Day 7 scope:
   - `rg -n "= \\{|\\{[^;]*SPARSE_.*_DEFAULT|opts = \\{" benchmarks/bench_colamd.c benchmarks/bench_chol_csc.c`
2. Re-read the public option-struct field definitions:
   - `sed -n '1,220p' include/sparse_qr.h`
   - `sed -n '1,220p' include/sparse_cholesky.h`
3. Replace the positional option initializers with designated initializers that name only the intended non-default fields.
4. Format the tree:
   - `make format`
5. Run targeted rebuild / behavior checks for the touched benchmarks:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --target bench_colamd bench_chol_csc`
   - `./build/sprint31-day1-cmake/bench_colamd`
   - `./build/sprint31-day1-cmake/bench_chol_csc --small-corpus --repeat 1`
6. Run a clean comparable rebuild and re-derive the benchmark/example warning counts:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

### Changes Made

1. `bench_colamd.c`
   - converted:
     - `opts_none`
     - `opts_amd`
     - `opts_colamd`
   - from positional `sparse_qr_opts_t` initialization to designated initialization using only:
     - `.reorder = ...`
2. `bench_chol_csc.c`
   - converted the forced linked-list `sparse_cholesky_opts_t` initializer from positional form to designated initialization using:
     - `.reorder = SPARSE_REORDER_AMD`
     - `.backend = SPARSE_CHOL_BACKEND_LINKED_LIST`
3. Chosen Day 7 public-facing initialization style:
   - set only the non-default fields the benchmark actually intends to override
   - leave trailing callback / context / telemetry fields implicit at zero / `NULL`
   - avoid copying the full current struct layout into benchmark code

### Validation Results

Targeted rebuild / behavior checks:

- `bench_colamd` rebuilt with `0` warnings in `artifacts/day7-build.stderr.txt`
- `bench_chol_csc` rebuilt with `0` warnings in `artifacts/day7-build.stderr.txt`
- `bench_colamd` still runs and prints the same comparison table shape and reorder labels
- `bench_chol_csc --small-corpus --repeat 1` still runs and prints the expected CSV output for the synthetic SPD corpus

Post-Day-7 clean CMake rebuild:

- full-tree warnings: `105` (down from `109` on Day 6)
- benchmark/example warnings: `7` (down from `11` on Day 6)
- `-Wmissing-field-initializers`: `6` (down from `10`)
- `-Wdouble-promotion`: `1` (unchanged)

Per-file Day-7 warning deltas:

- `benchmarks/bench_colamd.c`: `3 -> 0`
- `benchmarks/bench_chol_csc.c`: `1 -> 0`

Remaining benchmark/example warning-bearing files after Day 7:

- `benchmarks/bench_convergence.c`: `1` `-Wdouble-promotion`
- `benchmarks/bench_ldlt_csc.c`: `2` `-Wmissing-field-initializers`
- `benchmarks/bench_main.c`: `3` `-Wmissing-field-initializers`
- `examples/example_colamd.c`: `1` `-Wmissing-field-initializers`

### Day 7 Interpretation

- Day 7 removed the entire first initializer-cleanup batch exactly as intended.
- `bench_colamd.c` and `bench_chol_csc.c` are no longer coupled to the current trailing field layout of their public option structs.
- The designated-initializer pattern is now established for the remaining Sprint 31 initializer files:
  - name only the intended override fields
  - let trailing callback / context fields default implicitly

### Day 7 Outputs

- `artifacts/day7-designated-initializers-batch1.md`
- `artifacts/day7-build.stdout.txt`
- `artifacts/day7-build.stderr.txt`
- `artifacts/day7-bench-colamd.stdout.txt`
- `artifacts/day7-bench-colamd.stderr.txt`
- `artifacts/day7-bench-chol-csc.stdout.txt`
- `artifacts/day7-bench-chol-csc.stderr.txt`
- `artifacts/day7-cmake-build.stdout.txt`
- `artifacts/day7-cmake-build.stderr.txt`
- `artifacts/day7-tooling-warning-counts-by-file.txt`
- `artifacts/day7-tooling-warning-counts-by-class.txt`
- `artifacts/day7-tooling-warning-counts-by-file-and-class.txt`

## Day 8

**Objective:** Finish the Sprint 31 public-facing initializer cleanup by converting the remaining benchmark/example option-struct sites to designated initialization, then sweep the full benchmark/example queue to confirm the initializer warning class is fully closed.

### Files Edited

- `benchmarks/bench_ldlt_csc.c`
- `benchmarks/bench_main.c`
- `examples/example_colamd.c`

### Commands Run

1. Reconfirm the remaining live positional initializer sites in the Sprint 31 queue:
   - `rg -n "= \\{|\\{[^;]*SPARSE_.*_DEFAULT|opts = \\{" benchmarks/bench_ldlt_csc.c examples/example_colamd.c benchmarks/bench_colamd.c benchmarks/bench_chol_csc.c`
   - `rg -n "opts = \\{|_opts_t .*\\{" benchmarks/bench_main.c`
2. Re-read the public option-struct field definitions:
   - `sed -n '1,220p' include/sparse_ldlt.h`
   - `sed -n '1,140p' include/sparse_qr.h`
   - `sed -n '1,180p' include/sparse_lu.h`
3. Convert the remaining benchmark/example positional initializers to designated form.
4. Format the tree:
   - `make format`
5. Run targeted rebuild / behavior checks for the touched tools:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --target bench_ldlt_csc example_colamd bench_main`
   - `./build/sprint31-day1-cmake/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/sprint31-day1-cmake/example_colamd`
   - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder nd`
6. Run a clean comparable rebuild and re-derive the benchmark/example warning counts:
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

### Changes Made

1. `bench_ldlt_csc.c`
   - converted the linked-list `sparse_ldlt_opts_t` initializer to:
     - `.reorder = SPARSE_REORDER_AMD`
     - `.backend = SPARSE_LDLT_BACKEND_LINKED_LIST`
   - converted the dispatch-mode `sparse_ldlt_opts_t` initializer to:
     - `.reorder = SPARSE_REORDER_AMD`
     - `.backend = backend`
     - `.used_csc_path = &used_csc`
2. `example_colamd.c`
   - converted the QR options initializer to:
     - `.reorder = SPARSE_REORDER_COLAMD`
3. `bench_main.c`
   - folded into Day 8 after the first clean rebuild showed it still carried the last three benchmark/example initializer warnings
   - converted both `sparse_lu_opts_t` initializers to explicit `.pivot`, `.reorder`, `.tol`
   - converted the `sparse_cholesky_opts_t` initializer to explicit `.reorder`
4. Consolidated Sprint 31 initializer style after the Day 7 and Day 8 sweeps:
   - name only the non-default fields a benchmark/example actually intends to override
   - let trailing callback / context / telemetry fields default implicitly to zero / `NULL`
   - avoid copying full public struct layouts positionally into benchmark/example code

### Validation Results

Targeted rebuild / behavior checks:

- `bench_ldlt_csc`, `example_colamd`, and `bench_main` rebuilt with `0` warnings in `artifacts/day8-build.stderr.txt`
- `bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1` still ran and printed the expected CSV row
- `example_colamd` still ran and printed the expected COLAMD ordering / LU fill-in / QR residual output
- `bench_main --size 8 --repeat 1 --reorder nd` still ran and printed `Reorder: nd`

Post-Day-8 clean CMake rebuild:

- full-tree warnings: `99` (down from `105` on Day 7)
- benchmark/example warnings: `1` (down from `7` on Day 7)
- `-Wmissing-field-initializers`: `0` (down from `6`)
- `-Wdouble-promotion`: `1` (unchanged)

Per-file Day-8 warning deltas:

- `benchmarks/bench_ldlt_csc.c`: `2 -> 0`
- `examples/example_colamd.c`: `1 -> 0`
- `benchmarks/bench_main.c`: `3 -> 0`

Remaining benchmark/example warning-bearing file after Day 8:

- `benchmarks/bench_convergence.c`: `1` `-Wdouble-promotion`

### Day 8 Interpretation

- Day 8 closed the Sprint 31 benchmark/example initializer warning class completely.
- All Sprint 31 public-facing benchmark/example option-struct sites now use designated initialization.
- The benchmark/example warning queue is now down to one non-initializer mechanical site in `bench_convergence.c`.

### Day 8 Outputs

- `artifacts/day8-designated-initializers-batch2.md`
- `artifacts/day8-build.stdout.txt`
- `artifacts/day8-build.stderr.txt`
- `artifacts/day8-bench-ldlt-csc.stdout.txt`
- `artifacts/day8-bench-ldlt-csc.stderr.txt`
- `artifacts/day8-example-colamd.stdout.txt`
- `artifacts/day8-example-colamd.stderr.txt`
- `artifacts/day8-bench-main.stdout.txt`
- `artifacts/day8-bench-main.stderr.txt`
- `artifacts/day8-cmake-build.stdout.txt`
- `artifacts/day8-cmake-build.stderr.txt`
- `artifacts/day8-tooling-warning-counts-by-file.txt`
- `artifacts/day8-tooling-warning-counts-by-class.txt`
- `artifacts/day8-tooling-warning-counts-by-file-and-class.txt`

## Day 9

**Objective:** Re-run the Sprint 31 benchmark/example entry points after the main cleanup work, confirm that their accepted flags and emitted labels match the documented contract, and close any small remaining user-facing drift that survived the earlier implementation days.

### Files Edited

- `examples/example_colamd.c`

### Commands Run

1. Re-read the current parser / label surfaces:
   - `sed -n '1,220p' benchmarks/bench_reorder.c`
   - `sed -n '1,180p' benchmarks/bench_colamd.c`
   - `sed -n '1,180p' benchmarks/bench_main.c`
   - `sed -n '1,220p' benchmarks/bench_chol_csc.c`
   - `sed -n '1,220p' benchmarks/bench_ldlt_csc.c`
   - `sed -n '1,140p' examples/example_colamd.c`
2. Re-run the touched benchmark/example entry points and capture their current behavior:
   - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder nd`
   - `./build/sprint31-day1-cmake/bench_main --size 8 --repeat 1 --reorder colamd`
   - `./build/sprint31-day1-cmake/bench_reorder --only nos4 --skip-factor`
   - `./build/sprint31-day1-cmake/bench_colamd`
   - `./build/sprint31-day1-cmake/bench_chol_csc --small-corpus --repeat 1`
   - `./build/sprint31-day1-cmake/bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1`
   - `./build/sprint31-day1-cmake/example_colamd`
3. Read the benchmark overview doc for any obvious contract mismatch:
   - `sed -n '1,220p' benchmarks/README.md`
4. Close the one remaining output-label glitch found during the live audit:
   - `example_colamd.c` equal-fill case now prints `0% change` instead of `-0% increase`
5. Validation after the small output fix:
   - `make format`
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --target example_colamd bench_main bench_reorder bench_colamd bench_chol_csc bench_ldlt_csc`
   - rerun `example_colamd`
   - `cmake --build build/sprint31-day1-cmake --parallel 1 --clean-first`

### Changes Made

1. `example_colamd.c`
   - normalized the equal-fill case in the LU comparison output:
     - from `(-0% increase)`
     - to `(0% change)`
   - added `<math.h>` so the comparison can treat near-zero percentages explicitly

No other Day 9 code change was needed.

### Post-Cleanup Behavior Matrix

| tool | accepted / exposed reorder modes | observed post-Day-9 behavior |
|---|---|---|
| `bench_main` | `none`, `rcm`, `amd`, `nd` | accepts `nd`; prints lowercase `Reorder: nd`; still rejects `colamd` with an explicit error naming `none`, `rcm`, `amd`, `nd` |
| `bench_reorder` | `none`, `rcm`, `amd`, `colamd`, `nd` | CSV rows use lowercase reorder names across the full five-mode comparison set |
| `bench_colamd` | `none`, `amd`, `colamd` | title, headers, row labels, and percentage text all use lowercase `none` / `amd` / `colamd` wording |
| `bench_chol_csc` | fixed internal AMD benchmark baseline | output remains CSV backend-comparison data; no broader reorder CLI is advertised |
| `bench_ldlt_csc` | fixed internal AMD benchmark baseline | output remains CSV backend-comparison data; no broader reorder CLI is advertised |
| `example_colamd` | `none (natural order)`, `colamd` | public example now uses the same lowercase spelling convention and prints `0% change` when the fill counts are equal |

### Validation Results

Live behavior checks confirmed:

- `bench_main --reorder nd` succeeds and prints `Reorder: nd`
- `bench_main --reorder colamd` still exits nonzero with:
  - `Error: unknown reorder mode 'colamd' (use 'none', 'rcm', 'amd', or 'nd')`
- `bench_reorder --only nos4 --skip-factor` prints lowercase CSV rows for:
  - `none`
  - `rcm`
  - `amd`
  - `colamd`
  - `nd`
- `bench_colamd` prints:
  - `=== QR Fill-In Comparison: colamd vs amd vs none ===`
  - `nnz(R)none`, `nnz(R)amd`, `nnz(R)colamd`
- `bench_chol_csc --small-corpus --repeat 1` still prints the expected backend-comparison CSV rows
- `bench_ldlt_csc tests/data/suitesparse/nos4.mtx --repeat 1` still prints the expected backend-comparison CSV row
- `example_colamd` now prints:
  - `none (natural order)`
  - `colamd`
  - `0% change` in the equal-fill case

Post-Day-9 clean CMake rebuild:

- full-tree warnings: `99` (unchanged from Day 8)
- benchmark/example warnings: `1` (unchanged from Day 8)
- remaining benchmark/example warning:
  - `benchmarks/bench_convergence.c`: `1` `-Wdouble-promotion`

### Residual / Deferred Notes

- `bench_main` remains intentionally narrower than `bench_reorder`:
  - it does **not** accept `colamd`
  - this is not leftover drift
  - it reflects the actual LU / Cholesky factorization contract established in Day 2
- `bench_chol_csc` and `bench_ldlt_csc` remain backend-comparison tools, not general reorder harnesses:
  - their fixed AMD setup is intentional for apples-to-apples backend timing
  - broad reorder-coverage CLI work would be a separate scope expansion, not a Day 9 fix

### Day 9 Interpretation

- The main benchmark tools now behave consistently with the Sprint 31 contract.
- The only live user-facing drift found in the audit was the `example_colamd` equal-fill message, and that is now closed.
- The remaining Sprint 31 benchmark/example queue is no longer behavior drift; it is one isolated numeric-promotion warning in `bench_convergence.c`.

### Day 9 Outputs

- `artifacts/day9-benchmark-behavior-audit.md`
- `artifacts/day9-build.stdout.txt`
- `artifacts/day9-build.stderr.txt`
- `artifacts/day9-bench-main-nd.stdout.txt`
- `artifacts/day9-bench-main-nd.stderr.txt`
- `artifacts/day9-bench-main-colamd.stdout.txt`
- `artifacts/day9-bench-main-colamd.stderr.txt`
- `artifacts/day9-bench-reorder-nos4.stdout.txt`
- `artifacts/day9-bench-reorder-nos4.stderr.txt`
- `artifacts/day9-bench-colamd.stdout.txt`
- `artifacts/day9-bench-colamd.stderr.txt`
- `artifacts/day9-bench-chol-csc.stdout.txt`
- `artifacts/day9-bench-chol-csc.stderr.txt`
- `artifacts/day9-bench-ldlt-csc.stdout.txt`
- `artifacts/day9-bench-ldlt-csc.stderr.txt`
- `artifacts/day9-example-colamd.stdout.txt`
- `artifacts/day9-example-colamd.stderr.txt`
- `artifacts/day9-cmake-build.stdout.txt`
- `artifacts/day9-cmake-build.stderr.txt`
- `artifacts/day9-tooling-warning-counts-by-file.txt`
- `artifacts/day9-tooling-warning-counts-by-class.txt`
- `artifacts/day9-tooling-warning-counts-by-file-and-class.txt`

## Day 10

**Objective:** Define the compile-only benchmark/example quality gate for Sprint 31 in a way that fits the existing Makefile-driven local validation flow, catches the drift classes surfaced in Days 1-9, and avoids turning routine quality checks into full benchmark execution.

### Commands Run

1. Inspect the existing local quality flow and build targets:
   - `sed -n '1,260p' Makefile`
   - `sed -n '389,460p' Makefile`
2. Inspect the current CMake target coverage for benchmarks/examples:
   - `sed -n '1,260p' CMakeLists.txt`
3. Inspect the Sprint 30 rebuild / warning workflow:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/REBUILD_WORKFLOW.md`
   - `sed -n '1,260p' scripts/epic3_warning_workflow.sh`
4. Inspect benchmark-facing docs for current workflow assumptions:
   - `sed -n '1,220p' benchmarks/README.md`

### Current-State Findings

1. The existing default local quality flow is still:
   - `make format && make lint && make test`
   - and `make check` currently expands to:
     - `format-check`
     - `lint`
     - `test`
2. `lint` currently covers:
   - `src/*.c` with `-fsyntax-only -Werror`
   - `clang-tidy` on `src/*.c`
   - `cppcheck` on `src/` and `tests/`
   - but **not** benchmark or example compile/link coverage
3. The Makefile already has a compile-only benchmark target:
   - `bench-build`
   - builds all `$(BENCH_BINS)`
   - does **not** execute them
4. The Makefile also has an `examples` target that builds all examples, but:
   - it is not named as a quality gate
   - it is not integrated into `lint` or `check`
5. The Sprint 30 warning workflow is intentionally heavier:
   - clean CMake configure/build
   - `ctest`
   - Makefile `all`
   - warning summarization
   - it is the authoritative warning-capture path, but not the right default insertion point for a fast compile-only tooling gate
6. CMake already knows how to build the benchmark/example executables, but:
   - the existing local quality flow is Makefile-centric
   - benchmark/example compile drift in Sprint 31 was caught through local Makefile/CMake rebuilds, not through `ctest`

### Chosen Day 10 Design

#### Gate Placement

The Sprint 31 compile-only tooling gate should live in **both** places:

1. as a new explicit Makefile target for focused reruns
2. inside the default `make lint` path so the existing `make format && make lint && make test` workflow catches benchmark/example compile drift automatically

Reason:

- putting the gate only in a new explicit target would require maintainers to learn and remember a fourth routine validation command
- putting it only inside `lint` would make targeted reruns awkward
- doing both preserves the current validation habit while still giving a narrow compile-only entry point for benchmark/example-only edits

#### Proposed Target Contract

Proposed explicit target shape:

- `examples-build`
  - builds all `$(EX_BINS)`
  - no execution
- `tooling-build`
  - depends on:
    - `bench-build`
    - `examples-build`
  - no execution

`lint` integration:

- `lint` should invoke or depend on `tooling-build`
- `check` then inherits the tooling gate automatically through `lint`

#### Chosen Scope

The gate should compile:

- all Makefile benchmark binaries (`$(BENCH_BINS)`)
- all Makefile example binaries (`$(EX_BINS)`)

Reason:

- Sprint 31 drift was not limited to one file or one sub-area:
  - `bench_main.c`
  - `bench_colamd.c`
  - `bench_chol_csc.c`
  - `bench_ldlt_csc.c`
  - `example_colamd.c`
- building only the Sprint 31-touched subset would recreate the same drift risk on the next untouched benchmark/example file
- building all bench/example entry points is still practical because the gate does **not** execute them

### Gate Contract

What the gate **does** prove:

- benchmark/example sources still compile against the current public and internal headers
- benchmark/example binaries still link against the library
- option-struct growth, feature-test-macro changes, include drift, and API signature drift are caught on the local quality path
- benchmark/example compile coverage is no longer limited to ad hoc manual rebuilds

What the gate **does not** prove:

- benchmark runtime correctness
- benchmark numerical behavior
- benchmark performance stability
- long-running fixture execution
- CMake warning inventory deltas

Those remain covered by:

- targeted tool reruns
- `bench-fast` / `bench`
- Sprint 30 warning workflow
- normal test/benchmark investigation when a change warrants it

### Rejected Alternatives

#### Put the gate only in `check`

Rejected because:

- the repo’s documented day-to-day validation habit is `make format && make lint && make test`
- many maintainers will never call `make check`
- the drift would remain invisible on the most common validation path

#### Put the gate only in a new explicit target

Rejected because:

- it depends on maintainers remembering an extra command
- Sprint 31’s goal is to catch tooling drift earlier without retraining the whole workflow

#### Expand `lint` with `-fsyntax-only` over benchmarks/examples instead of building binaries

Rejected because:

- Sprint 31 drift included entry-point and option-surface issues where full compile/link coverage is the more honest signal
- a syntax-only pass would miss link-layer and binary-target wiring failures
- the Makefile already has the stronger `bench-build` precedent

#### Reuse `bench-fast`

Rejected because:

- `bench-fast` executes benchmarks
- Sprint 31 needs compile-only coverage, not runtime wall-time work

#### Reuse the Sprint 30 warning workflow as the default tooling gate

Rejected because:

- it is intentionally heavier
- it cleans build directories and runs `ctest`
- it is the right before/after warning-capture workflow, not the right default “did benchmark/example entry points still compile?” gate

### Day 10 Interpretation

- The repository already has half of the desired gate in `bench-build`; Sprint 31 mainly needs to complete the example side and wire the combined compile-only signal into the default quality flow.
- The most pragmatic Day 11 implementation is Makefile-first:
  - add `examples-build`
  - add `tooling-build`
  - have `lint` invoke `tooling-build`
- This design catches the exact Sprint 31 drift class earlier while keeping benchmark execution out of normal quality checks.

### Day 10 Outputs

- `artifacts/day10-compile-only-gate-design.md`

## Day 11

**Objective:** Implement the compile-only benchmark/example tooling gate chosen on Day 10, wire it into the normal local quality flow, and document how maintainers should use it.

### Files Edited

- `Makefile`
- `benchmarks/README.md`

### Commands Run

1. Re-read the Day 10 insertion points in the Makefile and benchmark docs:
   - `sed -n '180,240p' Makefile`
   - `sed -n '389,440p' Makefile`
   - `sed -n '1,120p' benchmarks/README.md`
2. Implement the new Makefile targets and `lint` integration.
3. Validate the explicit compile-only gate:
   - `make tooling-build`
4. Validate the `lint` integration path without waiting on the full analyzer runtime:
   - `make -n lint`
5. Spot-check the live integrated path startup:
   - start `make lint`
   - confirm from captured stdout that `bench-build`, `examples-build`, and `tooling-build` run before the existing source-only lint phases

### Changes Made

1. Added `examples-build`:
   - builds all `$(EX_BINS)`
   - does not execute them
2. Added `tooling-build`:
   - depends on:
     - `bench-build`
     - `examples-build`
   - does not execute benchmarks or examples
3. Changed `examples` to depend on `examples-build` so the user-facing examples entry point keeps working without duplicating the build logic.
4. Wired `lint` to depend on `tooling-build`.
5. Updated `benchmarks/README.md` with the new compile-only workflow:
   - `make tooling-build`
   - `make bench-build`
   - `make examples-build`
   - `make lint` now includes the compile-only tooling gate automatically

### Validation Results

Explicit gate validation:

- `make tooling-build` completed successfully
- built:
  - `14` benchmark binaries
  - `12` example binaries
- `artifacts/day11-tooling-build.stderr.txt` is empty

Integrated-path validation:

- `make -n lint` now shows, in order:
  - `Built 14 bench binaries (no execution).`
  - `Built 12 example binaries (no execution).`
  - `tooling-build: benchmark and example binaries built (no execution).`
  - `Compiling with strict warnings (-Werror)...`
  - `Running clang-tidy...`
  - `Running cppcheck...`
- a live `make lint` startup was also captured and showed the same gate ordering before the pre-existing analyzer phases

Important Day-11 validation note:

- I did **not** wait for the full `clang-tidy` / `cppcheck` portion of `make lint` to finish
- Day 11’s implementation changes are limited to:
  - compile-only target wiring
  - benchmark-facing workflow docs
- the new gate itself was fully validated through `make tooling-build`
- the `lint` integration point was validated through:
  - `make -n lint`
  - the captured live startup ordering in `artifacts/day11-lint.stdout.txt`

### Day 11 Interpretation

- Sprint 31’s compile-only tooling gate is now real, not just designed.
- Maintainers can run:
  - `make tooling-build`
  - for a focused benchmark/example compile-only check
- Maintainers running the normal:
  - `make format && make lint && make test`
  - workflow now automatically get the same compile-only tooling coverage during `lint`
- This closes the workflow gap that allowed benchmark/example compile drift to survive outside the normal quality path.

### Day 11 Outputs

- `artifacts/day11-tooling-gate-implementation.md`
- `artifacts/day11-tooling-build.stdout.txt`
- `artifacts/day11-tooling-build.stderr.txt`
- `artifacts/day11-lint.stdout.txt`
- `artifacts/day11-lint.stderr.txt`
- `artifacts/day11-lint-dryrun.stdout.txt`
- `artifacts/day11-lint-dryrun.stderr.txt`

## Day 12

**Objective:** Refresh the benchmark-facing and public-facing docs so they match the Sprint 31 reorder contract, designated-initializer style, and compile-only tooling gate that now exist on this branch.

### Commands Run

1. Re-read the Day 12 scope and the current benchmark/public docs:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_31/PLAN.md`
   - `sed -n '1,260p' benchmarks/README.md`
   - `sed -n '1,260p' README.md`
   - `sed -n '1,260p' include/sparse_lu.h`
   - `sed -n '1,260p' include/sparse_cholesky.h`
   - `sed -n '1,320p' include/sparse_analysis.h`
   - `sed -n '680,740p' docs/algorithm.md`
   - `sed -n '1268,1310p' docs/algorithm.md`
2. Sweep for stale reorder guidance and positional option snippets:
   - `rg -n "SPARSE_REORDER_AMD|tooling-build|bench_main|bench_reorder|colamd|make lint" docs include README.md benchmarks examples`
   - `rg -n "_opts_t opts = \\{|_opts_t aopts = \\{" README.md docs include examples benchmarks`
3. Re-check the benchmark entry-point contracts before editing:
   - `sed -n '1,160p' benchmarks/bench_main.c`
   - `sed -n '1,140p' benchmarks/bench_reorder.c`
   - `sed -n '1,120p' benchmarks/bench_colamd.c`
4. After the edits, run a narrow doc sanity sweep:
   - `git diff -- benchmarks/README.md README.md include/sparse_lu.h include/sparse_cholesky.h include/sparse_analysis.h docs/algorithm.md docs/planning/EPIC_3/SPRINT_31/WORKING_NOTES.md docs/planning/EPIC_3/SPRINT_31/artifacts/day12-documentation-refresh.md`
   - `rg -n "tooling-build|make lint|bench_main --reorder|COLAMD, or ND|factor_type = SPARSE_FACTOR_CHOLESKY|pivot = SPARSE_PIVOT_PARTIAL|reorder = SPARSE_REORDER_AMD" benchmarks/README.md README.md include/sparse_lu.h include/sparse_cholesky.h include/sparse_analysis.h docs/algorithm.md`

### Changes Made

1. Updated `benchmarks/README.md`:
   - added an explicit reorder-coverage section
   - documented the intentional contract split between:
     - `bench_main`
     - `bench_reorder`
     - the QR/COLAMD comparison tools
     - the fixed-reorder backend-comparison benchmarks
2. Updated the top-level `README.md` build workflow:
   - added `make tooling-build`
   - added `make lint`
   - documented that `lint` includes the compile-only tooling gate
3. Updated the public header examples to use designated initialization:
   - `include/sparse_lu.h`
   - `include/sparse_cholesky.h`
   - `include/sparse_analysis.h`
4. Refreshed the `sparse_analysis_opts_t` reorder-field documentation:
   - expanded the documented accepted set to:
     - `NONE`
     - `RCM`
     - `AMD`
     - `COLAMD`
     - `ND`
   - clarified that `sparse_analyze()` applies `COLAMD` symmetrically and that QR remains the column-only path
5. Updated `docs/algorithm.md` so the LU, Cholesky, and analyze-once examples no longer teach brittle positional option-struct initialization.

### Validation Results

- The targeted doc sanity sweep shows the intended new strings in the touched files:
  - benchmark reorder contract
  - compile-only tooling gate
  - designated initializer examples
  - `sparse_analysis` reorder coverage
- I did **not** run `make format`, `make lint`, or `make test` because Day 12 is documentation-only.

### Day 12 Interpretation

- Benchmark-facing guidance now matches the Sprint 31 behavior that shipped earlier on this branch instead of leaving the reorder split implicit.
- The main public header/manual examples no longer teach the positional-init pattern that Sprint 31 just removed from the benchmark/example code.
- The compile-only tooling gate is now documented both where benchmark users look first and in the top-level Make workflow.

### Day 12 Outputs

- `artifacts/day12-documentation-refresh.md`

## Day 13

**Objective:** Re-run the compile-only tooling gate and the authoritative serialized CMake warning path, close the last residual benchmark/example warning if it still exists, then prove the standard local validation flow passes cleanly.

### Commands Run

1. Re-read the Day 13 scope and current validation wiring:
   - `sed -n '320,420p' docs/planning/EPIC_3/SPRINT_31/PLAN.md`
   - `sed -n '180,260p' Makefile`
   - `sed -n '1,220p' benchmarks/README.md`
2. Re-run the compile-only benchmark/example gate:
   - `make tooling-build`
3. Inspect the only remaining candidate warning site and verify it against a fresh serialized CMake capture:
   - `sed -n '1,240p' benchmarks/bench_convergence.c`
   - `rg -n "double-promotion|NAN|1e-9|1e-10|1e-15|float|0\\.0f|1\\.0f|fabsf|sqrtf|powf" benchmarks/bench_convergence.c`
   - `rm -rf build/sprint31-day13-cmake`
   - `cmake -S . -B build/sprint31-day13-cmake`
   - `cmake --build build/sprint31-day13-cmake --parallel 1 --clean-first`
   - `rg -n "warning:" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-cmake-build.stderr.txt`
   - `rg -n "benchmarks/|examples/" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-cmake-build.stderr.txt`
4. Apply the residual mechanical cleanup in `benchmarks/bench_convergence.c`:
   - replace `return NAN;` with `return nan("");`
5. Re-run the standard validation flow and the serialized CMake comparison path:
   - `make format`
   - `cmake --build build/sprint31-day13-cmake --parallel 1 --clean-first`
   - `make lint`
   - `make test`
6. Derive final Day 13 metrics for the deferred queue:
   - `rg -c "warning:" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt`
   - `rg -n "benchmarks/|examples/" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt`
   - `rg -n "warning:" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt | grep -v '/tests/'`
   - `rg -o "\\[-W[^]]+\\]" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt | sort | uniq -c`
   - `rg -n "warning:" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt | sed 's#^.*/tests/##' | cut -d: -f1 | sort | uniq -c | sort -nr | head -n 15`

### Changes Made

1. Re-ran `make tooling-build` and confirmed the Sprint 31 compile-only gate still builds all benchmark and example binaries cleanly.
2. Reproduced the authoritative serialized CMake warning inventory and confirmed the only remaining benchmark/example warning was:
   - `benchmarks/bench_convergence.c:43`
   - `-Wdouble-promotion`
3. Closed that residual warning mechanically by changing the allocation-failure return in `compute_rel_residual()` from:
   - `NAN`
   - to:
   - `nan("")`
4. Re-ran the serialized CMake build after the fix and verified the benchmark/example warning queue is now empty.
5. Ran the standard local validation flow:
   - `make format`
   - `make lint`
   - `make test`

### Validation Results

Compile-only tooling gate:

- `make tooling-build` passed
- benchmark/example binaries still build cleanly with no execution

Authoritative serialized CMake warning path:

- before the final `bench_convergence.c` fix:
  - total warnings: `99`
  - benchmark/example warnings: `1`
  - residual site: `benchmarks/bench_convergence.c:43`
- after the fix:
  - total warnings: `98`
  - benchmark/example warnings: `0`
  - non-test warning lines: `0`

Standard local validation:

- `make format` passed
- `make lint` passed
  - stderr contains the normal `clang-tidy` progress/suppression summary, not a failing diagnostic
- `make test` passed

Final deferred warning queue from the Day 13 serialized CMake capture:

- remaining warnings are all in `tests/`
- class breakdown:
  - `62` `-Wmissing-field-initializers`
  - `33` `-Wdouble-promotion`
  - `3` `-Wunused-function`
- largest remaining files:
  - `test_ldlt.c` `18`
  - `test_sprint20_integration.c` `9`
  - `test_colamd.c` `8`
  - `test_chol_csc.c` `8`
  - `test_reorder_nd.c` `7`

### Day 13 Interpretation

- Sprint 31’s benchmark/example queue is now fully closed on the authoritative Apple Clang serialized CMake path.
- The Day 11 compile-only tooling gate remains valid after the full cleanup, and the standard `make format && make lint && make test` flow now passes with the Sprint 31 changes in place.
- The remaining warning debt is no longer mixed across benchmark/example tooling; it is an explicitly test-only deferred queue for a later sprint.

### Day 13 Outputs

- `artifacts/day13-validation-sweep.md`
- `artifacts/day13-cmake-configure.stdout.txt`
- `artifacts/day13-cmake-configure.stderr.txt`
- `artifacts/day13-cmake-build.stdout.txt`
- `artifacts/day13-cmake-build.stderr.txt`
- `artifacts/day13-final-cmake-build.stdout.txt`
- `artifacts/day13-final-cmake-build.stderr.txt`
- `artifacts/day13-make-format.stdout.txt`
- `artifacts/day13-make-format.stderr.txt`
- `artifacts/day13-make-lint.stdout.txt`
- `artifacts/day13-make-lint.stderr.txt`
- `artifacts/day13-make-test.stdout.txt`
- `artifacts/day13-make-test.stderr.txt`

## Day 14

**Objective:** Turn the Sprint 31 results into durable closeout inputs by writing the retrospective and the Sprint 32+ handoff from the validated Day 13 end state.

### Commands Run

1. Re-read the Day 14 closeout requirement and the current Sprint 31 notes:
   - `sed -n '360,440p' docs/planning/EPIC_3/SPRINT_31/PLAN.md`
   - `tail -n 120 docs/planning/EPIC_3/SPRINT_31/WORKING_NOTES.md`
2. Re-read the previous Epic 3 closeout shape to match style and scope:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_30/RETROSPECTIVE.md`
3. Re-read the Sprint 31 project-plan entry and current sprint outputs:
   - `sed -n '43,90p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `ls -R docs/planning/EPIC_3/SPRINT_31`
4. Pull the exact Day 1 and Day 13 metrics from Sprint 31 artifacts for the final write-up:
   - `cat docs/planning/EPIC_3/SPRINT_31/artifacts/day1-tooling-warning-counts-by-class.txt`
   - `cat docs/planning/EPIC_3/SPRINT_31/artifacts/day1-tooling-warning-counts-by-file.txt`
   - `rg -o "\\[-W[^]]+\\]" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt | sort | uniq -c`
   - `rg -n "warning:" docs/planning/EPIC_3/SPRINT_31/artifacts/day13-final-cmake-build.stderr.txt | sed 's#^.*/##' | cut -d: -f1 | sort | uniq -c | sort -nr | head -n 20`
   - area-count derivation confirming all remaining warnings are in `tests/`
5. After writing the closeout docs, do a narrow sanity read:
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_31/HANDOFF.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_31/RETROSPECTIVE.md`
   - `tail -n 140 docs/planning/EPIC_3/SPRINT_31/WORKING_NOTES.md`

### Changes Made

1. Added `HANDOFF.md` for Sprint 31:
   - records the validated Sprint 31 end state
   - captures the benchmark/tooling contracts that should not regress
   - routes the remaining deferred warning queue into explicit Sprint 32 priorities
2. Added `RETROSPECTIVE.md` for Sprint 31:
   - summarizes the sprint's scope, outcomes, metrics, and tradeoffs
   - records the final before/after warning delta
   - captures what worked, what did not, and why the remaining debt was deferred
3. Updated `WORKING_NOTES.md` with the Day 14 closeout record.

### Validation Results

- Day 14 itself is docs-only, so no new code-validation commands were needed.
- The closeout documents are grounded in the validated Day 13 end state:
  - `make tooling-build`: passed
  - `make format`: passed
  - `make lint`: passed
  - `make test`: passed
- Final authoritative warning state carried into the handoff:
  - full-tree warnings: `98`
  - `tests`: `98`
  - `benchmarks`: `0`
  - `examples`: `0`

### Day 14 Interpretation

- Sprint 31 now ends with the same durable artifacts Sprint 30 produced:
  - working notes
  - retrospective
  - handoff
- The benchmark/example queue is not just fixed on the branch; it is now summarized in a form Sprint 32 can use directly without re-triage.
- The remaining deferred queue is explicitly test-only and attached to named files and warning classes.

### Day 14 Outputs

- `HANDOFF.md`
- `RETROSPECTIVE.md`
- `WORKING_NOTES.md`
