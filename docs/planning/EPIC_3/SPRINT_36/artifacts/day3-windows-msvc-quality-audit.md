# Sprint 36 Day 3: Windows/MSVC Quality Audit

## Scope

Audit the current Windows/MSVC quality surface against the Sprint 34
reviewed-wrapper contract and define the real Windows parity queue for Sprint
36.

The point of Day 3 is to separate:

- real repo-side MSVC portability debt
- workflow/contract gaps in Windows CI
- intentionally staged or explicitly excluded surfaces that should be reported
  honestly instead of treated as silent failures

## Commands and Surfaces Reviewed

### Plan and baseline inputs

- `docs/planning/EPIC_3/SPRINT_36/PLAN.md`
- `docs/planning/EPIC_3/SPRINT_36/artifacts/day1-cross-platform-baseline.md`
- `docs/planning/EPIC_3/SPRINT_36/artifacts/day2-macos-warning-parity-audit.md`

### Windows workflow

- `.github/workflows/windows-ci.yml`

### build-system and compatibility surfaces

- `CMakeLists.txt`
- `_WIN32` / MSVC compatibility references across:
  - `src/`
  - `tests/`
  - `include/`
  - workflow comments

### comparison surfaces

- `.github/workflows/ci.yml`
- `make -n quality-review-cmake-compile`

## Main Result

The dominant Windows gap is **reviewed-contract expression**, not evidence of a
large hidden MSVC code-failure queue.

The repo already contains explicit MSVC/Win32 accommodations in both CMake and
the codebase. The bigger problem is that Windows CI still validates only a
narrow direct CMake path instead of expressing the broader reviewed-quality
story now in force on Linux/local paths.

## What The Audit Found

### 1. MSVC support is already represented explicitly in the repo

Key existing accommodations:

- CMake gates non-MSVC `-W*` flags away from `cl.exe`
- CMake uses MSVC-specific options and definitions:
  - `/W3`
  - `_CRT_SECURE_NO_WARNINGS`
  - `/experimental:c11atomics`
- `_WIN32` timing fallbacks exist for progress-timer paths
- `tests/test_framework.h` routes env-var helpers through `_putenv_s`
- several Win32/Posix boundaries are documented directly in comments and
  conditional build logic

Interpretation:

- Sprint 36 should not assume Windows support is merely aspirational
- the current Windows surface is partial, but it is deliberate and already
  partially engineered

### 2. Windows CI is the narrowest current quality contract

Current `windows-ci.yml` only runs:

- CMake configure
- CMake build
- CMake `ctest`

It does **not** yet express:

- reviewed Makefile compile-quality path
- reviewed local end-to-end path
- reviewed CMake parity wrapper naming
- Makefile-vs-CMake test-count parity
- dead-code reporting/check expectations

Interpretation:

- the first Windows parity work is primarily workflow alignment and reporting
- Windows currently says less about quality expectations than Linux or even
  macOS

### 3. Several Windows exclusions are explicit and partly legitimate

Current CMake test exclusions on Win32/MSVC:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

Current benchmark exclusions on Win32:

- `bench_main`
- `bench_scaling`
- `bench_convergence`
- `bench_refactor`
- `bench_bicgstab`
- `bench_chol_csc`
- `bench_ldlt_csc`
- `bench_refactor_csc`
- `bench_eigs`
- `bench_amd_qg`

Why these are not automatically "bugs":

- thread tests currently use pthread-specific APIs
- fuzz test currently depends on POSIX temp-file behavior
- several benchmarks use POSIX-only APIs with no current direct MSVC path

Interpretation:

- Sprint 36 should classify these as staged exclusions first
- only later evidence should decide whether each one belongs to portability
  work, workflow framing, or a later broader expansion sprint

### 4. The MSVC parity queue splits cleanly

#### Workflow / CI

- Windows workflow step naming and intent do not yet reflect the reviewed
  wrapper contract
- no explicit parity signal exists for:
  - reviewed CMake path naming
  - test-count parity
  - staged exclusions vs enforced surfaces

#### Code / build-system

- MSVC still uses a lighter warning contract (`/W3`) than the Linux Makefile
  strict-compile path
- several surfaces are excluded rather than ported:
  - thread tests
  - fuzz test
  - the POSIX-heavy benchmark set

#### Documentation / reporting

- the current workflow does not explain what Windows validates relative to
  Linux/macOS
- Win32 exclusions are discoverable only by reading comments and CMake guards,
  not through a compact parity report

## Keep / Fix / Document Classification

### Fix

- align Windows workflow wording with the reviewed parity contract
- expose stronger CMake parity interpretation where feasible
- make staged Windows exclusions explicit in parity reporting

### Keep

- CMake-first Windows path as the current practical entry surface
- explicit exclusion of pthread- and POSIX-tempfile-dependent tests until a
  real portability decision is made
- explicit benchmark gating where the bench code still depends on POSIX-only
  helpers

### Document

- current Windows test/benchmark exclusion set
- MSVC warning-level difference versus Linux reviewed Makefile path
- what Windows CI currently enforces vs what remains staged

## Likely Day 6 / Day 9 Queue

Most likely first Windows follow-on surfaces:

- `.github/workflows/windows-ci.yml`
  - clearer step naming
  - explicit parity framing
- parity-report artifacts/docs
  - enforced vs staged vs excluded Windows surface map

Conditional implementation surfaces only if later evidence requires them:

- `CMakeLists.txt`
- source/test files behind current Win32 gates

## Bottom Line

Day 3 narrowed Sprint 36's Windows work substantially:

- the repo already contains meaningful MSVC/Win32 accommodations
- the current Windows quality gap is mostly about reviewed-contract expression
  and truthful staged-scope reporting
- the right Sprint 36 follow-on is therefore workflow/report alignment first,
  with code changes only where concrete MSVC evidence justifies them
