# Sprint 36 Day 2: macOS Warning-Parity Audit

## Scope

Audit the current Apple Clang/macOS quality surface against the Sprint 34
reviewed-wrapper contract and define the real macOS parity queue for Sprint 36.

The point of Day 2 is not to assume macOS is "behind" in every way. It is to
separate:

- reviewed-path gaps that need alignment
- platform-specific value that should be kept
- documented compiler/runtime limits that should stay explicit

## Commands and Surfaces Reviewed

### Plan and baseline inputs

- `docs/planning/EPIC_3/SPRINT_36/PLAN.md`
- `docs/planning/EPIC_3/SPRINT_36/artifacts/day1-cross-platform-baseline.md`

### macOS workflow

- `.github/workflows/macos-ci.yml`

### reviewed target and macOS helper surfaces

- `Makefile`
  - reviewed wrapper targets
  - sanitizer targets
  - OpenMP/libomp handling
  - TSan notes
- `scripts/ci.sh`

### comparison surfaces

- `.github/workflows/ci.yml`
- `make -n CC=cc quality-review-compile`
- `make -n CC=cc quality-review-cmake-compile`
- `make -n CC=cc deadcode-check`

## Main Result

The dominant macOS parity gap is **workflow entrypoint alignment**, not target
absence.

The reviewed Sprint 34 targets are already callable on macOS-style local
`CC=cc` paths. The problem is that macOS CI still uses older direct build/test
entrypoints instead of expressing the reviewed wrapper contract directly.

## What The Audit Found

### 1. Reviewed targets are already available on macOS locally

The dry-run checks showed that macOS-style local invocations can already reach:

- `quality-review-compile`
- `quality-review-cmake-compile`
- `deadcode-check`

Interpretation:

- the repo does not currently show evidence that Apple Clang is blocked from
  the maintained reviewed wrapper layer
- Sprint 36 should treat macOS parity first as a CI/workflow contract problem
  rather than assuming a source-level compiler breakage queue

### 2. macOS CI still does not express the reviewed wrapper contract

Compared to Linux CI, `macos-ci.yml` still does not run:

- `make quality-review-compile`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

It also does not produce the reviewed CMake parity signals:

- `ctest -N`
- Makefile-vs-CMake test-count parity

Current macOS CI instead runs:

- direct `make`
- direct `make test`
- `make wall-check`
- Apple Clang `make sanitize`
- Homebrew GCC matrix leg
- install/pkg-config validation

Interpretation:

- macOS already has meaningful coverage
- but its quality contract is still older and less explicit than Linux's

### 3. Several current macOS differences should be preserved

The audit identified multiple macOS CI differences that are legitimate keeps:

- Homebrew GCC matrix leg
  - useful second-compiler coverage on macOS
- `wall-check`
  - real performance/regression signal already proven useful on macOS
- install/pkg-config validation job
  - distinct packaging value not provided by the Linux reviewed wrapper jobs
- no TSan in macOS CI
  - still justified by the documented Apple Clang/macOS TSan runtime limits

These are not parity failures. They are platform-specific value or truthful
platform limitations.

### 4. The biggest macOS communication debt is expectation wording

The current macOS surface still has a few explicit wording/reporting issues:

- the workflow names and steps do not describe the reviewed wrapper contract
  the way Linux CI does
- Homebrew GCC is pinned via `gcc-15`, which is a portability/reporting risk if
  the Homebrew default moves again
- sanitizer expectations are split across:
  - `make sanitize`
  - `make asan`
  - `scripts/ci.sh`
- OpenMP/libomp expectations are present, but only in comments and target help,
  not yet in a compact parity report

## Keep / Fix / Document Classification

### Fix

- align macOS CI with reviewed wrapper entrypoints where feasible
- add explicit reviewed-path expectation wording for macOS
- add explicit CMake parity interpretation if Sprint 36 chooses to expose it in
  macOS CI

### Keep

- Homebrew GCC matrix coverage
- `wall-check`
- install/pkg-config validation
- no TSan in macOS CI while the documented runtime limitations remain true

### Document

- Homebrew GCC pin drift risk
- Apple Clang sanitizer-path differences
- libomp/OpenMP expectations on macOS
- reviewed-wrapper availability vs current CI usage

## Likely Day 5 Queue

Most likely implementation surfaces for the first macOS parity batch:

- `.github/workflows/macos-ci.yml`
  - reviewed path alignment
  - clearer step naming
  - explicit expectation wording
- parity-report artifacts/docs
  - capture what macOS actually enforces vs what Linux enforces

Conditional implementation surfaces only if later evidence requires them:

- `Makefile`
- `scripts/ci.sh`

## Bottom Line

Day 2 narrowed Sprint 36's macOS work substantially:

- the reviewed wrapper layer already exists and is callable on macOS
- the main gap is that macOS CI still speaks in older direct build/test terms
- the right Sprint 36 follow-on is therefore workflow/report alignment first,
  not indiscriminate Apple Clang code churn
