# Sprint 36 Working Notes

## Day 1

**Objective:** Turn the Sprint 34 and Sprint 35 closeout state into a concrete
Sprint 36 starting inventory by confirming the inherited validated-quality
guarantees, auditing the current Makefile/CMake/CI parity surfaces, and naming
the first implementation targets for cross-platform work.

### Commands Run

1. Confirm branch and starting state:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `git branch --show-current`
2. Re-read the Sprint 36 plan and inherited handoff constraints:
   - `sed -n '1,120p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_34/HANDOFF.md`
   - `cat docs/planning/EPIC_3/SPRINT_35/HANDOFF.md`
3. Inventory the current workflow and target surfaces:
   - `ls .github/workflows`
   - `rg -n "quality-review|deadcode|ctest -N|quality-review-cmake|quality-review-compile" .github/workflows -g '*.yml'`
   - `rg -n "^\\.PHONY:|quality-review|deadcode|tooling-build|wall-check|format-check|lint:|test:|check:" Makefile`
4. Confirm local tool and suite baseline:
   - `command -v cppcheck`
   - `command -v clang-tidy`
   - `command -v xunused`
   - `command -v ctest`
   - `ctest -N --test-dir build/quality-review-cmake`
5. Re-read the current macOS, Windows, and Linux CI workflows:
   - `cat .github/workflows/macos-ci.yml`
   - `cat .github/workflows/windows-ci.yml`
   - `cat .github/workflows/ci.yml`

### Day 1 Findings

#### 1. Sprint 36 starts from a validated baseline, not a cleanup backlog

Sprint 36 inherits the Sprint 34/Sprint 35 closeout state as intended:

- reviewed Makefile wrapper contract already exists
- reviewed CMake parity contract already exists
- Linux CI already maps to the reviewed local/CMake/dead-code paths
- public-doc and example surface from Sprint 35 is already reconciled
- active `ctest` registry remains `53`

Interpretation:

- Sprint 36 is not a warning-debt sprint
- Sprint 36 is not a public-doc cleanup sprint
- Sprint 36 is a parity and portability sprint

#### 2. The reviewed-quality contract is still Linux-first in CI

Current CI split at Day 1:

- Linux (`ci.yml`)
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make deadcode-report`
  - `make deadcode-check`
- macOS (`macos-ci.yml`)
  - direct `make`
  - direct `make test`
  - `make wall-check`
  - Apple Clang `make sanitize`
- Windows (`windows-ci.yml`)
  - direct CMake configure/build/`ctest`

Interpretation:

- Linux already expresses the reviewed Sprint 34 contract directly
- macOS and Windows still validate real build/test paths, but not through the
  same reviewed wrapper layer
- the largest Day 1 parity gap is expectation alignment, not total absence of
  platform CI

#### 3. Local tool and suite prerequisites are already satisfied

Day 1 local tool availability:

- `cppcheck`: present
- `clang-tidy`: present
- `xunused`: present
- `ctest`: present

Maintained suite baseline:

- `ctest -N --test-dir build/quality-review-cmake`: `53` tests

Interpretation:

- Sprint 36 does not need a prerequisite tool-install phase locally
- parity work can start from workflow and behavior auditing immediately

#### 4. The main Sprint 34 carry-forward constraints are still live

Two inherited constraints remain load-bearing:

- dead-code compile-db coverage gap still excludes:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`
- dead-code execution still relies on shared paths:
  - `build/deadcode-cmake`
  - `build/deadcode/`

Interpretation:

- Sprint 36 should treat these as truthful portability/reporting constraints
- Sprint 36 should not pretend dead-code portability is already fully solved
- broader dead-code maturity still belongs mainly to the later Sprint 38 queue

#### 5. The first implementation surfaces are already clear

Highest-value Sprint 36 files at Day 1:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Likely first audit/fix split:

- Day 2:
  - Apple Clang/macOS reviewed-path parity
- Day 3:
  - Windows/MSVC quality audit
- Day 4:
  - reviewed cross-platform contract and CI/tooling design

### Day 1 Interpretation

- Sprint 36 starts from a strong validated local/Linux baseline and a clear
  platform mismatch in how that baseline is expressed elsewhere.
- The most important current gap is not “macOS and Windows are untested.” They
  are tested, but through older direct flows rather than the reviewed wrapper
  contract now in force on Linux.
- The day-one inventory supports a narrow, truthful parity sprint:
  - reviewed-path alignment
  - portability of scripts/targets
  - explicit platform expectation reporting

### Day 1 Outputs

- `artifacts/day1-cross-platform-baseline.md`
- `artifacts/day1-parity-surface-inventory.txt`

## Day 2

**Objective:** Audit the maintained Apple Clang/macOS quality surface against
the Sprint 34 reviewed-wrapper contract, distinguish real parity debt from
acceptable platform variance, and define the concrete macOS queue for later
Sprint 36 fixes.

### Commands Run

1. Re-read the Day 2 plan and the Day 1 baseline:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '39,63p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day1-cross-platform-baseline.md`
2. Re-read the current macOS workflow:
   - `cat .github/workflows/macos-ci.yml`
3. Re-read the reviewed Makefile targets and macOS-relevant target logic:
   - `sed -n '1,80p' Makefile`
   - `sed -n '288,585p' Makefile`
4. Compare Linux and macOS CI command surfaces:
   - inspect `.github/workflows/ci.yml`
   - inspect `.github/workflows/macos-ci.yml`
   - compare key hits for:
     - `quality-review`
     - `deadcode`
     - `format-check`
     - `lint`
     - `ctest`
     - `sanitize`
     - `wall-check`
     - `pkg-config`
5. Check whether the reviewed wrappers are already callable on macOS-style local
   `CC=cc` paths:
   - `make -n CC=cc quality-review-compile`
   - `make -n CC=cc quality-review-cmake-compile`
   - `make -n CC=cc deadcode-check`
6. Re-read the macOS-specific helper notes:
   - `cat scripts/ci.sh`
   - `rg -n "__APPLE__|APPLE|darwin|clang|gcc-15|libomp|sanitize-thread|tsan" Makefile scripts .github/workflows -g '*.*'`

### Day 2 Findings

#### 1. The main macOS parity gap is workflow entrypoint alignment, not target absence

Day 2's most important result is that the maintained reviewed targets are
already available on macOS locally:

- `quality-review-compile`
- `quality-review-cmake-compile`
- `deadcode-check`

The `make -n CC=cc ...` checks showed that those paths are callable with the
default macOS `cc`-style toolchain and do not depend on Linux-only target
names.

Interpretation:

- the dominant macOS parity gap is not "the reviewed wrapper contract cannot
  run on macOS"
- it is that `macos-ci.yml` still drives older direct entrypoints:
  - `make`
  - `make test`
  - `make wall-check`
  - Apple Clang `make sanitize`

#### 2. macOS CI still lags the reviewed Sprint 34 contract in three concrete ways

Compared to Linux CI, the current macOS workflow does **not** yet express:

- reviewed Makefile compile-quality path
  - no `make quality-review-compile`
- reviewed CMake parity path
  - no `make quality-review-cmake`
  - no `ctest -N` / Makefile-vs-CMake test-count parity signal
- dead-code reporting/check expectation
  - no `make deadcode-report`
  - no `make deadcode-check`

Interpretation:

- these are the highest-value Day 5/Day 9 macOS follow-on surfaces
- they are CI contract gaps first, not necessarily code-level warning failures

#### 3. Several macOS workflow differences are legitimate keeps, not fix debt

Day 2 also identified differences that should stay explicit rather than being
forced into fake parity:

- keep:
  - Homebrew GCC matrix leg
    - it covers a useful second compiler on macOS that Linux parity work does
      not replace
  - `wall-check`
    - this remains a real regression signal already used in macOS CI
  - install/pkg-config verification job
    - this is macOS-specific value, not workflow noise
  - no TSan in macOS CI
    - the existing notes still describe real Apple Clang/macOS TSan runtime
      limits

Interpretation:

- Sprint 36 should align reviewed-quality expectations on macOS without
  deleting the extra macOS-specific value already present

#### 4. The main macOS document/report queue is now explicit

The current macOS surface still carries a few platform-communication issues
that are not necessarily code bugs:

- document or normalize:
  - the Homebrew GCC pin uses `gcc-15` in workflow comments and job config
  - Apple Clang sanitizer expectations are split across:
    - `make sanitize`
    - `make asan`
    - `scripts/ci.sh`
  - OpenMP/libomp expectations are documented in Makefile comments but not yet
    presented as part of a reviewed cross-platform parity report

Interpretation:

- Sprint 36 should keep these as explicit parity-report and CI-wording items
- they do not yet justify source-code edits by themselves

#### 5. The likely Day 5 macOS fix batch is narrow

Based on the audit, the first macOS implementation batch should focus on:

- `.github/workflows/macos-ci.yml`
  - reviewed-path entrypoint alignment
  - clearer platform expectation wording
- possibly `Makefile`
  - only if later work finds a real Apple Clang reviewed-path incompatibility
- supporting docs/reporting
  - to explain the resulting macOS reviewed-path interpretation truthfully

This is narrower than a generic "fix all macOS warnings" queue.

### Day 2 Interpretation

- Day 2 did **not** find evidence that Apple Clang is presently blocked from
  the reviewed wrapper contract locally.
- Day 2 **did** find that macOS CI still expresses an older build/test contract
  than Linux CI, even though the newer reviewed targets are already available.
- The right next step is therefore not broad code churn. It is a disciplined
  macOS parity batch centered on workflow entrypoints and truthful reporting.

### Day 2 Outputs

- `artifacts/day2-macos-warning-parity-audit.md`
