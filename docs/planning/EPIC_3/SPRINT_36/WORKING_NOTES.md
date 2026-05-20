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

## Day 3

**Objective:** Audit the current Windows/MSVC quality surface against the
Sprint 34 reviewed-wrapper contract, distinguish real repo portability debt
from staged workflow limitations, and define the concrete MSVC queue for later
Sprint 36 implementation work.

### Commands Run

1. Re-read the Day 3 plan and the Day 1/Day 2 baseline:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '64,88p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day1-cross-platform-baseline.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day2-macos-warning-parity-audit.md`
2. Re-read the current Windows workflow:
   - `cat .github/workflows/windows-ci.yml`
3. Re-read the MSVC-relevant CMake and compatibility surfaces:
   - `cat CMakeLists.txt`
   - `rg -n "MSVC|_WIN32|clock_gettime|CLOCK_MONOTONIC|unistd.h|sys/time.h|_putenv_s|setenv|unsetenv|NOMINMAX|PATH_MAX|strcasecmp|snprintf|ssize_t|drand48|mkstemp|realpath|popen|fmemopen" CMakeLists.txt include src tests scripts .github/workflows -g '*.*'`
4. Compare Linux and Windows workflow entrypoints:
   - inspect `.github/workflows/ci.yml`
   - inspect `.github/workflows/windows-ci.yml`
   - compare key hits for:
     - `quality-review`
     - `deadcode`
     - `format-check`
     - `lint`
     - `ctest`
     - `cmake --build`
     - `sanitize`
     - `wall-check`
5. Re-read the reviewed CMake parity wrapper surface:
   - `make -n quality-review-cmake-compile`
6. Quantify the currently staged Win32/MSVC exclusions in CMake:
   - inspect the conditional `add_sparse_test(...)` gates
   - inspect the conditional benchmark `add_executable(...)` gates

### Day 3 Findings

#### 1. Windows already has explicit MSVC accommodations in code and CMake

Day 3 did **not** uncover a naive Unix-only build system pretending Windows
support exists. The repo already contains explicit MSVC/Win32 accommodations,
including:

- MSVC-specific CMake warning handling:
  - non-MSVC `-W*` flags are gated away from `cl.exe`
  - MSVC uses `/W3`
  - `_CRT_SECURE_NO_WARNINGS` is set
  - `/experimental:c11atomics` is enabled
- `_WIN32` timing fallbacks for the progress-timer paths
- `_putenv_s` compatibility in `tests/test_framework.h`
- known POSIX-only test and benchmark surfaces already gated out in CMake

Interpretation:

- the dominant Windows issue is not "the repo has no MSVC portability work"
- it is that the Windows path still expresses only a narrower CMake build/test
  contract than the reviewed Linux/local contract

#### 2. The current Windows workflow is the furthest from the reviewed contract

Compared to Linux CI, `windows-ci.yml` still does not run:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake`
- `make deadcode-report`
- `make deadcode-check`

Current Windows CI only runs:

- CMake configure
- CMake build
- CMake `ctest`

Interpretation:

- Windows is further behind the reviewed-wrapper story than macOS
- the highest-value Sprint 36 Windows work is workflow/contract alignment first
- code changes should only follow where MSVC-specific evidence says they are
  needed

#### 3. The staged Windows exclusions are explicit and partly legitimate

CMake currently excludes three tests on Windows/MSVC:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

And it gates out ten benchmark binaries on Win32:

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

Interpretation:

- some of this is legitimate staged scope, not silent failure:
  - `pthread`-based thread tests are explicitly POSIX-only today
  - `test_fuzz` depends on POSIX temp-file behavior
  - several benchmarks use POSIX-only APIs with no current MSVC equivalents
- Sprint 36 should document these exclusions truthfully before trying to erase
  them indiscriminately

#### 4. The main Windows parity debt splits three ways

##### Workflow / CI debt

- Windows does not yet express the reviewed wrapper contract
- no explicit reviewed CMake parity naming or test-count parity signal exists in
  the workflow
- no dead-code expectation is represented on Windows

##### Code / build-surface debt

- current CMake warning level on MSVC is only `/W3`
- the repo does not yet provide a Windows-reviewed analogue to the stricter
  Linux Makefile compile-quality path
- several surfaces are still intentionally excluded rather than ported:
  - thread tests
  - fuzz test
  - POSIX-bound benchmark set

##### Documentation / reporting debt

- the workflow currently does not explain what Windows is validating relative
  to Linux/macOS
- the Win32 exclusions live mostly in code comments/CMake comments rather than
  in a compact parity report

#### 5. The likely Day 6 / Day 9 Windows queue is bounded

Based on the audit, the first Windows follow-on should focus on:

- `.github/workflows/windows-ci.yml`
  - clearer reviewed-path expectation wording
  - stronger CMake parity framing
- `CMakeLists.txt`
  - only if Sprint 36 chooses to make the current Win32 exclusions/reporting
    more explicit in the maintained parity surface
- parity-report/docs work
  - classify enforced vs staged vs intentionally excluded Windows surfaces

This is narrower than a generic "make Windows equal to Linux in one sprint"
queue.

### Day 3 Interpretation

- Day 3 found a more mature Windows surface than the old workflow comments
  alone suggest: MSVC-specific accommodations already exist in CMake and the
  codebase.
- The main current gap is still contract expression and reporting, not evidence
  of a large hidden MSVC source-bug backlog.
- Sprint 36 should therefore treat Windows like macOS in one important sense:
  reviewed-path alignment and truthful parity reporting come first, with code
  changes driven only by concrete MSVC evidence.

### Day 3 Outputs

- `artifacts/day3-windows-msvc-quality-audit.md`

## Day 4

**Objective:** Define the reviewed cross-platform quality contract before
changing CI or helper scripts, including what Sprint 36 will enforce now, what
it will report explicitly, and what remains intentionally staged for later
work.

### Commands Run

1. Re-read the Day 4 plan and the Day 2/Day 3 audit outputs:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '89,113p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day2-macos-warning-parity-audit.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day3-windows-msvc-quality-audit.md`
2. Re-read the current Linux, macOS, and Windows workflows together:
   - `cat .github/workflows/ci.yml`
   - `cat .github/workflows/macos-ci.yml`
   - `cat .github/workflows/windows-ci.yml`
3. Reconfirm the reviewed local/CMake wrapper surface:
   - inspect `Makefile`
   - inspect `make -n quality-review-cmake-compile`

### Day 4 Design Decisions

#### 1. "Parity" in Sprint 36 means explicit reviewed-path interpretation, not identical commands everywhere

Sprint 36 should not pretend that Linux, macOS, and Windows have identical
tooling and runtime constraints today.

Chosen Day 4 interpretation:

- parity means each platform must have an explicit, truthful mapping to:
  - reviewed compile-quality expectations
  - reviewed CMake/test parity expectations
  - dead-code expectation status
  - named staged exclusions or unavailable surfaces
- parity does **not** mean every platform must immediately run the exact same
  command list

Interpretation:

- Sprint 36 is a contract-alignment sprint first
- stronger same-command enforcement can come later once staged exclusions are
  better closed or isolated

#### 2. The platform contract now splits into enforced, staged, and excluded layers

Chosen reporting model by platform:

- Linux
  - enforced reviewed compile-quality path
  - enforced reviewed CMake parity path
  - enforced dead-code report/check path
- macOS
  - enforced real build/test/sanitize/wall-check/install coverage
  - staged reviewed-wrapper alignment
  - staged dead-code parity
- Windows
  - enforced real CMake configure/build/ctest coverage
  - staged reviewed-wrapper naming/parity alignment
  - staged dead-code parity
  - explicit excluded test/benchmark surfaces

Interpretation:

- Sprint 36 should make these states visible instead of flattening them into a
  false "all platforms equal" story

#### 3. The reviewed local contract remains the source-of-truth baseline

Sprint 34/Sprint 35 local invariants remain authoritative:

- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`
- `53` registered CTest tests

Day 4 decision:

- later platform-specific CI wording should point back to these as the baseline
  reviewed contract
- platform workflows may map to subsets or staged analogues, but they should
  say so directly

#### 4. Sprint 36 should not pull dead-code into fake cross-platform enforcement

Dead-code remains a special case because of:

- the compile-db coverage gap
- the shared `build/deadcode-cmake` and `build/deadcode/` execution model
- `xunused` setup differences

Day 4 decision:

- Linux keeps dead-code as the enforced path
- macOS and Windows should document dead-code as staged/unavailable in parity
  reporting unless Sprint 36 explicitly implements a safe path there
- Sprint 36 should improve truthfulness of this status, not overclaim portable
  dead-code support

#### 5. The implementation order for Days 5 through 10 is now fixed

Chosen sequence:

- Day 5:
  - macOS workflow alignment batch
  - focus on `.github/workflows/macos-ci.yml`
- Day 6:
  - Windows workflow/reporting alignment batch
  - focus on `.github/workflows/windows-ci.yml`
- Day 7:
  - script/target portability audit
  - focus on `Makefile`, `scripts/deadcode_workflow.sh`, `scripts/deadcode_report.py`
- Day 8:
  - first portability fix batch
- Day 9:
  - CI expectation wording/reporting refinement across all three workflows
- Day 10:
  - compact parity report that classifies enforced vs staged vs excluded

This order is load-bearing because it:

- aligns platform CI entrypoints before broader reporting
- keeps portability fixes informed by the real platform contract
- delays the parity report until the workflow state is closer to final

### Day 4 Interpretation

- Day 4 closes the biggest ambiguity left by Days 2 and 3: Sprint 36 is not
  trying to force identical commands on all platforms immediately.
- It is making the reviewed contract explicit per platform, while keeping Linux
  as the enforced source-of-truth baseline.
- That gives the next implementation days a disciplined shape: workflow
  alignment first, portability fixes second, compact parity reporting last.

### Day 4 Outputs

- `artifacts/day4-cross-platform-parity-design.md`

## Day 5

**Objective:** Implement the first macOS parity batch by aligning the Apple
Clang CI leg with the reviewed Sprint 34 contract while preserving the extra
macOS-specific value already present in the workflow.

### Commands Run

1. Re-read the current macOS workflow and the reviewed wrapper surfaces:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `sed -n '1,220p' .github/workflows/macos-ci.yml`
   - `sed -n '454,520p' Makefile`
2. Update `.github/workflows/macos-ci.yml` to:
   - align the Apple Clang leg with `make quality-review-compile`
   - align the Apple Clang leg with `make quality-review-cmake`
   - preserve the Homebrew GCC direct build/test role
   - preserve `wall-check`, sanitize, and install/pkg-config coverage
3. Validate the workflow syntax and resulting diff:
   - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/macos-ci.yml"); puts "yaml_ok"'`
   - `sed -n '1,220p' .github/workflows/macos-ci.yml`
   - `git diff -- .github/workflows/macos-ci.yml`

### Day 5 Findings

#### 1. Apple Clang is now the explicit reviewed macOS baseline

The Apple Clang matrix leg now expresses the reviewed contract directly:

- installs reviewed-path tools:
  - Homebrew `llvm`
  - Homebrew `cppcheck`
  - adds Homebrew LLVM `bin` to `PATH`
- runs:
  - `make quality-review-compile`
  - `make quality-review-cmake`
  - `make wall-check`
  - `make sanitize`

Interpretation:

- the macOS workflow now names the same reviewed compile-quality and reviewed
  CMake parity concepts that Linux CI already enforces
- this closes the largest Day 2 macOS contract gap

#### 2. The Homebrew GCC leg remains intentionally direct coverage

The Homebrew GCC leg was **not** forced into a fake reviewed-wrapper copy.

It still provides:

- direct `make`
- direct `make test`
- `make wall-check`

Interpretation:

- this preserves useful second-compiler coverage on macOS
- Sprint 36 keeps the distinction between:
  - reviewed baseline compiler path
  - additional compiler coverage path

#### 3. macOS-specific value was preserved rather than collapsed

Day 5 kept the macOS-only or macOS-high-value surfaces intact:

- `wall-check`
- Apple Clang sanitize
- Homebrew GCC matrix coverage
- install/pkg-config validation job

Interpretation:

- the parity batch aligned the contract without deleting the extra diagnostic
  value already present in `macos-ci.yml`

#### 4. Dead-code remains staged on macOS by design

Day 5 intentionally did **not** add:

- `make deadcode-report`
- `make deadcode-check`

Interpretation:

- this matches the Day 4 contract
- macOS dead-code parity remains staged/unavailable pending later portability
  and execution-model work

### Day 5 Interpretation

- Day 5 converted the biggest macOS parity issue from an implicit mismatch into
  an explicit reviewed contract:
  - Apple Clang = reviewed baseline
  - Homebrew GCC = direct second-compiler signal
- The workflow is now more truthful and more aligned with Linux, while still
  preserving the macOS-specific checks that make this workflow useful.
- The next cross-platform implementation step should mirror this approach on
  Windows: clarify the reviewed contract without pretending staged surfaces are
  already enforced.

### Day 5 Outputs

- `artifacts/day5-macos-workflow-alignment.md`

## Day 6

**Objective:** Implement the first Windows parity batch by making the current
Win32/MSVC workflow express its reviewed CMake subset more explicitly, while
preserving the staged exclusion model surfaced by the Day 3 audit.

### Commands Run

1. Re-read the current Windows workflow and the Day 3/Day 4 design inputs:
   - `git status --short --branch`
   - `git rev-parse --short HEAD`
   - `cat .github/workflows/windows-ci.yml`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day3-windows-msvc-quality-audit.md`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day4-cross-platform-parity-design.md`
2. Update `.github/workflows/windows-ci.yml` to:
   - frame the job as a reviewed CMake parity subset
   - expose `ctest -N` directly in the workflow
   - assert the current staged Windows test-count expectation
   - surface the named Windows staged exclusions explicitly
3. Validate the workflow syntax and resulting diff:
   - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/windows-ci.yml"); puts "yaml_ok"'`
   - `sed -n '1,220p' .github/workflows/windows-ci.yml`
   - `git diff -- .github/workflows/windows-ci.yml`

### Day 6 Findings

#### 1. Windows now names its reviewed CMake subset explicitly

The workflow now describes the current Win32/MSVC contract as:

- reviewed CMake configure path
- reviewed CMake build path
- reviewed Windows CTest surface via `ctest -N`
- reviewed CMake execution path via full `ctest`

Interpretation:

- this closes the most important Day 3 wording gap
- Windows still remains CMake-first, but no longer looks like an unqualified
  generic build/test job

#### 2. The staged Windows suite count is now visible and asserted

Day 6 added a workflow-level expectation:

- `EXPECTED_WINDOWS_CTEST_COUNT=50`

And the workflow now:

- runs `ctest -N`
- prints the resulting registered test count
- fails if the count drifts from `50`

Interpretation:

- Windows now exposes a real parity signal instead of only running the final
  suite blindly
- this is the Windows analogue to "make the staged scope explicit" without
  claiming full Linux-count parity

#### 3. The current staged exclusions are now surfaced directly in workflow output

The workflow now prints the named staged Windows exclusions:

- `test_threads`
- `test_sprint4_integration`
- `test_fuzz`

Interpretation:

- Sprint 36 no longer leaves these exclusions discoverable only through CMake
  comments or prior sprint notes
- this improves operator truthfulness without pretending those exclusions are
  already closed

#### 4. Day 6 deliberately did not overclaim Windows parity

The workflow still does **not** add:

- `make quality-review-compile`
- `make quality-review`
- `make deadcode-report`
- `make deadcode-check`

Interpretation:

- this matches the Day 4 design
- Windows remains a reviewed CMake subset plus explicit staged exclusions, not
  a fake copy of the Linux contract

### Day 6 Interpretation

- Day 6 did for Windows what Day 5 did for macOS, but at the narrower Windows
  CMake-first scope.
- The workflow is now more truthful: it says what Windows actually enforces,
  shows the staged test-count surface directly, and names the current excluded
  tests.
- The next major step is not more platform-specific workflow churn by itself;
  it is the shared portability audit that will decide whether some staged
  surfaces can be generalized cleanly across platforms.

### Day 6 Outputs

- `artifacts/day6-windows-workflow-alignment.md`

## Day 7

**Objective:** Audit the reviewed-quality and dead-code helper surfaces for
shell, path, environment, and tool-discovery assumptions that still bind the
quality flow to POSIX-first behavior, and separate real portability debt from
intentional staged exclusions.

### Commands Run

1. Re-read the Sprint 36 Day 7 plan plus the current platform workflows:
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `sed -n '1,260p' .github/workflows/macos-ci.yml`
   - `sed -n '1,220p' .github/workflows/windows-ci.yml`
   - `sed -n '1,260p' .github/workflows/ci.yml`
2. Audit the maintained reviewed-quality and dead-code surfaces:
   - `sed -n '1,620p' Makefile`
   - `sed -n '620,820p' Makefile`
   - `sed -n '1,260p' scripts/deadcode_workflow.sh`
   - `sed -n '1,320p' scripts/deadcode_report.py`
   - `sed -n '1,260p' scripts/ci.sh`
   - `sed -n '1,260p' scripts/wall_check.sh`
   - `sed -n '1,260p' scripts/epic3_warning_workflow.sh`
3. Search for reviewed-path portability assumptions:
   - `rg -n "quality-review|deadcode|wall-check|sanitize|SHELL|bash|python3|xunused|llvm|clang-tidy|cppcheck|ctest -N|uname|brew|/tmp|realpath|sed -i|mktemp|grep -P|readlink|pwd -P" Makefile scripts .github/workflows include src tests docs/planning/EPIC_3/SPRINT_36 -g '!build/**'`
   - `rg -n "^\\.PHONY: clean|^clean:|rm -rf|find \\(|awk |for t in|for b in|python3|bash scripts|scripts/.*sh" Makefile`

### Day 7 Findings

#### 1. The reviewed CMake parity path is already the genuinely portable baseline

The current cross-platform story is much stronger on the CMake side than on the
Makefile side:

- Linux explicitly runs `make quality-review-cmake`
- macOS Apple Clang now explicitly runs `make quality-review-cmake`
- Windows now explicitly runs reviewed CMake configure/build plus `ctest -N`
  and full `ctest`

Interpretation:

- Sprint 36 does **not** need to invent a new portable quality path
- the existing portable reviewed baseline is already the CMake path
- the main portability debt sits in the Makefile-reviewed and dead-code helper
  layers, not in the CMake parity layer

#### 2. The Makefile reviewed wrappers are still intentionally POSIX-first

The maintained Makefile quality flow still depends on POSIX shell tools and
syntax in multiple places:

- shell loops in `test`, `bench`, `bench-fast`, `sanitize-thread`,
  `coverage-lcov`, and `coverage-gcovr`
- inline `awk` parsing in `quality-review-cmake-compile`, `wall-check`, and
  coverage thresholds
- `find`-derived file lists in `format`, `format-check`, and `lint`
- explicit `/bin/rm`, `/bin/mkdir`, and `bash` helper invocations

Interpretation:

- this is the real reason Windows does not yet claim `make quality-review-*`
  parity
- the problem is not "MSVC can't build the code"; it is that the Makefile path
  is still a POSIX-maintainer path
- only a subset of this is worth changing in Sprint 36, because the reviewed
  CMake path already carries the portable enforcement contract

#### 3. The dead-code workflow remains Linux-enforced by design, not by accident

The dead-code surface still has stronger portability constraints than the other
reviewed paths:

- `scripts/deadcode_workflow.sh` is an explicit `bash` workflow
- it requires `cppcheck`, `python3`, and `xunused`
- it has Darwin-specific `xcrun` / SDK / resource-dir handling
- Linux CI still builds and installs `xunused` from source at runtime
- the workflow still shares `build/deadcode-cmake` and `build/deadcode/`
  across `deadcode*` targets
- the compile-db coverage gap remains `bench_svd` plus six examples

Interpretation:

- Sprint 36 should keep dead-code as:
  - enforced on Linux CI
  - local/staged on macOS
  - excluded on Windows
- treating dead-code as fake all-platform parity would be less truthful than
  the current contract

#### 4. `wall-check` and the warning-capture helpers are explicit maintenance tools, not cross-platform reviewed gates

Two helper workflows still carry clear POSIX-only assumptions:

- `scripts/wall_check.sh`
  - `bash`
  - `mktemp`
  - `awk`
- `scripts/epic3_warning_workflow.sh`
  - `bash`
  - `awk`
  - `date`
  - POSIX clean-build directory handling

Interpretation:

- this is acceptable today because neither helper is part of the Windows
  reviewed-parity contract
- they should be documented/report-classified as Unix-maintainer tools, not
  treated as hidden Windows debt

#### 5. The highest-value Sprint 36 portability fixes are narrower than a “port everything” pass

The audit narrows the next implementation surface to a small set of
high-value moves:

- reduce avoidable POSIX-only parsing inside the maintained reviewed Makefile
  path where practical
- make Unix-only helper assumptions more explicit where they are intentional
- keep dead-code and maintenance-only helpers truthfully staged/excluded by
  platform instead of overclaiming parity

Interpretation:

- Day 8 should focus on reviewed-path helper portability and explicitness
- Day 9 should focus on CI/reporting alignment around enforced vs staged vs
  excluded surfaces

### Day 7 Interpretation

- The most important portability result is negative: the repo does **not** have
  a hidden cross-platform build failure in its maintained reviewed baseline.
- The portable baseline is already the CMake reviewed path; the remaining debt
  is in the older POSIX-maintainer wrappers and helper workflows.
- Sprint 36 should therefore tighten and clarify those wrappers rather than try
  to force dead-code or Makefile maintenance tools into fake all-platform
  symmetry.

### Day 7 Outputs

- `artifacts/day7-script-target-portability-audit.md`

## Day 8

**Objective:** Implement the first portability batch in the maintained reviewed
Makefile path by removing avoidable external Unix-tool dependencies, while
preserving the current Linux/macOS reviewed contract and not overclaiming
Windows parity for the Unix-maintainer helper flows.

### Commands Run

1. Re-read the Day 7 audit and inspect the maintained reviewed-path touchpoints:
   - `git status --short --branch`
   - `sed -n '1,620p' Makefile`
   - `sed -n '620,820p' Makefile`
   - `cat docs/planning/EPIC_3/SPRINT_36/artifacts/day7-script-target-portability-audit.md`
2. Update `Makefile` to:
   - remove reviewed-path `find` use where repo-native file lists already exist
   - remove hardcoded `/bin/mkdir` and `/bin/rm` paths
   - keep the current reviewed command meanings unchanged
3. Validate the touched reviewed paths directly:
   - `make quality-review-compile`
   - `make quality-review-cmake-compile`

### Day 8 Findings

#### 1. The maintained reviewed Makefile path no longer depends on external `find`

Day 8 replaced `find`-based file discovery in the maintained reviewed path with
repo-native Makefile file lists and globs:

- `ALL_SRC` now derives from `$(LIB_SRCS)` plus `src/*.h`
- `ALL_TEST_SRC` now derives from `$(TEST_SRCS)` plus `tests/*.h`
- `ALL_BENCH_SRC` now uses `$(BENCH_SRCS)`
- `ALL_EX_SRC` now uses `$(EX_SRCS)`
- `ALL_HEADERS` now uses `include/*.h`
- `lint` strict-compile and `clang-tidy` now run over `$(LIB_SRCS)`

Interpretation:

- this closes the most avoidable Unix-tool dependency in the maintained
  reviewed Makefile path
- the reviewed compile-quality path is now anchored more directly to the same
  repo-native file lists already used for the actual build

#### 2. Hardcoded `/bin/mkdir` and `/bin/rm` paths are gone from the maintained path

Day 8 also removed hardcoded absolute tool paths in the touched Makefile
surfaces:

- `mkdir -p` now replaces `/bin/mkdir -p`
- `rm -rf` / `rm -f` now replace `/bin/rm ...`

Interpretation:

- this is a smaller change than the `find` removal, but it is still valuable
- it reduces unnecessary POSIX-environment assumptions without changing
  semantics on the validated Linux/macOS paths

#### 3. The portable reviewed baseline stayed intact

Direct reruns of the touched reviewed paths both passed:

- `make quality-review-compile`
- `make quality-review-cmake-compile`

Observed parity state after the batch:

- Makefile reviewed compile-quality path still passed end to end
- CMake reviewed parity path still passed end to end
- `ctest -N` still reported `53` tests
- Makefile/CMake test-count parity remained `53` vs `53`

Interpretation:

- the batch improved the reviewed-path portability surface without changing the
  maintained quality contract

#### 4. The Unix-maintainer helper flows remain intentionally staged

Day 8 deliberately did **not** try to port or overclaim these surfaces:

- `deadcode*`
- `wall-check`
- `warning-workflow`
- coverage helpers

Interpretation:

- this matches the Day 7 audit
- those helpers still carry real POSIX or toolchain assumptions, and pretending
  they are now Windows-ready would make the contract less truthful

### Day 8 Interpretation

- Day 8 took the most justifiable portability win available in Sprint 36's
  reviewed Makefile path: remove avoidable shell-tool coupling where the repo
  already maintained the authoritative file lists anyway.
- The result is narrower but stronger than a cosmetic portability pass because
  the touched reviewed commands were rerun successfully after the change.
- The next step is now reporting and contract refinement, not more speculative
  portability churn.

### Day 8 Outputs

- `artifacts/day8-portability-batch1.md`

## Day 9

**Objective:** Refine CI and operator-facing reporting so the platform quality
contract is explicit in the workflows themselves and in the main README, with a
clear enforced/staged/supplemental split that matches the actual Sprint 36
implementation state.

### Commands Run

1. Re-read the Sprint 36 Day 9 plan and the current platform-contract inputs:
   - `sed -n '224,258p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day4-cross-platform-parity-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day7-script-target-portability-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day8-portability-batch1.md`
2. Update the workflow comments and job display names in:
   - `.github/workflows/ci.yml`
   - `.github/workflows/macos-ci.yml`
   - `.github/workflows/windows-ci.yml`
3. Add a concise cross-platform CI contract section to `README.md`.
4. Validate the workflow syntax and wording alignment:
   - `ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'`
   - `sed -n '680,735p' README.md`
   - `git diff -- .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml README.md`

### Day 9 Findings

#### 1. Linux now names its role as the enforced reviewed baseline directly

The Linux workflow now says, in both comments and job display names, that it
contains:

- enforced reviewed Makefile compile-quality path
- enforced reviewed CMake parity path
- enforced dead-code report/check path

And that it also carries Linux-specific supplemental signals:

- direct runtime + `bench-fast`
- TSan
- coverage

Interpretation:

- this closes the remaining "Linux is stronger, but only implicitly" gap
- later parity work can now compare other platforms to a named baseline instead
  of inferring it from job contents

#### 2. macOS and Windows now use the same enforced/staged/supplemental language

Day 9 tightened the non-Linux workflow wording so it matches the Sprint 36
design vocabulary directly:

- macOS:
  - Apple Clang leg = enforced reviewed baseline
  - Homebrew GCC leg = supplemental second-compiler signal
  - dead-code = staged
- Windows:
  - reviewed CMake subset = enforced
  - reviewed Makefile wrappers + dead-code = staged
  - named test exclusions remain explicit

Interpretation:

- CI no longer leaves the macOS/Windows contract partly implicit in comments and
  partly implicit in the job graph
- the workflow language now matches the Day 4 design model directly

#### 3. The README now carries the same platform contract as CI

Day 9 added a compact `Cross-Platform CI Contract` section to `README.md`
covering:

- Linux enforced reviewed paths
- macOS enforced reviewed Apple Clang paths
- Windows enforced reviewed CMake subset
- staged dead-code and reviewed-Makefile gaps where applicable
- supplemental or excluded surfaces by platform

Interpretation:

- operator-facing docs and workflow YAML now tell the same story
- Day 10 can build a compact parity report from an already-consistent contract
  instead of reconciling contradictory wording first

### Day 9 Interpretation

- Day 9 did not change the underlying platform enforcement behavior; it changed
  the truthfulness and readability of that behavior.
- That is the correct shape for this day because the remaining Sprint 36 gap was
  expectation clarity, not missing CI mechanics.
- The repo now has a consistent enforced/staged/supplemental vocabulary across
  the platform workflows and the main operator docs.

### Day 9 Outputs

- `artifacts/day9-ci-expectation-refinement.md`

## Day 10

**Objective:** Produce a compact cross-platform parity report that shows which
reviewed-quality surfaces are available from Make, from CMake, and in CI on
Linux, macOS, and Windows, using the enforced/staged/excluded model fixed in
Days 4-9.

### Commands Run

1. Re-read the parity-contract inputs and current workflow/docs state:
   - `sed -n '224,258p' docs/planning/EPIC_3/PROJECT_PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day4-cross-platform-parity-design.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day7-script-target-portability-audit.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day8-portability-batch1.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day9-ci-expectation-refinement.md`
2. Re-check the live operator/workflow surface:
   - `sed -n '680,760p' README.md`
   - `sed -n '1,120p' .github/workflows/ci.yml`
   - `sed -n '1,140p' .github/workflows/macos-ci.yml`
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`
3. Write the compact parity map and identify the bounded Day 11 follow-on queue.

### Day 10 Findings

#### 1. The parity surface is now compact enough to summarize directly

By Day 10, the reviewed-quality contract is no longer scattered across sprint
history:

- the local reviewed Makefile wrappers are named
- the local reviewed CMake parity path is named
- Linux, macOS, and Windows CI now all classify their surfaces explicitly

Interpretation:

- Sprint 36 can now carry one compact parity map instead of relying on prose
  spread across several artifacts

#### 2. The reviewed CMake path is the only fully honest cross-platform reviewed baseline

The report makes one structural fact explicit:

- reviewed CMake parity is the only surface that all three platforms can claim
  honestly today

By contrast:

- reviewed Makefile wrappers are enforced on Linux and macOS Apple Clang
  locally/CI, but still staged on Windows
- dead-code is enforced on Linux, staged on macOS, and excluded on Windows
- Unix-maintainer helper flows remain supplemental or excluded rather than fake
  parity surfaces

Interpretation:

- this confirms the Day 7 audit
- later portability work should keep anchoring on the reviewed CMake path first

#### 3. The remaining Day 11 queue is narrow and naming/reporting-focused

The report does not surface a new broad implementation queue. It narrows the
follow-on to a small final batch:

- reconcile any remaining wording mismatch between local target names, workflow
  comments, and the new parity report
- tighten any residual contract wording around staged dead-code / staged
  Windows Makefile reviewed paths if needed
- avoid reopening dead-code maturity or fake Windows Makefile parity work

Interpretation:

- Day 11 should be a compact consistency pass, not another audit cycle

### Day 10 Interpretation

- Day 10 succeeded because Days 4-9 already turned the platform contract into a
  stable enforced/staged/excluded model.
- The important output is not just the table; it is that the repo now has one
  operationally truthful parity map covering local Make, local CMake, and CI.
- The remaining Sprint 36 implementation queue is now bounded tightly enough
  that Day 11 can focus on final polish before validation.

### Day 10 Outputs

- `artifacts/day10-cross-platform-parity-report.md`

## Day 11

**Objective:** Close the remaining small naming/reporting mismatches between
the parity report, the workflow comments, the workflow step names, and the main
README so Sprint 36 enters validation with one consistent platform-contract
vocabulary.

### Commands Run

1. Re-read the Day 10 parity report and the current live contract surfaces:
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day10-cross-platform-parity-report.md`
   - `sed -n '680,760p' README.md`
   - `sed -n '1,120p' .github/workflows/ci.yml`
   - `sed -n '1,140p' .github/workflows/macos-ci.yml`
   - `sed -n '1,120p' .github/workflows/windows-ci.yml`
   - `sed -n '450,520p' Makefile`
2. Tighten workflow step names and README wording so they match the
   enforced/staged/supplemental model directly.
3. Validate workflow syntax and resulting wording:
   - `ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'`
   - `sed -n '714,732p' README.md`
   - `git diff -- .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml README.md`

### Day 11 Findings

#### 1. Workflow step names now match the contract model instead of only the comments

Day 11 pushed the enforced/supplemental distinction down from top-of-file
comments into the workflow step labels themselves:

- Linux:
  - reviewed compile-quality / CMake / dead-code steps now say `enforced`
  - runtime, sanitizer, benchmark, TSan, and coverage paths now say
    `supplemental` where appropriate
- macOS:
  - Apple Clang reviewed steps now say `enforced`
  - GCC direct build/test and install/pkg-config paths now say `supplemental`
- Windows:
  - reviewed CMake configure/build/`ctest -N`/`ctest` steps now say `enforced`

Interpretation:

- the job graph now communicates the same contract as the workflow comments
- CI output will be easier to interpret during Sprint 36 validation and later
  maintenance

#### 2. The README now states the Windows distinction more precisely

Day 11 tightened the README’s cross-platform contract wording by stating that
the Windows enforced subset is:

- direct CI configure + build + `ctest -N` + full `ctest`

and is **not** yet the same thing as:

- full named local Makefile reviewed-wrapper parity

Interpretation:

- this closes the most important remaining wording ambiguity from the Day 10
  report
- the README now matches both the Windows workflow and the parity report

#### 3. No broader implementation queue reopened

Day 11 intentionally did **not** reopen:

- dead-code maturity work
- Windows Makefile reviewed-wrapper portability work
- broader public-doc cleanup

Interpretation:

- the final pre-validation queue remains bounded
- Sprint 36 can now move into validation without carrying unresolved contract
  drift

### Day 11 Interpretation

- Day 11 was the correct final batch shape: small naming/reporting fixes only.
- The important result is not more mechanics; it is that the workflows, README,
  and parity report now all describe the same platform contract at the same
  level of precision.

### Day 11 Outputs

- `artifacts/day11-final-parity-consistency-pass.md`

## Day 12

**Objective:** Run the practical local validation paths that Sprint 36 changed
or re-framed, re-check the parity-report claims against live command behavior,
and pin the exact Day 13 full-sweep command set before closeout.

### Commands Run

1. Re-read the Day 12 plan and the live parity-contract surfaces:
   - `sed -n '240,360p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `sed -n '1,260p' docs/planning/EPIC_3/SPRINT_36/artifacts/day10-cross-platform-parity-report.md`
   - `sed -n '1,220p' docs/planning/EPIC_3/SPRINT_36/artifacts/day11-final-parity-consistency-pass.md`
   - `tail -n 120 docs/planning/EPIC_3/SPRINT_36/WORKING_NOTES.md`
2. Validate the current workflow/YAML and platform-focused local commands:
   - `ruby -e 'require "yaml"; %w[.github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml].each { |p| YAML.load_file(p); puts "yaml_ok #{p}" }'`
   - `make wall-check`
   - `make quality-review-compile`
   - `make quality-review-cmake-compile`
   - `make deadcode-report`
   - `make deadcode-check`
   - `make sanitize`
3. Record the Day 12 validation note and the exact Day 13 full-sweep command
   set.

### Day 12 Findings

#### 1. The reviewed local paths Sprint 36 aligned still pass live

The local reviewed-quality commands that matter most for Sprint 36 all passed:

- `make quality-review-compile`
- `make quality-review-cmake-compile`
- `make deadcode-report`
- `make deadcode-check`

Interpretation:

- the sprint’s cross-platform contract work did not break the maintained local
  reviewed entry points
- the Linux-style enforced baseline remains real, not just described

#### 2. The CMake parity contract is still exact

`make quality-review-cmake-compile` still reported:

- `ctest -N`: `53`
- Makefile tests: `53`
- CMake tests: `53`
- `PASS: test counts match`

Interpretation:

- the strongest honest cross-platform reviewed baseline is unchanged
- Sprint 36 did not drift the audited active-suite parity count while updating
  workflow and README contract wording

#### 3. The staged dead-code story is still truthful

The dead-code commands passed in their supported serialized local mode:

- `make deadcode-report`
- `make deadcode-check`

Interpretation:

- Sprint 36 still describes dead-code honestly:
  - enforced on Linux
  - staged on macOS
  - excluded on Windows
- Day 12 did not uncover any basis for broadening that claim prematurely

#### 4. The macOS-side helper/supporting path still has live evidence

Two Apple-Clang-adjacent supporting validations also passed:

- `make wall-check`
- `make sanitize`

`make wall-check` ended with `wall-check: PASS`.

`make sanitize` completed cleanly and ended with `All tests passed.` The long
tail of the suite stayed green through the heavier reordering, CSC, eigensolver,
graph, and framework tests.

Interpretation:

- the Sprint 36 wording around enforced Apple Clang reviewed paths and live
  supplemental helper coverage remains grounded in working commands
- no new platform regression surfaced in the local validation pass

#### 5. Workflow structure remains syntactically valid

Ruby YAML loading still passed for:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Interpretation:

- the Day 5/6/9/11 CI alignment work remains structurally sound
- the final sweep does not need to reopen workflow syntax concerns

### Day 12 Interpretation

- Day 12 succeeded because Sprint 36’s alignment work stayed attached to real
  maintained commands instead of introducing a speculative portability layer.
- The main result is that the Day 10 parity report is still operationally
  truthful when checked against live commands.
- No new parity debt or portability queue reopened, so Day 13 can stay a clean
  full validation sweep.

### Day 12 Outputs

- `artifacts/day12-platform-focused-validation.md`

### Day 13 Planned Full Sweep

- `make format`
- `make lint`
- `make test`
- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

## Day 13

**Objective:** Run the full Sprint 36 validation set, capture the measured end
state, and re-confirm that the reviewed local/CMake baseline still holds after
the parity and portability work.

### Commands Run

1. Re-read the Day 13 plan and the Day 12 validation record:
   - `sed -n '300,360p' docs/planning/EPIC_3/SPRINT_36/PLAN.md`
   - `sed -n '1,240p' docs/planning/EPIC_3/SPRINT_36/artifacts/day12-platform-focused-validation.md`
   - `git status --short --branch`
2. Run the full validation set:
   - `/usr/bin/time -p make format`
   - `/usr/bin/time -p make lint`
   - `/usr/bin/time -p make test`
   - `/usr/bin/time -p make quality-review-compile`
   - `/usr/bin/time -p make quality-review`
   - `/usr/bin/time -p make quality-review-cmake-compile`
   - `/usr/bin/time -p make quality-review-cmake`
3. After the first `make lint` failure, reset the stale sanitizer build tree and
   rerun from a clean baseline:
   - `make clean`

### Day 13 Findings

#### 1. The full maintained command set passed after the clean-baseline rerun

The authoritative rerun passed completely:

- `make format`
- `make lint`
- `make test`
- `make quality-review-compile`
- `make quality-review`
- `make quality-review-cmake-compile`
- `make quality-review-cmake`

Measured wall times:

- `make format`: `real 3.31`
- `make lint`: `real 303.32`
- `make test`: `real 264.17`
- `make quality-review-compile`: `real 696.45`
- `make quality-review`: `real 487.59`
- `make quality-review-cmake-compile`: `real 93.11`
- `make quality-review-cmake`: `real 817.17`

Interpretation:

- Sprint 36 ends with the full maintained local validation surface still green
- both the direct commands and the wrapper entry points remain valid

#### 2. The one Day 13 failure was a stale sanitizer artifact, not a source regression

The first `make lint` attempt failed because the prior Day 12 `make sanitize`
pass had left a UBSan-instrumented `build/libsparse_lu_ortho.a` in place, and
the benchmark link step then tried to reuse it without the UBSan runtime.

The fix was simply:

- `make clean`

Then the full Day 13 sweep was rerun successfully.

Interpretation:

- this was an operational build-tree issue, not a Sprint 36 code or workflow
  regression
- Day 14 should carry a small note that direct authoritative sweeps should
  start from a clean `build/` tree if a sanitizer path ran immediately before

#### 3. The reviewed CMake parity contract remains exact

`make quality-review-cmake-compile` again reported:

- `ctest -N`: `53`
- Makefile tests: `53`
- CMake tests: `53`
- `PASS: test counts match`

`make quality-review-cmake` completed the full suite successfully:

- `53 / 53` tests passed
- `Total Test time (real) = 703.03 sec`

Interpretation:

- the strongest honest cross-platform reviewed baseline is unchanged
- Sprint 36 did not drift the active-suite parity story while aligning CI,
  README, and workflow naming

#### 4. The reviewed wrapper contract is still fully live

Wrapper end states:

- `quality-review-compile: passed (format-check + lint)`
- `quality-review: passed (format-check + lint + test + deadcode-check)`
- `quality-review-cmake-compile: passed (configure + clean rebuild + ctest -N + test-count parity)`
- `quality-review-cmake: passed (configure + clean rebuild + ctest -N + ctest)`

Interpretation:

- the Sprint 34 reviewed wrapper contract survived Sprint 36 intact
- the platform-parity work improved truthfulness without weakening the
  maintained operator entry points

#### 5. The inherited baseline invariants remain intact

The Day 13 end state still preserves the important inherited invariants:

- active CTest registry remains `53`
- Makefile/CMake test-count parity remains `53` vs `53`
- `deadcode-check` still passes inside the reviewed local wrapper path
- the Sprint 32 opt-in/self-check surface remains live through
  `test_framework_optin`

Interpretation:

- Sprint 36 did not regress the validated baseline it inherited from Sprints
  32, 34, and 35

### Day 13 Interpretation

- Day 13 succeeded because Sprint 36 stayed bounded to contract alignment and
  portability truthfulness, not feature churn.
- The only operational lesson is narrow and manageable:
  sanitizer-built artifacts can contaminate a later direct lint sweep unless the
  build tree is cleaned first.
- Outside that one note, Sprint 36 is now in a clean validated state for
  closeout.

### Day 13 Outputs

- `artifacts/day13-full-validation-sweep.md`
