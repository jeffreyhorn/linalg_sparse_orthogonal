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
