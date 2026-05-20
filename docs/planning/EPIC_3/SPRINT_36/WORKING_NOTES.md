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
