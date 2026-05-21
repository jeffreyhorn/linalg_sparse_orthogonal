# Sprint 36 Day 7: Script & Target Portability Audit

## Scope

Audit the maintained reviewed-quality and dead-code helper surfaces for shell,
path, environment, and tool-discovery assumptions that still bind the quality
flow to POSIX-first behavior.

Files reviewed:

- `Makefile`
- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- `scripts/ci.sh`
- `scripts/wall_check.sh`
- `scripts/epic3_warning_workflow.sh`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

## Main Result

Sprint 36 does not have a hidden “portable reviewed path missing” problem.
It already has one:

- reviewed CMake parity

That is the path Linux, macOS, and Windows can all express honestly today.

The remaining portability debt is mostly outside that path:

- Makefile reviewed wrappers are still POSIX-maintainer flows
- dead-code remains Linux-enforced and platform-staged elsewhere
- maintenance helpers like `wall-check` and the warning workflow are Unix tools,
  not cross-platform reviewed gates

## Findings

### 1. Reviewed CMake parity is the real cross-platform baseline

Current platform story:

- Linux:
  - `make quality-review-cmake`
- macOS Apple Clang:
  - `make quality-review-cmake`
- Windows MSVC:
  - reviewed CMake configure/build
  - `ctest -N`
  - full `ctest`

Implication:

- cross-platform parity should continue to anchor on the CMake reviewed path
- Sprint 36 does not need to retrofit identical Makefile commands onto Windows
  to claim honest parity

### 2. The Makefile reviewed-quality path still assumes POSIX tools

Observed assumptions in `Makefile`:

- shell loops:
  - `for t in ...; do ...; done`
  - `for b in ...; do ...; done`
- inline `awk` parsing:
  - `quality-review-cmake-compile`
  - coverage thresholds
- `find`-based file discovery:
  - `format`
  - `format-check`
  - `lint`
- explicit Unix binaries:
  - `/bin/rm`
  - `/bin/mkdir`
- explicit `bash` script execution:
  - `warning-workflow`
  - `deadcode`

Implication:

- the reviewed Makefile contract is currently portable across Linux/macOS but
  not a truthful Windows claim
- that is a real staged gap, but not a sign that the core product code lacks
  Windows portability

### 3. Dead-code remains intentionally Linux-first

Observed constraints:

- `scripts/deadcode_workflow.sh` is a `bash` workflow
- requires:
  - `cppcheck`
  - `python3`
  - `xunused`
- Darwin-specific SDK/resource-dir handling is already present
- Linux CI still compiles and installs `xunused` from source
- shared-path constraint remains:
  - `build/deadcode-cmake`
  - `build/deadcode/`
- compile-db coverage gap remains:
  - `bench_svd`
  - `example_basic_solve`
  - `example_condition`
  - `example_iterative`
  - `example_least_squares`
  - `example_matrix_free`
  - `example_svd_lowrank`

Implication:

- dead-code should stay:
  - enforced on Linux
  - staged on macOS
  - excluded on Windows
- any stronger claim would be less truthful than the current contract

### 4. `wall-check` and warning-capture are POSIX maintenance helpers

Observed assumptions:

- `scripts/wall_check.sh`
  - `bash`
  - `mktemp`
  - `awk`
- `scripts/epic3_warning_workflow.sh`
  - `bash`
  - `awk`
  - `date`
  - POSIX directory cleanup and shell flow

Implication:

- these tools are valid Unix maintainer workflows
- they are not a useful first-pass Windows parity target
- Sprint 36 should classify them explicitly instead of silently treating them
  as universal quality gates

## Portability Queue For Day 8 / Day 9

### Day 8: Portability Batch I

Target the highest-value reviewed-path portability debt:

- reduce avoidable POSIX-only parsing inside the maintained reviewed Makefile
  path where practical
- make intentional Unix-only helper assumptions explicit where they remain
- preserve the current Linux/macOS reviewed behavior while tightening the
  contract language

### Day 9: CI / Reporting Alignment

Make the platform contract explicit in reporting:

- `enforced`
- `staged`
- `excluded`

Especially for:

- Makefile reviewed wrappers
- reviewed CMake parity
- dead-code
- maintenance-only helpers

## Conclusion

The Sprint 36 portability problem is not “the repo is not cross-platform.”
It is:

- the portable reviewed baseline already exists on the CMake side
- the older Makefile/dead-code/helper surfaces still carry POSIX assumptions
- only part of that surface deserves actual portability fixes this sprint

That is a much narrower and safer queue than a broad “Windows-ize all local
tooling” effort.
