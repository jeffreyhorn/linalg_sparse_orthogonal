# Sprint 99 Day 5: Package, Usability, and Workflow Evidence

## Purpose

Day 5 executes the package/install/export proof commands and inspects public,
maintainer, benchmark, and workflow claim surfaces against the Day 3 final
comparison scope. The goal is to verify the static-first package story,
documentation coherence, and platform-scope fences before the Day 6 final-fix
decision.

## Environment

- Date: 2026-06-30
- Branch: `sprint-99`
- Host context: local macOS development tree

Platform-specific workflow syntax remains finally proven by CI. Day 5 checks
the source-of-truth workflow text, expected-count assertions, and local
Unix-side package proof commands.

## Package and Install/Export Proof

### Make Install/Export

Command:

```sh
bash tests/test_install.sh
```

Result:

- 14 passed
- 0 failed
- final line: `ALL INSTALL TESTS PASSED`

Captured pass signals:

- static library installed
- no shared-library artifacts installed
- all 19 headers installed
- `pkg-config` file installed
- `pkg-config --cflags` returns include path
- `pkg-config --libs` returns library flag
- `pkg-config --modversion` returns `2.2.0`
- basic `pkg-config` consumer compiles, links, and runs
- maintained example source compiles and runs with the installed package
- uninstall removes library, headers, and pkg-config file

Classification:

- closeout-ready for the maintained static-first Make install/export proof
- residual/non-claim for shared-library package maturity and dynamic ABI
  guarantees

### CMake Install/Export and Consumer Proof

Command:

```sh
bash tests/test_cmake_install.sh
```

Result:

- 16 passed
- 0 failed
- 0 skipped
- final line: `ALL CMAKE INSTALL TESTS PASSED`

Captured pass signals:

- CMake configure, build, and install passed
- static library installed
- no shared-library artifacts installed
- 19 headers installed
- `SparseConfig.cmake` installed
- `SparseConfigVersion.cmake` installed
- `SparseTargets.cmake` installed
- `sparse.pc` installed
- `examples/cmake_example` configures with `find_package(Sparse)`
- `examples/cmake_example` builds and runs
- exact installed version lookup works
- mismatched version is rejected
- `pkg-config` version is `2.2.0`

Classification:

- closeout-ready for the maintained CMake install/export and consumer proof
- residual/non-claim for Windows install-validation and package-manager
  integration

## Documentation and Usability Scans

### Stale-Claim Scan

Command:

```sh
rg -n "best-in-class|benchmark supremacy|full platform parity|shared-library-first|dynamic ABI|broad complex|mixed-precision|portable timing|universal .*superiority|all solver" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md include .github/workflows
```

Result:

- no positive unsupported product claims found
- matches were negative/boundary language:
  - `docs/maintainer_guide.md` notes `portable timing gate` as something not
    claimed
  - `docs/maintainer_guide.md` notes canonical benchmark output must not be
    read as a portable timing claim

Classification:

- closeout-ready docs/usability surface
- no final-fix candidate

### Boundary Scan

Command:

```sh
rg -n "shared.*library|dynamic.*ABI|complex support|mixed precision|platform parity|timing superiority|benchmark.*superiority|best.*class|reviewed.*Windows|reviewed.*macOS|source-of-truth|static-first" README.md INSTALL.md benchmarks/README.md docs/maintainer_guide.md include .github/workflows
```

Result:

- static-first package contract is explicit in `INSTALL.md`,
  `docs/maintainer_guide.md`, workflows, and README CI text
- dynamic ABI and shared-library language appears as a non-claim or future
  separate product decision
- complex support language appears as an explicit non-claim in public headers
  and maintainer guidance
- platform parity language appears as a negative fence, not as a product claim
- Linux source-of-truth and macOS/Windows scope wording are present

Classification:

- closeout-ready claim-boundary surface
- no final-fix candidate

## Workflow and CI Scope Evidence

### Windows Reviewed Count

Workflow source:

- `.github/workflows/windows-ci.yml`

Current assertion:

- `EXPECTED_WINDOWS_CTEST_COUNT: "51"`

Source registration check:

- `CMakeLists.txt` contains 54 `add_sparse_test(...)` call sites.
- Windows excludes exactly three staged tests:
  - `test_threads`
  - `test_sprint4_integration`
  - `test_fuzz`

Expected count:

- `54 - 3 = 51`

Classification:

- closeout-ready Windows reviewed CMake subset assertion
- no final-fix candidate

### Platform Scope Wording

Checked surfaces:

- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

Result:

- Linux remains documented as the strongest reviewed source of truth.
- macOS remains the reviewed Apple Clang path plus supplemental confidence
  lanes, not full install/export parity.
- Windows remains the reviewed CMake-first consumer subset, not Makefile
  parity or a separate install-validation lane.
- `test_fuzz` remains explicitly outside reviewed Windows evidence.

Classification:

- closeout-ready workflow/platform-scope wording
- residual/non-claim for symmetric platform parity, Windows Makefile parity,
  and Windows install-validation

## Day 5 Classification Summary

| Evidence lane | Status | Reason |
|---|---|---|
| Make install/export | Closeout-ready | `tests/test_install.sh` passed 14/0 and confirmed static-first package shape |
| CMake install/export | Closeout-ready | `tests/test_cmake_install.sh` passed 16/0/0 and confirmed CMake consumer proof |
| public and maintainer docs | Closeout-ready | stale-claim scan found only negative guardrails |
| benchmark documentation | Closeout-ready | benchmark timing language remains threshold-free and non-portable |
| workflow scope | Closeout-ready | platform asymmetry is explicit and current |
| Windows expected CTest count | Closeout-ready | 54 CMake tests minus 3 staged Windows exclusions equals expected 51 |
| shared-library package maturity | Residual/non-claim | scripts explicitly assert no shared-library artifacts |
| dynamic ABI guarantee | Residual/non-claim | documented as not reviewed |
| symmetric platform parity | Residual/non-claim | Linux/macOS/Windows roles remain intentionally scoped |

## Final-Fix Candidates

Day 5 produced no final-fix candidates.

The evidence supports the Day 3 allowed language:

- Static-first install/export and CMake consumer proof are maintained and
  validated.
- Linux remains the strongest reviewed source of truth.
- macOS and Windows have intentionally narrower reviewed or supplemental proof
  roles.

The evidence does not support any widened shared-library, dynamic ABI,
platform parity, or portable benchmark/timing claim.
