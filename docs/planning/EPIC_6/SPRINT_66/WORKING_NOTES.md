# Sprint 66 Working Notes

## Day 2 - Validation Baseline & Install/Platform Rerun Recheck

### Goal

Reconfirm the reviewed baseline and rerun set that Sprint 66 packaging, ABI,
install, workflow, and platform-quality changes must preserve before any
implementation work lands.

### Actions

1. Rechecked the reviewed CMake parity anchor:
   - `ctest -N --test-dir build/quality-review-cmake`
2. Dry-ran the current strongest reviewed baseline wrapper:
   - `make -n quality-review-full`
3. Re-read the live Sprint 66 plan section and current Epic 6 handoff state
   from the merged Sprint 65 close.
4. Reconfirmed the current build-tree availability of the most relevant Sprint
   66 proof surfaces:
   - direct and CSC proof binaries
   - representative examples
   - canonical maintained benchmark binaries
5. Fixed the authoritative validation split for docs-only, bounded code-day,
   and substantial packaging/platform work.

### Findings

#### 1. The strongest reviewed baseline is unchanged at Sprint 66 start

The strongest local reviewed baseline is still:

- `make quality-review-full`

The reviewed CMake parity anchor remains exact:

- `ctest -N --test-dir build/quality-review-cmake` = `53`

Interpretation:

- Sprint 66 starts from the same reviewed truthfulness baseline as the Sprint
  65 close
- packaging, ABI, workflow, and platform work do not get a weaker local
  validation contract just because the main sprint topic is productization

#### 2. The Day 2 authority split is now explicit

The authoritative split for Sprint 66 is:

- bounded `*.c` / `*.h` days:
  - `make format`
  - `make lint`
  - `make test`
- stronger default for substantial packaging, install/export, workflow, or
  platform-quality work:
  - `make quality-review-full`
- docs-only days:
  - targeted sanity checks only

Interpretation:

- Sprint 66 should treat build/install/workflow/platform touches as closer to
  contract-sensitive work than to ordinary docs edits
- the stronger reviewed baseline remains the default for any change that could
  distort packaging or platform truthfulness

#### 3. The high-signal Sprint 66 rerun set is now fixed around the actual productization-risk surface

The high-signal rerun set at Sprint 66 start is:

- direct lifecycle and CSC proof surfaces:
  - `./build/test_integration`
  - `./build/test_sparse_lu`
  - `./build/test_cholesky`
  - `./build/test_ldlt`
  - `./build/test_chol_csc`
  - `./build/test_ldlt_csc`
- adjacent numerical sentinels that should not drift under build/install work:
  - `./build/test_qr`
  - `./build/test_svd`
- representative examples:
  - `./build/example_analysis`
  - `./build/example_basic_solve`
  - `./build/example_ldlt`
  - `./build/example_svd_lowrank`
- canonical maintained benchmark surfaces:
  - `./build/bench_refactor`
  - `./build/bench_refactor_csc`
  - `./build/bench_chol_csc`
  - `./build/bench_ldlt_csc`
  - `./build/bench_iterative_reuse`
  - `./build/bench_eigs_reuse`

Interpretation:

- the Sprint 66 rerun set is anchored to productization-sensitive user and
  maintainer proof surfaces rather than to every executable in the repo
- the canonical maintained benchmark lane from Sprint 65 remains part of the
  live Sprint 66 validation story

#### 4. The strongest likely Sprint 66 touch surfaces remain packaging and workflow truth surfaces, not solver APIs

The highest-signal likely Sprint 66 touch surfaces at Day 2 are:

- packaging/install/build:
  - `CMakeLists.txt`
  - `Makefile`
  - `INSTALL.md`
- workflow/platform truth surfaces:
  - `.github/workflows/ci.yml`
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `README.md`
  - `docs/maintainer_guide.md`
- likely narrow version/error or contract-adjacent headers only if the audit
  proves they need touching:
  - `include/sparse_types.h`
  - `include/sparse_cholesky.h`
  - `include/sparse_ldlt.h`

Measured hotspot sizes at Sprint 66 start:

- `README.md` = `1000`
- `INSTALL.md` = `206`
- `docs/maintainer_guide.md` = `511`
- `CMakeLists.txt` = `397`
- `Makefile` = `897`
- `.github/workflows/ci.yml` = `221`
- `.github/workflows/windows-ci.yml` = `57`
- `.github/workflows/macos-ci.yml` = `111`
- `include/sparse_types.h` = `233`
- `include/sparse_cholesky.h` = `232`
- `include/sparse_ldlt.h` = `334`

Interpretation:

- Sprint 66 still starts from a productization and workflow surface, not a
  broad solver-implementation surface
- the heaviest likely touched truth surfaces are already explicit before the
  packaging audit begins

### Day 2 Close

Sprint 66 now has:

- one explicit reviewed validation contract for packaging and platform work
- one fixed rerun set centered on productization-sensitive proofs and canonical
  maintained benchmarks
- one clear Day 3 starting point for the packaging and ABI surface audit
