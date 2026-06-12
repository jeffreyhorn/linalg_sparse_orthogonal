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

## Day 3 - Packaging and ABI Surface Audit

### Goal

Reduce Sprint 66's broad packaging and ABI question to the live repo seams that
actually define product maturity today: release shape, install/export truth,
versioning signals, and platform-claim asymmetries.

### Actions

1. Re-read the current build/install/export surface in `CMakeLists.txt`.
2. Re-read the user-facing install contract in `INSTALL.md` and the top-level
   packaging claims in `README.md`.
3. Re-read the maintainer and workflow truth surfaces most likely to constrain
   any later packaging or platform work:
   - `docs/maintainer_guide.md`
   - `.github/workflows/windows-ci.yml`
   - `.github/workflows/macos-ci.yml`
4. Ran targeted `rg` scans across the build/docs/workflow surfaces for:
   - install/export/package config
   - static/shared wording
   - version and downstream-consumption claims
   - Windows/macOS install-path verification
5. Re-ranked the likely Sprint 66 first implementation target from the live
   repo state instead of from generic productization language.

### Findings

#### 1. The repo already has a real install/export surface; the strongest gap is not "missing packaging"

The current packaging surface is materially real:

- `CMakeLists.txt` installs the library target, public headers, generated
  `sparse_version.h`, CMake package config files, and a `pkg-config`
  descriptor
- `INSTALL.md` documents both Makefile install and CMake install flows
- `README.md` advertises downstream `pkg-config` and `find_package(Sparse)`
  consumption
- macOS CI already includes a supplemental install and `pkg-config`
  verification lane

Interpretation:

- Sprint 66 should not behave as if the repo has no packaging story
- the live repo already supports credible developer-install consumption through
  both Make and CMake
- the real audit question is how narrow that shipped story still is

#### 2. The strongest packaging/productization gap is the static-first release shape

The current primary library target is still:

- `add_library(sparse_lu_ortho STATIC ...)`

The install/export surface is therefore real but intentionally narrow:

- static archive install is first-class
- CMake exported target is first-class
- `pkg-config` support is first-class
- a broader shared-library / ABI-distribution promise is not yet present

Interpretation:

- the strongest Day 3 gap is not "can consumers install this?"
- the strongest Day 3 gap is that the shipped release shape is still
  static-first and effectively static-only
- Sprint 66 should treat any shared-library or ABI widening as an explicit
  product decision, not as a hidden side effect of install cleanup

#### 3. The versioning source of truth is healthier than the sprint headline implies, but the ABI story is still narrow

The current versioning chain is already coherent:

- root `VERSION` file is the single source of truth
- `project(... VERSION ...)` reads from that file
- generated `sparse_version.h` is installed
- `SparseConfigVersion.cmake` is generated with `SameMajorVersion`
  compatibility
- `sparse.pc` is generated from the same project version

Interpretation:

- the versioning surface itself is not the strongest weak point
- the weaker point is that the repo still does not present a broader shared ABI
  promise that would make those version signals carry more distribution weight
- Sprint 66 should distinguish "version metadata exists" from "ABI contract is
  mature"

#### 4. The downstream-consumption story is already stronger than the static/shared story

The repo already supports two real downstream consumption paths:

- Makefile install + `pkg-config`
- CMake install + `find_package(Sparse)` + `Sparse::sparse_lu_ortho`

Platform truth is also narrower and more explicit than a generic packaging
review would suggest:

- Windows currently enforces the reviewed CMake subset only
- macOS already carries a supplemental install + `pkg-config` verification lane
- README and `INSTALL.md` already steer Windows to the CMake workflow

Interpretation:

- the strongest Day 3 platform/productization gap is not absent consumption
  paths
- the stronger remaining gap is convergence and truthfulness across those paths
- Sprint 66 should prioritize reconciling release/install claims and reviewed
  platform lanes before inventing new packaging fronts

#### 5. The likely first Sprint 66 implementation target is now explicit

From the live repo state, the strongest likely first target is:

- packaging/productization convergence around the existing static-first
  install/export surface

That means the highest-value first-touch surfaces are likely:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- reviewed workflow files only where platform truth must move with the
  packaging contract

Measured Day 3 hotspot sizes for the main packaging/ABI truth surfaces:

- `README.md` = `1000`
- `INSTALL.md` = `206`
- `docs/maintainer_guide.md` = `511`
- `CMakeLists.txt` = `397`
- `.github/workflows/windows-ci.yml` = `57`
- `.github/workflows/macos-ci.yml` = `111`

Interpretation:

- Sprint 66's first landing should start with release/install truth and
  bounded build-surface convergence
- broader shared-library ambition, ABI widening, and platform follow-through
  should only happen where the audit proves they are justified

### Day 3 Close

Sprint 66 now has:

- one explicit packaging/ABI baseline grounded in the live install/export
  surface
- one ranked gap map that separates "real install support exists" from
  "release shape is still narrow"
- one clear Day 4 starting point for the platform-residual recheck and the
  later packaging/productization batch

## Day 4 - Platform Residual Recheck

### Goal

Reassess the live macOS, Windows, and dead-code residual queue against the
current reviewed truthfulness contract so Sprint 66 carries only the bounded
platform/productization work that is actually justified.

### Actions

1. Re-read the compact cross-platform truth surfaces in:
   - `README.md`
   - `docs/maintainer_guide.md`
   - `INSTALL.md`
   - `Makefile`
   - `.github/workflows/ci.yml`
   - `.github/workflows/macos-ci.yml`
   - `.github/workflows/windows-ci.yml`
2. Re-read the current dead-code workflow topology:
   - `Makefile`
   - `scripts/deadcode_workflow.sh`
   - `scripts/deadcode_report.py`
3. Re-ranked the active residuals into:
   - real Sprint 66 platform-quality work
   - operational limits that remain explicit but should not drive the sprint
   - later stretch/non-goal platform work
4. Checked whether the current Windows and macOS lanes already verify the
   install/package claims from the Day 3 packaging audit.
5. Fixed the first platform/dead-code target set in writing before Day 5-6
   design begins.

### Findings

#### 1. The platform contract is already intentionally asymmetric, and that asymmetry is truthful

The current cross-platform contract is already explicit:

- Linux enforces:
  - reviewed Makefile compile-quality path
  - reviewed CMake parity path
  - dead-code report and completeness path
- macOS enforces:
  - reviewed Apple Clang quality path
  - reviewed CMake parity path
  - supplemental install and `pkg-config` verification
  - dead-code remains staged
- Windows enforces:
  - reviewed CMake configure/build/`ctest -N`/full `ctest`
  - Makefile reviewed wrappers remain staged
  - dead-code remains staged

Interpretation:

- Sprint 66 should not treat the current asymmetry as accidental drift
- the live repo already distinguishes enforced, staged, and supplemental lanes
  on purpose
- the correct question is which staged limits are still productization-relevant
  enough to touch now

#### 2. The strongest Day 4 platform gap is not "missing Windows Makefile parity"

Windows still routes to the reviewed CMake subset only, and that is consistent
across:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/windows-ci.yml`

Interpretation:

- the Windows reviewed-wrapper gap is real only if the repo were claiming a
  reviewed Windows Makefile path
- it is not the strongest Sprint 66 contradiction because the repo already says
  Windows should use the CMake workflow exclusively
- forcing Windows Makefile parity in Sprint 66 would be broader build-surface
  expansion, not bounded productization cleanup

#### 3. macOS dead-code remains a real residual, but still reads as staged-by-design rather than the first implementation target

The current macOS contract still says:

- dead-code is staged pending fresh measurement
- Apple Clang and Homebrew GCC cover build/test/wall/sanitize and install
  support, but not the dead-code workflow

Interpretation:

- macOS dead-code is still a real residual
- the repo does not currently present fresh measurement or a maintained macOS
  dead-code toolchain path that would justify claiming closure
- this should remain a bounded residual unless later Sprint 66 work proves a
  narrower, measurement-backed change is affordable

#### 4. Windows dead-code is also still a real residual, but weaker than the packaging/productization lane

Windows currently keeps dead-code staged rather than reviewed, and the active
dead-code workflow still depends on:

- a generated compile database
- `bash`
- `python3`
- `cppcheck`
- `xunused`
- one serialized shared-path artifact topology

Interpretation:

- Windows dead-code is not just a missing CI step
- it is tied to the current Linux-centered dead-code execution model
- that makes it a weaker first Sprint 66 target than the packaging/install
  contract itself

#### 5. Serialized dead-code execution remains the strongest real operational limit

The dead-code workflow still shares:

- `build/deadcode-cmake`
- `build/deadcode/`

And the maintained docs/workflow surfaces still state that the `deadcode*`
targets must run serially.

Interpretation:

- serialized dead-code execution remains the clearest active operational limit
- but it is still an execution-topology limit, not automatically the highest
  productization target
- Sprint 66 should keep that limit explicit and only touch it if a later batch
  can improve truthfulness without widening into a broad dead-code redesign

#### 6. The first platform/dead-code target set is now explicit

From the live repo state, the highest-value platform follow-through set is:

- docs/workflow/contract reconciliation around the staged platform lanes
- install/package/platform truth alignment where the packaging batch changes the
  released story
- bounded dead-code/platform wording cleanup where current residual language is
  still too generic

The weaker or deferred set is now explicit too:

- Windows Makefile reviewed-wrapper parity
- Windows dead-code enforcement
- macOS dead-code enforcement
- broad dead-code topology redesign
- fake cross-platform closure beyond measured reviewed evidence

Measured Day 4 hotspot sizes for the main platform/dead-code truth surfaces:

- `README.md` = `1000`
- `docs/maintainer_guide.md` = `511`
- `INSTALL.md` = `206`
- `Makefile` = `897`
- `.github/workflows/ci.yml` = `221`
- `.github/workflows/macos-ci.yml` = `111`
- `.github/workflows/windows-ci.yml` = `57`

Interpretation:

- the strongest Sprint 66 platform work is still contract convergence, not a
  platform-expansion sprint
- Day 5-6 design should stay bounded to packaging/productization plus explicit
  staged-lane reconciliation

### Day 4 Close

Sprint 66 now has:

- one sharper platform/dead-code residual map
- one explicit split between real Sprint 66 platform-quality work and deferred
  platform-expansion work
- one fixed starting point for the Day 5 packaging design and Day 6 platform
  follow-through design
