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

## Day 5 - Packaging and Productization Design

### Goal

Define the maintained Sprint 66 packaging, install, export, and release-shape
contract tightly enough that later implementation can improve product maturity
without inventing a broader ABI or cross-platform promise than the repo can
actually support.

### Actions

1. Reconciled the Day 3 packaging audit with the Day 4 platform residual map.
2. Fixed the intended Sprint 66 packaging contract across:
   - release shape
   - install/export consumer story
   - versioning and ABI wording
   - platform claim boundaries
3. Separated what belongs to:
   - build files
   - install docs
   - top-level README wording
   - maintainer policy
   - workflows and regression checks
4. Ranked which possible widenings are justified now versus explicitly deferred.
5. Fixed the first implementation fence for the Day 6-10 landing set.

### Findings

#### 1. The maintained Sprint 66 packaging contract should stay static-first unless a stronger proof burden is accepted explicitly

The current release shape is already:

- installable
- exported through CMake package files
- consumable through `pkg-config`
- intentionally static-first

The Day 5 design fixes that as the maintained Sprint 66 default:

- static archive install remains first-class
- CMake export/install remains first-class
- `pkg-config` consumption remains first-class
- no broad shared-library or SONAME-style promise is implied by default

Interpretation:

- Sprint 66 should improve productization around the shipped static-first
  surface, not silently pivot the repo into a shared-library sprint
- if any shared-library or ABI widening ever happens, it must be a separate
  explicit promise with its own validation and platform ownership

#### 2. The install/export consumer story should be tightened, not reinvented

The current downstream-consumption story is already good enough to preserve:

- Makefile install for Unix-like consumers
- `pkg-config` downstream consumption
- CMake install/export with `find_package(Sparse)`
- Windows CMake-first consumption

The Day 5 design therefore fixes the consumer contract as:

- keep the current consumer paths
- tighten wording where docs could read broader than the actual reviewed shape
- make the static-first install/export story read intentionally productized
  rather than incidentally available

Interpretation:

- Sprint 66 should converge the wording of the install/export story before
  adding more surface area
- "consumer ergonomics" now means truthfulness and consistency first

#### 3. The ABI/version contract should stay narrow and explicit

The version metadata chain is already coherent, but the ABI promise is still
intentionally narrow.

The Day 5 design fixes the ABI/version contract as:

- keep `VERSION`, generated `sparse_version.h`, and package-version files as
  the authoritative version metadata chain
- do not imply stable shared-library ABI compatibility beyond the current
  static-first exported package surface
- keep `SameMajorVersion` as package-config metadata, not as a broader release
  guarantee than the repo actually validates

Interpretation:

- Sprint 66 should improve clarity around what the version metadata does and
  does not promise
- the design should avoid wording that over-reads CMake package-version support
  into a broad binary-compatibility claim

#### 4. The platform truth fence stays explicit

The Day 5 design preserves the current platform boundary:

- Linux remains the strongest reviewed source of truth
- macOS remains reviewed but narrower, with supplemental install validation
- Windows remains the reviewed CMake subset and install-consumer lane
- dead-code asymmetries remain staged and explicit

Interpretation:

- packaging work must not imply stronger platform closure than the workflows
  actually review
- a more polished install/release story is acceptable only if it still reads
  truthfully through those platform boundaries

#### 5. Ownership of the converged packaging story is now explicit

The Day 5 ownership split is:

- `CMakeLists.txt`:
  - library release shape
  - install/export topology
  - package-config generation truth
- `INSTALL.md`:
  - operator-facing install and downstream-consumption instructions
  - platform-specific install-path caveats
- `README.md`:
  - compact top-level packaging/productization story
  - compact downstream consumption summary
- `docs/maintainer_guide.md`:
  - authoritative interpretation of the narrow ABI/platform contract
  - what remains staged or deferred
- workflows and regression checks:
  - only the reviewed evidence for the claimed install/platform lanes

Interpretation:

- Sprint 66 should not let the packaging story drift across five surfaces with
  conflicting strength of claim
- docs and workflows should state one converged contract, with the build files
  remaining the executable truth

#### 6. The first implementation fence is now fixed

The highest-value first implementation set is:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`

Likely support or reconciliation surfaces only if the landing proves they are
needed:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Explicitly not part of the first packaging batch:

- broad shared-library enablement
- broad ABI guarantee widening
- Windows Makefile reviewed-wrapper parity
- macOS dead-code enablement
- Windows dead-code enablement
- dead-code topology redesign

Interpretation:

- Day 6 should begin with bounded packaging/productization convergence on the
  current install/export surface
- Day 7+ can absorb workflow or install-regression follow-through only where
  the first landing actually changes the contract

### Day 5 Close

Sprint 66 now has:

- one explicit packaging/productization contract
- one fixed static-first safety fence
- one clear ownership split across build files, docs, maintainer policy, and
  workflows
- one bounded Day 6-10 implementation fence for the first landing

## Day 6 - Platform and Dead-Code Follow-Through Design

### Goal

Convert the remaining platform and dead-code residual queue into one bounded
implementation plan that stays inside the reviewed truth fence and names the
later proof surfaces precisely.

### Actions

1. Reconciled the Day 4 residual map with the Day 5 packaging contract.
2. Re-read the focused install/package regression homes:
   - `tests/test_install.sh`
   - `tests/test_cmake_install.sh`
3. Fixed which platform/dead-code residuals actually move in Sprint 66 and
   which stay deferred.
4. Defined what each bounded follow-through batch should prove:
   - workflow truthfulness
   - staged-lane wording alignment
   - install/package regression ownership
   - bounded operational cleanup only where the packaging batch moves the
     contract
5. Fixed the exact platform-quality implementation fence and the later
   regression-coverage shortlist.

### Findings

#### 1. The strongest Sprint 66 platform follow-through is contract reconciliation, not platform expansion

The Day 6 design fixes the first platform batch around:

- docs/workflow/contract alignment for enforced versus staged lanes
- packaging/install wording alignment where the Day 8+ batch changes the
  product story
- focused install/package regression support for the touched release surfaces

Interpretation:

- Sprint 66 should close the remaining contract drift first
- the platform lane is still about truthfulness and bounded regression support,
  not about forcing every staged path into the reviewed baseline

#### 2. The residuals that move in Sprint 66 are now explicit

The bounded residual set that may move in Sprint 66 is:

- wording alignment across:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - relevant workflow comments/job labels
- install/package regression ownership where the packaging batch changes the
  shipped contract
- narrow Makefile or workflow follow-through only if the packaging landing
  changes the reviewed command story materially

Interpretation:

- Sprint 66 can legitimately improve platform-quality clarity without claiming
  new enforcement lanes
- the moved set is now narrow enough to support a bounded Day 7 landing fence

#### 3. The residuals that stay deferred are also fixed explicitly

The deferred set remains:

- Windows Makefile reviewed-wrapper parity
- Windows dead-code enforcement
- macOS dead-code enforcement
- broad dead-code topology redesign
- broad wrapper redesign beyond the audited seams
- fake cross-platform closure beyond reviewed evidence

Interpretation:

- Sprint 66 should not consume these items just because they remain unsolved
- later work may revisit them, but the current sprint should keep them visible
  rather than silently half-solving them

#### 4. The bounded follow-through batch now has a proof contract

Each bounded follow-through batch should prove one of:

- reviewed workflow truthfulness:
  - comments, job names, and docs still match what Linux/macOS/Windows
    actually enforce
- install/package regression truth:
  - Make/pkg-config install path still works through `tests/test_install.sh`
  - CMake install/export/find-package path still works through
    `tests/test_cmake_install.sh`
- bounded operational cleanup:
  - only where the packaging batch changes the touched command or workflow
    story directly

Interpretation:

- later verification should stay attached to concrete regression surfaces
- Sprint 66 does not need a new generic platform-proof harness

#### 5. The exact implementation fence is now fixed

Required platform/dead-code follow-through surfaces are now:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`

Likely support only if the landing proves they must move:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Explicit non-touch set for this lane:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- broad dead-code artifact topology
- Windows-specific Makefile wrapper support
- new platform-specific benchmark or solver validation lanes

Interpretation:

- the platform-quality batch stays document/workflow/regression centered unless
  the packaging batch proves a narrower code or script move is necessary

#### 6. The later regression-coverage shortlist is now concrete

The focused later regression shortlist is:

- `make quality-review-full`
- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`
- platform-truth sanity checks on:
  - `.github/workflows/windows-ci.yml`
  - `.github/workflows/macos-ci.yml`
  - `.github/workflows/ci.yml`

Interpretation:

- the install/package regression story is now attached to concrete existing
  proof homes
- Day 7 can define the touched-file fence and validation order without having
  to rediscover the proof surface

### Day 6 Close

Sprint 66 now has:

- one bounded platform/dead-code implementation plan
- one explicit deferred residual list
- one concrete install/package regression shortlist
- one clear Day 7 starting point for the exact landing fence

## Day 7 - Exact Landing Fence and Regression Plan

### Goal

Collapse the Day 5-6 design into one exact touched-file fence, proof plan, and
validation order so the remaining Sprint 66 implementation days can land
without improvising surface area late.

### Actions

1. Collapsed the Day 5 packaging contract and Day 6 platform follow-through
   plan into one exact Day 8-12 landing sequence.
2. Separated the touched surfaces into:
   - required first-batch files
   - optional support files only if proof burden forces them
   - explicit non-touch set for Sprint 66
3. Fixed the proof plan across:
   - reviewed baseline gates
   - install/package regression checks
   - workflow truth checks
4. Fixed the intended validation order for later code-touching and docs-only
   days.
5. Recorded the implementation sequence so the sprint can stay bounded after
   Day 7.

### Findings

#### 1. The first landing should start on packaging truth surfaces, not on workflows or regression scripts

The exact required first-batch surface is now:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`

Interpretation:

- Day 8 should resolve the highest-value packaging/productization contradiction
  on the core build/install/docs surfaces first
- workflows and regression scripts should only move if that first landing
  actually changes the reviewed contract enough to require reconciliation

#### 2. Optional support surfaces are now bounded explicitly

Optional support surfaces only if the proof burden forces them:

- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Interpretation:

- these surfaces are not part of the first batch by default
- they become valid Sprint 66 touches only if the Day 8-10 landing changes the
  shipped install/platform contract materially enough that existing workflow or
  regression wording becomes stale

#### 3. The explicit non-touch set is now fixed for Sprint 66

The explicit non-touch set is:

- `scripts/deadcode_workflow.sh`
- `scripts/deadcode_report.py`
- broad dead-code artifact topology
- Windows-specific Makefile wrapper support
- broad shared-library enablement
- broad ABI guarantee widening
- macOS dead-code enforcement
- Windows dead-code enforcement
- new platform-specific benchmark or solver validation lanes

Interpretation:

- Sprint 66 now has a concrete fence against accidental productization sprawl
- later days should not consume these items just because they remain visible in
  the residual queue

#### 4. The proof plan is now concrete and ordered

The proof plan for the remaining sprint is:

- required reviewed baseline for substantial packaging/platform work:
  - `make quality-review-full`
- focused install/package regression checks when install/export behavior or
  contract wording moves materially:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`
- workflow truth checks when workflow comments/job labels or platform-claim
  wording moves:
  - direct review of `.github/workflows/ci.yml`
  - direct review of `.github/workflows/macos-ci.yml`
  - direct review of `.github/workflows/windows-ci.yml`

Interpretation:

- proof remains attached to concrete maintained surfaces
- Sprint 66 still does not need a new platform-proof harness

#### 5. The Day 8-12 sequence is now explicit

The remaining implementation order is now:

1. Day 8:
   - first packaging/productization batch on the required build/install/docs
     surfaces
2. Day 9:
   - post-landing audit and rerank of any remaining packaging/platform
     contradictions
3. Day 10:
   - second bounded batch only if the Day 8 landing leaves one real
     contract-level contradiction unresolved
4. Day 11:
   - workflow/CI/contract reconciliation plus focused install/package
     regression support only where the landed contract requires it
5. Day 12:
   - docs and maintainer-story follow-through on the converged contract

Interpretation:

- Sprint 66 should not widen into multiple parallel implementation fronts
- each later day now has one clear ownership focus tied to the bounded fence

#### 6. The validation order is now fixed

For later `*.c` / `*.h` changes, the required minimum remains:

- `make format`
- `make lint`
- `make test`

For substantial packaging/platform/build/workflow changes, the maintained
default remains:

- `make quality-review-full`

For docs-only landing or reconciliation days:

- targeted sanity checks only

Interpretation:

- the remaining sprint days now have an explicit validation order before
  implementation resumes
- Day 13 can later close from a known proof surface rather than from an ad hoc
  rerun set

### Day 7 Close

Sprint 66 now has:

- one exact touched-file fence
- one concrete proof and validation plan
- one explicit Day 8-12 landing order
- one bounded implementation map for the rest of the sprint

## Day 8 - Packaging and Productization Batch 1

### Goal

Land the highest-value first Sprint 66 packaging/productization slice inside
the Day 7 fence: make the maintained static-first install/export contract
explicit in the build and docs surfaces, and align the focused CMake install
regression with the repo's single `VERSION` source of truth.

### Actions

1. Updated `CMakeLists.txt` so the configure path states the maintained
   static-first package contract explicitly:
   - `BUILD_SHARED_LIBS=ON` now emits an explicit status note that the shipped
     package surface still remains the static archive output
   - the library export/output naming is now set explicitly on the maintained
     target
2. Tightened the operator-facing install story in `INSTALL.md`:
   - the shipped install/export surface is now described directly as
     static-first
   - the current version metadata propagation chain is stated explicitly
   - Windows remains explicitly CMake-first
3. Tightened the top-level package/install summary in `README.md` so the
   downstream `pkg-config` and `find_package(Sparse)` story reads as one
   intentional static-first package surface instead of an implied broader
   release promise.
4. Added the matching maintainer-policy interpretation in
   `docs/maintainer_guide.md`:
   - real install/export surface
   - static-first release shape
   - narrow ABI promise
   - platform truth fence
5. Updated `tests/test_cmake_install.sh` to read the expected installed package
   version from the repo `VERSION` file instead of hardcoding `1.0.0`.

### Findings

#### 1. The highest-value first contradiction is now closed

Before Day 8:

- the repo shipped a real static-first install/export surface
- but the build/docs surfaces did not state that maintained release shape as
  directly as they should
- and the focused CMake install regression still hardcoded a package version
  instead of following the single version source of truth

After Day 8:

- the build configure path states the static-first package shape explicitly
- the user-facing install/docs surfaces state the same contract explicitly
- the maintainer guide now owns the narrow ABI/platform interpretation
- the focused CMake install regression now follows `VERSION`

Interpretation:

- the first Sprint 66 productization batch resolved a real packaging truth
  contradiction instead of just rephrasing the same ambiguity

#### 2. The landed batch stayed inside the Day 7 fence

Touched Day 8 surfaces:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `tests/test_cmake_install.sh`

Untouched Day 8 surfaces:

- workflows
- `Makefile`
- `tests/test_install.sh`
- dead-code scripts
- any shared-library or ABI-widening surface

Interpretation:

- the batch stayed inside the required first-batch surface, with only one
  optional proof surface moving because the version-source-of-truth check
  required it

#### 3. The proof burden for this landing is now explicit and passed

Because this was substantial packaging/productization work, the stronger
reviewed baseline was used:

- `make quality-review-full`

Retained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 820.96 sec`

Because the package/install contract moved materially, the focused install
regressions were also run:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained focused install/package proof points:

- Make install/uninstall path passed
- `pkg-config --modversion sparse` reported `2.2.0`
- CMake install/export/find-package path passed
- the CMake install regression now verified `pkg-config` version against the
  repo `VERSION` value instead of a stale literal

Interpretation:

- Day 8 closes from both the strongest reviewed baseline and the exact install
  proof homes fixed in the Day 7 plan

### Day 8 Close

Sprint 66 now has:

- one landed static-first packaging/productization batch
- one resolved version-source-of-truth regression contradiction
- one explicit maintained packaging contract shared across build, install, top
  level, and maintainer surfaces
- one concrete Day 9 starting point for the post-landing rerank

## Day 9 - Post-Landing Audit and Rerank

### Goal

Reassess the Sprint 66 queue after the Day 8 packaging landing and determine
whether any real packaging/productization contradiction still justifies a
second core implementation batch, or whether the remaining work has narrowed to
workflow/install-contract reconciliation.

### Actions

1. Re-read the Day 8 landing notes in this file and the Day 8 artifact to
   confirm the exact contract that actually shipped.
2. Re-ran targeted `rg` scans across the live packaging, install, maintainer,
   workflow, and focused regression surfaces for:
   - `static-first`
   - `BUILD_SHARED_LIBS`
   - `shared-library`
   - `dynamic-ABI`
   - `VERSION`
   - `pkg-config`
   - `find_package(Sparse)`
   - `reviewed CMake subset`
3. Re-read the current top-level packaging summary in `README.md` and the
   focused CMake install regression in `tests/test_cmake_install.sh`.
4. Rechecked the current Sprint 66 branch shape with:
   - `git diff --stat master...HEAD`
5. Re-ranked the remaining Sprint 66 queue from the landed repo state instead
   of assuming a second packaging batch was automatically required.

### Findings

#### 1. Day 8 closed the strongest packaging contradiction

After the Day 8 landing, the maintained package story now reads coherently
across the live build and docs surfaces:

- `CMakeLists.txt` states that the maintained package surface remains
  static-first even when `BUILD_SHARED_LIBS=ON` is requested
- `INSTALL.md` treats `make install`, `cmake --install`, `pkg-config`, and
  `find_package(Sparse)` as one intentional static archive distribution story
- `README.md` states the same compact top-level package contract directly
- `docs/maintainer_guide.md` now owns the narrow ABI/platform interpretation
- `tests/test_cmake_install.sh` now verifies installed package version against
  the repo `VERSION` file

Interpretation:

- the highest-value Day 8 target was real and is now closed
- Sprint 66 no longer has a first-order contradiction around whether the repo
  has a real install/export surface or what release shape that surface implies

#### 2. No new core packaging contradiction is visible after the Day 8 landing

The post-landing reread did not uncover a second unresolved build/install
contradiction of the same weight as the Day 8 batch:

- the release shape still intentionally stays static-first
- the version metadata chain still stays coherent and single-sourced from
  `VERSION`
- downstream `pkg-config` and `find_package(Sparse)` consumption still point at
  the same maintained archive surface
- the repo still does not imply a broader shared-library or dynamic-ABI promise

Interpretation:

- Day 10 should not invent a second packaging batch just to make the sprint
  look symmetrical
- a broad shared-library or wider ABI move would still be a separate product
  decision, not normal Sprint 66 cleanup

#### 3. The strongest remaining queue is now contract reconciliation, not package-shape redesign

The strongest remaining residual after Day 8 is now the cross-surface
ownership/truth story around enforced, supplemental, and staged lanes:

- workflow comments and job labels still need to read as one coherent reviewed
  platform contract
- install/package regression ownership should stay explicit:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- top-level and maintainer wording should stay aligned with the platform fence:
  - Linux strongest reviewed source of truth
  - macOS reviewed quality plus supplemental install/`pkg-config`
  - Windows reviewed CMake subset only

Interpretation:

- the next highest-value Sprint 66 target is workflow/CI/contract
  reconciliation
- the remaining work now sits above the shipped package surface, not inside the
  core release-shape machinery

#### 4. The current branch shape confirms Sprint 66 has stayed bounded

Current branch-diff shape against `master...HEAD` is still narrow:

- first batch packaging/build/docs surfaces:
  - `CMakeLists.txt`
  - `INSTALL.md`
  - `README.md`
  - `docs/maintainer_guide.md`
  - `tests/test_cmake_install.sh`
- planning/doc surfaces for Sprint 66

Interpretation:

- Sprint 66 has not drifted into broad platform or ABI sprawl
- the Day 9 rerank can stay honest about what has and has not actually moved so
  far

#### 5. The exact Day 10 target is now fixed

The next strongest target is:

- workflow/CI/install-contract reconciliation around the shipped packaging and
  platform truth story

Likely touched surfaces:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Support only if the landing proves it is required:

- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`

Explicit non-goals remain:

- broad shared-library enablement
- ABI guarantee widening
- Windows Makefile reviewed-wrapper parity
- macOS dead-code enforcement
- Windows dead-code enforcement
- dead-code topology redesign

Interpretation:

- the remaining Sprint 66 work now has one exact ownership focus
- Day 10 should change only the surfaces needed to keep packaging, platform,
  and regression truth aligned

### Day 9 Close

Sprint 66 now has:

- one confirmed closed Day 8 packaging contradiction
- one reranked remaining queue centered on workflow/install-contract
  reconciliation
- one explicit Day 10 target set with a bounded support surface

## Day 10 - Workflow and Install-Contract Reconciliation Batch

### Goal

Land the bounded Day 10 follow-through from the Day 9 rerank: reconcile the
shipped static-first package story with the cross-platform workflow and proof
ownership surfaces, without reopening build/install mechanics or widening the
repo's ABI/platform claims.

### Actions

1. Tightened the `Cross-Platform CI Contract` table and installation proof
   ownership wording in `README.md`.
2. Tightened `INSTALL.md` so the focused install/package regression scripts are
   described explicitly as Unix-oriented local proof surfaces for the static-
   first package contract.
3. Added the matching install/package regression ownership section to
   `docs/maintainer_guide.md`.
4. Reconciled the workflow commentary in:
   - `.github/workflows/ci.yml`
   - `.github/workflows/macos-ci.yml`
   - `.github/workflows/windows-ci.yml`
   so the enforced/supplemental/staged wording matches the shipped
   package/platform interpretation directly.
5. Ran the stronger reviewed baseline plus the focused install/package
   regressions:
   - `make quality-review-full`
   - `bash tests/test_install.sh`
   - `bash tests/test_cmake_install.sh`

### Findings

#### 1. The remaining contradiction was proof ownership, not package mechanics

Before Day 10:

- the maintained static-first install/export story was already coherent across
  the main build/docs surfaces
- but the ownership of install/package proof was still too implicit and uneven
  across:
  - `README.md`
  - `INSTALL.md`
  - `docs/maintainer_guide.md`
  - workflow comments/job names

After Day 10:

- the local Unix-side install proof surfaces are explicit:
  - `tests/test_install.sh`
  - `tests/test_cmake_install.sh`
- macOS CI is now described more precisely as a narrower supplemental Make
  install/`pkg-config` verification lane
- Windows is now described more precisely as the reviewed CMake subset and
  CMake-first consumer story, not as a separate reviewed install-validation
  lane
- Linux remains the strongest reviewed source of truth without implicitly
  claiming a separate install-validation CI lane

Interpretation:

- the Day 10 batch closed a real contract-ownership contradiction
- no build/install behavior change was needed to do it honestly

#### 2. The landed batch stayed bounded to the Day 9 surface

Touched Day 10 surfaces:

- `README.md`
- `INSTALL.md`
- `docs/maintainer_guide.md`
- `.github/workflows/ci.yml`
- `.github/workflows/macos-ci.yml`
- `.github/workflows/windows-ci.yml`

Untouched Day 10 surfaces:

- `CMakeLists.txt`
- `Makefile`
- `tests/test_install.sh`
- `tests/test_cmake_install.sh`
- dead-code scripts/topology
- any shared-library or ABI-widening surface

Interpretation:

- the reconciliation stayed inside the exact Day 9 fence
- Sprint 66 still has not drifted into platform or packaging sprawl

#### 3. The stronger reviewed baseline and focused install/package proofs both passed

Because this was substantial packaging/platform/workflow contract work, the
stronger reviewed baseline was used:

- `make quality-review-full`

Retained reviewed anchors:

- `ctest -N --test-dir build/quality-review-cmake` = `53`
- Makefile/CMake parity = `53 vs 53`
- full reviewed CMake `ctest` = `53 / 53`
- `Total Test time (real) = 523.37 sec`

Because the install/package contract wording moved materially, the focused
proof surfaces were also rerun:

- `bash tests/test_install.sh`
- `bash tests/test_cmake_install.sh`

Retained focused proof points:

- Make install/uninstall path passed
- CMake install/export/find-package path passed
- installed `pkg-config` version stayed `2.2.0`

One non-blocking note remains the same as the recent reviewed baselines:

- `test_reorder_nd` still dominated the reviewed CMake path at `369.10 sec`
  out of `523.37 sec`, but the full reviewed path completed cleanly and all
  parity anchors stayed exact

Interpretation:

- the Day 10 contract batch closes from both the strongest reviewed baseline
  and the exact install/package proof surfaces it names

#### 4. The strongest remaining Sprint 66 queue is now residual tightening, not another implementation batch

After Day 10, the strongest remaining queue is now:

- residual packaging/platform interpretation tightening
- closeout-oriented docs/maintainer follow-through
- final validation and handoff

Still explicitly deferred:

- broad shared-library enablement
- ABI guarantee widening
- Windows Makefile reviewed-wrapper parity
- macOS dead-code enforcement
- Windows dead-code enforcement
- dead-code topology redesign

Interpretation:

- Sprint 66 no longer needs another core implementation batch to stay honest
- the remaining days should close the converged contract, not search for new
  scope

### Day 10 Close

Sprint 66 now has:

- one reconciled workflow/install-contract ownership story
- one validated proof chain for the shipped static-first package surface
- one much smaller remaining queue centered on residual tightening and closeout

## Day 11 - CI and Command-Surface Reconciliation

### Goal

Align the remaining maintained command and CI truth surfaces with the landed
Sprint 66 packaging/platform contract, remove stale sprint-era commentary that
now obscures the reviewed lane model, and fix the exact Day 12-14 queue.

### Actions

1. Re-read the current Day 10 landed state across:
   - `README.md`
   - `INSTALL.md`
   - `docs/maintainer_guide.md`
   - `.github/workflows/ci.yml`
   - `.github/workflows/macos-ci.yml`
   - `.github/workflows/windows-ci.yml`
2. Re-read the maintained command/help surface in `Makefile` to check whether
   any command-story contradiction remained after Day 10.
3. Ran targeted `rg` scans across the command/docs/workflow surfaces for:
   - `install`
   - `pkg-config`
   - `find_package(Sparse)`
   - `supplemental`
   - `reviewed CMake subset`
   - `install-validation`
4. Tightened the stale contract wording that still remained:
   - top-level CI summary in `README.md`
   - platform notes table in `INSTALL.md`
5. Re-ranked the remaining queue from the landed tree, including the exact
   proof gap still visible on the Unix-side Make install regression path.

### Findings

#### 1. The remaining stale commentary was in command-facing summaries, not in the reviewed contract core

After Day 10, the main packaging/platform contract surfaces were already
aligned.

The remaining stale wording was narrower:

- the top-level CI summary in `README.md` was still too generic for the current
  reviewed/enforced/supplemental split
- the `INSTALL.md` supported-platform table still leaned on older sprint-era
  notes instead of describing the live reviewed lane model directly

After Day 11:

- `README.md` now summarizes CI in the same contract language as the maintained
  Sprint 66 state:
  - Linux strongest reviewed source of truth
  - macOS reviewed Apple Clang path plus supplemental GCC and static-first
    install/`pkg-config`
  - Windows reviewed CMake subset and CMake-first consumer story
- `INSTALL.md` now uses the supported-platform table to describe actual current
  lane ownership instead of historical sprint references

Interpretation:

- the remaining contradiction was presentation drift in operator-facing
  summaries
- the Day 11 batch closed that without widening the implementation surface

#### 2. No Makefile or workflow behavior change was required

The Day 11 reread did not uncover a remaining contradiction that required:

- `Makefile` command-surface changes
- workflow job behavior changes
- install/export implementation changes

Interpretation:

- the Day 10-11 convergence work is now primarily about keeping the touched
  truth surfaces coherent
- Sprint 66 still does not need speculative CI expansion or packaging behavior
  churn

#### 3. The exact Day 12 proof gap is now explicit

The strongest remaining Day 12 proof gap is now narrower than the original
Sprint 66 wording implied:

- `tests/test_cmake_install.sh` already checks installed `pkg-config` version
  against the repo `VERSION` file
- `tests/test_install.sh` still only proves that `pkg-config --modversion
  sparse` is non-empty, not that it matches the same source of truth

Interpretation:

- the highest-value Day 12 target is focused install/package regression
  tightening on the Unix-side Make install path
- Sprint 66 does not need broad new assurance surfaces to close honestly

#### 4. The exact remaining Day 12-14 queue is now fixed

Day 12:

- focused install/package regression tightening on:
  - `tests/test_install.sh`
- support only if the landed proof burden requires it:
  - `tests/test_cmake_install.sh`
  - `INSTALL.md`
  - `README.md`

Day 13:

- full validation sweep:
  - `make format`
  - `make lint`
  - `make test`
  - `make quality-review-full`
- targeted install/package proof reruns:
  - `bash tests/test_install.sh`
  - `bash tests/test_cmake_install.sh`

Day 14:

- closeout and handoff from the Day 13 validated baseline

Interpretation:

- the remaining queue is now narrowed to proof tightening, validation, and
  close
- Sprint 66 no longer has another real implementation-front hiding in the
  residual set

### Day 11 Close

Sprint 66 now has:

- one reconciled command-facing CI/platform summary story
- one explicit Day 12 proof gap on the Unix-side Make install regression path
- one fixed Day 12-14 close sequence
