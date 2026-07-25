# Sprint 133 Working Notes

## Sprint Goal

Decide whether Epic 11 adds shared-library/dynamic ABI support or explicitly
preserves static-first support, then implement the selected package contract.

Sprint 133 is a package, ABI, install/export, downstream consumer, validation,
and closeout sprint. It must not infer shared-library support, dynamic ABI
stability, package-manager support, or platform install parity from the current
static-first install surface unless the selected contract implements and
validates those claims.

## Starting Constraints

- Treat Epic 10 package/platform work as the current static-first support
  baseline.
- Treat Sprint 118 package/ABI residual owner map as the dependency map for
  deciding shared-library support versus explicit deferral.
- Treat Sprint 131-132 report, freshness, runtime, and non-claim decisions as
  claim-boundary context, not package proof.
- Treat `INSTALL.md` and `docs/maintainer_guide.md` as current package support
  truth: the maintained install/export shape is static-first.
- Treat `tests/test_install.sh` as the local Unix Make install, uninstall, and
  `pkg-config` consumer proof.
- Treat `tests/test_cmake_install.sh` as the local CMake install/export,
  `find_package(Sparse)`, version, and installed-consumer proof.
- Treat explicit absence of shared-library artifacts in install tests as a
  current static-first proof, not as evidence that shared support is
  implemented.
- If any `.c` or `.h` file changes, run `make format && make lint && make
  test`. Documentation-only changes require `git diff --check` and a focused
  markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_133`.

## Input Artifact Inventory

| Input | Role in Sprint 133 |
| --- | --- |
| `docs/planning/EPIC_11/PROJECT_PLAN.md` Sprint 133 | Defines the seven Sprint 133 items for package/ABI audit, shared-library design or deferral, build/install contract work, ABI/symbol proof, downstream consumer proof, validation, and closeout. |
| `docs/planning/EPIC_11/SPRINT_133/PLAN.md` | Provides day-level execution order and 168-hour budget. |
| `docs/planning/EPIC_11/SPRINT_118/artifacts/day6-residual-owner-map.md` | Assigns package/ABI decision work to this owner path and records proof gates for shared-library support or static-first deferral. |
| `docs/planning/EPIC_11/SPRINT_118/templates/package-abi-decision-template.md` | Provides reusable decision evidence fields for static/shared/ABI/package-manager support. |
| `docs/planning/EPIC_10/SPRINT_112/artifacts/` | Prior package surface audit, Make install proof, and CMake install/export proof baseline. |
| `docs/planning/EPIC_10/SPRINT_115/artifacts/` | Prior package/platform residual intake, package-manager decision, and platform install deferral context. |
| `README.md` | Front-door package support summary and current shared-library deferral wording. |
| `INSTALL.md` | Operational install, staged install, static-first package contract, and local install-validation truth. |
| `docs/maintainer_guide.md` | Maintainer package/platform support truth, validation ownership, and non-claim boundaries. |
| `CMakeLists.txt` | CMake target, static-first `BUILD_SHARED_LIBS` handling, install/export rules, version generation, and pkg-config generation. |
| `cmake/SparseConfig.cmake.in` | Installed CMake package config owner. |
| `sparse.pc.in` | pkg-config metadata template owner. |
| `tests/test_install.sh` | Make install/uninstall and `pkg-config` downstream consumer proof. |
| `tests/test_cmake_install.sh` | CMake install/export, `find_package`, version, and downstream consumer proof. |

## Candidate Source Areas

| Source area | Current package/ABI role | Day owner |
| --- | --- | --- |
| `include/*.h` and generated `sparse_version.h` | Installed public header surface and source-compatibility contract. | Days 2, 5-6, 9-13 |
| `CMakeLists.txt` | Static library target, install/export rules, `BUILD_SHARED_LIBS` warning, version generation, CMake package generation, and pkg-config generation. | Days 3, 5-7, 9-13 |
| `cmake/SparseConfig.cmake.in` | Installed CMake package config entry point. | Days 3-4, 8, 11, 13 |
| `sparse.pc.in` | Installed `pkg-config` metadata and downstream link flags. | Days 3-4, 8, 12-13 |
| `Makefile` install targets | Unix static archive install, generated version header install, `sparse.pc` install, and uninstall behavior. | Days 3, 7, 10, 12-13 |
| `tests/test_install.sh` | Local Make install/uninstall, installed headers, no-shared-artifact, version, and `pkg-config` consumer proof. | Days 4, 7, 10, 12-13 |
| `tests/test_cmake_install.sh` | Local CMake install/export, no-shared-artifact, CMake package, version, and installed example proof. | Days 4, 7, 10-13 |
| `examples/cmake_example/` | Maintained downstream CMake consumer example. | Days 4, 11, 13 |
| `README.md` and `INSTALL.md` | User-facing package and install support wording. | Days 5, 8, 13-14 |
| `docs/maintainer_guide.md` | Maintainer-facing package validation, platform support, and non-claim policy. | Days 5, 8, 13-14 |
| `.github/workflows/` | Reviewed and supplemental platform install/build lane context. | Days 5-6, 13-14 if platform wording changes |
| `VERSION` and `include/sparse_version.h.in` | Single-source version metadata and installed version macros. | Days 2-3, 9-13 |

## Day-Level Ownership

| Day | Owner focus | Project-plan items |
| --- | --- | --- |
| 1 | Sprint intake, package/ABI source-area baseline, item map, validation lanes, static-first and dynamic-ABI fences | Items 1-7 |
| 2 | Public header, version macro, and symbol exposure audit | Item 1 |
| 3 | Install shape, CMake export, pkg-config, version, and package metadata audit | Item 1 |
| 4 | Downstream CMake/pkg-config consumer expectation audit | Items 1 and 5 |
| 5 | Shared-library versus static-first product decision | Item 2 |
| 6 | Selected shared-library design or static-first deferral/enforcement design | Item 2 |
| 7 | First build/install contract implementation batch | Item 3 |
| 8 | Package metadata and documentation alignment batch | Item 3 |
| 9 | ABI/symbol/version proof or static-first deferral proof design | Item 4 |
| 10 | ABI/symbol/version proof or static-first deferral proof implementation | Items 4 and 6 |
| 11 | CMake downstream consumer proof strengthening | Item 5 |
| 12 | pkg-config downstream consumer proof strengthening | Item 5 |
| 13 | Integrated package, install, source/build, and quality validation | Item 6 |
| 14 | Closeout, package/ABI support truth, residual queue, and Sprint 134 handoff | Item 7 |

## Validation Expectations

| Change type | Required validation |
| --- | --- |
| Documentation-only Sprint 133 artifacts | `git diff --check` and focused markdown whitespace scan over `docs/planning/EPIC_11/SPRINT_133`. |
| Public header edits | Focused compile/install consumer proof plus `make format && make lint && make test`. |
| C source edits | Focused behavior proof plus `make format && make lint && make test`. |
| Make install target edits | `bash tests/test_install.sh`, affected uninstall checks, and docs hygiene. |
| CMake install/export edits | `bash tests/test_cmake_install.sh`, focused CMake configure/build/install proof, and docs hygiene. |
| pkg-config metadata edits | `bash tests/test_install.sh` or focused staged `pkg-config` compile/link/run proof. |
| CMake package metadata edits | `bash tests/test_cmake_install.sh` and focused `find_package(Sparse)` proof. |
| Shared-library support changes | Library artifact inspection, loader/runtime proof, symbol/version proof, CMake/pkg-config consumer proof, and full C quality if `.c`/`.h` files change. |
| Static-first deferral/enforcement changes | No-shared-artifact proof, support-wording drift scan, relevant install proof, and docs hygiene. |
| Platform workflow changes | Workflow-equivalent proof, expected-count impact, staged-exclusion update, support wording scan, and relevant package validation. |

## Scope Boundaries

- Sprint 133 may decide shared-library support or explicitly preserve
  static-first support.
- Sprint 133 may change build/install/package metadata only after the product
  decision is recorded.
- Sprint 133 must not infer ABI stability from source compatibility.
- Sprint 133 must not infer dynamic-loader behavior from static archive
  install proof.
- Sprint 133 must not infer package-manager support from CMake or pkg-config
  metadata.
- Sprint 133 must not infer macOS or Windows install parity from local Unix
  install scripts.
- Sprint 133 must keep reviewed, supplemental, local-only, deferred, and
  unsupported package evidence separate.

## Day 1 Notes

- Created the Sprint 133 working-notes baseline.
- Created the Sprint 133 artifact directory and Day 1 intake artifact.
- Re-read the Sprint 133 project-plan section and mapped Items 1-7 to
  day-level owners.
- Reviewed Sprint 118 package/ABI residual owner map and decision template.
- Reviewed current package support wording in `README.md`, `INSTALL.md`, and
  `docs/maintainer_guide.md`.
- Inventoried candidate package/ABI source areas across installed public
  headers, generated version metadata, CMake install/export rules,
  `pkg-config` metadata, Make install/uninstall rules, install tests, CMake
  example consumers, and package documentation.
- Recorded validation lanes for documentation-only changes, public header
  edits, Make install edits, CMake install/export edits, pkg-config metadata
  edits, shared-library support changes, static-first deferral/enforcement
  changes, and platform workflow changes.
- Recorded duplicate fences so static install proof, shared-library support,
  dynamic ABI stability, package-manager support, and platform install parity
  are not conflated.

## Day 2 Notes

- Wrote the public header and symbol exposure audit artifact.
- Inventoried all source headers under `include/` plus generated
  `sparse_version.h` as the current installed header set.
- Identified the maintained installed consumer include set used by
  `tests/test_install.sh`, `examples/cmake_example/main.c`, and install docs:
  `sparse_types.h`, `sparse_matrix.h`, `sparse_lu.h`, and
  `sparse_lu_csr.h`.
- Classified headers by ABI risk if shared-library support is selected, with
  high-risk ownership on public storage/result structs, public handles,
  callback payloads, solver option structs, and core typedefs.
- Separated the opaque `SparseMatrix` handle from public layout structs such
  as CSR/CSC, dense, ILU, LDLT, QR, bidiag, SVD, eigensolver, and iterative
  handle structs.
- Recorded version and feature macros visible to installed consumers,
  including generated `SPARSE_VERSION*` macros, `SPARSE_IDX_BITS`,
  `SPARSE_SCALAR_BITS`, index format macros, and solver threshold constants.
- Recorded that no installed macro currently declares shared-library
  availability, ABI version, soname, symbol visibility, or package-manager
  support.
- Handed Day 3 exact install-shape questions for header count,
  `sparse_vector.h` install intent, generated version-header parity,
  CMake/pkg-config metadata, static archive wording, and potential shared
  artifact naming.

## Day 3 Notes

- Wrote the install shape and package metadata audit artifact.
- Audited Make install rules, CMake install/export rules,
  `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, install tests, package docs,
  and maintainer package truth.
- Confirmed current source header count is 18 and current installed header
  proof is 19 including generated `sparse_version.h`.
- Confirmed Make and CMake install paths both install the static archive,
  installed headers, generated version metadata, and `sparse.pc`; the CMake
  path additionally installs `SparseTargets.cmake`, `SparseConfig.cmake`, and
  `SparseConfigVersion.cmake`.
- Confirmed shared-library artifacts are intentionally absent and checked by
  both install validation scripts.
- Confirmed CMake declares `sparse_lu_ortho` as a `STATIC` target and treats
  `BUILD_SHARED_LIBS=ON` as an explicit static-first warning path.
- Mapped `sparse.pc` fields and recorded that it exposes include/link flags
  and version metadata but no library-type, ABI, index-width, scalar-width, or
  package-manager contract fields.
- Reconciled `README.md`, `INSTALL.md`, and `docs/maintainer_guide.md` with
  observed install behavior; no Day 3 wording drift requiring immediate docs
  changes was found.
- Recorded gap queue items for CMake header-count proof tightening,
  `sparse_vector.h` install disposition, pkg-config metadata identity,
  `Libs.private` policy, CMake static/shared component negotiation,
  `BUILD_SHARED_LIBS` enforcement strength, and missing symbol/ABI metadata.

## Day 4 Notes

- Wrote the downstream consumer expectation audit artifact.
- Audited `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `examples/cmake_example/`, `examples/README.md`, package docs, and
  maintainer package validation wording.
- Separated installed consumer proof from local build-tree examples,
  benchmarks, tests, and CMake build-tree parity targets.
- Confirmed current installed CMake proof uses `find_package(Sparse REQUIRED)`
  and links `Sparse::sparse_lu_ortho` from a temporary install prefix.
- Confirmed current `pkg-config` proof resolves staged `sparse.pc`, compiles
  and runs a generated matrix consumer, then compiles and runs
  `examples/cmake_example/main.c` through `pkg-config`.
- Recorded the static consumer contract around `libsparse_lu_ortho.a`,
  `${prefix}/include/sparse`, generated version metadata,
  `Sparse::sparse_lu_ortho`, and `pkg-config --cflags --libs sparse`.
- Recorded shared-library consumer proof requirements for artifact creation,
  static/shared selection, symbol export policy, ABI/version policy,
  loader/runtime proof, dependency metadata, header/layout policy, platform
  proof, and redesign of current no-shared-artifact checks.
- Added consumer gap queue items for broader installed-header compile proof,
  exact pkg-config path and link-flag checks, CMake header-count tightening,
  installed target path-origin checks, static/shared selection, loader proof,
  and package-manager consumer proof.
- Handed Day 5 the decision facts that current downstream proof is functional
  but static-first, while reviewed shared-library support would need new
  artifact, selection, symbol, loader, dependency, ABI-layout, and platform
  proof.

## Day 5 Notes

- Wrote the package and ABI product decision artifact.
- Decided Sprint 133 will preserve the static-first package contract and
  strengthen static-first deferral/enforcement proof instead of implementing
  shared-library or dynamic ABI support.
- Applied decision criteria across ABI stability, symbol visibility,
  versioning, install behavior, downstream selection, loader/runtime proof,
  dependency metadata, platform ownership, validation cost, and static-first
  value.
- Recorded that current install/export behavior intentionally emits the
  static archive, validates absence of shared artifacts, and exposes
  downstream CMake/`pkg-config` consumers for that static contract.
- Recorded blockers for shared support: no export/import macro, no symbol
  inventory, no ABI epoch or soname/install-name policy, no static/shared
  selection contract, no loader proof, no public/private dependency policy,
  no layout compatibility policy, and no platform shared-runtime ownership.
- Selected implementation days should focus on static-first strengthening:
  `BUILD_SHARED_LIBS=ON` deferral behavior, exact CMake header-count proof,
  installed CMake path-origin checks, exact `pkg-config` output checks, and
  documentation alignment only if implementation changes require it.
- Preserved package-manager support as a residual non-claim because no
  manager recipes or manager-specific consumer proof exist.

## Day 6 Notes

- Wrote the selected static contract design artifact.
- Translated the Day 5 static-first product decision into build, install,
  package metadata, downstream validation, rollback, and deferral-check
  requirements.
- Selected a narrow Day 7 implementation path: change `BUILD_SHARED_LIBS=ON`
  from warning-only behavior to an explicit configure-time static-first
  deferral failure.
- Preserved the existing static `sparse_lu_ortho` target, Make install path,
  CMake install/export shape, `sparse.pc` metadata, public headers, exact
  CMake package version behavior, and no-shared-artifact install checks.
- Rejected adding CMake static/shared components, `pkg-config` ABI fields,
  export/import macros, symbol maps, soname/install-name policy, or
  package-manager files during the static-first implementation batch.
- Defined candidate failure wording for `BUILD_SHARED_LIBS=ON` that names the
  rejected input, the maintained static archive contract, and the deferred
  shared-library/dynamic-ABI proof requirements.
- Defined Day 7 validation: focused failing configure probe for
  `BUILD_SHARED_LIBS=ON`, then `bash tests/test_cmake_install.sh` for the
  normal static install/export path if `CMakeLists.txt` changes.
- Deferred exact CMake header-count checks to Day 11 or Day 13 and exact
  `pkg-config` path/link checks to Day 12.

## Day 7 Notes

- Implemented the first build/install contract batch.
- Updated `CMakeLists.txt` so `BUILD_SHARED_LIBS=ON` now fails configure with
  explicit static-first/shared-library deferral wording instead of continuing
  after a warning.
- Preserved the static `sparse_lu_ortho` target, install/export rules,
  generated version metadata, `sparse.pc` generation, public headers,
  install tests, package docs, workflows, and no-shared-artifact checks.
- Ran a focused `cmake -S . -B <tmp> -DBUILD_SHARED_LIBS=ON` probe; the
  configure failed as expected and emitted wording for `BUILD_SHARED_LIBS=ON`,
  the static archive package surface, shared-library packaging, and deferred
  dynamic ABI support.
- The first probe harness checked one CMake-wrapped phrase as a contiguous
  string; the rerun used line-wrapping-safe checks and passed.
- Ran `bash tests/test_cmake_install.sh`; it passed 16 checks with 0 failures
  and 0 skips, including default static configure/build/install,
  no-shared-artifact proof, 19 installed headers, CMake package files,
  installed `cmake_example` configure/build/run, exact-version package
  behavior, mismatched-version rejection, and `pkg-config` version `2.2.0`.
- No `.c` or `.h` files changed, so the full C quality gate was not required.
- Left exact CMake header-count assertion, installed target path-origin
  checks, exact `pkg-config` output checks, docs alignment, deferral proof,
  and integrated package validation for Days 8-13.

## Day 8 Notes

- Wrote the package metadata and documentation batch artifact.
- Updated `INSTALL.md` so the maintained install contract states CMake rejects
  `BUILD_SHARED_LIBS=ON` and the CMake options section explains that shared
  library packaging and dynamic ABI support remain deferred.
- Updated `README.md` so the front-door installation summary says CMake
  rejects `BUILD_SHARED_LIBS=ON` rather than silently treating a shared-library
  request as supported.
- Updated `docs/maintainer_guide.md` so the authoritative package contract
  records `BUILD_SHARED_LIBS=ON` as a configure-time rejection and notes the
  Sprint 133 change from warning-only behavior.
- Did not change `sparse.pc.in`, `cmake/SparseConfig.cmake.in`, public
  headers, install scripts, workflows, source files, or package-manager files.
- Preserved the Day 5 static-first decision and kept shared-library support,
  dynamic ABI compatibility, and package-manager support as explicit
  non-claims.
- Handed Day 9 the need to design local ABI/symbol/static-deferral proof that
  checks for absence of shared artifacts, unsupported shared selectors, export
  macros, symbol maps, soname/install-name policy, and accidental ABI claims.

## Day 9 Notes

- Wrote the static deferral proof design artifact.
- Selected a local static-deferral proof script design rather than a shared
  ABI/symbol proof because Day 5 preserved static-first support and deferred
  shared-library/dynamic ABI support.
- Designed proof checks for `BUILD_SHARED_LIBS=ON` configure rejection,
  token-level deferral wording, explicit static CMake target declaration,
  absence of unsupported export/import macros, absence of shared ABI metadata
  or static/shared selectors, and package docs that keep shared-library,
  dynamic ABI, and package-manager support deferred.
- Recorded that existing header comments using `@warning **ABI break ...**`
  should be treated as source/layout migration notes, not maintained dynamic
  ABI compatibility promises.
- Classified the proposed script as local maintainer/package proof, not
  reviewed CI support.
- Handed Day 10 a concrete implementation path for
  `scripts/static_package_deferral_check.sh` with syntax check, main script
  execution, and documentation hygiene validation.

## Day 10 Notes

- Implemented `scripts/static_package_deferral_check.sh` as the local
  static-first package deferral proof.
- The script verifies `BUILD_SHARED_LIBS=ON` configure rejection, token-level
  deferral wording, explicit static CMake target declaration, absence of
  unsupported public export/import macros, absence of shared ABI metadata,
  absence of static/shared package selectors, and deferred support wording in
  package docs.
- Documented the new script in `docs/maintainer_guide.md` as local
  package-contract proof, not reviewed CI support.
- Ran `bash -n scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash scripts/static_package_deferral_check.sh`; the first run exposed a
  Markdown-backtick pattern issue in the script's INSTALL wording check, then
  the corrected script passed all checks.
- Did not change public headers, C source, install scripts, package metadata,
  workflows, or package-manager files.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 11 Notes

- Strengthened `tests/test_cmake_install.sh` as the downstream CMake
  installed-package proof.
- Changed the installed header check from `>= 14` to the exact current public
  header inventory: source `include/*.h` files plus generated
  `sparse_config.h`, currently 19 installed headers.
- Added installed CMake package metadata checks for
  `Sparse::sparse_lu_ortho` as `STATIC IMPORTED`,
  `${_IMPORT_PREFIX}/include` include directories, and
  `${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a` imported archive location.
- Added source-tree and build-tree path leak scans over the installed CMake
  package directory.
- Updated `docs/maintainer_guide.md` so maintainer ownership describes the
  stronger CMake install/export proof.
- Ran `bash -n tests/test_cmake_install.sh`; it passed.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips, including the new static imported-target, installed-prefix, and
  path-leak assertions.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 12 Notes

- Strengthened `tests/test_install.sh` as the downstream `pkg-config`
  installed-package proof.
- Added `pkg-config --print-errors --exists sparse` and exact
  `pkg-config --exists "sparse = $VERSION"` checks.
- Added `.pc` variable checks for `prefix`, `libdir`, and `includedir`.
- Tightened `pkg-config --cflags` and `pkg-config --libs` checks to require
  the installed include path and static archive link flags.
- Added `pkg-config --libs --static sparse` parity with `--libs` for the
  current self-contained link surface.
- Added a `Libs.private` absence check and an unsupported shared-library,
  ABI, and package-manager wording scan over installed `sparse.pc`.
- Kept the existing compile/link/run proof for a small installed consumer and
  the maintained example source.
- Updated `docs/maintainer_guide.md` so maintainer ownership describes the
  stronger Make install and `pkg-config` proof.
- Ran `bash -n tests/test_install.sh`; it passed.
- Ran `bash tests/test_install.sh`; the first two local runs exposed path
  normalization assumptions in exact flag checks, then the corrected test
  passed 22 checks with 0 failures.
- No `.c` or `.h` files changed, so the full C quality gate was not required.

## Day 13 Notes

- Ran integrated package validation for the Sprint 133 static-first package
  contract surfaces.
- Ran `bash -n tests/test_install.sh && bash -n tests/test_cmake_install.sh &&
  bash -n scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --name-only -- '*.c' '*.h'`; it returned no paths, so the full
  C `make format && make lint && make test` gate was not required by the
  Sprint instruction.
- Ran `bash tests/test_install.sh`; it passed 22 checks with 0 failures,
  including Make install/uninstall, no shared artifacts, exact 19-header
  install, `.pc` field semantics, exact version resolution, no
  `Libs.private`, unsupported package/ABI claim scanning, and downstream
  compile/link/run consumers.
- Ran `bash tests/test_cmake_install.sh`; it passed 21 checks with 0 failures
  and 0 skips, including CMake install/export, static imported-target
  metadata, installed-prefix include/archive paths, source/build path leak
  scans, `find_package(Sparse)` consumer configure/build/run, version
  compatibility checks, and installed pkg-config version.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed all
  static-first deferral checks.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over touched package, script, docs, and Sprint
  133 paths; it passed.
- Wrote the integrated package validation artifact for closeout and PR review.

## Day 14 Notes

- Wrote the Sprint 133 closeout and package ABI handoff artifact.
- Published the final support truth: Sprint 133 maintains a static-first
  install/export surface, rejects `BUILD_SHARED_LIBS=ON`, keeps package
  version metadata exact and source/package-scoped, and does not claim
  shared-library packaging, dynamic ABI compatibility, package-manager
  support, runtime-loader behavior, or broader platform install parity.
- Confirmed README, INSTALL, maintainer guide, CMake behavior,
  `sparse.pc.in`, and `cmake/SparseConfig.cmake.in` remain aligned with the
  selected static-first contract.
- Recorded residual queues for shared-library packaging, dynamic ABI policy,
  package-manager recipes, CMake static/shared selectors, `Libs.private`
  semantics, optional thread/OpenMP package proof, and reviewed CI promotion.
- Prepared Sprint 134 handoff notes and PR review summary material.
- Ran `git diff --name-only -- '*.c' '*.h'` and
  `git ls-files --others --exclude-standard -- '*.c' '*.h'`; both returned no
  paths, so the full C `make format && make lint && make test` gate was not
  required by the Sprint instruction.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `git diff --check`; it passed.
- Ran a trailing-whitespace scan over Sprint 133 and touched package/doc/script
  surfaces; it passed.

## PR Fix Notes

- Fixed the macOS supplemental Make install/`pkg-config` CI failure in
  `tests/test_install.sh`.
- GitHub macOS `pkg-config` preserved the doubled slash from the temporary
  install prefix in emitted `--cflags` and `--libs` output, while the local
  tool collapsed it.
- Replaced string equality against one path-normalization form with semantic
  checks: the emitted `-I` path must resolve to the installed include
  directory, the emitted `-L` path must resolve to the installed library
  directory, and the remaining link flags must be exactly
  `-lsparse_lu_ortho -lm`.
- Updated the Day 12 retrospective wording to describe semantic installed-path
  validation instead of normalized path-string validation.
