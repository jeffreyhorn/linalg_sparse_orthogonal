# Sprint 143 Working Notes

## Sprint Goal

Make and implement the Epic 12 package/ABI product decision: shared-library
ABI support with proof, or stricter static-first-only contract with no
ambiguity.

## Initial Constraints

- Start from the Sprint 142 runtime/backend package/ABI handoff and keep
  runtime/backend sentinel evidence separate from package proof.
- Audit public headers, symbol surface, versioning, visibility, CMake exports,
  `pkg-config`, install scripts, loader behavior, and platform risks before
  making the package decision.
- Choose one package path for Sprint 143: shared-library ABI support with
  proof, or stricter static-first-only support. Do not partially implement
  both.
- Preserve existing static downstream consumers unless the selected product
  decision explicitly changes them with validation.
- Do not widen package-manager, platform-parity, portable performance,
  runtime/backend, optional-backend, or state-of-the-art claims.
- Treat macOS and Windows package confidence as support-tier and platform-lane
  questions unless Sprint 144 promotes a reviewed platform lane.
- Run install/package, CMake export, `pkg-config`, static/shared guard,
  script, report, docs, and full C quality gates according to touched
  surfaces.

## Inherited Sprint 142 Handoff

| Surface | Sprint 142 result | Sprint 143 use |
| --- | --- | --- |
| Runtime/backend public-control boundary | Existing typed controls are caller-facing; env/build/report controls are non-API unless explicitly promoted. | Prevent package docs from accidentally making env/build/report controls ABI-stable. |
| Static-first install baseline | Current README/INSTALL state describes a maintained static archive package surface. | Audit and decide whether to preserve or widen that contract. |
| Sentinel non-claim boundary | `S2`/`S3` are local advisory timing rows; `S5` is a local wall-check hard gate. | Do not use sentinel rows as package, ABI, platform, or portable performance proof. |
| Build/report controls | `SPARSE_OPENMP`, `OMP_NUM_THREADS`, package/link settings, and benchmark opt-ins are build/report context. | Keep them out of package ABI claims unless a package decision owns them. |
| CMake/pkg-config consumer proof | Existing install tests cover static downstream use through `pkg-config` and `find_package(Sparse)`. | Revalidate and strengthen downstream consumer proof for the selected path. |
| Shared-library decision gate | Choose shared-library ABI support with proof, or strengthen static-first-only deferral. | Primary Sprint 143 product decision. |
| Platform tier dependency | Sprint 143 package work must not imply macOS/Windows reviewed parity. | Route platform promotion to Sprint 144. |

## Initial Package Surface Map

| Surface | Current owner area | Current signal |
| --- | --- | --- |
| Public installed headers | `include/*.h`, `include/sparse_version.h.in`, generated `build/include/sparse_version.h` | 18 source headers plus generated version header are installed under `include/sparse`; public structs, enums, typedefs, macros, and function declarations define the ABI-risk surface. |
| Version source of truth | `VERSION`, `include/sparse_version.h.in`, `CMakeLists.txt`, `Makefile`, `sparse.pc.in`, `SparseConfigVersion.cmake` generation | Repo `VERSION` drives generated header, CMake project version, exact CMake package version file, and `pkg-config --modversion`. |
| Make install/uninstall | `Makefile` install block | Installs `build/libsparse_lu_ortho.a`, headers, generated version header, and `sparse.pc`; uninstalls those files. |
| CMake install/export | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in` | Explicit `add_library(... STATIC ...)`, exports `Sparse::sparse_lu_ortho`, installs archive/header/package config files, and writes exact-version package metadata. |
| `pkg-config` metadata | `sparse.pc.in`, generated installed `sparse.pc` | Describes include path and static archive link flags with optional build flags; currently no `Libs.private` or static/shared selector. |
| Static/shared guard | `CMakeLists.txt`, `scripts/static_package_deferral_check.sh`, README, INSTALL, maintainer guide | `BUILD_SHARED_LIBS=ON` is rejected; guard checks no public export/import macros, no SOVERSION/visibility metadata, no package selector, and preserved deferral wording. |
| Install proof scripts | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Validate Make install/`pkg-config` and CMake install/export/downstream consumer behavior, exact version behavior, no shared artifacts, and uninstall cleanup where applicable. |
| CI package lanes | `.github/workflows/ci.yml`, `.github/workflows/macos-ci.yml`, `.github/workflows/windows-ci.yml` | Linux carries reviewed static-first package contract; macOS and Windows carry supplemental package confidence with narrower support-tier wording. |
| Package report rows | `tests/corpus/manifests/report_families.tsv`, `scripts/normalize_report_index.py`, related docs/tests | Package rows are source-controlled proof-owner metadata, not fresh local install-run evidence unless commands are run. |
| Public package docs | `README.md`, `INSTALL.md` | Current docs describe maintained static package surface and shared-library deferral. |
| Maintainer package docs | `docs/maintainer_guide.md` | Explains package proof ownership, support-tier split, static-first contract, and non-claims. |

## Item-To-Day Owner Map

| Project-plan item | Day owner(s) | Notes |
| --- | --- | --- |
| Item 1: ABI Feasibility Audit | Days 1-4 | Intake, public header/symbol audit, install/export metadata audit, and platform/loader risk audit. |
| Item 2: Product Decision | Day 5 | Decide shared-library ABI support versus stricter static-first-only support. |
| Item 3: Implementation Batch | Days 6-8 | Design and implement the selected package path. |
| Item 4: Downstream Consumer Proof | Day 9 | Strengthen Make, `pkg-config`, CMake, version, loader if applicable, and unsupported-artifact proof. |
| Item 5: CI/Packaging Alignment | Day 10 | Align Linux/macOS/Windows CI support-tier wording and package report rows. |
| Item 6: Documentation Alignment | Day 11 | Update README, INSTALL, package metadata comments, maintainer guidance, and non-claim wording. |
| Item 7: Validation and Closeout | Days 12-14 | Focused package validation, full quality gate and claim closure, final closeout, and Sprint 144 handoff. |

## Initial Validation Expectations

| Touched surface | Required checks |
| --- | --- |
| Documentation and planning artifacts only | `git diff --check`, trailing-whitespace scan, and focused path/reference review. |
| Shell package/install scripts | `bash -n`, focused script execution where feasible, generated-output hygiene, and relevant package proof commands. |
| Python report-index or corpus scripts/tests | `python3 -m py_compile`, focused script tests, `python3 scripts/validate_corpus_schema.py`, and normalized report-index/freshness checks for affected families. |
| Make/CMake/package metadata | Focused configure/build/install/export checks, `tests/test_install.sh`, `tests/test_cmake_install.sh`, static/shared guard checks, and package report checks as applicable. |
| CI workflow changes | YAML/syntax review, command/path review, support-tier wording review, and local feasible proof commands. |
| C or header files | Focused tests for touched behavior, then `make format && make lint && make test`. |
| Shared-library implementation path | Build/install/export, symbol/visibility, loader, downstream consumer, platform-specific, and negative unsupported-artifact checks defined by Day 5 decision. |

## Initial Non-Claim Register

| Non-claim | Boundary |
| --- | --- |
| Shared-library ABI support | Not claimed unless Sprint 143 explicitly selects and proves it. Current state rejects `BUILD_SHARED_LIBS=ON`. |
| Dynamic loader behavior | Not claimed unless shared-library path adds loader proof for supported platforms. |
| Long-term ABI compatibility | Not claimed without symbol/versioning/layout policy and compatibility proof. |
| Package-manager availability | Not claimed; local install/export proof does not imply distro, Homebrew, vcpkg, conan, MSI, or binary package distribution. |
| Platform parity | Not claimed; Sprint 144 owns platform promotion. |
| Portable performance | Package proof must not cite local sentinel timing as portable evidence. |
| Runtime/backend behavior | Package docs must not turn maintainer env/build/report controls into public ABI. |
| State-of-the-art status | Package/ABI work is not competitive sparse linear algebra evidence. |

## Stop Conditions

- Stop and ask if shared-library support appears feasible only by widening
  platform, ABI, loader, or package-manager claims beyond Sprint 143 scope.
- Stop and ask if both package paths remain attractive after Day 5; the sprint
  must select one path rather than implementing partial support for both.
- Stop and ask if package docs or metadata would imply ABI stability before
  symbol, visibility, versioning, loader, and downstream proof exist.
- Stop and ask if CI wording would promote macOS or Windows support tiers that
  belong to Sprint 144.
- Stop and ask if runtime/backend sentinel rows are being used as package,
  ABI, platform, or portable performance proof.
- Stop and ask if required package or quality gates fail.

## Day 1 Notes

- Created Sprint 143 working notes and artifact directory structure.
- Re-read the Sprint 143 project-plan section and Sprint 143 day-by-day plan.
- Reviewed Sprint 142 Day 13 package/ABI handoff and Sprint 142 retrospective
  readiness requirements.
- Confirmed Sprint 143 starts from a static-first package baseline:
  - `CMakeLists.txt` rejects `BUILD_SHARED_LIBS=ON`.
  - CMake declares `sparse_lu_ortho` as an explicit `STATIC` target.
  - Make install installs `build/libsparse_lu_ortho.a`, public headers,
    generated `sparse_version.h`, and `sparse.pc`.
  - CMake install/export installs static archive, headers, generated version
    header, `SparseTargets.cmake`, `SparseConfig.cmake`,
    `SparseConfigVersion.cmake`, and `sparse.pc`.
  - `sparse.pc.in` has no shared/static selector and no `Libs.private` stanza.
  - `scripts/static_package_deferral_check.sh` guards static-first wording,
    shared request rejection, absence of public export/import macros, and
    absence of shared ABI metadata.
- Mapped the current package surfaces across headers, versioning, Make/CMake,
  `pkg-config`, guard scripts, install tests, CI workflows, package report
  rows, README, INSTALL, and maintainer guide.
- Mapped Sprint 143 Items 1-7 to day-level owners.
- Recorded initial validation expectations, non-claims, and stop conditions
  before audit or implementation work begins.

## Day 2 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day2-public-header-symbol-audit.md`
  as the public header and symbol audit artifact.
- Accounted for the 19 installed public headers:
  18 source headers plus generated `sparse_version.h` from
  `include/sparse_version.h.in`.
- Recorded the public declaration shape from focused scans:
  - 130 function declarations in the simple public scan.
  - 31 public structs.
  - 15 public enums.
  - 4 callback typedefs.
  - 5 generated version macro surfaces.
- Identified ABI-sensitive surfaces:
  `idx_t` width via `SPARSE_IDX_BITS`, `sparse_scalar_t`, public struct
  layout, enum values, opaque `SparseMatrix` allocation/free semantics,
  callback signatures, ownership/lifecycle rules, build-option macros, and
  installed non-`sparse_` helper declarations.
- Inspected the current static archive symbol surface with
  `nm -g --defined-only build/libsparse_lu_ortho.a`.
- Recorded current archive symbol counts:
  - 359 defined global symbols.
  - 222 `sparse_`-prefixed defined global symbols.
  - 137 non-`sparse_` or internal-looking defined global symbols.
- Confirmed no current export/import or shared ABI metadata was found in
  `include/`, `CMakeLists.txt`, `cmake/`, or `sparse.pc.in` for
  `SPARSE_API`, `SPARSE_EXPORT`, `SPARSE_IMPORT`, `__declspec`, visibility
  attributes, `SOVERSION`, `WINDOWS_EXPORT_ALL_SYMBOLS`,
  `C_VISIBILITY_PRESET`, or `VISIBILITY_INLINES_HIDDEN`.
- Identified Day 5 shared-library proof requirements:
  explicit export macro policy, hidden-by-default implementation visibility,
  reviewed exported-symbol allowlist, struct/enum ABI policy, versioning
  policy, platform loader proof, dynamic downstream consumer proof,
  static/shared coexistence decision, and negative unsupported-artifact checks.
- Routed Day 3 to install/export metadata audit: Make install/uninstall,
  CMake target/export/config/version metadata, `sparse.pc.in`, install proof
  scripts, CMake install proof, and static package deferral guard.

## Day 3 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day3-install-export-metadata-audit.md`
  as the Make/CMake/`pkg-config` install/export metadata audit artifact.
- Audited Make install/uninstall behavior:
  installs `build/libsparse_lu_ortho.a`, all source headers, generated
  `sparse_version.h`, and generated `sparse.pc`; uninstall removes the static
  archive, `include/sparse`, and `sparse.pc`.
- Audited CMake install/export behavior:
  explicit `STATIC` target, configure-time `BUILD_SHARED_LIBS=ON` rejection,
  installed archive/header/package config/export files, exact-version package
  config, static imported target, and installed `sparse.pc`.
- Confirmed `sparse.pc.in` describes the static archive link surface with
  `-lsparse_lu_ortho -lm @SPARSE_PC_LIBS_EXTRA@`, no `Libs.private`, and no
  static/shared selector.
- Audited proof scripts:
  - `tests/test_install.sh` owns Unix-side Make install/`pkg-config`
    downstream proof, no shared artifacts, exact version, and uninstall.
  - `tests/test_cmake_install.sh` owns Unix-side CMake install/export,
    static imported target, exact/mismatched version behavior, and installed
    `find_package(Sparse)` consumer proof.
  - `scripts/static_package_deferral_check.sh` owns shared-library deferral,
    explicit static target, no export/import macro, no shared ABI metadata,
    no package selectors, and support wording checks.
- Ran shell syntax validation:
  `bash -n tests/test_install.sh tests/test_cmake_install.sh scripts/static_package_deferral_check.sh`;
  it passed.
- Ran package report validation:
  - `python3 scripts/normalize_report_index.py --family package --check`:
    passed with 6 rows.
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`:
    passed with 6 source-controlled advisory rows.
- Recorded that package report rows are source-controlled proof-owner rows,
  not evidence that install validation commands were just run.
- Identified selected-path metadata requirements:
  shared-library ABI support would require new shared build/install/export
  metadata, visibility/export policy, static/shared selection semantics,
  loader/version proof, and dynamic downstream consumers; stricter
  static-first support would preserve rejection and strengthen negative
  checks, docs, and diagnostics.
- Routed Day 4 to platform and loader risk audit across Linux, macOS, Windows,
  CI support-tier wording, and Sprint 144 platform-promotion separation.

## Day 4 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day4-platform-loader-risk-audit.md`
  as the platform, loader, and CI support-tier risk audit artifact.
- Audited package and platform lanes in the CI workflows:
  - Linux carries the reviewed static-first package contract through
    `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
    `scripts/static_package_deferral_check.sh`.
  - Linux CMake parity remains a reviewed source/build baseline, not separate
    package-manager or shared-library proof.
  - macOS Make install/`pkg-config` and CMake install/export lanes are
    supplemental static-first confidence paths.
  - Windows reviewed scope remains the CMake source/build/test subset with
    `EXPECTED_WINDOWS_CTEST_COUNT=56`.
  - Windows CMake install/downstream proof is supplemental and explicitly
    checks installed `.lib`, no `*.dll`, 19 headers, CMake package files,
    installed example execution, exact version acceptance, and mismatched
    version rejection.
- Identified shared-library blockers by platform:
  - Linux lacks soname/SOVERSION, RPATH/RUNPATH, dynamic downstream loader
    proof, hidden visibility, exported-symbol allowlist, and
    static/shared-aware `pkg-config` semantics.
  - macOS lacks `.dylib` install-name and `@rpath` policy, reviewed
    install/export parity, Darwin exported-symbol policy, and dynamic
    downstream loader proof.
  - Windows lacks public `__declspec` export/import macros, DLL/import-library
    policy, runtime install proof, DLL search-path proof, reviewed install
    parity, Makefile parity, and `pkg-config` proof.
- Recorded cross-platform ABI risks: public struct layout, enum values,
  callback signatures, `idx_t` width, generated version macros, exact package
  version semantics, static/shared coexistence, and source-controlled package
  report rows that are not fresh dynamic loader evidence.
- Identified static-first strengthening opportunities:
  preserve `BUILD_SHARED_LIBS=ON` rejection, keep negative shared-artifact
  checks, clarify Linux reviewed versus macOS/Windows supplemental support
  tiers, keep CMake/`pkg-config` metadata static-only, and strengthen docs/CI
  wording around deferred shared ABI support.
- Routed platform promotion to Sprint 144:
  Sprint 143 can decide the package/ABI product path, but should not promote
  macOS or Windows package parity or hide existing Windows staged blockers.
- Drafted validation scenarios for both possible Day 5 decisions:
  shared-library support would require full build/install/export,
  symbol/visibility, loader, downstream, selection, and ABI-version proof;
  static-first support requires install/export proof, negative shared-library
  proof, artifact exclusion, docs/CI wording, and package report boundary
  checks.

## Day 5 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day5-package-abi-product-decision.md`
  as the Sprint 143 package/ABI product decision artifact.
- Selected the stricter static-first-only package contract for Sprint 143:
  maintained static archive install/export, installed public headers, CMake
  package metadata, `pkg-config` metadata, downstream consumer proof,
  exact-version checks, and explicit unsupported shared-artifact proof.
- Rejected shared-library ABI support for Sprint 143 implementation because
  the Day 2-4 evidence showed unresolved symbol visibility, exported-symbol
  allowlist, ABI layout/versioning, CMake/`pkg-config` selector, loader, and
  platform support-tier work.
- Scored shared-library ABI support at 12 and static-first strengthening at
  37 across user value, proof feasibility, platform risk, implementation risk,
  documentation burden, claim risk, Sprint 137 fit, and Sprint 142 fit.
- Defined the selected implementation path for Days 6-14:
  strengthen unsupported shared-library guards, no-shared-artifact checks,
  static-only CMake and `pkg-config` metadata, package diagnostics, CI
  support-tier comments, README/INSTALL/maintainer wording, and package report
  boundaries.
- Defined required validation:
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `scripts/static_package_deferral_check.sh`, package report index and
  freshness checks, shell syntax checks for changed scripts, focused
  Make/CMake/package metadata checks, docs claim review, and full C quality
  gate if any C or public header files change.
- Recorded explicit deferrals for shared-library build/install/export,
  dynamic ABI compatibility, export/import macro policy, loader
  compatibility, static/shared selectors, package-manager distribution, macOS
  reviewed install/export parity, and Windows reviewed install-validation
  parity.
- Added implementation stop conditions:
  stop before installing/documenting shared artifacts, adding ABI metadata,
  adding package selectors, promoting macOS/Windows package lanes, using report
  or runtime rows as loader/platform proof, or changing C/header files without
  the full C quality gate.
- Routed Day 6 to static-first guard design and implementation planning before
  build/script/doc surfaces are edited.

## Day 6 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day6-static-first-implementation-design.md`
  as the selected-path implementation design artifact.
- Converted the Day 5 static-first decision into a bounded edit order:
  static-first guard design, install/export metadata, install proof
  diagnostics, CI support-tier alignment, public/maintainer docs, report
  boundary checks, and validation/closeout.
- Defined file ownership for `CMakeLists.txt`,
  `cmake/SparseConfig.cmake.in`, `sparse.pc.in`, `Makefile`,
  `tests/test_install.sh`, `tests/test_cmake_install.sh`,
  `scripts/static_package_deferral_check.sh`, Linux/macOS/Windows CI
  workflows, README, INSTALL, maintainer guide, and package report rows.
- Preserved compatibility rules for existing static consumers:
  `pkg-config --cflags --libs sparse`, `find_package(Sparse REQUIRED)`,
  `Sparse::sparse_lu_ortho`, exact CMake package version behavior, installed
  19-header shape, static archive names, `DESTDIR`/`PREFIX` staging, and
  optional build-flag metadata.
- Defined negative-proof behavior for unsupported shared artifacts and ABI
  claims:
  reject `BUILD_SHARED_LIBS=ON`, preserve explicit `STATIC` target, install no
  `.so`/`.dylib`/`.dll` artifacts, add no public export/import macro, add no
  shared ABI metadata, add no package selectors, and keep docs non-claims.
- Mapped focused validation commands to each touched surface:
  documentation hygiene, shell syntax checks, `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, `scripts/static_package_deferral_check.sh`,
  package report checks, CI wording review, docs claim scan, and full C
  quality gate if C/header files are changed.
- Recorded implementation risks around overbroad guard regexes, accidental
  shared metadata, `pkg-config` output changes, CI platform overclaiming,
  report freshness overclaims, and static consumer command drift.
- Routed Day 7 to the first package implementation batch with the unsupported
  shared-library guard as the highest-value first target.

## Day 7 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day7-package-batch1-static-first-metadata.md`
  as the first selected-path package implementation artifact.
- Updated `CMakeLists.txt` install metadata by removing the unused
  `LIBRARY DESTINATION` entry from the explicit `STATIC` target install rule.
  The installed target now carries the archive destination only.
- Clarified the CMake `pkg-config` metadata comment so it names the maintained
  static archive package surface.
- Updated `sparse.pc.in` description to identify the installed `.pc` file as
  static archive package metadata without adding ABI, loader, package-manager,
  or shared-library claims.
- Strengthened `scripts/static_package_deferral_check.sh`:
  - rejects non-static `sparse_lu_ortho` target declarations;
  - requires static archive install metadata;
  - rejects runtime/shared-library install destinations;
  - requires the static archive `.pc` description.
- Preserved static consumer compatibility:
  `pkg-config --cflags --libs sparse`, `find_package(Sparse REQUIRED)`,
  `Sparse::sparse_lu_ortho`, target names, archive names, installed headers,
  exact-version behavior, `PREFIX`, and `DESTDIR` staging remain unchanged.
- Routed Day 8 to install proof diagnostics in `tests/test_install.sh` and
  `tests/test_cmake_install.sh`, with no-shared-artifact and metadata failure
  messages as the first review targets.

## Day 8 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day8-package-batch2-install-proof-diagnostics.md`
  as the completed selected-path package implementation artifact.
- Strengthened `tests/test_install.sh`:
  - missing `sparse.pc` failures now include the expected file path;
  - installed `.pc` metadata must describe the static archive package contract.
- Strengthened `tests/test_cmake_install.sh`:
  - missing `sparse.pc` failures now include the expected file path;
  - installed CMake package metadata must not contain shared/module imported
    targets or shared imported locations;
  - installed `.pc` metadata must describe the static archive package
    contract;
  - installed `.pc` metadata must not contain unsupported package or ABI
    wording.
- Preserved compatibility for existing static consumers:
  `pkg-config --cflags --libs sparse`, `find_package(Sparse REQUIRED)`,
  `Sparse::sparse_lu_ortho`, archive names, installed headers, exact-version
  behavior, and downstream example execution remain unchanged.
- Confirmed the selected static-first product path is mechanically complete at
  the package metadata/guard/proof-script level; Day 9 should focus on
  downstream consumer proof review and deterministic skip/fail behavior rather
  than adding shared-loader checks.

## Day 9 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day9-downstream-consumer-proof.md`
  as the downstream consumer proof artifact.
- Strengthened `tests/test_install.sh` downstream runtime assertions:
  - the basic `pkg-config` consumer must emit version text, encoded version
    text, `nnz: 1`, and `OK`;
  - the maintained example compiled with `pkg-config` must emit library
    version text, solution output, and `OK`.
- Strengthened `tests/test_cmake_install.sh` downstream runtime assertions:
  - installed CMake example output must include library version text, solution
    output, and `OK`;
  - exact-version `find_package(Sparse VERSION EXACT REQUIRED)` consumer now
    configures, builds, and runs.
- Preserved unsupported-artifact boundaries:
  Make and CMake install proofs still reject shared artifacts, CMake package
  metadata still rejects shared imported metadata, and `pkg-config` metadata
  still rejects unsupported package/ABI wording.
- Did not add loader/runtime dynamic-link checks because Sprint 143 selected
  static-first-only support and deferred shared-library ABI support.
- Routed Day 10 to CI and package report alignment so workflow comments and
  report rows match the static downstream proof without implying platform
  parity, loader support, shared ABI, or package-manager availability.

## Day 10 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day10-ci-package-report-alignment.md`
  as the CI and package-report alignment artifact.
- Updated Linux workflow comments to describe the reviewed package lane as
  static archive install/export/downstream proof with static `pkg-config` and
  CMake metadata plus unsupported shared-artifact checks.
- Updated macOS workflow comments to clarify that supplemental package lanes
  include downstream consumer and unsupported shared-artifact proof while
  remaining outside reviewed macOS install/export parity.
- Updated Windows workflow comments to clarify that the supplemental
  install/downstream lane is CMake-first static package confidence only and
  does not imply Makefile parity, `pkg-config` parity, shared-library support,
  or dynamic ABI support.
- Strengthened the Windows supplemental install/downstream script:
  - checks installed `lib/pkgconfig/sparse.pc`;
  - checks the static archive `.pc` description;
  - rejects unsupported package/ABI wording in `sparse.pc`;
  - rejects shared/module imported CMake target metadata and shared imported
    locations;
  - requires installed example output to include version text, solution output,
    and `OK`;
  - builds and runs the exact-version consumer after exact-version configure.
- Left package report row semantics unchanged:
  package rows remain source-controlled proof-owner metadata for
  `tests/test_install.sh`, `tests/test_cmake_install.sh`, `sparse.pc.in`,
  `cmake/SparseConfig.cmake.in`, and
  `scripts/static_package_deferral_check.sh`.
- Routed Day 11 to README, INSTALL, and maintainer guide alignment with the
  final static-first package implementation and preserved non-claims.

## Day 11 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day11-documentation-alignment.md`
  as the public and maintainer documentation alignment artifact.
- Updated `README.md` installation summary to call out static-archive scoped
  `sparse.pc` metadata, downstream compile/link/run proof, exact package
  version handling, and the package-manager/dynamic-loader non-claim boundary.
- Updated `INSTALL.md` maintained install contract to describe the static
  `.pc` metadata description, absence of `Libs.private` and static/shared
  selectors, stricter `pkg-config` runtime proof, and CMake exact-version
  configure/build/run proof.
- Updated `docs/maintainer_guide.md` proof-owner wording for:
  - static archive `.pc` description checks;
  - stricter runtime output checks for Make/`pkg-config` consumers;
  - no shared imported CMake metadata;
  - no unsupported package/ABI wording in installed `.pc` metadata;
  - exact-version CMake configure/build/run behavior.
- Updated the maintainer guide Windows reviewed CTest count from 54 to 56 so
  docs match the current Windows workflow.
- Preserved non-claims for shared-library packaging, dynamic ABI,
  runtime-loader behavior, package-manager availability, Windows Makefile or
  `pkg-config` parity, macOS/Windows reviewed install/export parity, portable
  performance, and state-of-the-art status.
- Routed Day 12 to focused package validation and generated-output hygiene.

## Day 12 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day12-focused-package-validation.md`
  as the focused package validation artifact.
- Ran shell syntax checks for `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`; they passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran package report validation:
  - `python3 scripts/normalize_report_index.py --family package --check`:
    passed with 6 rows;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`:
    passed with 6 source-controlled advisory rows.
- Ran `bash tests/test_install.sh`; it passed with 23 checks and 0 failures.
- Ran `bash tests/test_cmake_install.sh`; it passed with 26 checks, 0
  failures, and 0 skips.
- Inspected generated-output hygiene:
  `git ls-files -o --exclude-standard` showed only intentional Sprint 143
  planning files, and `git status --ignored --short build` showed `build/`
  remains ignored.
- No Day 12 scoped repairs were needed.
- Routed Day 13 to full quality gate and claim closure: determine whether
  C/header gates are required, rerun required touched-surface checks, and
  publish earned static-first package claims plus residual non-claims.

## Day 13 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day13-quality-gate-claim-closure.md`
  as the quality gate and package/ABI claim closure artifact.
- Confirmed no `.c` or `.h` files changed, so `make format && make lint &&
  make test` was not required for Day 13.
- Ran shell syntax checks for `tests/test_install.sh`,
  `tests/test_cmake_install.sh`, and
  `scripts/static_package_deferral_check.sh`; they passed.
- Ran `bash scripts/static_package_deferral_check.sh`; it passed.
- Ran `bash tests/test_install.sh`; it passed with 23 checks and 0 failures.
- Ran `bash tests/test_cmake_install.sh`; it passed with 26 checks, 0
  failures, and 0 skips.
- Ran package report checks:
  - `python3 scripts/normalize_report_index.py --family package --check`:
    passed with 6 rows;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`:
    passed with 6 source-controlled advisory rows.
- Parsed Linux, macOS, and Windows workflow YAML with Ruby; all parsed.
- Ran a package/docs/workflow claim-boundary scan; matches were explicit
  non-claims, support-tier boundaries, or unrelated bounded algorithm notes.
- Ran `git diff --check` and a trailing-whitespace scan; both passed.
- Published earned static-first package claims:
  static archive install/export, static `pkg-config` metadata, static CMake
  imported target, unsupported shared-artifact absence, `BUILD_SHARED_LIBS`
  rejection, Make/`pkg-config` downstream compile/link/run, CMake downstream
  configure/build/run, exact-version proof, and Linux reviewed package lane.
- Preserved non-claims for shared libraries, dynamic ABI, runtime loader,
  package managers, static/shared selectors, Windows Makefile/`pkg-config`
  parity, macOS/Windows reviewed install/export parity, portable performance,
  and state-of-the-art status.
- Routed Day 14 to final artifact consistency review, final hygiene, and
  Sprint 144 platform-promotion handoff.

## Day 14 Notes

- Created
  `docs/planning/EPIC_12/SPRINT_143/artifacts/day14-closeout-validation-summary.md`
  as the Sprint 143 closeout validation summary and Sprint 144 handoff
  artifact.
- Verified the complete Sprint 143 artifact package is present:
  `PLAN.md`, `WORKING_NOTES.md`, and Day 1 through Day 14 artifacts.
- Mapped Sprint 143 Items 1-7 to closeout evidence:
  ABI feasibility audit, product decision, implementation batch, downstream
  consumer proof, CI/package alignment, documentation alignment, and
  validation/closeout are all complete.
- Re-ran final validation checks:
  - `bash scripts/static_package_deferral_check.sh`: passed;
  - `python3 scripts/normalize_report_index.py --family package --check`:
    passed with 6 rows;
  - `python3 scripts/normalize_report_index.py --family package --check-freshness`:
    passed with 6 source-controlled advisory rows;
  - workflow YAML parse for Linux, macOS, and Windows workflows: passed;
  - final package/docs/workflow/artifact claim-boundary scan: passed with
    only explicit non-claims, support-tier boundaries, sprint planning text,
    or unrelated bounded algorithm notes;
  - `git diff --check`: passed;
  - final trailing-whitespace scan: passed.
- Confirmed generated-output hygiene:
  only intentional Sprint 143 planning files are untracked, and `build/`
  remains ignored.
- Carried forward Day 13 install proof as the current full focused install
  evidence:
  `bash tests/test_install.sh` passed 23 checks with 0 failures, and
  `bash tests/test_cmake_install.sh` passed 26 checks with 0 failures and
  0 skips.
- Confirmed no `.c` or `.h` files changed during Sprint 143, so
  `make format && make lint && make test` was not required.
- Published final implemented package behavior:
  explicit static CMake target, archive-only CMake install metadata, static
  `.pc` metadata, no package selectors, no shared artifacts, no shared imported
  CMake metadata, Make/`pkg-config` downstream compile/link/run, CMake
  downstream configure/build/run, exact-version configure/build/run, and
  source-controlled package proof-owner rows.
- Preserved non-claims for shared-library support, dynamic ABI, runtime
  loader, package-manager distribution, static/shared selectors, Windows
  Makefile/`pkg-config` parity, Windows reviewed install-validation parity,
  macOS reviewed install/export parity, portable performance, and
  state-of-the-art status.
- Routed Sprint 144 to platform-promotion decisions for macOS reviewed
  install/export parity and Windows reviewed install-validation parity, using
  Sprint 143 static-first package semantics as input.
