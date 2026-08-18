# Sprint 165 Day 12: Quality Gate And Drift Scan

## Purpose

Day 12 validates the accumulated Sprint 165 package-boundary work before the
closeout days. Because the current sprint change set includes
`include/sparse_cholesky.h`, the full source quality gate was required in
addition to focused install/package checks and claim-drift scans.

## Changed Surface Reviewed

The local changed tracked files at the start of the Day 12 gate were:

- `CMakeLists.txt`
- `INSTALL.md`
- `README.md`
- `docs/maintainer_guide.md`
- `include/sparse_cholesky.h`
- `scripts/static_package_deferral_check.sh`
- `tests/test_install.sh`

The Day 12 scan also covered `docs/planning/EPIC_14/SPRINT_165/` so sprint
artifacts and working notes could not introduce unsupported package claims.

## Required Full Gate

Command:

```sh
make format && make lint && make test
```

Result: passed.

Evidence:

- `make format` completed successfully.
- strict warning compile completed successfully.
- `clang-tidy` completed successfully; visible warning totals were the
  existing suppressed non-user-code/NOLINT analyzer output.
- `cppcheck` completed successfully across source and test files.
- `make test` completed successfully and ended with `All tests passed.`

## Focused Package Checks

Command:

```sh
bash scripts/static_package_deferral_check.sh
```

Result: passed.

Evidence:

- `BUILD_SHARED_LIBS` rejection remained fail-closed.
- static target and static install metadata checks passed.
- shared export/ABI metadata remained absent.
- package metadata retained no static/shared selector.
- support wording and Windows package non-claim wording remained bounded.
- Windows workflow retained no unselected package execution.

Command:

```sh
bash tests/test_install.sh
```

Result: passed with 23 passes and 0 failures.

Evidence:

- static library, all 19 headers, and `sparse.pc` installed;
- no shared-library artifacts installed;
- `pkg-config` resolved the installed package and exact version `2.2.0`;
- prefix, libdir, includedir, cflags, and libs matched installed paths;
- `--static` libs matched the current self-contained link flags;
- no private dependency stanza or unsupported package/ABI wording appeared;
- generated and maintained downstream consumers compiled, linked, and ran;
- uninstall removed library, headers, and `sparse.pc`.

Command:

```sh
bash tests/test_cmake_install.sh
```

Result: passed with 27 passes, 0 failures, and 0 skips.

Evidence:

- CMake configure, build, and install succeeded;
- installed archive, headers, CMake package files, and `sparse.pc` were
  present;
- installed CMake target metadata remained static and install-prefix based;
- no shared imported metadata, unsupported loader/static-shared selector
  metadata, source-tree paths, or build-tree paths appeared;
- maintained CMake example configured, built, and ran through
  `find_package(Sparse)`;
- exact-version consumer configured, built, and ran;
- mismatched same-major lower package version was rejected.

## Report Index Freshness

Commands:

```sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
```

Result: passed.

Evidence:

- package report index check reported 6 rows ok;
- package freshness reported freshness ok for 6 source-controlled advisory
  rows.

## Claim Drift Scan

Command:

```sh
rg -n 'BUILD_SHARED_LIBS|shared-library|shared library|dynamic ABI|dynamic-ABI|ABI guarantee|ABI stable|ABI stability|binary compatible|runtime-loader|dynamic-loader|SONAME|install-name|RPATH|DLL|import-library|package-manager|Homebrew|apt|dnf|pacman|vcpkg|conan|pkg-config execution parity|Makefile parity|metadata-only|static/shared selector' \
  CMakeLists.txt README.md INSTALL.md docs/maintainer_guide.md \
  include/sparse_cholesky.h scripts/static_package_deferral_check.sh \
  tests/test_install.sh docs/planning/EPIC_14/SPRINT_165
```

Result: reviewed; no unsupported claims found.

The hits were expected:

- CMake `BUILD_SHARED_LIBS=ON` rejection and missing-policy diagnostics;
- static-package guard checks that reject unsupported shared-library, ABI,
  runtime-loader, package-manager, and static/shared selector drift;
- install test checks that reject unsupported wording in installed metadata;
- README, INSTALL, and maintainer-guide text preserving explicit non-claims;
- sprint planning and artifact evidence that records the same boundaries.

No changed file was found to claim shared-library support, dynamic ABI
compatibility, runtime-loader behavior, package-manager distribution, Windows
Makefile parity, Windows `pkg-config` command parity, or a supported
static/shared package selector.

## Whitespace Check

Command:

```sh
git diff --check
```

Result: passed.

## Residual Boundaries

- Hosted Linux/macOS/Windows CI remains the authority for hosted platform lane
  behavior.
- Windows `sparse.pc` validation remains metadata-only inspection.
- Windows Makefile install/uninstall parity and Windows `pkg-config` command
  execution parity remain deferred non-claims.
- Shared-library support, dynamic ABI compatibility, runtime-loader behavior,
  package-manager distribution, and static/shared package selector UX remain
  deferred product decisions.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Checks pass or blockers are documented. | Complete | Full gate, package checks, report index checks, and `git diff --check` passed. |
| Changed files do not introduce unsupported package or ABI claims. | Complete | Drift scan found only expected guard, rejection, non-claim, and artifact text. |
| Validation evidence is sufficient for closeout. | Complete | Day 12 records full quality, install/downstream, CMake install, report-index, and claim-boundary evidence. |
