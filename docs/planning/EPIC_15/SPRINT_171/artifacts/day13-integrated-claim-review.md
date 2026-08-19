# Sprint 171 Day 13: Integrated Claim Review

## Purpose

Day 13 reconciles Sprint 171 package-manager decision artifacts, guard scripts,
documentation updates, validation results, and staging hygiene into one
coherent package-manager claim boundary.

## Integrated Decision

Sprint 171 selected formal package-manager deferral. The repository now has one
source-controlled package-manager readiness decision:

- package-manager support is not currently provided;
- source install via Make or CMake remains the maintained path;
- installed `pkg-config` and CMake package metadata remain static archive
  package metadata, not provider package-manager distribution proof;
- no vcpkg, Homebrew, Conan, pkgsrc, distro/system package, provider registry,
  tap, recipe, or binary package support is claimed.

## Artifact Reconciliation

| Artifact | Role | Day 13 Result |
| --- | --- | --- |
| `artifacts/day3-provider-selection.md` | Selected formal package-manager deferral. | Coherent with docs, guards, and validation. |
| `artifacts/day5-package-manager-deferral.md` | Canonical Sprint 171 deferral record. | Required by the executable package-manager deferral guard. |
| `scripts/package_manager_deferral_check.sh` | Enforces package-manager deferral, provider recipe absence, metadata neutrality, and public non-claims. | Coherent with Day 5 decision and Day 11 docs. |
| `scripts/static_package_deferral_check.sh` | Enforces static-first package and shared-library/dynamic ABI deferral. | Remains separate from provider package-manager deferral. |
| `scripts/normalize_report_index.py` | Emits package proof-owner rows. | Includes `package_package_manager_deferral_v1`. |
| `README.md` | Short user front door. | States package-manager support is not currently provided. |
| `INSTALL.md` | Operational package/install guide. | Names unsupported provider families and documents the deferral guard as a non-claim guard. |
| `docs/maintainer_guide.md` | Maintainer policy and evidence interpretation. | Records when maintainers must run the package-manager deferral guard. |
| `artifacts/day12-provider-deferral-validation.md` | Local validation record. | Shows package-manager guard, static guard, install proof, report-index checks, and diff hygiene passed. |

## Claim-Scan Results

Day 13 ran targeted scans across current user-facing docs, package metadata
templates, guard scripts, CMake config, and CI workflow text:

```sh
rg -n "package-manager|package manager|vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|binary package|registry|tap" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md \
  scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh sparse.pc.in cmake/SparseConfig.cmake.in
rg -n "shared-library|dynamic ABI|runtime-loader|BUILD_SHARED_LIBS|static/shared selector|Windows Makefile|Windows.*pkg-config|ABI support|shared ABI" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md \
  scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh sparse.pc.in cmake/SparseConfig.cmake.in \
  CMakeLists.txt .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml
```

Reviewed matches were allowed because they preserve one of these meanings:

- package-manager support is unsupported or deferred;
- provider names appear only as unsupported examples or unrelated platform
  compiler/tooling notes;
- package proof is static-first source install or installed metadata proof;
- Windows package proof remains CMake install/downstream scoped;
- Windows Makefile parity and Windows `pkg-config` command execution parity
  remain non-claims;
- shared-library packaging, dynamic ABI compatibility, runtime-loader
  behavior, and static/shared selectors remain non-claims or guarded
  configure-time rejections.

No scan result required a Day 13 documentation or guard change.

## Generated-Output Staging Check

Day 13 checked for accidental package-manager/provider artifacts outside
planning, `.git`, build directories, and archive content:

```sh
find . \( -path './.git' -o -path './docs/planning' -o -path './build' -o -path './build-*' -o -path './archive' \) -prune -o \
  \( -name 'vcpkg.json' -o -name 'vcpkg-configuration.json' -o -name 'portfile.cmake' \
  -o -name 'conanfile.py' -o -name 'conanfile.txt' -o -path '*/ports/*' \
  -o -path '*/Formula/*' -o -path '*/pkgsrc/*' -o -path '*/debian/control' \
  -o -path '*/debian/rules' -o -path '*/debian/changelog' -o -name '*.spec' \
  -o -name '*.tar.gz' -o -name '*.zip' -o -name '*.deb' -o -name '*.rpm' \
  -o -name '*.pkg' \) -print
```

Result: no generated package artifacts, provider recipes, source archives, or
package outputs were found for staging.

`git status --porcelain=v1` showed only intended Sprint 171 source-controlled
changes:

- README, INSTALL, maintainer guide, and package report-index script updates;
- Sprint 171 planning artifacts and working notes;
- `scripts/package_manager_deferral_check.sh`.

## Residuals

| Residual | Disposition |
| --- | --- |
| No package-manager provider is supported. | Intentional Sprint 171 decision. Future work must select exactly one provider and add recipe, provider proof, downstream consumer proof, cleanup proof, docs, and guards before claiming support. |
| No provider registry submission or binary-package output exists. | Intentional non-claim. Do not add registry, tap, recipe, or binary-package wording without provider proof. |
| Windows remains CMake install/downstream scoped. | Intentional retained boundary. No Windows Makefile parity or Windows `pkg-config` execution parity is implied. |
| Shared-library packaging and dynamic ABI support remain deferred. | Owned by Sprint 170 static package deferral guard, not reopened by Sprint 171. |

## Sprint 172 Handoff

Sprint 172 public-header coherence work should start from these package/adoption
boundaries:

1. Public header/API wording must not imply package-manager availability,
   shared-library support, dynamic ABI guarantees, runtime-loader support, or
   broad platform parity.
2. Any package/adoption wording changes should run:
   `bash scripts/package_manager_deferral_check.sh`.
3. Any static package/shared ABI wording changes should run:
   `bash scripts/static_package_deferral_check.sh`.
4. Package proof-owner index changes should run:
   `python3 scripts/normalize_report_index.py --family package --check` and
   `python3 scripts/normalize_report_index.py --family package --check-freshness`.

## Day 13 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Integrated claim review | Complete | Decision, artifacts, docs, guards, validation, and residuals are reconciled above. |
| Claim-scan results | Complete | Targeted package-manager and shared ABI scans were reviewed. |
| Generated-output staging check | Complete | No generated package outputs or provider recipe files were found. |
| Residual and handoff list | Complete | Residuals and Sprint 172 handoff rules are listed above. |
| Day 13 claim-review artifact | Complete | This file. |

## Validation

Day 13 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
rg -n "package-manager|package manager|vcpkg|Homebrew|Conan|pkgsrc|apt|dnf|pacman|binary package|registry|tap" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md \
  scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh sparse.pc.in cmake/SparseConfig.cmake.in
rg -n "shared-library|dynamic ABI|runtime-loader|BUILD_SHARED_LIBS|static/shared selector|Windows Makefile|Windows.*pkg-config|ABI support|shared ABI" \
  README.md INSTALL.md docs/maintainer_guide.md docs/tutorial.md docs/api_reference.md docs/cookbook.md \
  scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh sparse.pc.in cmake/SparseConfig.cmake.in \
  CMakeLists.txt .github/workflows/ci.yml .github/workflows/macos-ci.yml .github/workflows/windows-ci.yml
find . \( -path './.git' -o -path './docs/planning' -o -path './build' -o -path './build-*' -o -path './archive' \) -prune -o \
  \( -name 'vcpkg.json' -o -name 'vcpkg-configuration.json' -o -name 'portfile.cmake' \
  -o -name 'conanfile.py' -o -name 'conanfile.txt' -o -path '*/ports/*' \
  -o -path '*/Formula/*' -o -path '*/pkgsrc/*' -o -path '*/debian/control' \
  -o -path '*/debian/rules' -o -path '*/debian/changelog' -o -name '*.spec' \
  -o -name '*.tar.gz' -o -name '*.zip' -o -name '*.deb' -o -name '*.rpm' \
  -o -name '*.pkg' \) -print
git status --porcelain=v1
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Package-manager claim boundary is internally coherent. | Complete | Decision, docs, guards, report-index rows, and validation all preserve formal deferral. |
| Source install, CMake/`pkg-config`, and provider support remain distinct. | Complete | Source install and installed metadata are documented separately from provider package-manager support. |
| No generated artifacts are staged unintentionally. | Complete | Staging check found no provider recipes, generated archives, or package outputs. |
