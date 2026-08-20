# Sprint 171 Day 8: Package Claim Guard Design

## Purpose

Day 8 designs the package claim guard updates needed after the Sprint 171
package-manager decision. Sprint 171 selected formal package-manager deferral,
so the guard design must keep source install, CMake/`pkg-config` install, and
package-manager support as separate claim surfaces.

## Current Guard Baseline

| Guard Surface | Current Owner | Current Scope |
| --- | --- | --- |
| Unix Make install and `pkg-config` proof | `tests/test_install.sh` | Installed static archive, headers, `.pc` metadata, downstream compile/link/run, exact version behavior, and uninstall cleanup. |
| Unix CMake install/export proof | `tests/test_cmake_install.sh` | Installed static archive, exported CMake target, `find_package(Sparse)`, exact version behavior, and downstream compile/link/run. |
| Static package and shared ABI deferral | `scripts/static_package_deferral_check.sh` | Static-first package posture, `BUILD_SHARED_LIBS=ON` rejection, no shared-library metadata, and shared ABI non-claims. |
| Package-manager deferral | `scripts/package_manager_deferral_check.sh` | Formal Sprint 171 deferral record, provider recipe absence, package metadata neutrality, and public non-claim wording. |
| Normalized package proof-owner rows | `scripts/normalize_report_index.py --family package` | Source-controlled ownership rows for package proof scripts/templates. |

## Claim Boundaries To Preserve

The guard update must preserve these distinctions:

- source install is not package-manager support;
- installed `sparse.pc` metadata is not provider registry support;
- installed CMake package exports are not Homebrew, vcpkg, Conan, pkgsrc, or
  distro package recipes;
- Windows CMake install/downstream proof is not Windows Makefile parity or
  Windows `pkg-config` command execution parity;
- static archive install proof is not shared-library packaging, dynamic ABI
  compatibility, runtime-loader compatibility, or static/shared selector
  support.

## Positive Checks For The Selected Deferral Path

Day 9 implementation should ensure the maintained guard surface positively
checks:

1. The Sprint 171 Day 5 package-manager deferral record exists.
2. The deferral record states that package-manager support is formally
   deferred.
3. The deferral record names unsupported provider families: vcpkg, Homebrew,
   Conan, pkgsrc, Debian/Fedora/system packages, provider registries, taps,
   recipes, and binary packages.
4. The deferral record lists evidence needed to revisit provider support,
   including selected provider, source/archive input, checksum policy,
   license/version metadata, dependency policy, provider recipe, isolated
   install proof, downstream consumer proof, cleanup proof, docs, and guard
   coverage.
5. Public docs keep package-manager support as an explicit non-claim.
6. Package metadata templates stay provider-neutral.
7. Normalized package proof-owner rows include the package-manager deferral
   guard once Day 9 promotes it into the report-index surface.

## Negative Checks For Unsupported Claims

Day 9 implementation should keep or add failures for:

- provider recipe artifacts appearing without a support decision:
  `vcpkg.json`, `vcpkg-configuration.json`, `portfile.cmake`, `conanfile.py`,
  `conanfile.txt`, `Formula/`, `ports/`, `pkgsrc/`, `debian/`, and RPM spec
  files outside planning or archived historical material;
- package metadata templates mentioning provider support, package-manager
  distribution, binary packages, registry readiness, or package-manager
  support;
- public docs implying package-manager support without a provider proof;
- shared-library support, dynamic ABI compatibility, runtime-loader support,
  static/shared selectors, or `BUILD_SHARED_LIBS` support appearing without a
  later product decision;
- Windows package-manager, Makefile, or `pkg-config` execution parity wording
  being inferred from the maintained Windows CMake package proof.

## Surfaces To Update On Day 9

| Surface | Intended Day 9 Change | Rationale |
| --- | --- | --- |
| `scripts/package_manager_deferral_check.sh` | Keep as the focused executable guard; tighten only if Day 8 review found a missing assertion. | This script owns provider deferral enforcement. |
| `scripts/static_package_deferral_check.sh` | Preserve existing shared-library/static-first checks; do not duplicate provider-specific checks unless needed. | Avoid mixing shared ABI deferral with package-manager provider deferral. |
| `scripts/normalize_report_index.py` | Add `package_manager_deferral` as a package proof owner row. | Makes the new guard visible in normalized package evidence. |
| `INSTALL.md` normalized package rows | Add the new package-manager deferral guard to the listed proof owners. | Keeps user-facing proof-owner documentation aligned with generated package rows. |
| `docs/maintainer_guide.md` package ownership section | Add the new guard to focused install/package regression ownership and normalized package row interpretation. | Gives maintainers a direct owner for package-manager non-claims. |

## Validation Commands For Day 9

Day 9 should run:

```sh
bash -n scripts/package_manager_deferral_check.sh
bash scripts/package_manager_deferral_check.sh
bash scripts/static_package_deferral_check.sh
python3 scripts/normalize_report_index.py --family package --check
python3 scripts/normalize_report_index.py --family package --check-freshness
git diff --check
```

If Day 9 changes any `.c` or `.h` files, it must also run:

```sh
make format
make lint
make test
```

No `.c` or `.h` changes are expected for the Day 9 guard implementation.

## Day 8 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Package claim guard design | Complete | This artifact defines the guard split and Day 9 implementation surface. |
| Positive and negative check lists | Complete | Checks cover deferral evidence, provider recipe absence, package metadata neutrality, public non-claims, and shared ABI boundaries. |
| Validation command list | Complete | Focused script, report-index, and diff-hygiene commands are listed above. |
| Day 8 claim-guard design artifact | Complete | This file. |

## Validation

Day 8 changed planning artifacts only. No `.c` or `.h` files were modified, so
the full C quality gate is not required for this day.

Validation command:

```sh
git diff --check
```

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Guard changes are scoped before implementation. | Complete | Day 9 implementation surfaces and expected changes are identified. |
| Package-manager support cannot be inferred from static source install. | Complete | Source install, CMake/`pkg-config`, and provider package-manager support are explicitly separated. |
| Unsupported shared-library/ABI claims remain protected. | Complete | Existing Sprint 170 shared-library/static-first guard remains the owner for shared ABI non-claims. |
