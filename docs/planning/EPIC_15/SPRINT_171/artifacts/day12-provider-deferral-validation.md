# Sprint 171 Day 12: Provider Or Deferral Validation

## Purpose

Day 12 validates the selected Sprint 171 package-manager path. Because Sprint
171 selected formal package-manager deferral, validation focuses on the local
deferral guard, retained static-first package guard, Unix Make
install/`pkg-config` proof after install-doc changes, normalized package
proof-owner rows, and quality-gate applicability.

## Validation Matrix

| Validation | Command | Result |
| --- | --- | --- |
| Package-manager deferral guard | `bash scripts/package_manager_deferral_check.sh` | Passed |
| Static package/shared ABI deferral guard | `bash scripts/static_package_deferral_check.sh` | Passed |
| Unix Make install/`pkg-config` proof | `bash tests/test_install.sh` | Passed, 23 passed and 0 failed |
| Normalized package proof-owner rows | `python3 scripts/normalize_report_index.py --family package --check` | Passed, 7 rows ok |
| Package proof-owner freshness | `python3 scripts/normalize_report_index.py --family package --check-freshness` | Passed, 7 source-controlled rows ok |
| Diff hygiene | `git diff --check` | Passed |

## Install Proof Decision

Day 11 changed install/package documentation and package proof-owner
interpretation, so Day 12 ran the Unix Make install/`pkg-config` proof:

```sh
bash tests/test_install.sh
```

The install proof passed:

- static library installed;
- no shared-library artifacts installed;
- all 19 headers installed;
- `sparse.pc` installed and resolved;
- exact version constraint passed;
- prefix, libdir, includedir, cflags, and libs matched installed filesystem
  paths;
- `.pc` metadata retained the static archive package description;
- unsupported package/ABI claims stayed absent;
- downstream `pkg-config` consumers compiled, linked, and ran;
- uninstall cleanup removed library, headers, and package metadata.

## CMake Install Proof Decision

Day 12 did not rerun `tests/test_cmake_install.sh` because this day did not
change CMake package expectations, CMake install rules, exported target
metadata, or CMake downstream consumer behavior. The normalized package rows
still include the CMake install/export proof owner, and the static/package
guards still validate that the CMake package metadata remains within the
maintained static-first boundary.

## C Quality-Gate Decision

No `.c` or `.h` files were modified for Day 12. The full C quality gate
(`make format && make lint && make test`) is therefore not required by the
Sprint 171 rules for this day.

## Provider Tool Decision

No provider tooling was invoked. This is intentional: Sprint 171 selected
formal package-manager deferral, not a vcpkg, Homebrew, Conan, pkgsrc,
distro/system package, registry, tap, recipe, or binary package path.

## Validation Output Summary

The package-manager deferral guard passed these checks:

- Sprint 171 deferral record exists;
- provider recipe artifacts are absent outside planning/archive locations;
- `sparse.pc.in` and `cmake/SparseConfig.cmake.in` remain provider-neutral;
- README, INSTALL, and maintainer guide preserve package-manager non-claims.

The static package deferral guard passed these checks:

- Sprint 170 static-first product decision remains present;
- `BUILD_SHARED_LIBS=ON` remains rejected;
- `sparse_lu_ortho` remains an explicit static target;
- Makefile and CMake install metadata remain static archive scoped;
- shared-library, dynamic ABI, runtime-loader, and static/shared selector
  support remain deferred.

The normalized package report index passed with seven rows, including:

- `package_make_install_pkg_config_v1`;
- `package_cmake_install_export_v1`;
- `package_pkg_config_template_v1`;
- `package_cmake_package_config_v1`;
- `package_static_package_deferral_v1`;
- `package_package_manager_deferral_v1`;
- `report_contract_package_static_install_package_install_proof_owner_v1`.

## Day 12 Deliverables

| Deliverable | Status | Notes |
| --- | --- | --- |
| Provider/deferral validation log | Complete | Package-manager deferral guard passed. |
| Static package guard validation log | Complete | Static package/shared ABI deferral guard passed. |
| Install proof decision and results | Complete | Unix Make install/`pkg-config` proof passed after install-doc changes. |
| C quality-gate decision | Complete | No `.c` or `.h` edits; full C gate not required. |
| Day 12 validation artifact | Complete | This file. |

## Completion Criteria

| Criterion | Status | Notes |
| --- | --- | --- |
| Selected package-manager path or deferral validates locally. | Complete | Formal deferral guard passed. |
| Install/package proof remains green where relevant. | Complete | `tests/test_install.sh` passed with 23 checks and 0 failures. |
| Failures stop the sprint for user input. | Complete | No failures occurred. |
