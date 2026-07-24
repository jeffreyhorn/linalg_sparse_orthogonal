# Sprint 133 Day 14 - Closeout and Package ABI Handoff

## Purpose

Day 14 closes Sprint 133 by publishing the final package/ABI support truth,
residual queues, Sprint 134 handoff notes, and PR review summary material for
the static-first package contract work.

## Final Support Truth

| Surface | Sprint 133 truth |
| --- | --- |
| Package shape | Maintained static archive install/export surface. |
| Make install | Installs `libsparse_lu_ortho.a`, public headers, generated version header, and `sparse.pc`. |
| CMake install/export | Installs `libsparse_lu_ortho.a`, public headers, `SparseConfig.cmake`, `SparseConfigVersion.cmake`, `SparseTargets.cmake`, and `sparse.pc`. |
| CMake target | `Sparse::sparse_lu_ortho` is exported as a static imported target for installed consumers. |
| Version metadata | Single-sourced from `VERSION`; CMake package version behavior is exact-version only. |
| `pkg-config` metadata | Describes the installed static archive link route and package/source version metadata. |
| Shared-library support | Deferred; no `.so`, `.so.*`, `.dylib`, `.dll`, import-library, soname, install-name, or loader-runtime support is claimed. |
| Dynamic ABI compatibility | Deferred; current version metadata is not a broad dynamic ABI guarantee. |
| Package-manager support | Deferred; no Homebrew, apt/deb, rpm/dnf, pacman, vcpkg, Conan, or other package-manager support is claimed. |
| Platform install tiers | Linux remains the strongest reviewed source of truth; macOS carries narrower supplemental static-first Make install/`pkg-config` confidence; Windows remains the reviewed CMake-first consumer subset. |

## Implemented Sprint 133 Changes

| Area | Result |
| --- | --- |
| Product decision | Selected static-first package contract and deferred shared-library/dynamic ABI/package-manager support. |
| CMake enforcement | Changed `BUILD_SHARED_LIBS=ON` from warning-only behavior to configure-time rejection under the static-first contract. |
| Public docs | Updated README and INSTALL wording so shared-library requests are not described as supported. |
| Maintainer docs | Updated package/ABI ownership, proof locations, platform-tier interpretation, and support boundaries. |
| Static deferral proof | Added `scripts/static_package_deferral_check.sh` to guard CMake rejection, static target declaration, absence of shared ABI metadata/selectors, and bounded support wording. |
| CMake consumer proof | Strengthened `tests/test_cmake_install.sh` with exact header count, static imported target checks, installed-prefix include/archive checks, and source/build path leak scans. |
| pkg-config consumer proof | Strengthened `tests/test_install.sh` with exact `.pc` variable, cflag, lib flag, version, `--static`, no-`Libs.private`, unsupported-claim, and downstream compile/run checks. |
| Integrated validation | Day 13 passed all focused package, install, consumer, deferral, syntax, and hygiene gates. |

## Public and Maintainer Wording Alignment

| File | Alignment status |
| --- | --- |
| `README.md` | States downstream consumers can use `pkg-config` or `find_package(Sparse)` against the maintained static package surface and that CMake rejects `BUILD_SHARED_LIBS=ON`. |
| `INSTALL.md` | States the install surface is static-first, lists maintained artifacts, describes package-version metadata, and records shared-library/dynamic ABI deferral. |
| `docs/maintainer_guide.md` | Records the authoritative package/ABI contract, proof ownership, platform tier boundaries, and explicit non-claims. |
| `CMakeLists.txt` | Rejects `BUILD_SHARED_LIBS=ON`, keeps the library target static, and uses exact package-version compatibility. |
| `sparse.pc.in` | Emits static archive link flags and package/source version metadata without shared-library, dynamic ABI, package-manager, or `Libs.private` claims. |
| `cmake/SparseConfig.cmake.in` | Remains a narrow installed package config and does not introduce unsupported shared/static selectors. |

No wording drift remains in the maintained Sprint 133 surfaces.

## Validation Summary

| Command | Result |
| --- | --- |
| `bash -n tests/test_install.sh && bash -n tests/test_cmake_install.sh && bash -n scripts/static_package_deferral_check.sh` | Pass. |
| `git diff --name-only -- '*.c' '*.h'` | No tracked C/header changes. |
| `git ls-files --others --exclude-standard -- '*.c' '*.h'` | No untracked C/header changes. |
| `bash tests/test_install.sh` | Pass: 22 checks, 0 failures. |
| `bash tests/test_cmake_install.sh` | Pass: 21 checks, 0 failures, 0 skips. |
| `bash scripts/static_package_deferral_check.sh` | Pass. |
| `git diff --check` | Pass. |
| Trailing-whitespace scan over Sprint/package surfaces | Pass. |

Because no `.c` or `.h` files changed, `make format && make lint && make test`
was not required by the Sprint 133 validation rule.

## Residual Queue

| Residual | Owner | Blocker or required proof | Support tier |
| --- | --- | --- | --- |
| Shared-library packaging | Future product/package sprint | Build rules, artifact naming, loader/runtime proof, install/export metadata, platform support tiers, and docs. | Deferred non-claim. |
| Dynamic ABI compatibility | Future ABI policy sprint | ABI epoch, public layout policy, symbol inventory, export/import macros, soname/install-name policy, and compatibility tests. | Deferred non-claim. |
| Package-manager recipes | Future distribution sprint | Manager-specific recipes, install roots, dependency metadata, platform proof, upgrade/uninstall behavior, and downstream consumer tests. | Deferred non-claim. |
| CMake static/shared selectors | Future shared-package sprint only if shared support is selected | Package config selection semantics, target naming policy, and consumer tests. | Deferred non-claim. |
| `pkg-config` `Libs.private` split | Future dependency-policy sprint if public/private link semantics change | Dependency classification, static/shared behavior, and downstream link proof. | Deferred for current self-contained static surface. |
| Optional thread/OpenMP package matrix | Future package-validation sprint | Install and consumer proof with `SPARSE_MUTEX` and `SPARSE_OPENMP` modes across Make/CMake. | Not part of default package contract. |
| Reviewed CI promotion for install proofs | Future CI ownership sprint | Runtime cost budget, platform runners, failure triage ownership, and reviewed-platform scope. | Local proof today. |

## Sprint 134 Handoff

Recommended Sprint 134 starting position:

1. Treat Sprint 133 static-first package support as the maintained baseline.
2. Keep `BUILD_SHARED_LIBS=ON` rejection unless a new product decision funds the full shared-library proof stack.
3. Use `tests/test_install.sh`, `tests/test_cmake_install.sh`, and `scripts/static_package_deferral_check.sh` as the local package-contract gates for any package-surface change.
4. Do not add public shared-library, dynamic ABI, package-manager, platform install parity, or runtime-loader claims without matching implementation and validation.
5. If package work continues, the highest-value next steps are CI promotion analysis for the local install proofs, optional `SPARSE_MUTEX`/`SPARSE_OPENMP` package-metadata matrix proof, and explicit future distribution criteria.

## PR Review Summary Material

Sprint 133 implements a static-first package/ABI product decision and proof
hardening pass:

- preserves static archive install/export as the maintained package shape;
- rejects `BUILD_SHARED_LIBS=ON` at CMake configure time;
- updates README, INSTALL, and maintainer support wording;
- adds a static package deferral guard script;
- strengthens CMake installed-package consumer validation;
- strengthens Make install and `pkg-config` downstream consumer validation;
- records integrated validation evidence and residual support queues;
- leaves shared-library packaging, dynamic ABI compatibility, package-manager
  support, and broader platform install parity as explicit non-claims.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| The selected package contract is clear to users and maintainers. | Complete | README, INSTALL, maintainer guide, CMake behavior, package proofs, and closeout artifact agree on static-first support. |
| Residual package/ABI work has owners, blockers, and support-tier boundaries. | Complete | Residual queue records future owners, blockers, and deferred/non-claim status. |
| Sprint 133 can close without unresolved support wording drift. | Complete | Wording scan, static deferral proof, validation artifact, and final hygiene checks passed. |
