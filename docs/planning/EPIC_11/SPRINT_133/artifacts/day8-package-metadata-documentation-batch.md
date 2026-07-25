# Sprint 133 Day 8 - Package Metadata and Documentation Batch

## Purpose

Day 8 aligns package-facing documentation with the selected static-first
contract and the Day 7 CMake behavior change. The implementation still does
not add shared-library or dynamic ABI support.

## Updated Documentation

| File | Update |
| --- | --- |
| `INSTALL.md` | Added the maintained-install-contract statement that CMake rejects `BUILD_SHARED_LIBS=ON`, and added CMake option guidance explaining the static archive contract and deferred shared-library/dynamic-ABI proof. |
| `README.md` | Added the front-door package summary note that CMake rejects `BUILD_SHARED_LIBS=ON` rather than silently treating a shared-library request as supported. |
| `docs/maintainer_guide.md` | Added maintainer package-truth bullets recording `BUILD_SHARED_LIBS=ON` as a configure-time rejection under the static-first contract and noting the Sprint 133 change from warning-only behavior. |

No CMake package metadata, `pkg-config` metadata, public headers, install
scripts, workflows, package-manager files, or source files changed on Day 8.

## Package Truth After Day 8

| Surface | Current truth |
| --- | --- |
| Maintained package shape | Static archive install/export surface. |
| CMake shared request | `BUILD_SHARED_LIBS=ON` fails configure with shared-library/dynamic-ABI deferral wording. |
| CMake consumer route | `find_package(Sparse REQUIRED)` plus `Sparse::sparse_lu_ortho`. |
| `pkg-config` consumer route | `pkg-config --cflags --libs sparse`, linking `-lsparse_lu_ortho`. |
| Version metadata | Repo `VERSION` propagated through generated header, CMake package version, and `sparse.pc`; not ABI compatibility policy. |
| Shared-library support | Deferred and unclaimed. |
| Dynamic ABI compatibility | Deferred and unclaimed. |
| Package-manager support | Deferred and unclaimed. |

## Ambiguity Cleanup

| Ambiguity | Day 8 handling |
| --- | --- |
| `BUILD_SHARED_LIBS=ON` could be interpreted as a supported shared mode because CMake accepted it before Day 7. | Public and maintainer docs now say it is intentionally rejected. |
| Package version metadata could be interpreted as ABI compatibility. | Maintainer guidance still states exact package version is not a dynamic-ABI guarantee. |
| Installed CMake and `pkg-config` consumers could imply package-manager support. | No package-manager language was added; package-manager support remains a residual non-claim. |
| Static/shared package selection could be implied by CMake options. | Docs describe only the static archive contract and deferred shared proof. |

## Metadata Decision

Day 8 did not add new metadata fields to `sparse.pc` or
`cmake/SparseConfig.cmake.in`.

Reason: the Day 5 decision selected static-first support, and Day 6 rejected
adding unsupported static/shared selectors or ABI fields. Day 11 and Day 12
remain the right owners for tighter consumer proof of existing CMake and
`pkg-config` metadata.

## Validation

| Check | Result |
| --- | --- |
| `git diff --check` | Pass |
| Sprint 133 markdown trailing-whitespace scan | Pass |
| Claim-boundary scan | Pass; shared-library, dynamic ABI, and package-manager wording remains deferred/non-claim language. |

No `.c` or `.h` files changed, so `make format && make lint && make test` is
not required for Day 8. The Day 8 docs do not change install behavior, so the
Day 7 `bash tests/test_cmake_install.sh` result remains the latest behavior
validation for the CMake change.

## Day 9 Handoff

Day 9 should design the ABI/symbol/static-deferral proof around the selected
static-first contract:

- shared artifacts remain absent;
- `BUILD_SHARED_LIBS=ON` is an explicit configure-time deferral;
- no `SPARSE_API`, symbol map, soname/install-name, ABI epoch, or shared
  package selector should appear without a future product decision;
- generated version metadata must remain package/source metadata, not dynamic
  ABI proof;
- proof should be local and specific enough to catch support wording drift
  without inventing shared support.

## Completion Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Public support wording matches the Day 5 decision. | Complete | README and INSTALL now state static-first support and `BUILD_SHARED_LIBS=ON` rejection. |
| Package metadata does not imply unsupported ABI or package-manager claims. | Complete | Day 8 made no metadata-field additions and preserved shared ABI/package-manager non-claims. |
| Maintainers have clear validation commands for the selected contract. | Complete | Maintainer guide records the static-first rejection behavior; Day 7 and Day 8 artifacts point to the focused configure probe and `tests/test_cmake_install.sh`. |
