# Day 1 Residual Package and Platform Intake

## Purpose

Day 1 turns Sprint 112's package/platform deferred debt and Sprint 114's
package/platform-facing residual into a Sprint 115 execution boundary. The
main outcome is a duplicate fence: completed package proof remains evidence,
not unresolved debt, and non-package source-boundary work remains deferred out
of Sprint 115.

## Source Evidence Reviewed

| Source | Relevant evidence |
|---|---|
| `docs/planning/EPIC_10/PROJECT_PLAN.md` Sprint 115 | Ten Sprint 115 items and 168-hour project-plan scope. |
| `docs/planning/EPIC_10/SPRINT_115/PLAN.md` Day 1 | Required Day 1 tasks, deliverables, and completion criteria. |
| Sprint 112 retrospective | Package/platform residual deferred debt and explicit non-claims. |
| Sprint 112 package/install artifacts | Static-first decision, Make install proof, CMake install/export proof, downstream consumer proof, platform-tier contract, and validation closeout. |
| Sprint 114 retrospective | Package/platform-facing residual and non-package residuals. |
| Epic 10 Sprint 114 residual deferral decision | Routes package/platform residuals to Sprint 115 and source-boundary/helper-abstraction residuals to Sprint 117. |

## Duplicate-Work Exclusion Fence

The following Sprint 112 work is evidence for Sprint 115, not unresolved work:

| Excluded work | Why excluded | Sprint 115 handling |
|---|---|---|
| Package surface audit | Completed in Sprint 112. | Use as baseline for promotion decisions. |
| Static-first versus shared-library/ABI support decision | Completed in Sprint 112. | Revisit only as explicit product-contract future-work decision. |
| Install/consumer proof design | Completed in Sprint 112. | Use as proof template for reviewed-lane decisions. |
| Make install and pkg-config proof | Completed locally in Sprint 112. | Decide whether local proof should become reviewed CI evidence. |
| CMake install/export proof | Completed locally in Sprint 112. | Decide whether platform reviewed lanes should be added. |
| Downstream consumer proof | Completed in Sprint 112. | Use as installed-consumer proof baseline. |
| Platform-tier contract | Completed in Sprint 112. | Preserve unless Sprint 115 earns stronger reviewed evidence. |
| Windows reviewed-scope follow-through | Completed in Sprint 112. | Start from staged exclusions; do not claim broader parity. |
| macOS package/platform follow-through | Completed in Sprint 112. | Start from staged exclusions and toolchain notes. |
| Packaging documentation alignment | Completed in Sprint 112. | Update only when Sprint 115 decisions change support truth. |
| Integrated package/platform validation | Completed in Sprint 112. | Use as validation pattern. |
| Sprint 112 closeout and handoff | Completed in Sprint 112. | Use residual queue as Sprint 115 input. |

## Sprint 114 Deferral Boundary

Sprint 115 consumes only Sprint 114's package/platform-facing residual:

- verify package, ABI, Windows, CMake parity, install-header, and adoption
  claims remain fenced unless Sprint 115 adds reviewed evidence.

The following Sprint 114 residuals are explicitly out of Sprint 115 scope:

- eigensolver private-owner movement;
- `s20_select_indices` movement;
- `s20_lift_ritz_vectors` movement;
- shift-invert setup/conversion movement;
- `lanczos_iterate_op` movement;
- broad direct/iterative oracle abstraction;
- broad SVD proof-helper abstraction.

Those items remain routed to Sprint 117 residual queue review and post-Epic
handoff unless Sprint 117 explicitly promotes one item.

## Remaining Package/Platform Owners

| Owner | Primary surfaces | Dependency | Planned day(s) |
|---|---|---|---:|
| Linux install proof CI promotion | `tests/test_install.sh`, `tests/test_cmake_install.sh`, `.github/workflows/ci.yml` | Sprint 112 local install proof. | 2-3 |
| macOS CMake install/export parity | `.github/workflows/macos-ci.yml`, CMake install/export proof, README/INSTALL wording | Linux/local proof boundary and macOS staged exclusions. | 4-5 |
| Windows install validation | `.github/workflows/windows-ci.yml`, CMake install/export proof, downstream consumer proof | Windows reviewed CMake subset and staged install-validation exclusion. | 6-7 |
| Windows thread/fuzz portability | `tests/test_threads.c`, `tests/test_sprint4_integration.c`, `tests/test_fuzz.c`, CTest membership | Windows staged exclusions and reviewed-count contract. | 8-9 |
| macOS backend/toolchain follow-through | `Makefile`, `.github/workflows/macos-ci.yml`, README/INSTALL wording | macOS package/platform follow-through from Sprint 112. | 10 |
| Shared-library/dynamic ABI contract | `CMakeLists.txt`, `Makefile`, `cmake/SparseConfig.cmake.in`, README/INSTALL | Static-first support decision. | 11 |
| Package-manager support decision | README/INSTALL/package docs and any recipe references | Static-first install truth and platform support tiers. | 12 |
| Sprint 114 package/platform residual intake | Sprint 114 retrospective and Epic 10 deferral decision | All Sprint 115 package/platform decisions. | 13 |
| Validation and handoff | all touched CI/build/script/doc/code surfaces | Days 1-13 decisions. | 14 |

## Affected Surface Inventory

| Surface | Files / paths | Sprint 115 risk |
|---|---|---|
| Local install scripts | `tests/test_install.sh`, `tests/test_cmake_install.sh` | Local proof may be promoted to reviewed CI or remain local-only. |
| Linux CI | `.github/workflows/ci.yml` | Reviewed install lane could add runtime and environment risk. |
| macOS CI | `.github/workflows/macos-ci.yml` | CMake install/export, coverage backend, GCC, libomp, and TSan claims need evidence. |
| Windows CI | `.github/workflows/windows-ci.yml` | Installed-package proof, reviewed CTest count, and staged exclusions require precision. |
| CMake package surface | `CMakeLists.txt`, `cmake/SparseConfig.cmake.in`, `examples/cmake_example/CMakeLists.txt` | Install/export and static-first package claims must remain coherent. |
| Make package surface | `Makefile` | install/uninstall, pkg-config, sanitizer, coverage, and macOS toolchain notes may need updates. |
| Adoption docs | `README.md`, `INSTALL.md`, `docs/` package/platform references | Public claims must match reviewed evidence. |
| Windows portability tests | `tests/test_threads.c`, `tests/test_sprint4_integration.c`, `tests/test_fuzz.c` | Moving staged exclusions into reviewed scope can change CTest membership. |
| Platform helper comments | `tests/test_framework.h` and test skip comments | Comments should not imply support broader than reviewed lanes. |

## Dependency Order

1. Intake and duplicate fence.
2. Linux local install proof promotion/no-promotion.
3. macOS install/export parity proof or deferral.
4. Windows install-validation proof or deferral.
5. Windows thread/fuzz portability proof or staged-exclusion decision.
6. macOS backend/toolchain follow-through.
7. Shared-library/dynamic ABI product-contract decision.
8. Package-manager support decision.
9. Sprint 114 package/platform residual intake and deferral boundary.
10. Final validation and handoff to Sprint 116 adoption QA and Sprint 117
    closeout.

## Non-Claims Preserved

- No shared-library package claim.
- No dynamic ABI compatibility claim.
- No SONAME/SOVERSION or symbol export stability claim.
- No runtime-loader behavior claim.
- No package-manager support claim.
- No Windows installed-package support claim without reviewed proof.
- No macOS full install/export parity claim without reviewed proof.
- No Windows thread/fuzz/property parity claim without reviewed proof.
- No public API or install-header change.
- No Sprint 114 eigensolver/source-boundary or broad helper-abstraction work
  is moved into Sprint 115.

## Day 1 Validation

Day 1 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 1 because no `.c` or `.h` files
changed.
