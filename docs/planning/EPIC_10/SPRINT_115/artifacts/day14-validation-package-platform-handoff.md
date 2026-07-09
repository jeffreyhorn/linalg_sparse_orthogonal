# Day 14: Validation and Package/Platform Handoff

## Purpose

Day 14 closes Sprint 115 by reviewing the sprint artifacts, validating the
touched surface, and publishing the final package/platform truth for Sprint 116
adoption QA and Sprint 117 Epic closeout.

## Touched Surface

Sprint 115 touched only planning documentation:

- `docs/planning/EPIC_10/SPRINT_115/PLAN.md`
- `docs/planning/EPIC_10/SPRINT_115/WORKING_NOTES.md`
- `docs/planning/EPIC_10/SPRINT_115/artifacts/*.md`
- `docs/planning/EPIC_10/PROJECT_PLAN.md` for the Sprint 117 residual-queue
  follow-up.

Sprint 115 did not change:

- `.c` or `.h` files;
- public headers or installed headers;
- `CMakeLists.txt`;
- `Makefile`;
- GitHub workflows;
- package metadata templates;
- install validation scripts;
- source-list metadata;
- helper targets;
- CTest registration or reviewed test counts.

## Final Decision Matrix

| Surface | Sprint 115 close decision | Resulting claim |
|---|---|---|
| Linux install proof | Keep `tests/test_install.sh` and `tests/test_cmake_install.sh` as local Unix-side proof. | No reviewed Linux install CI lane. |
| macOS CMake install/export | Defer reviewed macOS CMake install/export parity. | macOS install/export parity is not claimed. |
| Windows install-validation | Defer separate reviewed Windows install-validation lane. | Windows remains reviewed CMake-first consumer subset only. |
| Windows thread/fuzz/property | Keep `test_threads`, `test_sprint4_integration`, and `test_fuzz` staged. | No Windows thread/fuzz/property parity claim. |
| macOS backend/toolchain | Keep Apple Clang reviewed; keep Homebrew GCC/install/coverage/TSan supplemental or local. | No new macOS reviewed backend/toolchain lane. |
| Shared-library/dynamic ABI | Keep static-first package story; defer shared/dynamic ABI. | No shared-library or dynamic ABI promise. |
| Package managers | Keep package-manager support future work. | No Homebrew/vcpkg/distro/Windows package-manager claim. |
| Sprint 114 package/platform residual | Consumed as claim-fence validation only. | Non-package proof-owner residuals stay out of Sprint 115. |

## Unsupported-Claim Checklist

Sprint 115 closes without claiming:

- reviewed Linux install CI proof;
- full reviewed macOS CMake install/export parity;
- Windows installed-package support;
- Windows Makefile parity;
- Windows `pkg-config` parity;
- Windows thread/fuzz/property parity;
- macOS coverage reviewed-lane support;
- macOS TSan reviewed-lane support;
- Homebrew GCC reviewed-lane parity;
- shared-library package support;
- dynamic ABI compatibility;
- SONAME/SOVERSION policy;
- DLL/import-library support;
- dylib install-name/rpath support;
- runtime-loader validation;
- Homebrew formula support;
- vcpkg port support;
- distro package support;
- Chocolatey, winget, MSYS2, Conan, or Spack support;
- public API or install-header expansion.

## Sprint 116 Adoption QA Handoff

Sprint 116 should treat this artifact as the package/platform truth source for
adoption-facing documentation. User-facing docs may describe the maintained
static-first install/export surface, `pkg-config`, and `find_package(Sparse)`,
but should not advertise package-manager installation, shared libraries,
dynamic ABI compatibility, Windows install-validation parity, full macOS
install/export parity, or broader Windows thread/fuzz/property parity.

## Sprint 117 Closeout Handoff

Sprint 117 should carry Sprint 114's non-package residual queue unless it
explicitly promotes one bounded item with full proof. That queue remains:

- eigensolver private-owner movement;
- `s20_select_indices` movement;
- `s20_lift_ritz_vectors` movement;
- shift-invert setup/conversion movement;
- `lanczos_iterate_op` movement;
- broad direct/iterative generated-RHS oracle abstraction;
- broad SVD proof-helper abstraction.

Any promotion should include exact old/new owner files, source-list updates,
CMake updates, focused consumer proof, reviewed CTest count evidence where
applicable, and rollback instructions.

## Final Metrics

| Metric | Sprint 115 close state |
|---|---:|
| artifact files | 14 |
| working notes files | 1 |
| plan files | 1 |
| changed source/header files | 0 |
| changed workflow files | 0 |
| changed Make/CMake/package metadata files | 0 |
| reviewed CTest membership changes | 0 |
| public/install-header changes | 0 |
| package-manager recipes added | 0 |
| shared-library build rules added | 0 |
| dynamic ABI claims added | 0 |

## Validation

Documentation hygiene for the Sprint 115 package passed:

- `git diff --check`
- trailing-whitespace scan over `docs/planning/EPIC_10/SPRINT_115`

No `.c` or `.h` files changed, so the full C quality gate
`make format && make lint && make test` was not required.
