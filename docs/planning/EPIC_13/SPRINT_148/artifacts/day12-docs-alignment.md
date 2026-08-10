# Sprint 148 Day 12: Documentation Alignment

## Purpose

Align public and maintainer-facing Windows support wording with the Sprint 148
promotion of the former staged Windows CMake tests.

## Updated Documents

| Document | Change |
| --- | --- |
| `README.md` | Updated the CI summary so Windows is still described as CMake-first, but no longer says pthread/POSIX-backed tests are outside the reviewed subset. |
| `INSTALL.md` | Updated the supported-platforms Windows row to name the promoted `test_threads`, `test_sprint4_integration`, and `test_fuzz` targets while preserving the install/parity non-claims. |
| `docs/maintainer_guide.md` | Updated the platform-confidence owner text from `56` to `59` registered Windows CTest tests and replaced the stale staged-exclusion interpretation. |

## Windows Support Statement After Day 12

The reviewed Windows lane remains CMake-first:

- Visual Studio 2022 / MSVC CMake configure and build;
- `ctest -N` enumeration with `EXPECTED_WINDOWS_CTEST_COUNT=59`;
- full hosted Windows `ctest` execution;
- promoted coverage for `test_threads`, `test_sprint4_integration`, and
  `test_fuzz`.

The promoted tests are no longer documented as staged Windows exclusions.

## Preserved Non-Claims

Day 12 did not broaden Windows support claims beyond the reviewed CMake lane.
The documentation still avoids claiming:

- Windows Makefile parity;
- Windows `pkg-config` parity;
- separate reviewed Windows install-validation parity;
- package-manager support;
- shared-library support;
- dynamic ABI support;
- broad Windows parity beyond the hosted MSVC CMake proof surface.

## Sprint 149 Handoff

Sprint 149 can treat the staged-test portability gap as closed for the reviewed
Windows CMake lane. It should keep the install-validation decision separate:
promoting Windows install/package parity still requires its own evidence and
should not be inferred from Sprint 148 test promotion.

## Validation

Documentation-focused validation for Day 12:

- stale public/support wording search for old Windows staged-exclusion claims:
  passed;
- whitespace check over touched docs and Sprint 148 artifacts: passed;
- `git diff --check`: passed.

No `.c` or `.h` files were edited on Day 12, so the full C quality gate was
not required for this documentation-only update.
