# Day 9: CMake Installed-Consumer Matrix

**Date:** 2026-10-24

**Scope:** Sprint 219 item 219.4, 10 hours
**Host:** macOS, native CMake static-library install

## Fixture And Package Identity

`tests/test_cmake_idx_width_install.sh` builds separate 32- and 64-bit
archives, installs each into a clean prefix, and validates each package with
`check_static_install_contract.py` before relocating it. The validator checks
the generated header, exported target, and actual archive query agree on the
index width. The downstream fixture in `tests/cmake_width_consumer` requests
the exact version from `VERSION`, imports `Sparse::sparse_lu_ortho`, and links
the maintained CMake example. No consumer width is specified in the positive
case; the target selects it.

| Package | Positive result | Rejection case | Safe phase |
| --- | --- | --- | --- |
| 32-bit | Relocated consumer reports width 32, package version, and `OK` | Explicit 64-bit selector, via C flags and target definition | Compile |
| 32-bit | Own archive passes static identity validation | 64-bit header ahead of target include path | Compile |
| 32-bit | Own archive runs the solver | Imported target redirected to 64-bit archive | Executable pre-use check, before matrix calls |
| 32-bit | Exact installed version resolves | Wrong exact version (`2.3.0`) | Configure |
| 32-bit | Fresh cache finds 32-bit package | Reconfigure for 64-bit prefix and selector | Compile rejection if still bound to 32-bit package; otherwise verified 64-bit rebind and solve |
| 64-bit | Relocated consumer reports width 64, package version, and `OK` | Explicit 32-bit selector, via C flags and target definition | Compile |
| 64-bit | Own archive passes static identity validation | 32-bit header ahead of target include path | Compile |
| 64-bit | Own archive runs the solver | Imported target redirected to 32-bit archive | Executable pre-use check, before matrix calls |
| 64-bit | Exact installed version resolves | Wrong exact version (`2.3.0`) | Configure |
| 64-bit | Fresh cache finds 64-bit package | Reconfigure for 32-bit prefix and selector | Compile rejection if still bound to 64-bit package; otherwise verified 32-bit rebind and solve |

The archive override is fixture-only: it changes the imported target's
location to the opposite archive and verifies the verbose link uses that
path. The resulting executable exits with `consumer=N header=N archive=M`
before printing a version, solution, or `OK`. The reconfigure fixture first
builds and runs against its original prefix, then changes `CMAKE_PREFIX_PATH`
and the explicit width in the same build tree. It cleans before rebuilding
so an old consumer object cannot satisfy the second configuration. CMake can
either retain the original `Sparse_DIR`, which must reject the opposite width at
compile time, or resolve the new package. A successful reconfigure passes
only if the cache names the new package and the consumer runs at its width.

## Verification And Limits

`bash tests/test_cmake_idx_width_install.sh` passed **18/18** on the final
script. The independent 32- and 64-bit installations, relocated positive
consumers, exact-version rejection, and four mismatch classes per width all
passed. `bash -n tests/test_cmake_idx_width_install.sh`, `make format-check`,
and `git diff --check` passed. Day 8 had already passed full Make lint and
test on the underlying C/header implementation; Day 9 changes only the
CMake consumer fixture and shell test.

This is a native static-package check, not a shared-library or cross-version
ABI claim. The executable guard protects the maintained consumer fixture;
arbitrary manually linked callers must still follow the published width
contract. Day 10 owns the pkg-config and wider mixed-prefix matrix.
