# Day 10 Windows Follow-Through

## Purpose

Day 10 reviews the Windows validation surface after the Day 9 platform-tier
contract. The goal is to decide whether any Windows exclusions can be promoted
into reviewed truth during Sprint 112, or whether they should remain explicit
staged exclusions with bounded non-claims.

## Reviewed Windows Surface

| Surface | Evidence | Day 10 status |
|---|---|---|
| Runner image | `.github/workflows/windows-ci.yml` pins `windows-2022`. | Reviewed lane remains pinned because it depends on the VS 2022 generator. |
| Configure | `cmake -S . -B build -G "Visual Studio 17 2022" -A x64`. | Reviewed. |
| Build | `cmake --build build --config Release`. | Reviewed. |
| CTest registration | `ctest --test-dir build -C Release -N` with `EXPECTED_WINDOWS_CTEST_COUNT=51`. | Reviewed subset guard. |
| Test execution | `ctest --test-dir build -C Release --output-on-failure`. | Reviewed for registered Windows CMake tests. |
| Consumer interpretation | Workflow comments and output say CMake-first consumer proof only. | Reviewed wording is bounded and current. |

## CMake Registration Review

| CMake rule | Windows effect | Interpretation |
|---|---|---|
| `if(Threads_FOUND AND NOT WIN32)` around `test_threads` and `test_sprint4_integration` | Excludes pthread-based tests on Windows. | Correct staged exclusion because the sources include pthread APIs directly. |
| `if(NOT WIN32 AND NOT MSVC)` around `test_fuzz` | Excludes the fuzz/property binary from Windows. | Correct staged exclusion; do not claim bounded lifecycle property/fuzz evidence on Windows. |
| Regular `add_sparse_test(...)` calls outside those gates | Registered in the Windows CMake subset. | Count currently guarded at 51 by workflow output. |
| POSIX-only benchmark gate `if(NOT WIN32)` | Excludes nonportable benchmark binaries from Windows CMake build. | Benchmark parity is not a Windows reviewed claim. |
| Static target `add_library(sparse_lu_ortho STATIC ...)` | Builds a static library surface under MSVC. | Supports the static-first package boundary, not shared-library or ABI claims. |

## Staged-Exclusion Decisions

| Exclusion | Promote in Sprint 112? | Reason |
|---|---:|---|
| `test_threads` | No | Source uses pthread APIs directly; promoting requires a Windows thread-test implementation or portability layer. |
| `test_sprint4_integration` | No | Coupled to the same pthread-dependent test path. |
| `test_fuzz` | No | Remains gated off for Windows/MSVC; bounded lifecycle property/fuzz evidence stays Linux/macOS-side only. |
| Makefile reviewed wrappers | No | Makefile install and reviewed wrappers are Unix-oriented; Windows maintained path is CMake. |
| Dead-code flow | No | Current dead-code tooling and workflow ownership are Linux-side. |
| Separate Windows install validation | No | No reviewed `cmake --install` plus downstream package-consumer lane exists on Windows. |
| Shared-library or DLL/import-library behavior | No | Sprint 112 selected static-first support and no shared artifact proof exists. |
| Package-manager support | No | No vcpkg, Chocolatey, winget, or other package-manager proof exists. |

## Windows Non-Claims

- Windows support remains a reviewed MSVC CMake-first consumer subset.
- Windows does not claim Makefile parity.
- Windows does not claim Unix install-script parity.
- Windows does not claim a separate reviewed install-validation lane.
- Windows does not claim dead-code parity.
- Windows does not claim benchmark parity.
- Windows does not claim `test_threads`, `test_sprint4_integration`, or
  `test_fuzz` coverage.
- Windows does not claim bounded lifecycle property/fuzz evidence.
- Windows does not claim shared-library, dynamic ABI, DLL/import-library, or
  runtime-loader behavior.
- Windows does not claim package-manager support.

## Documentation and Workflow Assessment

| Surface | Assessment | Change needed on Day 10? |
|---|---|---:|
| `.github/workflows/windows-ci.yml` header comments | Already describe reviewed CMake subset, staged Makefile/dead-code paths, excluded tests, and no install-validation claim. | No |
| `.github/workflows/windows-ci.yml` job output | Already prints staged exclusions and reviewed-scope boundary. | No |
| `INSTALL.md` Windows section | Already directs Windows users to CMake and says Makefile targets are Unix-only. | No |
| `INSTALL.md` platform table | Already says Windows is reviewed CMake subset only and not separate install validation. | No |
| `README.md` compact CI summary | Already says Windows enforces reviewed CMake subset and CMake-first consumer story. | No |
| `docs/maintainer_guide.md` package/platform ownership | Already records Windows reviewed CMake subset and no separate install-validation lane. | No |

## Decision

No Windows exclusion should move into reviewed parity on Day 10. The current
workflow and documentation already expose the reviewed subset and staged
exclusions clearly. Promoting any excluded lane would require implementation
or CI evidence that does not exist in Sprint 112 so far.

## Residual Windows Queue

- Add a Windows-native thread-test owner before considering `test_threads` or
  `test_sprint4_integration` promotion.
- Add explicit Windows install validation only as a separate reviewed lane with
  `cmake --install`, installed target lookup, and downstream consumer
  compile/link/run proof.
- Add Windows fuzz/property coverage only after the fuzz binary is made
  portable and reviewed under MSVC.
- Add shared-library or DLL/import-library validation only if a future sprint
  changes the static-first support decision.

## Completion Criteria

- Windows reviewed coverage is explicit and evidence-bound.
- Staged exclusions remain visible and are not silently converted into support.
- No Windows support claim exceeds the reviewed checks in
  `.github/workflows/windows-ci.yml`.
