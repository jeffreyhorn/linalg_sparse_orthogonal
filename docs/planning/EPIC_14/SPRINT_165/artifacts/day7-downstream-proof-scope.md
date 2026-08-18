# Sprint 165 Day 7 Downstream Proof Scope Refresh

## Purpose

Day 7 refreshes the downstream proof scope for the maintained static-first
package surface before Day 8 implementation work. The goal is to make the proof
contract explicit enough that Unix, macOS, and Windows package checks are not
conflated.

## Reviewed Surfaces

| Surface | Role In Downstream Proof |
| --- | --- |
| `tests/test_install.sh` | Local Unix Make install/uninstall and `pkg-config` command-execution proof. |
| `tests/test_cmake_install.sh` | Local Unix CMake install/export and `find_package(Sparse)` proof. |
| `examples/cmake_example/CMakeLists.txt` | Maintained installed CMake consumer front door. |
| `examples/cmake_example/main.c` | Maintained downstream source used by CMake and pkg-config proof paths. |
| `.github/workflows/ci.yml` | Linux reviewed static-first package contract source of truth. |
| `.github/workflows/macos-ci.yml` | Hosted macOS reviewed execution of Unix Make/pkg-config and CMake install/export proofs. |
| `.github/workflows/windows-ci.yml` | Hosted Windows CMake-first install/downstream validation and metadata-only `sparse.pc` inspection. |
| `INSTALL.md` | Public operational install and proof interpretation owner. |
| `README.md` | Short package front-door and non-claim summary. |
| `docs/maintainer_guide.md` | Maintainer proof-owner and package-boundary policy owner. |

## Downstream Proof Requirement Table

| Requirement | Unix Make/pkg-config | Unix CMake install/export | macOS Hosted Proof | Windows Hosted Proof |
| --- | --- | --- | --- | --- |
| Static archive installed | `libsparse_lu_ortho.a` under installed `lib` | `libsparse_lu_ortho.a` under installed `lib` | Same proof scripts as Unix | `sparse_lu_ortho.lib` under installed `lib` |
| No shared artifacts installed | Rejects `.so`, `.so.*`, `.dylib`, `.dll` | Rejects `.so`, `.so.*`, `.dylib`, `.dll` | Same proof scripts as Unix | Rejects installed `.dll` artifacts |
| Public headers installed | Counts checked-in headers plus generated `sparse_version.h` | Counts checked-in headers plus generated `sparse_version.h` | Same proof scripts as Unix | Counts installed headers and checks generated `sparse_version.h` |
| CMake package metadata | Not primary owner | Checks config, version, targets, static imported target, installed include path, installed archive path, and no source/build path leaks | Same CMake script as Unix | Checks config, version, targets, static imported target, installed include path, installed `.lib`, and no source/build path leaks |
| pkg-config metadata | Checks installed `sparse.pc`, variables, cflags, libs, exact version, static archive description, no `Libs.private`, and no unsupported ABI/package wording | Checks installed `sparse.pc` version and metadata non-claims | Same Make/pkg-config script as Unix | Metadata-only inspection of installed `sparse.pc`; no `pkg-config` command execution |
| Downstream compile/link/run | Generated basic consumer and maintained example compile/link/run through `pkg-config` | Maintained example and exact-version consumer configure/build/run through `find_package(Sparse)` | Same proof scripts as Unix | Generated basic CMake consumer, maintained example, and exact-version consumer configure/build/run through installed CMake package metadata |
| Exact version handling | `pkg-config --exists "sparse = <version>"` and `--modversion` | `find_package(Sparse <version> EXACT REQUIRED)`, mismatched-version rejection, and `pkg-config --modversion` | Same proof scripts as Unix | Exact-version CMake consumer succeeds; lower same-major mismatched CMake consumer is rejected |
| Uninstall cleanup | Confirms static archive, headers, and `sparse.pc` are removed | Not primary owner | Same Make/pkg-config script as Unix | Not a supported Windows Makefile/uninstall proof |

## Platform-Specific Proof Boundaries

| Platform/Lane | Proves | Does Not Prove |
| --- | --- | --- |
| Linux reviewed package contract | Maintained static archive package contract through Make install/`pkg-config`, CMake install/export, installed downstream consumers, package metadata, and static deferral checks. | Shared-library support, dynamic ABI support, runtime-loader behavior, package-manager distribution, or broad platform parity. |
| macOS reviewed install/pkg-config and CMake install/export | Hosted macOS execution of the same maintained Unix proof scripts for the static archive package contract. | Homebrew packaging, macOS install-name/RPATH support, shared-library support, dynamic ABI support, or broad macOS package parity. |
| Windows reviewed CMake install/downstream | Installed static `.lib`, headers, CMake package metadata, metadata-only `sparse.pc` inspection, generated and maintained CMake consumers, exact-version behavior, mismatch-version rejection, and no DLL/shared metadata. | Windows Makefile install/uninstall parity, Windows `pkg-config` command execution parity, package-manager support, shared-library support, dynamic ABI support, runtime-loader behavior, or broad Windows parity. |

## Stale Expectation Register

| Risk | Failure Mode | Required Handling |
| --- | --- | --- |
| Raw path spelling comparisons | Temp paths can include spelling differences such as double slashes; direct string comparison can fail even when paths identify the same directory. | Use filesystem-equivalence checks for installed include and lib paths where possible. |
| Output shape assumptions | Maintained examples may print valid multi-line output; matching a single raw string can fail even when required tokens are present. | Match required semantic tokens such as version, solution, `nnz`, and `OK` after normalizing captured output. |
| Hard-coded Windows CTest count | Adding or removing reviewed CMake tests changes `ctest -N` total count and can fail the Windows inspected-surface gate. | Update `EXPECTED_WINDOWS_CTEST_COUNT` with every reviewed Windows CMake test-surface change and record the reason. |
| Header count drift | Generated `sparse_version.h` is installed in addition to checked-in public headers. | Derive expected Unix header counts from checked-in headers plus one generated version header; keep Windows count aligned with installed public headers. |
| Windows `sparse.pc` wording | Windows installs `sparse.pc` and inspects it as metadata, but does not execute `pkg-config`. | Keep Windows proof descriptions explicit: metadata-only `sparse.pc` inspection is not Windows `pkg-config` execution parity. |
| Exact-version metadata interpretation | Exact package version checks can be misread as a dynamic ABI policy. | Continue stating that exact package metadata is package resolution evidence, not a dynamic ABI guarantee. |

## Day 8 Implementation Handoff

Day 8 should only edit proof scripts or workflow expectations if it closes one
of these concrete drift risks:

1. A script uses raw path-string equality where filesystem-equivalence is
   required.
2. A script assumes one-line example output instead of semantic output tokens.
3. A Windows CTest count is stale relative to the reviewed CMake test surface.
4. A package proof omits a maintained static-first requirement listed above.
5. A proof description conflates Windows metadata-only `sparse.pc` inspection
   with Windows `pkg-config` command execution.

Avoid broad rewrites, new package-manager claims, shared-library selectors,
runtime-loader metadata, or dynamic ABI language unless a later product decision
selects those surfaces.

## Validation

Day 7 reviewed and documented proof scope. No `.c` or `.h` files were changed
for Day 7.

Focused proof-owner validation was run:

```text
bash tests/test_install.sh
```

Result:

```text
Passed: 23
Failed: 0
ALL INSTALL TESTS PASSED
```

```text
bash tests/test_cmake_install.sh
```

Result:

```text
Passed: 27
Failed: 0
Skipped: 0
ALL CMAKE INSTALL TESTS PASSED
```

## Completion Check

- Downstream proof scope is explicit before Day 8 edits.
- Unix `pkg-config`, Unix CMake, macOS hosted proof, Windows CMake-first proof,
  and Windows metadata-only `sparse.pc` inspection are separated.
- Deferred Windows Makefile and `pkg-config` execution parity remain
  documented.
- Known stale-expectation risks are captured for implementation follow-up.
