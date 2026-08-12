# Sprint 153 Day 8 Downstream Proof Design

## Purpose

Day 8 designs downstream proof for the Day 5 static-first product decision.
The goal is to prove the maintained installed consumer paths and unsupported
shared-loader boundaries without implying shared-library or dynamic ABI
support.

## Existing Downstream Proof Inventory

| Surface | Existing Proof | Current Claim |
| --- | --- | --- |
| Unix Make install | `bash tests/test_install.sh` | Installs static archive, headers, generated version header, and `sparse.pc`; verifies no shared artifacts. |
| Unix `pkg-config` metadata | `tests/test_install.sh` | Verifies prefix/libdir/includedir, exact version, static link flags, no `Libs.private`, static archive description, and no unsupported wording. |
| Unix `pkg-config` consumer | `tests/test_install.sh` | Compiles and runs a generated basic consumer and maintained example source against installed static package metadata. |
| Unix CMake install/export | `bash tests/test_cmake_install.sh` | Installs static archive, headers, generated version header, CMake package files, and `sparse.pc`; verifies no shared artifacts. |
| Unix CMake package metadata | `tests/test_cmake_install.sh` | Verifies static imported target, installed-prefix include/archive paths, no source/build path leaks, no shared imported metadata, exact version behavior, and mismatch rejection. |
| Unix CMake consumer | `tests/test_cmake_install.sh` | Builds and runs maintained `examples/cmake_example` plus an exact-version consumer against the installed static package. |
| Windows CMake install/downstream | `.github/workflows/windows-ci.yml` | Builds and installs a static `.lib`, verifies CMake/package metadata, rejects DLL/shared metadata, and runs generated and maintained installed CMake consumers. |
| Static deferral guard | `bash scripts/static_package_deferral_check.sh` | Proves `BUILD_SHARED_LIBS=ON` rejection, explicit static target, no shared install destinations, no export/import macro, no shared ABI metadata, no package selectors, and deferred wording. |

## CMake Downstream Proof Design

The selected CMake proof should remain installed-prefix and static-target
focused.

Day 9 should preserve these checks:

- configure, build, and install from a temporary build tree;
- verify installed static archive path;
- verify no `.so`, `.so.*`, `.dylib`, or `.dll` artifacts;
- verify exact installed header count;
- verify `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and
  `SparseTargets.cmake` exist;
- verify `Sparse::sparse_lu_ortho` is `STATIC IMPORTED`;
- verify imported include and archive paths use the install prefix, not source
  or build paths;
- verify no shared imported metadata appears;
- configure, build, and run maintained `examples/cmake_example` through
  `find_package(Sparse)`;
- configure, build, and run an exact-version installed consumer;
- verify mismatched-version configure fails.

### Day 9 CMake Enhancement Candidate

Add a focused unsupported-loader metadata check to
`tests/test_cmake_install.sh` if not already present in enough detail:

- scan installed CMake package files for shared selectors and loader metadata
  tokens such as `SOVERSION`, `IMPORTED_SONAME`, `INSTALL_NAME`,
  `MACOSX_RPATH`, `IMPORTED_IMPLIB`, and `RUNTIME`;
- fail with wording that names the static-first product decision if any token
  appears before a shared support decision.

This is a proof of absence for unsupported loader metadata, not a dynamic
loader test.

## `pkg-config` Downstream Proof Design

The selected `pkg-config` proof should remain Unix-side static archive proof.

Day 9 should preserve these checks:

- `pkg-config --exists sparse`;
- `pkg-config --exists "sparse = $EXPECTED_VERSION"`;
- `prefix`, `libdir`, and `includedir` resolve to the install prefix;
- `--cflags` resolves to the installed include path;
- `--libs` returns installed static archive link flags;
- `--static --libs` matches current self-contained link flags;
- no `Libs.private` stanza appears under the current package decision;
- static archive description is exact;
- unsupported shared-library, dynamic ABI, package-manager, and selector
  wording remains absent;
- generated and maintained consumers compile, link, and run.

### Day 9 `pkg-config` Enhancement Candidate

No `sparse.pc.in` behavior change is needed for Day 9. If a proof update is
made, it should only improve diagnostics:

- failure wording should say the current package is static-first;
- failure wording should direct shared-library requests to the exact blockers
  in `BUILD_SHARED_LIBS=ON` diagnostics rather than implying a supported
  selector.

## Loader Or Unsupported-Loader Proof Design

Because the selected product decision defers shared-library support, Day 9
should implement unsupported-loader proof instead of loader execution proof.

| Platform | Supported Day 9 Proof | Explicit Non-Claim |
| --- | --- | --- |
| Linux | Static install proofs reject `.so` and shared CMake metadata; deferral guard rejects `BUILD_SHARED_LIBS=ON` with exact blockers. | No `.so`, SONAME, shared consumer, or runtime loader support. |
| macOS | Static install proofs reject `.dylib`; macOS CI runs Make/pkg-config and CMake install/export static package proof. | No `.dylib`, install-name, RPATH, shared consumer, or runtime loader support. |
| Windows | Windows CMake install/downstream lane rejects `.dll` artifacts and shared imported metadata while running installed static CMake consumers. | No DLL/import-library, runtime lookup, Windows Makefile parity, or Windows `pkg-config` execution parity. |

## Exact Output And Failure Modes

Day 9 proof updates should fail with product-decision wording:

- If a shared artifact appears: fail with "unexpected shared-library artifact
  under the static-first package decision".
- If CMake metadata imports a shared target: fail with "shared imported target
  metadata appeared before shared-library support was selected".
- If CMake loader metadata appears: fail with "loader metadata appeared before
  runtime-loader support was selected".
- If `sparse.pc` gains shared/static selectors: fail with "package selector
  appeared before shared/static package semantics were selected".
- If a downstream consumer links from a source/build path: fail with "installed
  consumer proof leaked source/build path".

## Day 9 Implementation Checklist

1. Inspect current `tests/test_install.sh`, `tests/test_cmake_install.sh`, and
   `scripts/static_package_deferral_check.sh` for any missing unsupported-loader
   metadata checks.
2. Prefer `tests/test_cmake_install.sh` for installed CMake package metadata
   assertions.
3. Add a CMake package metadata scan for unsupported loader/static-shared
   selector tokens if needed.
4. Keep `tests/test_install.sh` behavior unchanged unless diagnostics need a
   narrow static-first wording improvement.
5. Keep Windows CI as CMake-first static install/downstream proof; do not add a
   Windows `pkg-config` execution claim.
6. Run focused validation:
   - `bash scripts/static_package_deferral_check.sh`;
   - `bash tests/test_install.sh`;
   - `bash tests/test_cmake_install.sh`;
   - `git diff --check`.
7. Run the full C source gate only if C or public header files are modified.
