# Sprint 188 Day 6: Install Surface Proof Hardening

## Purpose

Harden the Homebrew local proof install-surface checks so a future successful
formula install proves only the maintained static archive package surface and
rejects unsupported shared-library, provider, selector, or ABI metadata.

## Changes Made

| Surface | Change |
| --- | --- |
| `scripts/homebrew_local_formula_proof.sh` | Verifies the Homebrew prefix and installed `lib` directory before checking installed files. |
| `scripts/homebrew_local_formula_proof.sh` | Keeps existing checks for `libsparse_lu_ortho.a`, installed headers, CMake package files, target metadata, and `sparse.pc`. |
| `scripts/homebrew_local_formula_proof.sh` | Adds installed package metadata scanning for unsupported provider, shared-library, selector, SONAME/DLL/dylib, `Libs.private`, and dynamic ABI wording. |
| `scripts/package_manager_deferral_check.sh` | Guards that installed metadata rejection remains present in the proof script. |
| `packaging/homebrew/README.md` | Documents installed static package metadata expectations and rejected installed metadata surfaces. |

## Installed Static Package Checklist

Future successful Homebrew proof installs must contain:

- Homebrew prefix for the temporary formula;
- installed `lib` directory;
- `lib/libsparse_lu_ortho.a`;
- installed sparse public headers;
- `lib/cmake/Sparse/SparseConfig.cmake`;
- `lib/cmake/Sparse/SparseConfigVersion.cmake`;
- `lib/cmake/Sparse/SparseTargets.cmake`;
- `lib/cmake/Sparse/SparseTargets-noconfig.cmake`;
- `lib/pkgconfig/sparse.pc`;
- `Sparse::sparse_lu_ortho STATIC IMPORTED` target metadata; and
- target metadata pointing at `${_IMPORT_PREFIX}/lib/libsparse_lu_ortho.a`.

## Rejected Installed Surfaces

The proof must fail if installed metadata or installed files introduce:

- Homebrew/provider availability wording in installed package metadata;
- vcpkg, Conan, pkgsrc, apt, dnf, pacman, registry-ready, binary package, or
  package-manager support wording;
- `Libs.private` package metadata;
- shared-library selectors;
- `BUILD_SHARED_LIBS` selector behavior;
- `SPARSE_ABI`, `SPARSE_SHARED`, or `SPARSE_STATIC` metadata knobs;
- dynamic ABI wording;
- SONAME, DLL, or dylib policy;
- `Sparse::*shared` targets; or
- `.dylib`, `.so`, `.so.*`, or `.dll` installed artifacts.

## Cleanup and Retry Notes

The proof still sets `UNINSTALL_ON_EXIT=1` before the Homebrew install step.
If install, installed-surface validation, or `brew test` fails after the
formula is installed, the exit trap attempts `brew uninstall --force` for the
temporary formula. Temporary proof roots are removed by default unless
`--keep-temp` is selected for diagnostics.

Because Day 5 moved license metadata validation before archive creation, the
current missing-license blocker exits before temporary archives or install
attempts are created.

## Validation Results

| Command | Result | Interpretation |
| --- | --- | --- |
| `bash -n scripts/homebrew_local_formula_proof.sh scripts/package_manager_deferral_check.sh scripts/static_package_deferral_check.sh` | Passed | Changed shell scripts parse successfully. |
| `scripts/homebrew_local_formula_proof.sh` | Expected exit `2` | Proof remains unavailable because no standalone root license metadata exists; it stops before archive/install work. |
| `scripts/package_manager_deferral_check.sh` | Passed | Package-manager non-claims, Homebrew boundary, placeholder-license rejection, archive verification, and installed metadata rejection remain guarded. |
| `scripts/static_package_deferral_check.sh` | Passed | Static-first package contract and shared-library/dynamic ABI deferrals remain guarded. |

## Day 7 Handoff

Day 7 can focus on the formula `test do` downstream consumer proof:

1. keep exact-version `find_package(Sparse ...)`;
2. keep `Sparse::sparse_lu_ortho` as the linked target;
3. prove installed headers are sufficient for a minimal consumer;
4. keep post-test static artifact and shared-artifact checks; and
5. ensure `brew test` failure blocks package support promotion.

## Validation Scope

Day 6 changed shell scripts and documentation but no `.c` or `.h` files, so
the full C quality gate is not required.
