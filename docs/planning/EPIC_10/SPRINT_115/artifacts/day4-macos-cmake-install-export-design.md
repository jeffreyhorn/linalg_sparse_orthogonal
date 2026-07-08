# Day 4 macOS CMake Install and Export Parity Design

## Purpose

Day 4 designs the proof needed before claiming reviewed macOS CMake
install/export parity. It distinguishes the current reviewed Apple Clang path,
supplemental Homebrew GCC path, supplemental Make install/`pkg-config` path,
and local CMake install/export proof.

No workflow, documentation, build metadata, or package claim changes are made
on Day 4.

## Current macOS CI Surface

`.github/workflows/macos-ci.yml` currently provides:

| Lane | Current role |
|---|---|
| Apple Clang matrix leg | Reviewed macOS path: `make quality-review-compile`, `make quality-review-cmake`, `make wall-check`, and `make sanitize`. |
| Homebrew GCC matrix leg | Supplemental second-compiler direct build/test/wall-check coverage. |
| `install-and-pkgconfig` job | Supplemental static-first Make install/`pkg-config` confidence path. |

The workflow comments explicitly state that the Make install/`pkg-config` job
does not claim reviewed macOS install/export parity and does not replace the
broader local install regression scripts.

## Existing Local CMake Install/Export Proof

`tests/test_cmake_install.sh` already proves the local Unix-side CMake package
surface:

| Proof | Required behavior |
|---|---|
| configure/build/install | CMake configures, builds, and installs into a temp prefix. |
| package shape | installed output contains the static archive and public headers. |
| no shared artifacts | installed lib/bin paths do not contain `.so`, `.so.*`, `.dylib`, or `.dll`. |
| package config files | `SparseConfig.cmake`, `SparseConfigVersion.cmake`, and `SparseTargets.cmake` are installed. |
| installed consumer | `examples/cmake_example` configures with `find_package(Sparse)`, builds, links, and runs. |
| version contract | exact-version `find_package` succeeds and lower same-major mismatches are rejected when applicable. |
| `pkg-config` consistency | installed `pkg-config --modversion sparse` matches `VERSION`. |

Sprint 112 recorded this proof as local macOS evidence, not as reviewed macOS
install/export parity.

## Reviewed-Lane Requirements

To claim reviewed macOS CMake install/export parity, Day 5 would need a
reviewed macOS CI lane that proves at least:

1. `bash tests/test_cmake_install.sh` or equivalent install/export proof runs
   on `macos-latest`.
2. CMake configure/build/install uses an explicit temp prefix and does not
   depend on global package state.
3. Installed static archive, headers, `SparseConfig*.cmake`,
   `SparseTargets.cmake`, and `sparse.pc` are present.
4. Installed `find_package(Sparse)` consumer configures, builds, links, and
   runs.
5. Exact-version package behavior is checked.
6. Shared-library artifacts remain absent.
7. The workflow and docs name the lane as static-first CMake install/export
   proof, not dylib, dynamic ABI, package-manager, or runtime-loader support.
8. Runtime and Homebrew/toolchain assumptions are acceptable for reviewed PR
   execution.

## Toolchain and Environment Considerations

| Area | Current state | Design implication |
|---|---|---|
| compiler | Apple Clang reviewed path is already enforced. | CMake install/export proof should run under Apple Clang if promoted. |
| CMake | CMake is already used by `make quality-review-cmake`. | Existing CMake availability is likely sufficient, but install/export proof adds downstream consumer configuration. |
| SDK | `macos-latest` SDK can float. | Keep the proof static-first and avoid SDK-specific package claims. |
| Homebrew | Homebrew GCC/libomp are supplemental. | CMake install/export proof should not depend on Homebrew GCC. |
| install prefix | Local script uses a temp prefix. | Reviewed lane should also use a temp prefix and no global install. |
| runtime loader | Static package only. | No dylib/install-name/rpath claim should be introduced. |

## Deferral Criteria

Day 5 should defer reviewed macOS CMake install/export parity if any of these
hold:

1. The existing reviewed Apple Clang CMake parity path plus supplemental
   Make install/`pkg-config` path is enough for current public wording.
2. Adding `tests/test_cmake_install.sh` to macOS CI would duplicate local proof
   without changing a needed claim.
3. The new lane would increase CI runtime or platform ownership beyond the
   Sprint 115 support-truth goal.
4. The project is not ready to claim full reviewed macOS install/export parity
   across CMake package files and installed consumers.
5. The lane could be misread as shared-library, dynamic ABI, package-manager,
   or runtime-loader support.

## Support Claims to Fence Until Proof Lands

- Full reviewed macOS CMake install/export parity.
- macOS package-manager support.
- macOS shared-library or dylib support.
- Dynamic ABI compatibility.
- Runtime-loader, install-name, rpath, or symbol visibility behavior.
- Claim that local CMake install/export proof replaces reviewed CI evidence.
- Claim that Homebrew GCC supplemental coverage owns the package/install
  surface.

## Day 5 Decision Checklist

Before changing macOS CI on Day 5, answer:

1. Does reviewed macOS CMake install/export parity materially improve public
   support truth beyond the existing supplemental Make install path?
2. Can the lane run under Apple Clang on `macos-latest` with stable temp-prefix
   behavior?
3. Is the claim narrow enough to remain static-first CMake install/export
   proof?
4. Are workflow comments, README, INSTALL, and maintainer-guide wording ready
   to describe the promoted or deferred status?
5. Is the validation command set clear for all touched files?

## Non-Claims

- No macOS workflow was changed on Day 4.
- No macOS reviewed CMake install/export parity claim was added on Day 4.
- No package-manager, shared-library, dylib, dynamic ABI, or runtime-loader
  support claim changed on Day 4.
- No Linux or Windows install-validation scope is mixed into this macOS
  decision.

## Day 4 Validation

Day 4 changes documentation only. Required validation:

```text
git diff --check
rg -n '[ \t]+$' docs/planning/EPIC_10/SPRINT_115
```

Full C quality gates are not required for Day 4 because no `.c` or `.h` files
changed.
